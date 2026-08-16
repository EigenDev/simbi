// =============================================================================
// evolve_runtime_source_gpu.rs
//
// GPU validation of the runtime user-source path: a runtime-loaded user source (python -> json ->
// SourceConfig, no recompile) must run on the device via NVRTC and match the CPU per-cell
// interpreter. proves the
// runtime DAG -> `source_apply_from_built_gv` -> neutral IR -> render(Cuda) -> nvrtc-jit path closes
// on-device, the device twin of the CPU `evolve_runtime_source.rs`.
//
// the test isolates the source kernel (no multi-step evolve, so no FMA-induced dt drift): build two
// identical adiabatic sims (host = CpuSpace/HostMemory, device = CudaSpace/UnifiedMemory) with a
// smooth nonzero state, attach the same runtime source on both, then `snapshot_stage` (fill u_stage)
// + one `source_apply` at a fixed weight, and diff every conserved field. relative tolerance < 1e-9
// per the FMA-fusion budget (`project_fma_discipline`).
//
// two source shapes:
//   * position-independent force `a = [p0, 0..]` — stresses the scalar (p{i}) manifest path.
//   * position-dependent force `a = [x_0 * p0, 0..]` — the harder case: the device kernel resolves
//     `x_0` from the in-kernel centroid (`x_lo + i*dx`), the CPU pass from `cell_coord`; they must
//     agree, validating the runtime-built `cell_geometry_gv` centroid arithmetic under NVRTC.
//
// runs only with --features cuda; needs a CUDA GPU (rtx 2070 canonical env, NVCC_CCBIN=g++-15).
// run: cargo test -p symbi --features cuda --test evolve_runtime_source_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::{KernelSet, evolve};
use symbi::sim::state::*;
use symbi_algebra::{Domain, Tensor};
use symbi_geometry::{Cartesian, Metric};
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_user_source;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{NEWTONIAN_SPEC, SourceConfig};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const N: usize = 8;
const CFL: f64 = 0.4;
const GAMMA: f64 = 1.4;
const WEIGHT: f64 = 0.01; // the SSP stage weight ac*dt the driver passes; fixed here.

// per-cell GPU-vs-CPU diff in relative units. ULP-bounded modulo nvcc FMA. one ctx_sync before the
// host read (UnifiedMemory isn't ordered against pending device kernels by stream semantics alone).
fn cmp<const D: usize, MH: MemorySpace, MD: MemorySpace>(
    dom: &Domain<D>,
    host: &Field<f64, D, MH>,
    dev: &Field<f64, D, MD>,
    what: &str,
) {
    symbi_xpu::cuda::ctx_sync();
    for c in dom.iter() {
        let (h, g) = (*host.view().at(c), *dev.view().at(c));
        assert!(g.is_finite(), "{what} at {c:?} went non-finite on GPU: {g}");
        let rel = (g - h).abs() / h.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "{what} at {c:?}: gpu {g} != cpu {h} (rel {rel:e})"
        );
    }
}

// adiabatic Newton sim with a smooth nonzero state (nonzero velocity so the energy source v.g is
// also exercised). identical IC across backends. (mirrors substrate_fused_source_gpu.rs.)
fn build_adiabatic<S: ExecutionSpace, Mem: MemorySpace, const D: usize>()
-> SimState<Newtonian, D, Cartesian, IdealGas<f64>, S, Mem>
where
    Cartesian: Metric<f64, D>,
{
    let dx = 1.0 / N as f64;
    SimState::<Newtonian, D, Cartesian, IdealGas<f64>, S, Mem>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N; D])
    .spacing([dx; D])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("adiabatic sim construction failed")
    .set_initial(|x| {
        let r2: f64 = (0..D).map(|k| (x[k] - 0.5).powi(2)).sum();
        let rho = 1.0 + 0.3 * (-r2 / 0.05).exp();
        let nrg = (1.0 + 2.0 * (-r2 / 0.02).exp()) / (GAMMA - 1.0);
        let vel: [f64; D] = std::array::from_fn(|k| 0.02 * (k as f64 + 1.0) / rho);
        let vsq: f64 = (0..D).map(|k| vel[k] * vel[k]).sum();
        let pre = (GAMMA - 1.0) * (nrg - 0.5 * rho * vsq);
        Prim {
            rho,
            vel: Tensor::new(vel),
            pre,
        }
    })
    .build()
}

// position-independent force `a = [p0, 0, 0..]`: nodes [parameter p0, constant 0]; output 0 -> a_0,
// the rest -> a_k = 0.
fn force_const_json(dim: usize) -> String {
    let outputs: Vec<usize> = std::iter::once(0)
        .chain(std::iter::repeat(1).take(dim - 1))
        .collect();
    format!(
        r#"{{ "kind": "force", "dim": {dim}, "outputs": {outputs:?}, "params": [0.5],
            "nodes": [ {{"op": "PARAMETER", "param_idx": 0}}, {{"op": "CONSTANT", "value": 0.0}} ] }}"#,
    )
}

// position-dependent force `a = [x_0 * p0, 0, 0..]`: nodes [VARIABLE_X1, parameter p0, multiply,
// constant 0]; output 2 -> a_0 = x_0*p0, the rest -> a_k = 0.
fn force_posdep_json(dim: usize) -> String {
    let outputs: Vec<usize> = std::iter::once(2)
        .chain(std::iter::repeat(3).take(dim - 1))
        .collect();
    format!(
        r#"{{ "kind": "force", "dim": {dim}, "outputs": {outputs:?}, "params": [0.5],
            "nodes": [ {{"op": "VARIABLE_X1"}}, {{"op": "PARAMETER", "param_idx": 0}},
                       {{"op": "MULTIPLY", "left": 0, "right": 1}}, {{"op": "CONSTANT", "value": 0.0}} ] }}"#,
    )
}

// build two identical sims, attach the same runtime source, snapshot_stage + one source_apply, diff.
fn check_runtime_source<const D: usize>(json: &str)
where
    Cartesian: Metric<f64, D>,
{
    let cfg = SourceConfig::from_json(json).expect("parse config");
    let host = build_adiabatic::<CpuSpace, HostMemory, D>();
    let dev = build_adiabatic::<CudaSpace, UnifiedMemory, D>();
    // each kernel-set owns its own RuntimeSource (the lazy gpu-ir OnceLock is per-instance), so
    // build the source twice from the same config.
    let hset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, D>::new(GAMMA, CFL, &host.geom.allocated)
            .with_runtime_source(
                build_user_source(&cfg, &NEWTONIAN_SPEC).expect("wrap"),
                cfg.params.clone(),
            );
    let dset =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, D>::new(GAMMA, CFL, &dev.geom.allocated)
            .with_runtime_source(
                build_user_source(&cfg, &NEWTONIAN_SPEC).expect("wrap"),
                cfg.params.clone(),
            );
    let interior = &host.geom.interior;

    // fill u_stage (the stage-input snapshot the source reads), then apply the source once.
    hset.snapshot_stage(&host);
    dset.snapshot_stage(&dev);
    hset.source_apply(&host, WEIGHT);
    dset.source_apply(&dev, WEIGHT);

    cmp(
        interior,
        &host.fields.cons.den,
        &dev.fields.cons.den,
        "runtime src cons.den",
    );
    for k in 0..D {
        cmp(
            interior,
            &host.fields.cons.mom[k],
            &dev.fields.cons.mom[k],
            &format!("runtime src cons.mom_{k}"),
        );
    }
    let (hnrg, dnrg) = (
        host.fields.cons.nrg_field().unwrap(),
        dev.fields.cons.nrg_field().unwrap(),
    );
    cmp(interior, hnrg, dnrg, "runtime src cons.nrg");
}

#[test]
fn runtime_force_const_gpu_1d() {
    check_runtime_source::<1>(&force_const_json(1));
}
#[test]
fn runtime_force_const_gpu_2d() {
    check_runtime_source::<2>(&force_const_json(2));
}
#[test]
fn runtime_force_const_gpu_3d() {
    check_runtime_source::<3>(&force_const_json(3));
}

#[test]
fn runtime_force_posdep_gpu_1d() {
    check_runtime_source::<1>(&force_posdep_json(1));
}
#[test]
fn runtime_force_posdep_gpu_2d() {
    check_runtime_source::<2>(&force_posdep_json(2));
}
#[test]
fn runtime_force_posdep_gpu_3d() {
    check_runtime_source::<3>(&force_posdep_json(3));
}

// region (IF_THEN_ELSE -> Select) and relax (Max clamp) introduce carrier-dialect ops that must
// trace + render on device without panicking at trace time. force a = [p0, 0] masked chi = (x_0 < 0.5).
const REGION_JSON_2D: &str = r#"{
    "kind": "force", "dim": 2, "outputs": [0, 1], "region": 6, "params": [0.5],
    "nodes": [
        {"op": "PARAMETER", "param_idx": 0}, {"op": "CONSTANT", "value": 0.0},
        {"op": "VARIABLE_X1"}, {"op": "CONSTANT", "value": 0.5},
        {"op": "LT", "left": 2, "right": 3}, {"op": "CONSTANT", "value": 1.0},
        {"op": "IF_THEN_ELSE", "condition": 4, "true_case": 5, "false_case": 1}
    ]
}"#;

// relax velocity toward v_ref = 0, rate kappa = p0 = 2. outputs = [kappa, v_ref_0, v_ref_1].
const RELAX_JSON_2D: &str = r#"{
    "kind": "relax", "dim": 2, "outputs": [0, 1, 1], "params": [2.0],
    "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "CONSTANT", "value": 0.0} ]
}"#;

#[test]
fn runtime_force_region_gpu_2d() {
    check_runtime_source::<2>(REGION_JSON_2D);
}
#[test]
fn runtime_relax_gpu_2d() {
    check_runtime_source::<2>(RELAX_JSON_2D);
}

// =============================================================================
// the runtime-source evolve oracle across the device boundary.
//
// the device twin of `jit_fused_equals_two_pass.rs` (which pins the three CPU
// engines — interp / pointwise-JIT / fused-JIT — bit-for-bit). the 4th engine is
// GPU NVRTC (`apply_runtime_source_gpu` -> `build_runtime_source_ir` ->
// `prepared_to_ir`); the single-`source_apply` tests pin it at one stage, and
// this drives the runtime source inside the production `evolve()` loop
// on-device — the multi-step interaction of cfl -> flux -> godunov -> per-stage
// runtime `source_apply` -> c2p -> ghost, driven on the GPU and tracked against
// the CPU.
//
// tolerance is rel < 1e-6 (not bit-for-bit): unlike the CPU-vs-CPU twin, this
// crosses the device boundary, so nvcc FMA fusion in the godunov flux + the c2p
// compounds per RK2 step (`project_fma_discipline`). the anti-vacuous guard is
// structural: if the GPU silently skipped the NVRTC source, its trajectory would
// diverge from the CPU (which applied the source) and the rel-close assert would
// fire — a passing run means the device source ran and tracked the host.
// =============================================================================

// build host + device adiabatic sims with the same runtime source, run the full
// evolve() loop on each to t_final, and assert the conserved state tracks (rel <
// 1e-6, NaN-free) with matching step counts.
fn check_runtime_source_evolve<const D: usize>(json: &str)
where
    Cartesian: Metric<f64, D>,
{
    let cfg = SourceConfig::from_json(json).expect("parse config");
    let mut host = build_adiabatic::<CpuSpace, HostMemory, D>();
    let mut dev = build_adiabatic::<CudaSpace, UnifiedMemory, D>();
    let hset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, D>::new(GAMMA, CFL, &host.geom.allocated)
            .with_runtime_source(
                build_user_source(&cfg, &NEWTONIAN_SPEC).expect("wrap"),
                cfg.params.clone(),
            );
    let dset =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, D>::new(GAMMA, CFL, &dev.geom.allocated)
            .with_runtime_source(
                build_user_source(&cfg, &NEWTONIAN_SPEC).expect("wrap"),
                cfg.params.clone(),
            );

    // capture a pre-evolve momentum sample to prove the run actually moved
    // the gas (else the test would compare two static states and pass vacuously).
    let probe = host.geom.interior.iter().next().expect("nonempty interior");
    let mom0_before = *host.fields.cons.mom[0].view().at(probe);

    // short smoke: a handful of RK2 steps; dt is cfl-clamped to land both backends
    // exactly on t_final, so they take the same step count.
    let t_final = 0.2_f64;
    evolve(&mut host, &hset, t_final).expect("cpu evolve");
    evolve(&mut dev, &dset, t_final).expect("gpu evolve");
    symbi_xpu::cuda::ctx_sync(); // host-read barrier: the final step's c2p/ghost run async.

    assert!(
        host.iteration >= 3,
        "too few steps ({}) — smoke would be vacuous",
        host.iteration
    );
    assert_eq!(
        host.iteration, dev.iteration,
        "step count diverged: cpu {} vs gpu {}",
        host.iteration, dev.iteration,
    );

    let mom0_after = *host.fields.cons.mom[0].view().at(probe);
    assert!(
        (mom0_after - mom0_before).abs() > 1e-9,
        "gas never evolved (mom_0 {mom0_before} -> {mom0_after}) — the test exercised nothing",
    );

    // ~N steps of RK2 compound FMA drift across the boundary -> rel < 1e-6.
    let evolve_close = |g: f64, c: f64, what: &str| {
        assert!(g.is_finite(), "{what} went non-finite on GPU: {g}");
        let rel = (g - c).abs() / c.abs().max(1.0);
        assert!(rel < 1e-6, "{what}: gpu {g} != cpu {c} (rel {rel:e})");
    };
    let interior = &host.geom.interior;
    for coord in interior.iter() {
        evolve_close(
            *dev.fields.cons.den.view().at(coord),
            *host.fields.cons.den.view().at(coord),
            "evolve cons.den",
        );
        for k in 0..D {
            evolve_close(
                *dev.fields.cons.mom[k].view().at(coord),
                *host.fields.cons.mom[k].view().at(coord),
                "evolve cons.mom",
            );
        }
        let (hnrg, dnrg) = (
            host.fields.cons.nrg_field().unwrap(),
            dev.fields.cons.nrg_field().unwrap(),
        );
        evolve_close(
            *dnrg.view().at(coord),
            *hnrg.view().at(coord),
            "evolve cons.nrg",
        );
    }
}

// position-independent force through the full loop (mom + the v.a energy overlay).
#[test]
fn runtime_force_const_evolve_gpu_2d() {
    check_runtime_source_evolve::<2>(&force_const_json(2));
}
#[test]
fn runtime_force_const_evolve_gpu_3d() {
    check_runtime_source_evolve::<3>(&force_const_json(3));
}
// position-dependent force: the in-kernel centroid path inside the evolve loop.
#[test]
fn runtime_force_posdep_evolve_gpu_2d() {
    check_runtime_source_evolve::<2>(&force_posdep_json(2));
}
// region (Select) + relax (Max clamp) carrier-dialect ops driven multi-step on-device.
#[test]
fn runtime_force_region_evolve_gpu_2d() {
    check_runtime_source_evolve::<2>(REGION_JSON_2D);
}
#[test]
fn runtime_relax_evolve_gpu_2d() {
    check_runtime_source_evolve::<2>(RELAX_JSON_2D);
}
