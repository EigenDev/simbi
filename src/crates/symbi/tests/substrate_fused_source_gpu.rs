// =============================================================================
// substrate_fused_source_gpu.rs
//
// GPU validation: every fused-source godunov kernel must run on
// the GPU and match the CPU result modulo nvcc FMA fusion. proves that
// the spec -> AOT -> substrate pipeline closes on-device for BOTH position-
// independent overlays (uniform_accel) AND position-dependent overlays
// (point_mass_grav — the more interesting case, since `cell_geometry_gv`'s
// in-kernel centroid arithmetic from `x_lo + i*dx` has to compile and
// execute correctly under NVRTC).
//
// the test pattern mirrors `substrate_hydro_gpu.rs`: build TWO identical
// SimStates (host = CpuSpace/HostMemory, device = CudaSpace/UnifiedMemory),
// configure the SAME `FusedSourceBinding` on both kernel sets, run
// godunov_euler (and godunov_rk2 for the harder integrator path), diff
// every conserved field — relative tolerance < 1e-9 per the existing
// FMA-fusion budget (`project_fma_discipline`).
//
// runs ONLY with --features cuda; needs a CUDA-capable GPU (RTX 2070 in the
// canonical env). distrobox unnecessary — `NVCC_CCBIN=/usr/bin/g++-15` is
// already in the host env per `reference_symbi_cuda_distrobox`.
//
// run: cargo test -p symbi --features cuda --test substrate_fused_source_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_kernels::FusedSourceBinding;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::{Domain, Tensor};
use symbi_geometry::{Cartesian, Metric};
use symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const N: usize = 8;
const CFL: f64 = 0.4;
const GAMMA: f64 = 1.4;
const CS: f64 = 1.0;

// per-cell GPU-vs-CPU diff in relative units. ULP-bounded modulo nvcc FMA.
// host reads of UnifiedMemory aren't ordered against pending device
// kernels by stream semantics alone (per-launch `ctx_sync` was removed for
// the production pipelining win). cmp() syncs once before reading.
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
        assert!(rel < 1e-9, "{what} at {c:?}: gpu {g} != cpu {h} (rel {rel:e})");
    }
}

// ---- adiabatic Newton sim with a smooth nonzero state on both backends ------
fn build_adiabatic<S: ExecutionSpace, Mem: MemorySpace, const D: usize>(
) -> SimState<Newtonian, D, Cartesian, IdealGas<f64>, S, Mem>
where Cartesian: Metric<f64, D>,
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
    // conserved den=1+0.3*exp(-r2/0.05), mom[k]=0.02*(k+1), nrg=pre/(gamma-1) (no kinetic term)
    // inverted to prim: vel[k]=mom[k]/rho, pre'=(gamma-1)*nrg - (gamma-1)*0.5*rho*|v|^2 so that
    // prim->cons restores the same conserved nrg.
    .set_initial(|x| {
        let r2: f64 = (0..D).map(|k| (x[k] - 0.5).powi(2)).sum();
        let rho = 1.0 + 0.3 * (-r2 / 0.05).exp();
        let nrg = (1.0 + 2.0 * (-r2 / 0.02).exp()) / (GAMMA - 1.0);
        let vel: [f64; D] = std::array::from_fn(|k| 0.02 * (k as f64 + 1.0) / rho);
        let vsq: f64 = (0..D).map(|k| vel[k] * vel[k]).sum();
        let pre = (GAMMA - 1.0) * (nrg - 0.5 * rho * vsq);
        Prim { rho, vel: Tensor::new(vel), pre }
    })
    .build()
}

fn build_iso<S: ExecutionSpace, Mem: MemorySpace, const D: usize>(
) -> SimState<IsoNewtonian, D, Cartesian, Isothermal<f64>, S, Mem>
where Cartesian: Metric<f64, D>,
{
    let dx = 1.0 / N as f64;
    SimState::<IsoNewtonian, D, Cartesian, Isothermal<f64>, S, Mem>::build(
        IsoNewtonian,
        Isothermal { cs: CS },
        Cartesian,
    )
    .cells([N; D])
    .spacing([dx; D])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("iso sim construction failed")
    // conserved den=1+0.4*exp(-r2/0.05), mom[k]=0.02*(k+1) inverted to iso prim: vel[k]=mom[k]/rho.
    .set_initial(|x| {
        let r2: f64 = (0..D).map(|k| (x[k] - 0.5).powi(2)).sum();
        let rho = 1.0 + 0.4 * (-r2 / 0.05).exp();
        let vel: [f64; D] = std::array::from_fn(|k| 0.02 * (k as f64 + 1.0) / rho);
        PrimG::<f64, D, IsoModel> { rho, vel: Tensor::new(vel), pre: Default::default() }
    })
    .build()
}

// drive the SAME pre-godunov pipeline (c2p, ghost_fill, flux per dir) on BOTH
// sims, then run godunov_euler + godunov_rk2 with the supplied binding, and
// diff every cons field. the flux step builds non-uniform face fluxes — the
// fused source's spec contribution is then the load-bearing diff between the
// unfused output and the new output. asserts the FULL substrate path closes.
fn check_adiabatic_fused<const D: usize>(binding: FusedSourceBinding)
where Cartesian: Metric<f64, D>,
{
    let host = build_adiabatic::<CpuSpace, HostMemory, D>();
    let dev  = build_adiabatic::<CudaSpace, UnifiedMemory, D>();
    let hset = AdiabaticSubstrateKernelSet::<HostMemory, f64, D>::new(GAMMA, CFL, &host.geom.allocated)
        .with_fused_source(binding.clone());
    let dset = AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, D>::new(GAMMA, CFL, &dev.geom.allocated)
        .with_fused_source(binding);
    let interior = &host.geom.interior;

    // primitive + ghost + flux on both backends (the prerequisites godunov reads).
    hset.c2p(&host);        dset.c2p(&dev);
    hset.ghost_fill(&host); dset.ghost_fill(&dev);
    for dir in 0..D {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
    }

    // **the load-bearing step**: the FUSED godunov on BOTH backends.
    hset.godunov_stage(&host, 0.01, 0.0, 1.0);
    dset.godunov_stage(&dev, 0.01, 0.0, 1.0);
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "adiabatic fused cons.den (euler)");
    for k in 0..D {
        cmp(interior, &host.fields.cons.mom[k], &dev.fields.cons.mom[k],
            &format!("adiabatic fused cons.mom_{k} (euler)"));
    }
    let (hnrg, dnrg) = (host.fields.cons.nrg_field().unwrap(), dev.fields.cons.nrg_field().unwrap());
    cmp(interior, hnrg, dnrg, "adiabatic fused cons.nrg (euler)");

    // the rk2 integrator path (snapshot + euler-half + average) — a different
    // kernel name; proves dispatching `_rk2_with_{slug}` routes correctly on GPU.
    hset.snapshot(&host); dset.snapshot(&dev);
    hset.godunov_stage(&host, 0.01, 0.5, 0.5);
    dset.godunov_stage(&dev, 0.01, 0.5, 0.5);
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "adiabatic fused cons.den (rk2)");
    for k in 0..D {
        cmp(interior, &host.fields.cons.mom[k], &dev.fields.cons.mom[k],
            &format!("adiabatic fused cons.mom_{k} (rk2)"));
    }
    cmp(interior, hnrg, dnrg, "adiabatic fused cons.nrg (rk2)");
}

fn check_iso_fused<const D: usize>(binding: FusedSourceBinding)
where Cartesian: Metric<f64, D>,
{
    let host = build_iso::<CpuSpace, HostMemory, D>();
    let dev  = build_iso::<CudaSpace, UnifiedMemory, D>();
    let hset = IsoSubstrateKernelSet::<HostMemory, f64, D>::new(CS, CFL, &host.geom.allocated)
        .with_fused_source(binding.clone());
    let dset = IsoSubstrateKernelSet::<UnifiedMemory, f64, D>::new(CS, CFL, &dev.geom.allocated)
        .with_fused_source(binding);
    let interior = &host.geom.interior;

    hset.c2p(&host);        dset.c2p(&dev);
    hset.ghost_fill(&host); dset.ghost_fill(&dev);
    for dir in 0..D {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
    }

    hset.godunov_stage(&host, 0.01, 0.0, 1.0);
    dset.godunov_stage(&dev, 0.01, 0.0, 1.0);
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "iso fused cons.den (euler)");
    for k in 0..D {
        cmp(interior, &host.fields.cons.mom[k], &dev.fields.cons.mom[k],
            &format!("iso fused cons.mom_{k} (euler)"));
    }

    hset.snapshot(&host); dset.snapshot(&dev);
    hset.godunov_stage(&host, 0.01, 0.5, 0.5);
    dset.godunov_stage(&dev, 0.01, 0.5, 0.5);
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "iso fused cons.den (rk2)");
    for k in 0..D {
        cmp(interior, &host.fields.cons.mom[k], &dev.fields.cons.mom[k],
            &format!("iso fused cons.mom_{k} (rk2)"));
    }
}

// ----- uniform_accel: POSITION-INDEPENDENT fused source ---------------------
//
// the spec source declares only `g_ext_k` runtime scalars. these stress the
// scalar manifest path (the FusedSourceBinding's `scalars: HashMap` makes it
// to NVRTC unchanged) but not the in-kernel centroid arithmetic.

fn accel_binding<const D: usize>(g: &[f64]) -> FusedSourceBinding {
    let pairs: Vec<(&str, f64)>;
    let owned: Vec<(String, f64)> = (0..D).map(|k| (format!("g_ext_{k}"), g[k])).collect();
    pairs = owned.iter().map(|(s, v)| (s.as_str(), *v)).collect();
    FusedSourceBinding::new("uniform_accel", &pairs)
}

#[test] fn adiabatic_uniform_accel_gpu_1d() { check_adiabatic_fused::<1>(accel_binding::<1>(&[-9.81])); }
#[test] fn adiabatic_uniform_accel_gpu_2d() { check_adiabatic_fused::<2>(accel_binding::<2>(&[-9.81, 0.5])); }
#[test] fn adiabatic_uniform_accel_gpu_3d() { check_adiabatic_fused::<3>(accel_binding::<3>(&[-9.81, 0.5, 0.3])); }
#[test] fn iso_uniform_accel_gpu_1d() { check_iso_fused::<1>(accel_binding::<1>(&[-9.81])); }
#[test] fn iso_uniform_accel_gpu_2d() { check_iso_fused::<2>(accel_binding::<2>(&[-9.81, 0.5])); }
#[test] fn iso_uniform_accel_gpu_3d() { check_iso_fused::<3>(accel_binding::<3>(&[-9.81, 0.5, 0.3])); }

// ----- point_mass_grav: POSITION-DEPENDENT fused source ---------------------
//
// the harder case: the spec source declares `xm_k` + `gm` scalars and `x_k`
// Params — and the position-dependent binding ties `x_k` to in-kernel cell centroids
// computed from `x_lo_k + i*dx_k`. that arithmetic must compile and execute
// CORRECTLY on the GPU; otherwise position-dependent overlays are broken
// on-device. body offset slightly from the grid center so |x - xm| != 0
// everywhere (avoids the singularity in 1/|x-xm|^3).

fn grav_binding<const D: usize>(gm: f64, xm: &[f64]) -> FusedSourceBinding {
    let mut owned: Vec<(String, f64)> = (0..D).map(|k| (format!("xm_{k}"), xm[k])).collect();
    owned.push(("gm".to_string(), gm));
    owned.push(("eps".to_string(), 0.0)); // eps = 0 recovers the bare 1/r^3 (matches the analytic check)
    let pairs: Vec<(&str, f64)> = owned.iter().map(|(s, v)| (s.as_str(), *v)).collect();
    FusedSourceBinding::new("point_mass_grav", &pairs)
}

#[test] fn adiabatic_point_mass_grav_gpu_1d() { check_adiabatic_fused::<1>(grav_binding::<1>(1.0, &[0.1])); }
#[test] fn adiabatic_point_mass_grav_gpu_2d() { check_adiabatic_fused::<2>(grav_binding::<2>(1.0, &[0.1, 0.1])); }
#[test] fn adiabatic_point_mass_grav_gpu_3d() { check_adiabatic_fused::<3>(grav_binding::<3>(1.0, &[0.1, 0.1, 0.1])); }
#[test] fn iso_point_mass_grav_gpu_1d() { check_iso_fused::<1>(grav_binding::<1>(1.0, &[0.1])); }
#[test] fn iso_point_mass_grav_gpu_2d() { check_iso_fused::<2>(grav_binding::<2>(1.0, &[0.1, 0.1])); }
#[test] fn iso_point_mass_grav_gpu_3d() { check_iso_fused::<3>(grav_binding::<3>(1.0, &[0.1, 0.1, 0.1])); }
