// =============================================================================
// rigid_curvilinear_wall_gpu.rs
//
// the device twin of rigid_curvilinear_wall: the runtime-JIT shaped rigid wall on
// the cylindrical (R, phi) chart, run on device memory (NVRTC render of the same
// GvKernel the host cranelift-compiles) and asserted BIT-CLOSE to the CPU run.
// gates the three host discriminators through the GPU dispatch:
// - NO-PENETRATION: the penalized cons field matches the CPU cons field;
// - RECEIPT FRAME: the force receipt (a cartesian world vector) matches the CPU
//   receipt (an x-stream pushes +x with the y component cancelling);
// - TARGET FRAME: a wall translating through still gas produces the same receipt
//   on device as on host (the frame rotation is baked into the emitted kernel).
// the per-body delta reductions must land the same value on both backends (the
// fixed-order fold contract), so the receipt is a parity discriminator too.
//
// runs on the host GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test rigid_curvilinear_wall_gpu
// =============================================================================
#![cfg(feature = "cuda")]

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::CylindricalRPhi;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const NR: usize = 48;
const NP: usize = 96;
const R_LO: f64 = 1.0;
const R_HI: f64 = 3.0;
const DR: f64 = (R_HI - R_LO) / NR as f64;
const DP: f64 = 2.0 * PI / NP as f64;
const R_BODY: f64 = 0.35;
const V_INF: f64 = 0.3;
const BODY_X: f64 = 0.0;
const BODY_Y: f64 = 2.0;

// a uniform cartesian x-directed stream in the local physical basis.
fn stream_prim(phi: f64, v: f64) -> Prim<f64, 2> {
    Prim { rho: 1.0, vel: Tensor::new([v * phi.cos(), -v * phi.sin()]), pre: 1.0 }
}

// build the shaped-wall cylindrical sim on any (space, memory) — the host and device
// runs bake the SAME CSG sphere, differing only in where the kernel executes.
fn build<S: ExecutionSpace, Mem: MemorySpace>(
    vel_x: f64,
    with_body: bool,
    body_vel: [f64; 2],
    surface: SurfaceSpec,
) -> SimState<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, S, Mem> {
    let sim = SimState::<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, S, Mem>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        CylindricalRPhi,
    )
    .cells([NR, NP])
    .origin([R_LO, 0.0])
    .spacing([DR, DP])
    // the (r, phi) disk plane (the cylindrical 2d default is the r-z section); shaped CSG walls
    // live on r-phi, and this matches the host twin rigid_curvilinear_wall.rs.
    .cyl_plane(symbi_sim::state::CylPlane::RPhi)
    .boundaries(Boundaries(std::array::from_fn(|a| {
        if a == 1 { [BoundaryType::Periodic; 2] } else { [BoundaryType::Outflow; 2] }
    })))
    .cfl(CFL)
    .timestepping(Timestepping::Rk2)
    .allocate()
    .expect("cyl sim")
    .set_initial(|x| stream_prim(x[1], vel_x))
    .build();
    if !with_body {
        return sim;
    }
    let mut sim = sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(
            0,
            Tensor::new([BODY_X, BODY_Y]),
            Tensor::new(body_vel),
            1.0,
            R_BODY,
            0.1,
            false,
        )
        .with_surface(surface),
    ));
    // the CSG shape routes the runtime-JIT / NVRTC shaped kernel instead of the AOT sphere.
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], R_BODY));
    sim
}

type HostSim = SimState<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CpuSpace, HostMemory>;
type DevSim = SimState<Newtonian, 2, CylindricalRPhi, IdealGas<f64>, CudaSpace, UnifiedMemory>;

// the max relative discrepancy of a cons field between the host + device runs.
fn cons_rel_gap(h: &HostSim, d: &DevSim) -> f64 {
    let mut gap = 0.0_f64;
    let cmp = |hf: &symbi_grid::Field<f64, 2, HostMemory>,
               df: &symbi_grid::Field<f64, 2, UnifiedMemory>,
               gap: &mut f64| {
        for c in h.geom.interior.iter() {
            let (a, b) = (*hf.view().at(c), *df.view().at(c));
            assert!(b.is_finite(), "non-finite device cons at {c:?}: {b}");
            *gap = gap.max((a - b).abs() / a.abs().max(1.0));
        }
    };
    cmp(&h.fields.cons.den, &d.fields.cons.den, &mut gap);
    for k in 0..2 {
        cmp(&h.fields.cons.mom[k], &d.fields.cons.mom[k], &mut gap);
    }
    let (hn, dn) = (h.fields.cons.nrg_field().unwrap(), d.fields.cons.nrg_field().unwrap());
    cmp(hn, dn, &mut gap);
    gap
}

// run ONE shaped penalization on host + device and return their force receipts.
fn penalize_once(vel_x: f64, body_vel: [f64; 2], surface: SurfaceSpec) -> ([f64; 2], [f64; 2], f64) {
    let h = build::<CpuSpace, HostMemory>(vel_x, true, body_vel, surface);
    let d = build::<CudaSpace, UnifiedMemory>(vel_x, true, body_vel, surface);
    dispatch_penalize(&h, 1e-3, GAMMA, 1.0);
    dispatch_penalize(&d, 1e-3, GAMMA, 1.0);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    let hf = h.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let df = d.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let gap = cons_rel_gap(&h, &d);
    ([hf[0], hf[1]], [df[0], df[1]], gap)
}

// the host + device force receipts agree when each component matches to a combined
// rel+abs tolerance against the force MAGNITUDE. the transverse component is a
// numerical zero (mirror cancellation about phi = pi/2), where the host slab-order
// fold and the device block-order fold differ at round-off — bounded by the absolute
// floor, not a mismatch (the fixed-order fold is reproducible per backend, not bitwise
// across backends).
fn receipts_close(hf: [f64; 2], df: [f64; 2], tag: &str) {
    let scale = hf[0].abs().max(hf[1].abs());
    for k in 0..2 {
        let diff = (hf[k] - df[k]).abs();
        assert!(
            diff < 1e-9 * scale + 1e-11,
            "{tag} receipt[{k}] host {} != device {} (abs diff {diff:e}, scale {scale:e})",
            hf[k],
            df[k],
        );
    }
}

#[test]
fn shaped_wall_penalize_matches_cpu_on_device() {
    // the sealed no-slip wall in an x-stream: the host + device force receipts must
    // agree, and the penalized cons fields must agree cell-by-cell.
    let sealed = SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 };
    let (hf, df, gap) = penalize_once(V_INF, [0.0, 0.0], sealed);
    assert!(gap < 1e-9, "shaped-wall cons field host!=device: rel gap {gap:e}");
    receipts_close(hf, df, "sealed");
    // the host discriminator still holds through the device path: an x-stream pushes +x.
    assert!(df[0] > 0.0 && df[0] > 3.0 * df[1].abs(), "device receipt not world-frame +x: {df:?}");
}

#[test]
fn moving_shaped_wall_target_matches_cpu_on_device() {
    // a sealed no-slip wall translating along +x through still gas: the frame-rotated
    // velocity target is baked into the emitted kernel, so the device receipt must match
    // the host receipt and still react along -x.
    let sealed = SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 };
    let (hf, df, gap) = penalize_once(0.0, [V_INF, 0.0], sealed);
    assert!(gap < 1e-9, "moving shaped-wall cons field host!=device: rel gap {gap:e}");
    receipts_close(hf, df, "moving");
    assert!(df[0] < 0.0 && df[0].abs() > 3.0 * df[1].abs(), "device moving receipt not -x: {df:?}");
}

#[test]
fn shaped_wall_evolved_matches_cpu_on_device() {
    // the integration-level gate: evolve the sealed no-penetration wall to t = 1 on both
    // backends and assert the cons state agrees — the device shaped penalize rides the full
    // step loop (godunov + c2p + penalize + feedback) identically to the host run.
    let no_pen = SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 1.0e3, k_eta_t: 0.0 };
    let mut h = build::<CpuSpace, HostMemory>(V_INF, true, [0.0, 0.0], no_pen);
    let mut d = build::<CudaSpace, UnifiedMemory>(V_INF, true, [0.0, 0.0], no_pen);
    let hk = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &h.geom.allocated);
    let dk = AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(GAMMA, CFL, &d.geom.allocated);
    evolve(&mut h, &hk, 1.0).expect("host shaped-wall run");
    evolve(&mut d, &dk, 1.0).expect("device shaped-wall run");
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    let gap = cons_rel_gap(&h, &d);
    assert!(gap < 1e-7, "evolved shaped-wall cons host!=device: rel gap {gap:e}");
}
