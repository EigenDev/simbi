// =============================================================================
// mhd_shaped_wall_gpu.rs
//
// device parity for a SHAPED (CSG) rigid immersed wall on a 2.5D MHD sim
// (NewtonianMhd, D=2, DOF=3) — the magnetized shaped-wall front. the shaped
// penalize binds c_a2 = max |B|^2/rho to lift the wall relaxation to the fast
// magnetosonic speed; it rewrites the cell-centered gas state (den, all 3 mom,
// nrg) in place while the staggered CT faces stay untouched. this gates that the
// runtime NVRTC render of the 2.5D shaped kernel + the host-side c_a2 reduction
// behave identically on device.
// - PENALIZE: one shaped penalization matches host==device on cons + the force
//   receipt; the cell B is untouched by penalize, so it stays bit-equal;
// - EVOLVED: a handful of RK2 steps through the production loop (godunov + CT +
//   shaped penalize each step) keep cons + cell B bit-close host==device.
//
// runs on the host GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test mhd_shaped_wall_gpu
// =============================================================================
#![cfg(feature = "cuda")]

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;
const B0: f64 = 1.0;
const BZ0: f64 = 0.4;
const V0: f64 = 0.5;
const R_BODY: f64 = 0.18;

type MSim<S, Mem> = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, S, Mem, f64>;

fn build<S: ExecutionSpace, Mem: MemorySpace>() -> MSim<S, Mem> {
    let dx = 1.0 / N as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    let sim = MSim::<S, Mem>::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([dx, dx])
        .cfl(CFL)
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("2.5d mhd sim")
        .set_initial(|[x, y]| {
            let vel = Tensor::new([-V0 * (2.0 * PI * y).sin(), V0 * (2.0 * PI * x).sin(), 0.0]);
            let mag = Tensor::new([
                -B0 * (2.0 * PI * y).sin(),
                B0 * (4.0 * PI * x).sin(),
                BZ0 * (2.0 * PI * x).cos(),
            ]);
            MhdPrim { hydro: Prim { rho: rho0, vel, pre: p0 }, mag }
        })
        .seed_faces(|axis, x| match axis {
            0 => -B0 * (2.0 * PI * x[1]).sin(),
            _ => B0 * (4.0 * PI * x[0]).sin(),
        })
        .build();
    // a sealed shaped (SDF sphere) rigid wall placed where the Orszag-Tang velocity is maximal
    // (x = y = 0.25, |v| ~ V0), so the sealed wall sees a real flow to react against; the CSG
    // shape routes the runtime shaped kernel (host cranelift / device NVRTC) at dof = 3.
    let mut sim = sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(0, Tensor::new([0.25, 0.25]), Tensor::zeros(), 1.0, R_BODY, 1.0, true)
            .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 }),
    ));
    sim.immersed.as_mut().unwrap().shapes[0] = Some(SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], R_BODY));
    sim
}

type HostSim = MSim<CpuSpace, HostMemory>;
type DevSim = MSim<CudaSpace, UnifiedMemory>;

// max relative discrepancy over the interior of every conserved gas field + the cell B.
fn field_gap(h: &HostSim, d: &DevSim) -> f64 {
    let mut gap = 0.0_f64;
    let cmp = |hf: &symbi_grid::Field<f64, 2, HostMemory>, df: &symbi_grid::Field<f64, 2, UnifiedMemory>, gap: &mut f64| {
        for c in h.geom.interior.iter() {
            let (a, b) = (*hf.view().at(c), *df.view().at(c));
            assert!(b.is_finite(), "non-finite device field at {c:?}: {b}");
            *gap = gap.max((a - b).abs() / a.abs().max(1.0));
        }
    };
    cmp(&h.fields.cons.den, &d.fields.cons.den, &mut gap);
    for k in 0..3 {
        cmp(&h.fields.cons.mom[k], &d.fields.cons.mom[k], &mut gap);
    }
    let (hn, dn) = (h.fields.cons.nrg_field().unwrap(), d.fields.cons.nrg_field().unwrap());
    cmp(hn, dn, &mut gap);
    let (hm, dm) = (h.fields.mhd.as_ref().unwrap(), d.fields.mhd.as_ref().unwrap());
    for k in 0..3 {
        cmp(&hm.bcell[k], &dm.bcell[k], &mut gap);
    }
    gap
}

#[test]
fn mhd_shaped_wall_penalize_matches_cpu_on_device() {
    let h = build::<CpuSpace, HostMemory>();
    let d = build::<CudaSpace, UnifiedMemory>();
    dispatch_penalize(&h, 1e-3, GAMMA, 1.0);
    dispatch_penalize(&d, 1e-3, GAMMA, 1.0);
    symbi_xpu::cuda::ctx_sync();

    // penalize rewrites the gas cons (magnetosonic-stiff via c_a2) and leaves the cell B
    // untouched: both must agree host==device.
    let gap = field_gap(&h, &d);
    assert!(gap < 1e-9, "mhd shaped-wall penalize host!=device: rel gap {gap:e}");

    let hf = h.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let df = d.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let scale = (0..2).map(|k| hf[k].abs()).fold(0.0_f64, f64::max);
    assert!(scale > 1e-8, "the magnetized shaped wall never penalized (force {hf:?})");
    for k in 0..2 {
        let diff = (hf[k] - df[k]).abs();
        assert!(diff < 1e-9 * scale + 1e-11, "mhd force[{k}] host {} != device {} (diff {diff:e})", hf[k], df[k]);
    }
}

#[test]
fn mhd_shaped_wall_evolved_matches_cpu_on_device() {
    let mut h = build::<CpuSpace, HostMemory>();
    let mut d = build::<CudaSpace, UnifiedMemory>();
    let hk = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, 1.0, &h.geom.allocated);
    let dk = NewtonianMhdSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(GAMMA, CFL, 1.0, &d.geom.allocated);
    evolve(&mut h, &hk, 0.05).expect("host mhd shaped run");
    evolve(&mut d, &dk, 0.05).expect("device mhd shaped run");
    symbi_xpu::cuda::ctx_sync();

    assert!(h.iteration >= 3, "too few steps ({})", h.iteration);
    assert_eq!(h.iteration, d.iteration, "step count diverged: cpu {} gpu {}", h.iteration, d.iteration);
    let gap = field_gap(&h, &d);
    assert!(gap < 1e-6, "evolved mhd shaped-wall host!=device: rel gap {gap:e}");
}
