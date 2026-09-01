// =============================================================================
// shaped_wall_twoway_iso_gpu.rs
//
// device parity gates for two shaped-wall paths:
// - two-way torque spin-up: a free (two_way) shaped spinner in still gas is
//   dragged toward rest by the reaction torque. the spinning kernel's torque
//   diagnostic + the integrated body omega must match host==device — the
//   two-way path, where the reduced torque steers omega, as distinct from a
//   prescribed spin;
// - iso shaped wall: an energy-free shaped obstacle (no nrg channel) penalizes
//   identically on device, cons + force receipt bit-close to the CPU run.
//
// runs on the host GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test shaped_wall_twoway_iso_gpu
// =============================================================================
#![cfg(feature = "cuda")]

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::{Prim, PrimG};
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec, apply_body_deltas};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 1.4;
const N: usize = 48;
const L: f64 = 1.0;
const DX: f64 = 2.0 * L / N as f64;

// ----- two-way torque spin-up ---------------------------------------------------

const OMEGA0: f64 = 3.0;
const INERTIA: f64 = 10.0;

type TwSim<S, Mem> = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, S, Mem>;

fn build_twoway<S: ExecutionSpace, Mem: MemorySpace>() -> TwSim<S, Mem> {
    let mut sim = TwSim::<S, Mem>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([DX, DX])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(
            BodyCollection::new().add(
                Body::rigid_sphere(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.3, INERTIA, true)
                    .with_surface(SurfaceSpec::Porous {
                        porosity: 0.0,
                        k_eta_n: 50.0,
                        k_eta_t: 50.0,
                    })
                    .with_spin(OMEGA0)
                    .with_two_way_coupling(true),
            ),
        );
    // an asymmetric shape so the omega x r drag produces a real reaction torque.
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.25, 0.1, 1.0]));
    sim
}

// compare two f64 values against the force/torque magnitude with an absolute floor: the
// transverse / near-zero receipt components differ at round-off across the host slab-order and
// device block-order folds (allowed by the deterministic-fold contract).
fn close(a: f64, b: f64, scale: f64, tag: &str) {
    let diff = (a - b).abs();
    assert!(
        diff < 1e-9 * scale + 1e-11,
        "{tag}: host {a} != device {b} (abs diff {diff:e}, scale {scale:e})"
    );
}

#[test]
fn two_way_spin_torque_matches_cpu_on_device() {
    let mut h = build_twoway::<CpuSpace, HostMemory>();
    let mut d = build_twoway::<CudaSpace, UnifiedMemory>();
    let dt = 1e-3;
    dispatch_penalize(&h, dt, GAMMA, 1.0, 3.0);
    dispatch_penalize(&d, dt, GAMMA, 1.0, 3.0);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();

    // the penalized cons fields agree cell-by-cell.
    let mut gap = 0.0_f64;
    let cmp = |hf: &symbi_grid::Field<f64, 2, HostMemory>,
               df: &symbi_grid::Field<f64, 2, UnifiedMemory>,
               gap: &mut f64| {
        for c in h.geom.interior.iter() {
            let (a, b) = (*hf.view().at(c), *df.view().at(c));
            assert!(b.is_finite(), "non-finite device cons at {c:?}");
            *gap = gap.max((a - b).abs() / a.abs().max(1.0));
        }
    };
    cmp(&h.fields.cons.den, &d.fields.cons.den, &mut gap);
    for k in 0..2 {
        cmp(&h.fields.cons.mom[k], &d.fields.cons.mom[k], &mut gap);
    }
    assert!(
        gap < 1e-9,
        "two-way spinner cons host!=device: rel gap {gap:e}"
    );

    // the reaction-torque diagnostic (the spinning kernel's z-moment) agrees, then the
    // host-side apply_body_deltas integrates the same torque to the same omega on both.
    let dh = h.immersed.as_ref().unwrap().diagnostics.consolidate();
    let dd = d.immersed.as_ref().unwrap().diagnostics.consolidate();
    let scale = dh[0].torque_delta[2].abs().max(1e-12);
    close(
        dh[0].torque_delta[2],
        dd[0].torque_delta[2],
        scale,
        "two-way torque_z",
    );
    apply_body_deltas(&mut h.immersed.as_mut().unwrap().bodies, &dh, dt);
    apply_body_deltas(&mut d.immersed.as_mut().unwrap().bodies, &dd, dt);
    let (ho, dvo) = (
        h.immersed.as_ref().unwrap().bodies.get(0).omega[2],
        d.immersed.as_ref().unwrap().bodies.get(0).omega[2],
    );
    close(ho, dvo, OMEGA0, "two-way omega_z");
    // the physics still holds through the device path: drag decelerates, one step does not reverse.
    assert!(
        dvo < OMEGA0 && dvo > 0.0,
        "device two-way spinner not dragged toward rest: {OMEGA0} -> {dvo}"
    );
}

// ----- iso shaped wall ----------------------------------------------------------

type IsoSim<S, Mem> = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, S, Mem>;

fn build_iso<S: ExecutionSpace, Mem: MemorySpace>() -> IsoSim<S, Mem> {
    let mut sim = IsoSim::<S, Mem>::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([DX, DX])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| PrimG::<f64, 2, IsoModel> {
            rho: 1.5,
            vel: Tensor::new([0.2, -0.1]),
            pre: Default::default(),
        })
        .build()
        .with_bodies(
            BodyCollection::new().add(
                Body::rigid_sphere(
                    0,
                    Tensor::new([0.1, -0.05]),
                    Tensor::zeros(),
                    1.0,
                    0.2,
                    1.0,
                    true,
                )
                .with_surface(SurfaceSpec::Porous {
                    porosity: 0.0,
                    k_eta_n: 50.0,
                    k_eta_t: 50.0,
                }),
            ),
        );
    sim.immersed.as_mut().unwrap().shapes[0] = Some(SdfExpr::<f64, 3>::cuboid(
        [0.0, 0.0, 0.0],
        [0.15, 0.15, 1.0],
    ));
    sim
}

#[test]
fn iso_shaped_wall_matches_cpu_on_device() {
    let h = build_iso::<CpuSpace, HostMemory>();
    let d = build_iso::<CudaSpace, UnifiedMemory>();
    dispatch_penalize(&h, 1e-3, 1.0, 1.0, 3.0);
    dispatch_penalize(&d, 1e-3, 1.0, 1.0, 3.0);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();

    // the iso kernel drops the nrg channel: compare den + mom only.
    let mut gap = 0.0_f64;
    let cmp = |hf: &symbi_grid::Field<f64, 2, HostMemory>,
               df: &symbi_grid::Field<f64, 2, UnifiedMemory>,
               gap: &mut f64| {
        for c in h.geom.interior.iter() {
            let (a, b) = (*hf.view().at(c), *df.view().at(c));
            assert!(b.is_finite(), "non-finite device iso cons at {c:?}");
            *gap = gap.max((a - b).abs() / a.abs().max(1.0));
        }
    };
    cmp(&h.fields.cons.den, &d.fields.cons.den, &mut gap);
    for k in 0..2 {
        cmp(&h.fields.cons.mom[k], &d.fields.cons.mom[k], &mut gap);
    }
    assert!(
        gap < 1e-9,
        "iso shaped-wall cons host!=device: rel gap {gap:e}"
    );

    let hf = h.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let df = d.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let scale = hf[0].abs().max(hf[1].abs());
    for k in 0..2 {
        close(hf[k], df[k], scale, &format!("iso force[{k}]"));
    }
    // a sealed iso wall removes no mass on either backend.
    let (hm, dm) = (
        h.immersed.as_ref().unwrap().diagnostics.consolidate()[0].mass_delta,
        d.immersed.as_ref().unwrap().diagnostics.consolidate()[0].mass_delta,
    );
    assert_eq!(hm, 0.0, "host iso sealed wall removed mass: {hm}");
    assert_eq!(dm, 0.0, "device iso sealed wall removed mass: {dm}");
}
