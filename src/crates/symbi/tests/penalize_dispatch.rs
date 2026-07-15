// =============================================================================
// penalize_dispatch.rs
//
// the [Drain] penalization dispatch (docs/design/50 step 2b), end to end on a
// real sim: the kernel runs over the body's declared support box only, drains
// the masked cells in place, leaves the far field BIT-untouched, and the
// reduced deltas land in the diagnostics accumulator — gas loss == body gain
// to machine precision.
// =============================================================================

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 48;
const L: f64 = 1.0;
const GAMMA: f64 = 1.4;

#[test]
fn penalize_drains_the_mask_and_conserves_gas_plus_body() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| Prim {
            rho: 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos(),
            vel: Tensor::new([0.15, -0.1]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.1, -0.05]),
            Tensor::zeros(),
            1.0,
            0.08,
            0.04,
            0.5,
            0.0,
            0.12, // accretion radius = the mask radius
        )));

    let before: Vec<f64> = sim.geom.interior.iter().map(|c| *sim.fields.cons.den.view().at(c)).collect();
    let nrg_before: Vec<f64> =
        sim.geom.interior.iter().map(|c| *sim.fields.cons.nrg_field().unwrap().view().at(c)).collect();

    let dt = 1e-3;
    dispatch_penalize(&sim, dt, GAMMA, 1.0);

    let im = sim.immersed.as_ref().unwrap();
    let deltas = im.diagnostics.consolidate();
    assert!(deltas[0].mass_delta > 0.0, "the drain never removed mass");
    assert!(deltas[0].energy_delta > 0.0, "the drain never removed energy");

    // gas loss == body gain, machine-exact: sum the cell-integrated conserved
    // change over the interior against the accumulated delta.
    let dv = {
        // the kernel's volume spelling (face-difference widths, reciprocated
        // twice) — mirrored so the sum matches to the bit family.
        let w = ((-L) + dx) - (-L);
        1.0 / (1.0 / (w * w))
    };
    let mut lost_mass = 0.0;
    let mut lost_nrg = 0.0;
    let mut far_changed = 0usize;
    for (i, c) in sim.geom.interior.iter().enumerate() {
        let after = *sim.fields.cons.den.view().at(c);
        lost_mass += (before[i] - after) * dv;
        lost_nrg += (nrg_before[i] - *sim.fields.cons.nrg_field().unwrap().view().at(c)) * dv;
        // far outside the support ball (r_cut = 0.12 + 20 dx ~ 0.95): the
        // dispatch never touches the cell — bit-identical, not just close.
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let y = -L + (c[1] as f64 + 0.5) * dx;
        let r = ((x - 0.1f64).powi(2) + (y + 0.05).powi(2)).sqrt();
        if r > 0.12 + 22.0 * dx && after.to_bits() != before[i].to_bits() {
            far_changed += 1;
        }
    }
    assert_eq!(far_changed, 0, "cells beyond the support ball were touched");
    assert!(
        (lost_mass - deltas[0].mass_delta).abs() <= 1e-12 * deltas[0].mass_delta.abs(),
        "gas mass loss {lost_mass} != body gain {}",
        deltas[0].mass_delta,
    );
    assert!(
        (lost_nrg - deltas[0].energy_delta).abs() <= 1e-12 * deltas[0].energy_delta.abs().max(1e-30),
        "gas energy loss {lost_nrg} != body gain {}",
        deltas[0].energy_delta,
    );
}

// the LEDGER LAW through the full RK evolve loop: with periodic boundaries the
// drain is the only mass sink, so the accumulated per-step receipts must equal
// the gas's total conserved loss. this is the gate the legacy feedback failed
// (RK2 stage-weight over-count, measured 1.5x on bondi) and the gate a
// stage-blended penalize placement fails the same way — the penalize runs ONCE
// per step after the RK combination precisely so this holds.
#[test]
fn ledger_equals_gas_loss_through_the_rk_loop() {
    use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
    use symbi::sim::evolve::evolve;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| Prim {
            rho: 1.0 + 0.1 * (3.0 * x).sin() * (2.0 * y).cos(),
            vel: Tensor::new([0.05, -0.03]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.1, -0.05]),
            Tensor::zeros(),
            1.0, 0.08, 0.04, 0.5, 0.0, 0.12,
        )));
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);

    let dv = {
        let w = ((-L) + dx) - (-L);
        1.0 / (1.0 / (w * w))
    };
    let mass0: f64 = sim.geom.interior.iter().map(|c| *sim.fields.cons.den.view().at(c) * dv).sum();
    evolve(&mut sim, &sub, 0.05).expect("evolve");
    let mass1: f64 = sim.geom.interior.iter().map(|c| *sim.fields.cons.den.view().at(c) * dv).sum();

    let ledger: f64 = sim.immersed.as_ref().unwrap().history.mass_delta().iter().sum();
    let lost = mass0 - mass1;
    assert!(ledger > 0.0, "the drain never fired");
    assert!(
        (ledger - lost).abs() <= 1e-11 * lost.abs(),
        "ledger {ledger} != gas loss {lost} (ratio {})",
        ledger / lost,
    );
}

#[test]
fn iso_dispatch_drains_directly() {
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::isothermal::IsoNewtonian;
    use symbi_hydro::state::PrimG;
    type ISim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let sim = ISim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| PrimG::<f64, 2, IsoModel> {
            rho: 1.5,
            vel: Tensor::new([0.1, -0.05]),
            pre: Default::default(),
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.1, -0.05]),
            Tensor::zeros(),
            1.0, 0.08, 0.04, 0.5, 0.0, 0.12,
        )));
    dispatch_penalize(&sim, 1e-3, 1.0, 1.0);
    let deltas = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert!(deltas[0].mass_delta > 0.0, "iso dispatch removed no mass: {:e}", deltas[0].mass_delta);
}

// the ANGULAR-MOMENTUM receipt (docs/design/51): the reduced torque_delta is
// the moment of the gas's momentum loss about the body center, machine-exact.
// gas in rigid rotation about the body makes the receipt decisively nonzero
// (a drain removes the local angular momentum along with the mass).
#[test]
fn torque_receipt_equals_the_moment_of_the_momentum_loss() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let center = [0.1, -0.05];
    let w_z = 0.6;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| Prim {
            rho: 1.0 + 0.1 * (3.0 * x).cos() * (2.0 * y).sin(),
            // rigid rotation about the body center: v = w_z z-hat x r.
            vel: Tensor::new([-w_z * (y - center[1]), w_z * (x - center[0])]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new(center),
            Tensor::zeros(),
            1.0, 0.08, 0.04, 0.5, 0.0, 0.12,
        )));

    let mom_before: Vec<[f64; 2]> = sim
        .geom
        .interior
        .iter()
        .map(|c| [*sim.fields.cons.mom[0].view().at(c), *sim.fields.cons.mom[1].view().at(c)])
        .collect();

    let dt = 1e-3;
    dispatch_penalize(&sim, dt, GAMMA, 1.0);
    let deltas = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert!(deltas[0].mass_delta > 0.0, "the drain never fired");

    // the independent recomputation: sum (x - x_b) x dmom * dv / dt over the
    // interior — the same face-mid centroid and double-reciprocal volume the
    // kernel evaluates.
    let dv = {
        let w = ((-L) + dx) - (-L);
        1.0 / (1.0 / (w * w))
    };
    let mid = |i: isize| ((-L + i as f64 * dx) + (-L + (i as f64 + 1.0) * dx)) * 0.5;
    let mut torque_z = 0.0;
    for (i, c) in sim.geom.interior.iter().enumerate() {
        let dmx = mom_before[i][0] - *sim.fields.cons.mom[0].view().at(c);
        let dmy = mom_before[i][1] - *sim.fields.cons.mom[1].view().at(c);
        let rx = mid(c[0]) - center[0];
        let ry = mid(c[1]) - center[1];
        torque_z += (rx * dmy - ry * dmx) * dv / dt;
    }
    assert!(
        torque_z.abs() > 1e-6,
        "the rotating cloud produced no angular-momentum exchange: {torque_z:e}"
    );
    assert!(
        (torque_z - deltas[0].torque_delta[2]).abs() <= 1e-11 * torque_z.abs(),
        "torque receipt {} != moment of the momentum loss {torque_z}",
        deltas[0].torque_delta[2],
    );
    assert_eq!(deltas[0].torque_delta[0], 0.0);
    assert_eq!(deltas[0].torque_delta[1], 0.0);
}

// the porous surface through the dispatch (docs/design/50 zoo): the body's
// declared SurfaceSpec picks the kernel, and the porosity endpoints hold
// end to end — p = 0 books EXACTLY zero mass receipts (a sealed wall absorbs
// momentum but never mass), p > 0 drains.
#[test]
fn porous_surface_endpoints_hold_through_the_dispatch() {
    use symbi_ib::SurfaceSpec;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let build = |surface: SurfaceSpec| -> Sim {
        Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([N, N])
            .origin([-L, -L])
            .spacing([dx, dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("sim")
            .set_initial(|[x, y]| Prim {
                rho: 1.0 + 0.1 * (2.0 * x).sin() * (1.5 * y).cos(),
                vel: Tensor::new([0.2, -0.15]),
                pre: 1.0,
            })
            .build()
            .with_bodies(BodyCollection::new().add(
                Body::black_hole(
                    0,
                    Tensor::new([0.1, -0.05]),
                    Tensor::zeros(),
                    1.0, 0.08, 0.04, 0.5, 0.0, 0.12,
                )
                .with_surface(surface),
            ))
    };

    // sealed free-slip wall: zero mass receipts, exactly; momentum exchanged.
    let sealed = build(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 0.0 });
    dispatch_penalize(&sealed, 1e-3, GAMMA, 1.0);
    let d = sealed.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(d[0].mass_delta, 0.0, "a sealed surface must book zero mass, exactly");
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the sealed wall never pushed back: {f:?}"
    );

    // half-open: the drain fires.
    let porous = build(SurfaceSpec::Porous { porosity: 0.5, k_eta_n: 50.0, k_eta_t: 0.0 });
    dispatch_penalize(&porous, 1e-3, GAMMA, 1.0);
    let d = porous.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert!(d[0].mass_delta > 0.0, "the half-open surface never drained");
}

// a RIGID (non-accreting) body reaches the penalize dispatch through the mask_radius
// gate — the old accretion_radius gate would have skipped it. it penalizes (the wall
// pushes back) while removing exactly zero mass, and the feedback ledger consolidates a
// non-black-hole body without panicking.
#[test]
fn rigid_wall_non_accreting_penalizes_without_draining() {
    use symbi_ib::SurfaceSpec;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let rigid = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| Prim {
            rho: 1.0 + 0.1 * (2.0 * x).sin() * (1.5 * y).cos(),
            vel: Tensor::new([0.2, -0.15]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(
            // rigid sphere: no accretion capability, so accretion_radius() is None; it
            // masks to its physical radius. no-slip wall on both channels.
            Body::rigid_sphere(0, Tensor::new([0.1, -0.05]), Tensor::zeros(), 1.0, 0.12, 1.0, true)
                .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 }),
        ));
    assert_eq!(rigid.immersed.as_ref().unwrap().bodies.get(0).accretion_radius(), None);
    dispatch_penalize(&rigid, 1e-3, GAMMA, 1.0);
    let d = rigid.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(d[0].mass_delta, 0.0, "a rigid wall removes no mass, exactly");
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the non-accreting rigid wall never penalized — the mask_radius gate skipped it: {f:?}"
    );
}

// an ARBITRARY-SHAPE rigid body: a box, not a sphere. its penalization kernel is runtime-built +
// cranelift-JIT'd (the box geometry baked as constants, the body position a runtime scalar), then
// executed over the shape's bounding box through the standalone `run_parallel_raw`. it must
// penalize (the wall pushes back) while removing exactly zero mass — the full host runtime-JIT path
// end to end.
#[test]
fn shaped_box_rigid_wall_penalizes_via_runtime_jit() {
    use symbi_ib::sdf::SdfExpr;
    use symbi_ib::SurfaceSpec;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| Prim {
            rho: 1.0 + 0.1 * (2.0 * x).sin() * (1.5 * y).cos(),
            vel: Tensor::new([0.25, -0.15]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(
            Body::rigid_sphere(0, Tensor::new([0.1, -0.05]), Tensor::zeros(), 1.0, 0.2, 1.0, true)
                .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 }),
        ));
    // attach the arbitrary shape: a box in the body-local frame (a 0.3 x 0.3 square in the z=0
    // plane), NOT the sphere the AOT kernel would use.
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.15, 0.15, 1.0]));

    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(d[0].mass_delta, 0.0, "a sealed shaped wall removes no mass, exactly");
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the runtime-JIT'd box wall never penalized: {f:?}"
    );
}

// the ISO shaped wall: an arbitrary-shape rigid obstacle in an energy-free flow (the common
// obstacle case). the runtime-JIT'd iso kernel drops the nrg channel; the sealed wall still
// penalizes (force) and removes no mass.
#[test]
fn shaped_box_rigid_wall_iso_penalizes_via_runtime_jit() {
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::isothermal::IsoNewtonian;
    use symbi_hydro::state::PrimG;
    use symbi_ib::sdf::SdfExpr;
    use symbi_ib::SurfaceSpec;
    type ISim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let mut sim = ISim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| PrimG::<f64, 2, IsoModel> {
            rho: 1.5,
            vel: Tensor::new([0.2, -0.1]),
            pre: Default::default(),
        })
        .build()
        .with_bodies(BodyCollection::new().add(
            Body::rigid_sphere(0, Tensor::new([0.1, -0.05]), Tensor::zeros(), 1.0, 0.2, 1.0, true)
                .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 }),
        ));
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.15, 0.15, 1.0]));

    dispatch_penalize(&sim, 1e-3, 1.0, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(d[0].mass_delta, 0.0, "a sealed iso shaped wall removes no mass, exactly");
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the runtime-JIT'd iso box wall never penalized: {f:?}"
    );
}
