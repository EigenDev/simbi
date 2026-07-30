// =============================================================================
// penalize_dispatch.rs
//
// the [Drain] penalization dispatch, end to end on a
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

    let before: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .collect();
    let nrg_before: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.nrg_field().unwrap().view().at(c))
        .collect();

    let dt = 1e-3;
    dispatch_penalize(&sim, dt, GAMMA, 1.0);

    let im = sim.immersed.as_ref().unwrap();
    let deltas = im.diagnostics.consolidate();
    assert!(deltas[0].mass_delta > 0.0, "the drain never removed mass");
    assert!(
        deltas[0].energy_delta > 0.0,
        "the drain never removed energy"
    );

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
        // dispatch never touches the cell — bit-identical.
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
        (lost_nrg - deltas[0].energy_delta).abs()
            <= 1e-12 * deltas[0].energy_delta.abs().max(1e-30),
        "gas energy loss {lost_nrg} != body gain {}",
        deltas[0].energy_delta,
    );
}

// the LEDGER LAW through the full RK evolve loop: with periodic boundaries the
// drain is the only mass sink, so the accumulated per-step receipts must equal
// the gas's total conserved loss. an RK2 stage-weight over-count double-counts
// the receipts (measured 1.5x on bondi), and a stage-blended penalize placement
// fails the same way — the penalize runs ONCE per step after the RK combination
// precisely so this holds.
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
            // softening 0.10 keeps the run correction-free: a harder point mass
            // drives the sink-center c2p unphysical, and FOFC's response (a
            // first-order redo plus a freeze parachute) reverts conserved
            // variables in a way the mass sum sees, breaking the exact ledger
            // identity. the fofc-counter assert below guards that no correction
            // fired.
            1.0,
            0.08,
            0.10,
            0.5,
            0.0,
            0.12,
        )));
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);

    let dv = {
        let w = ((-L) + dx) - (-L);
        1.0 / (1.0 / (w * w))
    };
    let mass0: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c) * dv)
        .sum();
    symbi::regimes::fofc::fofc_reset_stats();
    evolve(&mut sim, &sub, 0.05).expect("evolve");
    let mass1: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c) * dv)
        .sum();

    let ledger: f64 = sim
        .immersed
        .as_ref()
        .unwrap()
        .history
        .mass_delta()
        .iter()
        .sum();
    let lost = mass0 - mass1;
    let (fb, fz) = symbi::regimes::fofc::fofc_stats();
    assert!(
        fb == 0 && fz == 0,
        "the ledger oracle's evolution tripped FOFC (fallback {fb}, freeze {fz}): the \
         freeze parachute is deliberately non-conservative, so the exact ledger == loss \
         identity only holds on a correction-free run — soften the setup, don't loosen \
         the tolerance"
    );
    assert!(ledger > 0.0, "the drain never fired");
    assert!(
        (ledger - lost).abs() <= 1e-11 * lost.abs(),
        "ledger {ledger} != gas loss {lost} (ratio {})",
        ledger / lost,
    );
}

#[test]
fn iso_dispatch_drains_directly() {
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::eos::Isothermal;
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
            1.0,
            0.08,
            0.04,
            0.5,
            0.0,
            0.12,
        )));
    dispatch_penalize(&sim, 1e-3, 1.0, 1.0);
    let deltas = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert!(
        deltas[0].mass_delta > 0.0,
        "iso dispatch removed no mass: {:e}",
        deltas[0].mass_delta
    );
}

// the ANGULAR-MOMENTUM receipt: the reduced torque_delta is
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
            1.0,
            0.08,
            0.04,
            0.5,
            0.0,
            0.12,
        )));

    let mom_before: Vec<[f64; 2]> = sim
        .geom
        .interior
        .iter()
        .map(|c| {
            [
                *sim.fields.cons.mom[0].view().at(c),
                *sim.fields.cons.mom[1].view().at(c),
            ]
        })
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

// the porous surface through the dispatch: the body's
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
            .with_bodies(
                BodyCollection::new().add(
                    Body::black_hole(
                        0,
                        Tensor::new([0.1, -0.05]),
                        Tensor::zeros(),
                        1.0,
                        0.08,
                        0.04,
                        0.5,
                        0.0,
                        0.12,
                    )
                    .with_surface(surface),
                ),
            )
    };

    // sealed free-slip wall: zero mass receipts, exactly; momentum exchanged.
    let sealed = build(SurfaceSpec::Porous {
        porosity: 0.0,
        k_eta_n: 50.0,
        k_eta_t: 0.0,
    });
    dispatch_penalize(&sealed, 1e-3, GAMMA, 1.0);
    let d = sealed.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(
        d[0].mass_delta, 0.0,
        "a sealed surface must book zero mass, exactly"
    );
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the sealed wall never pushed back: {f:?}"
    );

    // half-open: the drain fires.
    let porous = build(SurfaceSpec::Porous {
        porosity: 0.5,
        k_eta_n: 50.0,
        k_eta_t: 0.0,
    });
    dispatch_penalize(&porous, 1e-3, GAMMA, 1.0);
    let d = porous.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert!(d[0].mass_delta > 0.0, "the half-open surface never drained");
}

#[test]
fn porous_absorption_converges_under_grid_refinement() {
    use std::f64::consts::PI;
    use symbi_ib::SurfaceSpec;

    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    const RADIUS: f64 = 0.3;
    const EXPOSURE_TIME: f64 = 0.5;

    let run = |cells: usize| {
        let dx = 2.0 * L / cells as f64;
        let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([cells, cells])
            .origin([-L, -L])
            .spacing([dx, dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("sim")
            .set_initial(|_| Prim {
                rho: 1.0,
                vel: Tensor::new([0.2, -0.1]),
                pre: 1.0,
            })
            .build()
            .with_bodies(
                BodyCollection::new().add(
                    Body::black_hole(
                        0,
                        Tensor::zeros(),
                        Tensor::zeros(),
                        1.0,
                        0.08,
                        0.04,
                        0.5,
                        0.0,
                        RADIUS,
                    )
                    .with_surface(SurfaceSpec::Porous {
                        porosity: 0.5,
                        k_eta_n: 1.0,
                        k_eta_t: 1.0,
                    }),
                ),
            );

        dispatch_penalize(&sim, EXPOSURE_TIME, GAMMA, 1.0);
        let receipt = sim.immersed.as_ref().unwrap().diagnostics.consolidate()[0];
        let force = receipt.force_delta;
        assert!(
            receipt.mass_delta > 0.0,
            "the drain channel never fired at {cells} cells"
        );
        assert!(
            (force[0] * force[0] + force[1] * force[1]).sqrt() > 1e-8,
            "the wall channel never fired at {cells} cells"
        );
        receipt.mass_delta
    };

    let exact = PI * RADIUS * RADIUS;
    let coarse_error = (run(32) - exact).abs();
    let medium_error = (run(64) - exact).abs();
    let fine_error = (run(128) - exact).abs();

    assert!(
        medium_error < coarse_error && fine_error < medium_error,
        "absorbed disk mass did not converge: errors {coarse_error:e}, {medium_error:e}, \
         {fine_error:e}"
    );
    assert!(
        fine_error < 0.75 * coarse_error,
        "two refinements failed to contract the absorbed-mass error by 25%: \
         errors {coarse_error:e}, {medium_error:e}, {fine_error:e}"
    );
}

// a RIGID (non-accreting) body reaches the penalize dispatch through the mask_radius
// gate: mask_radius is defined for every body, while accretion_radius is None for a
// non-accreting one. it penalizes (the wall pushes back) while removing exactly zero
// mass, and the feedback ledger consolidates a non-black-hole body without panicking.
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
        .with_bodies(
            BodyCollection::new().add(
                // rigid sphere: no accretion capability, so accretion_radius() is None; it
                // masks to its physical radius. no-slip wall on both channels.
                Body::rigid_sphere(
                    0,
                    Tensor::new([0.1, -0.05]),
                    Tensor::zeros(),
                    1.0,
                    0.12,
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
    assert_eq!(
        rigid
            .immersed
            .as_ref()
            .unwrap()
            .bodies
            .get(0)
            .accretion_radius(),
        None
    );
    dispatch_penalize(&rigid, 1e-3, GAMMA, 1.0);
    let d = rigid.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(
        d[0].mass_delta, 0.0,
        "a rigid wall removes no mass, exactly"
    );
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the non-accreting rigid wall never penalized — the mask_radius gate skipped it: {f:?}"
    );
}

// an ARBITRARY-SHAPE rigid body: a box. its penalization kernel is runtime-built +
// cranelift-JIT'd (the box geometry baked as constants, the body position a runtime scalar), then
// executed over the shape's bounding box through the standalone `run_parallel_raw`. it must
// penalize (the wall pushes back) while removing exactly zero mass — the full host runtime-JIT path
// end to end.
#[test]
fn shaped_box_rigid_wall_penalizes_via_runtime_jit() {
    use symbi_ib::SurfaceSpec;
    use symbi_ib::sdf::SdfExpr;
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
    // attach the arbitrary shape: a box in the body-local frame (a 0.3 x 0.3 square in the z=0
    // plane).
    sim.immersed.as_mut().unwrap().shapes[0] = Some(SdfExpr::<f64, 3>::cuboid(
        [0.0, 0.0, 0.0],
        [0.15, 0.15, 1.0],
    ));

    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(
        d[0].mass_delta, 0.0,
        "a sealed shaped wall removes no mass, exactly"
    );
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
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::isothermal::IsoNewtonian;
    use symbi_hydro::state::PrimG;
    use symbi_ib::SurfaceSpec;
    use symbi_ib::sdf::SdfExpr;
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

    dispatch_penalize(&sim, 1e-3, 1.0, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(
        d[0].mass_delta, 0.0,
        "a sealed iso shaped wall removes no mass, exactly"
    );
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the runtime-JIT'd iso box wall never penalized: {f:?}"
    );
}

// TWO-WAY rotational coupling: a free (two_way) spinner in STILL fluid is dragged toward rest —
// the reaction torque of the gas it spins up decelerates it. one evolve step (dispatch fills the
// torque diagnostic, apply_body_deltas integrates I domega = torque dt) must reduce omega.
// this also pins the sign of the coupling.
#[test]
fn two_way_spin_is_dragged_to_a_stop() {
    use symbi_ib::SurfaceSpec;
    use symbi_ib::sdf::SdfExpr;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    const OMEGA0: f64 = 3.0;
    const INERTIA: f64 = 10.0;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
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
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.25, 0.1, 1.0]));

    let dt = 1e-3;
    dispatch_penalize(&sim, dt, GAMMA, 1.0);
    let deltas = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    symbi_ib::apply_body_deltas(&mut sim.immersed.as_mut().unwrap().bodies, &deltas, dt);
    // spin was about z, so the drag reduces the z component of the angular-velocity vector.
    let omega = sim.immersed.as_ref().unwrap().bodies.get(0).omega[2];
    assert!(
        omega < OMEGA0,
        "the free spinner should decelerate from drag: {OMEGA0} -> {omega}"
    );
    assert!(omega > 0.0, "one step should not reverse the spin: {omega}");
}

// force-driven TRANSLATION: a two-way rigid obstacle in a +x flow is pushed downstream — the drag
// it exerts on the gas reacts back (mass dv = force_delta dt), so its velocity and position advance
// in the flow direction over one evolve step.
#[test]
fn two_way_body_is_pushed_downstream_by_the_flow() {
    use symbi_ib::SurfaceSpec;
    use symbi_ib::sdf::SdfExpr;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.3, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(
            BodyCollection::new().add(
                Body::rigid_sphere(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.3, 1.0, true)
                    .with_surface(SurfaceSpec::Porous {
                        porosity: 0.0,
                        k_eta_n: 50.0,
                        k_eta_t: 50.0,
                    })
                    .with_two_way_coupling(true),
            ),
        );
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.2, 0.2, 1.0]));

    let dt = 1e-3;
    dispatch_penalize(&sim, dt, GAMMA, 1.0);
    let deltas = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    symbi_ib::apply_body_deltas(&mut sim.immersed.as_mut().unwrap().bodies, &deltas, dt);
    let b = sim.immersed.as_ref().unwrap().bodies.get(0);
    assert!(
        b.velocity[0] > 0.0,
        "the +x flow should accelerate the body downstream: {}",
        b.velocity[0]
    );
    assert!(
        b.position[0] > 0.0,
        "the body should drift downstream: {}",
        b.position[0]
    );
    // the transverse push is negligible for a symmetric obstacle in an axis-aligned flow.
    assert!(
        b.velocity[1].abs() < b.velocity[0],
        "the drift is predominantly downstream"
    );
}

// ARBITRARY-AXIS 3D spin: a box spinning about the X axis in still 3D fluid
// must book its reaction torque about X — proving the mask uses Rodrigues(axis, angle) and the wall
// velocity is omega x r about the config axis.
#[test]
fn spinning_box_about_x_axis_imparts_torque_3d() {
    use symbi_ib::SurfaceSpec;
    use symbi_ib::sdf::SdfExpr;
    type Sim3 = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let n = 24;
    let dx = 2.0 * L / n as f64;
    let mut sim = Sim3::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n, n])
        .origin([-L, -L, -L])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("3d sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(
            BodyCollection::new().add(
                Body::rigid_sphere(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.4, 1.0, true)
                    .with_surface(SurfaceSpec::Porous {
                        porosity: 0.0,
                        k_eta_n: 50.0,
                        k_eta_t: 50.0,
                    })
                    .with_spin_about(5.0, Tensor::new([1.0, 0.0, 0.0])),
            ),
        );
    // a box elongated in y, so spinning about x sweeps its arms through z and grabs the gas.
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.1, 0.3, 0.1]));

    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(
        d[0].mass_delta, 0.0,
        "a sealed spinning wall removes no mass"
    );
    let tau = d[0].torque_delta;
    assert!(
        tau[0].abs() > 1e-6,
        "spin about x should book a torque about x: {tau:?}"
    );
    // the reaction is about the spin axis (x), whichever axis the config names.
    assert!(
        tau[0].abs() > 10.0 * tau[2].abs(),
        "the x-axis spin torque must dominate the z component: {tau:?}"
    );
}

// an arbitrary-shape rigid wall on a CURVILINEAR (r-phi disk) grid: the mask distance is physical
// (the coordinate centroid maps to Cartesian), and off-Cartesian the dispatch runs the whole
// interior. a box at an off-origin Cartesian point must penalize the flow (mass 0, force nonzero).
#[test]
fn shaped_box_rigid_wall_cylindrical_penalizes() {
    use std::f64::consts::PI;
    use symbi_geometry::Cylindrical;
    use symbi_ib::SurfaceSpec;
    use symbi_ib::sdf::SdfExpr;
    use symbi_sim::state::CylPlane;
    type CylSim = SimState<Newtonian, 2, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, nphi) = (40usize, 32usize);
    let mut sim = CylSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, nphi])
        .origin([1.0, 0.0])
        .spacing([0.1, 2.0 * PI / nphi as f64])
        .cyl_plane(CylPlane::RPhi)
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .allocate()
        .expect("cyl sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.2, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(
            BodyCollection::new().add(
                // the body sits at a Cartesian point inside the annulus (physical radius ~2.06).
                Body::rigid_sphere(
                    0,
                    Tensor::new([2.0, 0.5]),
                    Tensor::zeros(),
                    1.0,
                    0.5,
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
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.4, 0.4, 1.0]));

    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(
        d[0].mass_delta, 0.0,
        "a sealed shaped wall on a cyl grid removes no mass"
    );
    let f = d[0].force_delta;
    assert!(
        (f[0] * f[0] + f[1] * f[1]).sqrt() > 1e-8,
        "the curvilinear box wall never penalized: {f:?}"
    );
}

// a SPINNING rigid box in initially-STILL fluid must drag the gas around: the no-slip surface
// relaxes the velocity toward omega x r, so the wall imparts angular momentum and books a nonzero
// reaction torque about z. an identical NON-spinning wall in still fluid imparts ~nothing.
#[test]
fn spinning_box_wall_imparts_torque_to_still_fluid() {
    use symbi_ib::SurfaceSpec;
    use symbi_ib::sdf::SdfExpr;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let build = |omega: f64| -> Sim {
        let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([N, N])
            .origin([-L, -L])
            .spacing([dx, dx])
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
                    Body::rigid_sphere(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.3, 1.0, true)
                        .with_surface(SurfaceSpec::Porous {
                            porosity: 0.0,
                            k_eta_n: 50.0,
                            k_eta_t: 50.0,
                        })
                        .with_spin(omega),
                ),
            );
        // an elongated box, so the spin grabs the fluid.
        sim.immersed.as_mut().unwrap().shapes[0] =
            Some(SdfExpr::<f64, 3>::cuboid([0.0, 0.0, 0.0], [0.25, 0.1, 1.0]));
        sim
    };

    let spinning = build(5.0);
    dispatch_penalize(&spinning, 1e-3, GAMMA, 1.0);
    let d = spinning
        .immersed
        .as_ref()
        .unwrap()
        .diagnostics
        .consolidate();
    assert_eq!(
        d[0].mass_delta, 0.0,
        "a sealed spinning wall removes no mass"
    );
    let tau = d[0].torque_delta[2];
    assert!(
        tau.abs() > 1e-6,
        "the spinning wall imparted no torque to the still fluid: {tau}"
    );

    // baseline: the SAME wall with no spin leaves still fluid still — negligible torque.
    let still = build(0.0);
    dispatch_penalize(&still, 1e-3, GAMMA, 1.0);
    let d0 = still.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert!(
        d0[0].torque_delta[2].abs() < 1e-12,
        "a non-spinning wall in still fluid should impart no torque: {}",
        d0[0].torque_delta[2],
    );
    assert!(
        tau.abs() > 1e3 * d0[0].torque_delta[2].abs().max(1e-30),
        "the spin torque must dominate the still baseline",
    );
}

// the (r, z) AXISYMMETRIC section: an on-axis sphere sink is the axisymmetric
// point body. the mask distance is the plain euclidean |(r, z - z0)| (identity
// section embedding), the cell volume is the ring measure r dr dz, the net
// world force is z only (ring-radial cancels identically), and the axis torque
// is r * f_phi from the drained out-of-plane momentum. gates:
// - mass ledger: gas loss == the booked receipt to machine precision;
// - localization: cells beyond the mask ball are bit-untouched;
// - a swirling gas (v_phi != 0, dof = 3) books a POSITIVE axis torque (the
//   sink absorbs prograde angular momentum) and zero radial world force.
#[test]
fn rz_on_axis_sink_drains_conserves_and_books_axis_torque() {
    use symbi_geometry::Cylindrical;
    type RzSim = SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, nz) = (48usize, 48usize);
    let (r_lo, dr) = (0.05f64, 2.0 / nr as f64);
    let (z_lo, dz) = (-1.0f64, 2.0 / nz as f64);
    let z0 = 0.0f64;
    let racc = 0.25f64;
    let sim = RzSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, nz])
        .origin([r_lo, z_lo])
        .spacing([dr, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("rz sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.3, 0.1]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.0, z0]),
            Tensor::zeros(),
            1.0,
            0.08,
            0.04,
            0.5,
            0.0,
            racc,
        )));

    // ring-measure cell volume mirror: the FULL ring, dv = pi (r_hi^2 - r_lo^2) dz —
    // the same finite-volume weight the kernel's geometry scaffold integrates.
    let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
    let dv = |c: [isize; 2]| -> f64 {
        let rl = r_lo + (c[0] - ilo[0]) as f64 * dr;
        let rh = rl + dr;
        std::f64::consts::PI * (rh * rh - rl * rl) * dz
    };
    let dist = |c: [isize; 2]| -> f64 {
        let r = r_lo + ((c[0] - ilo[0]) as f64 + 0.5) * dr;
        let z = z_lo + ((c[1] - ilo[1]) as f64 + 0.5) * dz;
        (r * r + (z - z0) * (z - z0)).sqrt()
    };
    let mass = |s: &RzSim| -> f64 {
        s.geom
            .interior
            .iter()
            .map(|c| *s.fields.cons.den.view().at(c) * dv(c))
            .sum()
    };
    let before_mass = mass(&sim);
    let far_before: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .filter(|c| dist(*c) > racc + 25.0 * dr.min(dz))
        .map(|c| *sim.fields.cons.den.view().at(c))
        .collect();

    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);

    let after_mass = mass(&sim);
    let d = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
    let lost = before_mass - after_mass;
    assert!(lost > 0.0, "the on-axis rz sink drained nothing");
    assert!(
        ((lost - d[0].mass_delta) / lost).abs() < 1e-11,
        "rz mass ledger broken: gas lost {lost:e} vs receipt {:e}",
        d[0].mass_delta
    );
    let far_after: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .filter(|c| dist(*c) > racc + 25.0 * dr.min(dz))
        .map(|c| *sim.fields.cons.den.view().at(c))
        .collect();
    assert!(
        !far_before.is_empty(),
        "the far-field probe covers no cells"
    );
    for (b, a) in far_before.iter().zip(&far_after) {
        assert!(
            b.to_bits() == a.to_bits(),
            "far field touched: {b:e} -> {a:e}"
        );
    }
    // the swirling gas gives up prograde angular momentum: positive z torque,
    // zero radial world force (ring cancellation is exact in-kernel).
    assert!(
        d[0].torque_delta[2] > 0.0,
        "prograde swirl must book a positive axis torque: {:?}",
        d[0].torque_delta
    );
    assert_eq!(
        d[0].force_delta[0], 0.0,
        "ring-radial receipt must cancel exactly"
    );
    // uniform +z momentum drains: the receipt on the body points along +z.
    assert!(
        d[0].force_delta[1] > 0.0,
        "the +z momentum drain must book a +z receipt"
    );
}

// off-axis and shaped bodies remain outside the axisymmetric contract.
#[test]
#[should_panic(expected = "must sit ON the symmetry axis")]
fn rz_off_axis_body_fails_loud() {
    use symbi_geometry::Cylindrical;
    type RzSim = SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let sim = RzSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([16, 8])
        .origin([1.0, 0.0])
        .spacing([1.0 / 16.0, 0.5 / 8.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("rz sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([1.5, 0.2]),
            Tensor::zeros(),
            1.0,
            0.08,
            0.04,
            0.5,
            0.0,
            0.1,
        )));
    dispatch_penalize(&sim, 1e-3, GAMMA, 1.0);
}
