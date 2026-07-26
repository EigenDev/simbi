// =============================================================================
// refine_body_accretion.rs
//
// per-level immersed bodies on the static-refinement hierarchy (the accretion
// spec sim's wiring): the FINEST level owns the sink + diagnostics, coarser
// levels carry gravity-only proxies (same mass/softening/motion, sink_rate=0),
// body motion advances once per root step on the finest and syncs outward.
//
// gates, on a central black hole in dense quiescent gas with the fine level
// covering the sink:
//   (a) accretion is recorded on the finest level's body; the gravitating
//       mass stays fixed (fixed-potential sink),
//   (b) the coarse proxy has gravity but NO accretion capability,
//   (c) gravity acts on the coarse level too (inward momentum outside the
//       coverage) and the sink drains fluid on the fine level,
//   (d) restriction consistency holds with body sources active,
//   (e) a prescribed binary's positions advance and stay synced across levels.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, BodyKind, SurfaceSpec};
use symbi_ib::sdf::SdfExpr;
use symbi_sim::tracers::ACCRETION_RESERVOIR;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 16;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn dense_gas(sim: &Sim) {
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        sim.fields.cons.den.view_mut().set(c, 2.0);
        for dd in 0..3 {
            sim.fields.cons.mom[dd].view_mut().set(c, 0.0);
        }
        cnrg.view_mut().set(c, 1.0 / (GAMMA - 1.0));
    }
}

/// a 2-level hierarchy on [x0, x0+1)^3 with the middle half refined. periodic
/// walls: an outflow boundary feeds gravity-driven inflow that swamps the sink
/// in the mass budget. the prescribed binary rotates about the coordinate
/// ORIGIN, so the binary test centers the domain there (x0 = -0.5); the static
/// central-mass test keeps the unit box.
fn two_level(x0: f64, bodies: BodyCollection<f64, 3>) -> Hier {
    let dx = 1.0 / N as f64;
    // dense_gas inverted to a prim closure: rho=2, vel=0, pre=1 (nrg=1/(gamma-1)).
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([x0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|_x: [f64; 3]| Prim { rho: 2.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions =
        [RefinementRegion { x_lo: [x0 + 0.25; 3], x_hi: [x0 + 0.75; 3] }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .unwrap()
    .with_bodies(bodies);
    dense_gas(&hier.levels[1].state);
    hier
}

/// composite gas mass: coarse outside the coverage + fine interior.
fn composite_mass(hier: &Hier) -> f64 {
    let mut mass = 0.0;
    for lvl in &hier.levels {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let cov = lvl.coverage.as_ref();
        for c in lvl.state.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            mass += *lvl.state.fields.cons.den.view().at(c) * vol;
        }
    }
    mass
}

#[test]
fn refined_sink_receipt_moves_fine_tracers_into_the_accretion_reservoir() {
    let mut hierarchy = two_level(0.0, BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.5; 3]),
        Tensor::zeros(),
        0.05,
        0.05,
        0.1,
        20.0,
        1.0,
        0.1,
    )));
    hierarchy.attach_mass_tracers(32_768);
    let mass_before = composite_mass(&hierarchy);

    hierarchy.evolve_steps(1).unwrap();

    let removed_mass = mass_before - composite_mass(&hierarchy);
    let tracers = hierarchy.levels[1].state.tracers.as_ref().unwrap();
    let accreted = tracers
        .owner
        .iter()
        .filter(|&&owner| owner == ACCRETION_RESERVOIR)
        .count();
    let represented_mass = accreted as f64 * tracers.weight;

    assert!(removed_mass > 0.0, "the refined sink removed no fluid mass");
    assert!(
        accreted > 100,
        "the refined sink produced no statistically useful tracer receipt: {accreted}"
    );
    assert!(
        (represented_mass - removed_mass).abs() / removed_mass < 0.15,
        "tracer reservoir mass {represented_mass:e} does not represent removed composite mass \
         {removed_mass:e}"
    );
    assert!(
        tracers
            .flags
            .iter()
            .filter(|flags| flags.crossed_sink)
            .count()
            == accreted,
        "sink flags and reservoir ownership disagree"
    );
}

#[test]
fn central_bh_accretes_on_the_finest_level_only() {
    let m_init = 0.05;
    let mut hier = two_level(0.0, BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.5; 3]),
        Tensor::zeros(),
        m_init,
        0.05,
        0.1,  // softening: ~2 coarse cells — the first gravity kick must stay subsonic
        5.0,  // sink_rate
        1.0,  // sink_delta
        0.1,  // accretion_radius — sink sphere [0.4, 0.6] inside the fine level
    )));

    // capability split: the coarse proxy must have gravity but no sink.
    let coarse_body = *hier.levels[0].state.immersed.as_ref().unwrap().bodies.get(0);
    assert!(coarse_body.has_gravity() && !coarse_body.has_accretion());
    let fine_body = *hier.levels[1].state.immersed.as_ref().unwrap().bodies.get(0);
    assert!(fine_body.has_accretion());

    let m0 = composite_mass(&hier);
    hier.evolve_steps(6).unwrap();

    // accretion recorded on the finest; gravitating mass fixed.
    let body = *hier.levels[1].state.immersed.as_ref().unwrap().bodies.get(0);
    assert_eq!(body.mass, m_init, "fixed-potential sink: gravitating mass drifted");
    let BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } = body.kind else {
        panic!("finest body lost its accretion capability");
    };
    assert!(
        total_accreted_mass > 0.0 && total_accreted_mass.is_finite(),
        "no accretion recorded on the finest level: {total_accreted_mass:e}"
    );
    assert!(accretion_rate > 0.0, "mdot not recorded: {accretion_rate:e}");

    // the mass budget: the sink removed fluid (gravity-driven inflow piles gas
    // up at the center, so a local density check is meaningless — the COMPOSITE
    // mass must drop, and by an amount commensurate with the recorded
    // accretion; the feedback reduction runs at root cadence while the source
    // removes per stage, so the two agree only to a factor).
    let fine = &hier.levels[1].state;
    let removed = m0 - composite_mass(&hier);
    assert!(removed > 0.0, "the sink removed no composite mass ({removed:e})");
    let ratio = removed / total_accreted_mass;
    assert!(
        (0.2..=5.0).contains(&ratio),
        "removed mass {removed:e} inconsistent with recorded accretion \
         {total_accreted_mass:e} (ratio {ratio:.2})"
    );

    // gravity acts on the COARSE level outside the coverage: the cell just
    // outside the box on the x axis gains momentum toward the center.
    let coarse = &hier.levels[0].state;
    let momx_out = *coarse.fields.cons.mom[0].view().at([3, 8, 8]);
    assert!(momx_out > 0.0, "no inward pull outside the coverage (momx {momx_out:e})");

    // restriction consistency with body sources active.
    let cov = hier.levels[0].coverage.as_ref().unwrap();
    for c in cov.iter() {
        let coarse_den = *coarse.fields.cons.den.view().at(c);
        let mut sum = 0.0;
        for oi in 0..2isize {
            for oj in 0..2isize {
                for ok in 0..2isize {
                    sum += *fine.fields.cons.den.view().at([2 * c[0] + oi, 2 * c[1] + oj, 2 * c[2] + ok]);
                }
            }
        }
        let rel = ((coarse_den - sum / 8.0) / coarse_den).abs();
        assert!(rel < 1e-12, "restriction drift at {c:?}: rel {rel:e}");
    }

    // physical everywhere.
    for (ll, lvl) in hier.levels.iter().enumerate() {
        for c in lvl.state.geom.interior.iter() {
            let den = *lvl.state.fields.cons.den.view().at(c);
            assert!(den.is_finite() && den > 0.0, "level {ll} {c:?}: den {den}");
        }
    }
}

#[test]
fn shaped_rigid_wall_crossing_refinement_boundary_acts_on_both_levels() {
    let radius = 0.12_f64;
    let half_height = 0.35_f64;
    let bound = (radius * radius + half_height * half_height).sqrt();
    let body = Body::rigid_sphere(
        0,
        Tensor::new([0.5; 3]),
        Tensor::zeros(),
        1.0,
        bound,
        0.1,
        false,
    )
    .with_surface(SurfaceSpec::Porous {
        porosity: 0.0,
        k_eta_n: 1.0e3,
        k_eta_t: 1.0e3,
    });
    let mut hierarchy = two_level(0.0, BodyCollection::new().add(body));
    hierarchy.attach_body_shapes(vec![Some(SdfExpr::capped_cylinder(
        [0.0; 3],
        radius,
        half_height,
    ))]);

    for level in &hierarchy.levels {
        let energy = level.state.fields.cons.nrg_field().unwrap();
        for cell in level.state.geom.interior.iter() {
            level.state.fields.cons.den.view_mut().set(cell, 2.0);
            level.state.fields.cons.mom[0].view_mut().set(cell, 2.0);
            level.state.fields.cons.mom[1].view_mut().set(cell, 0.0);
            level.state.fields.cons.mom[2].view_mut().set(cell, 0.0);
            energy.view_mut().set(cell, 2.5);
        }
    }

    hierarchy.evolve_steps(1).unwrap();

    let coarse_momentum = *hierarchy.levels[0]
        .state
        .fields
        .cons
        .mom[0]
        .view()
        .at([8, 8, 2]);
    let fine_momentum = *hierarchy.levels[1]
        .state
        .fields
        .cons
        .mom[0]
        .view()
        .at([16, 16, 16]);
    assert!(
        coarse_momentum < 1.9,
        "uncovered coarse cylinder segment was not rigidly enforced: {coarse_momentum}"
    );
    assert!(
        fine_momentum < 1.9,
        "fine cylinder segment was not rigidly enforced: {fine_momentum}"
    );
}

#[test]
fn prescribed_binary_positions_stay_synced_across_levels() {
    // the prescribed keplerian advance rotates about the ORIGIN: domain
    // centered there, binary on the x axis.
    let sep = 0.08f64;
    let omega_v = (0.05 / sep.powi(3)).sqrt() * (sep / 2.0);
    let bodies = BodyCollection::new()
        .add(Body::black_hole(
            0, Tensor::new([-sep / 2.0, 0.0, 0.0]), Tensor::new([0.0, -omega_v, 0.0]),
            0.025, 0.03, 0.08, 5.0, 1.0, 0.05,
        ))
        .add(Body::black_hole(
            1, Tensor::new([sep / 2.0, 0.0, 0.0]), Tensor::new([0.0, omega_v, 0.0]),
            0.025, 0.03, 0.08, 5.0, 1.0, 0.05,
        ))
        .with_binary_params(symbi_ib::BinaryParams::new(0.05, sep, 0.0, 1.0))
        .as_binary();
    let p0 = *bodies.get(0);
    let mut hier = two_level(-0.5, bodies);

    hier.evolve_steps(4).unwrap();

    let fine_b = *hier.levels[1].state.immersed.as_ref().unwrap().bodies.get(0);
    let coarse_b = *hier.levels[0].state.immersed.as_ref().unwrap().bodies.get(0);
    // the prescribed orbit moved the bodies.
    let moved: f64 = (0..3).map(|ax| (fine_b.position[ax] - p0.position[ax]).powi(2)).sum::<f64>().sqrt();
    assert!(moved > 1e-6, "binary did not advance (moved {moved:e})");
    // and every level sees the same positions/velocities.
    for ax in 0..3 {
        assert_eq!(
            fine_b.position[ax].to_bits(), coarse_b.position[ax].to_bits(),
            "body positions diverged across levels on axis {ax}"
        );
        assert_eq!(
            fine_b.velocity[ax].to_bits(), coarse_b.velocity[ax].to_bits(),
            "body velocities diverged across levels on axis {ax}"
        );
    }
}

// the ISOTHERMAL twin of the finest-owns-bodies gate: the iso kernel set's
// body source / feedback / penalize run on a refined hierarchy, accretion is
// recorded on the finest level, and the gravitating mass stays fixed — the
// certificate for un-gating iso + refinement + bodies in the binding.
#[test]
fn central_bh_accretes_on_the_finest_level_iso() {
    use symbi::regimes::substrate::IsoSubstrateKernelSet;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::isothermal::IsoNewtonian;
    use symbi_hydro::state::PrimG;
    type ISim = SimState<IsoNewtonian, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    type IKset = IsoSubstrateKernelSet<HostMemory, f64, 3>;
    let cs = 1.0;
    let m_init = 0.05;
    let dx = 1.0 / N as f64;
    let coarse = ISim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|_x: [f64; 3]| PrimG::<f64, 3, IsoModel> {
            rho: 2.0,
            vel: Tensor::new([0.0; 3]),
            pre: Default::default(),
        })
        .build();
    let ck = IKset::new(cs, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion { x_lo: [0.25; 3], x_hi: [0.75; 3] }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        IKset::new(cs, CFL, &s.geom.allocated)
    })
    .unwrap()
    .with_bodies(BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.5; 3]),
        Tensor::zeros(),
        m_init,
        0.05,
        0.1,
        5.0,
        1.0,
        0.1,
    )));

    // seed the fine level's gas (the harness convention — with_refinement's
    // IC prolongation is not relied on here); iso has no energy slot.
    {
        let fine = &hier.levels[1].state;
        for c in fine.geom.interior.iter() {
            fine.fields.cons.den.view_mut().set(c, 2.0);
            for dd in 0..3 {
                fine.fields.cons.mom[dd].view_mut().set(c, 0.0);
            }
        }
    }

    let coarse_body = *hier.levels[0].state.immersed.as_ref().unwrap().bodies.get(0);
    assert!(coarse_body.has_gravity() && !coarse_body.has_accretion());

    hier.evolve_steps(6).unwrap();

    let body = *hier.levels[1].state.immersed.as_ref().unwrap().bodies.get(0);
    assert_eq!(body.mass, m_init, "fixed-potential sink: gravitating mass drifted");
    let BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } = body.kind else {
        panic!("finest body lost its accretion capability");
    };
    assert!(
        total_accreted_mass > 0.0 && total_accreted_mass.is_finite(),
        "no iso accretion recorded on the finest level: {total_accreted_mass:e}"
    );
    assert!(accretion_rate > 0.0, "iso mdot not recorded: {accretion_rate:e}");

    // every level finite (the restriction + penalize interplay held together).
    for lvll in &hier.levels {
        for c in lvll.state.geom.interior.iter() {
            assert!(lvll.state.fields.cons.den.view().at(c).is_finite());
        }
    }
}
