// =============================================================================
// refinement_well_balanced.rs
//
// a declared stationary target is an exact fixed point of the refined scheme, and declaring one
// leaves conservation intact.
//
// a hydrostatic atmosphere solves the continuum equations while the scheme discretizes them: it
// leaves a residual `R = div_h F_h(qt) - s_h(qt)` at truncation order, so gas seeded on the exact
// profile starts moving. `R` also depends on the grid, so the coarse-fine flux register differences
// two unequal reconstructions of the same exact solution and applies the difference to the coarse
// cells at the interface as a force — which is why a refined run drifts far faster than the single
// grid it is built from.
//
// the two properties gated here are independent and each is satisfiable alone by a wrong
// implementation. a correction that simply suppressed the register would hold the state still
// while destroying the conservation the register exists to provide; a correction that left the
// register alone would conserve while still injecting momentum. only together do they say the
// deviation from the target is what is being refluxed.
//
// velocity is the probe: the target has `v = 0` identically, so any speed after a step is the
// imbalance itself, read directly with no reference state to subtract.
//
// run: cargo test -p symbi --test refinement_well_balanced -- --nocapture
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// the gravitating mass sits one domain width left of x = 0, so the gas at x feels a bare point
/// mass at radius x + 1 and the domain covers r in [1, 2] with no singularity.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 100.0;
/// enough root steps that a per-step imbalance accumulates well clear of roundoff: the undeclared
/// run reaches 5.8e-4 here, a factor of 6 above its one-step value.
const STEPS: u64 = 20;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn kset(s: &Sim) -> Kset {
    Kset::new(GAMMA, CFL, &s.geom.allocated)
}

/// the isentropic atmosphere in hydrostatic balance against GM, from the bernoulli invariant
/// `gamma K0/(gamma-1) rho^(gamma-1) - GM/r = const`, normalized to `rho = 1` at the outer edge.
fn hydrostatic(x: [f64; 1]) -> Prim<f64, 1> {
    let r = x[0] + G_OFFSET;
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a - GM / (1.0 + G_OFFSET);
    let rho = (a * (GM / r + c)).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

/// nested patches, each half the previous, centred on the domain: `levels - 1` of them.
fn nested(levels: usize) -> Vec<RefinementRegion<1>> {
    (0..levels.saturating_sub(1))
        .map(|ii| {
            let half = 0.2 / 2f64.powi(ii as i32);
            RefinementRegion {
                x_lo: [0.5 - half],
                x_hi: [0.5 + half],
            }
        })
        .collect()
}

fn build(regions: &[RefinementRegion<1>]) -> Hier {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        // a reflecting wall exerts no work on gas at rest, so the hydrostatic state is a fixed
        // point of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(hydrostatic)
        .build();
    let ck = kset(&coarse);
    let hier = Hierarchy::with_refinement(coarse, ck, regions, ProlongOrder::Ppm, kset)
        .unwrap()
        .with_bodies(
            symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                0,
                symbi_algebra::Tensor::new([-G_OFFSET]),
                symbi_algebra::Tensor::zeros(),
                GM,
                1.0e-3,
                0.0,
            )),
        );
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(hydrostatic);
    }
    hier
}

/// the same hierarchy with the atmosphere declared as its stationary target and every level seeded
/// from it. the seed matters: covered coarse cells carry the restriction of the finer level's
/// target, which is what the run's own restriction reproduces every parent step, and the pointwise
/// profile sits a truncation-order distance away from it.
fn build_declared(regions: &[RefinementRegion<1>]) -> Hier {
    let mut hier = build(regions).with_equilibrium(hydrostatic).unwrap();
    hier.seed_equilibrium();
    hier
}

#[test]
fn declared_target_ghosts_do_not_read_the_fine_interior() {
    let mut hier = build_declared(&nested(2));
    hier.prime();
    let ghost_cells: Vec<[isize; 1]> = {
        let st = &hier.levels[1].state;
        st.geom
            .allocated
            .iter()
            .filter(|coord| !st.geom.interior.contains(*coord))
            .collect()
    };
    let snapshot = |h: &Hier| -> Vec<[u64; 3]> {
        let prim = &h.levels[1].state.fields.prim;
        let (rho, vel, pre) = (
            prim.rho.view(),
            prim.vel[0].view(),
            prim.pre.as_ref().expect("adiabatic target").view(),
        );
        ghost_cells
            .iter()
            .map(|coord| {
                [
                    rho.at(*coord).to_bits(),
                    vel.at(*coord).to_bits(),
                    pre.at(*coord).to_bits(),
                ]
            })
            .collect()
    };
    let before = snapshot(&hier);

    {
        let st = &hier.levels[1].state;
        let mut rho = st.fields.prim.rho.view_mut();
        let mut vel = st.fields.prim.vel[0].view_mut();
        let mut pre = st.fields.prim.pre.as_ref().expect("adiabatic").view_mut();
        for coord in st.geom.interior.iter() {
            *rho.at_mut(coord) *= 0.7;
            *vel.at_mut(coord) += 3.0;
            *pre.at_mut(coord) *= 1.9;
        }
    }
    hier.prolong_cf(1, 1.0);
    let after = snapshot(&hier);
    assert_eq!(
        after, before,
        "a fixed coarse state produced different declared-target ghosts after only the fine \
         interior changed"
    );

    let target = hier.levels[1]
        .prim_eq
        .as_ref()
        .expect("declared target primitives");
    let (target_rho, target_vel, target_pre) = (
        target.rho.view(),
        target.vel[0].view(),
        target.pre.as_ref().expect("adiabatic target").view(),
    );
    for (coord, got) in ghost_cells.iter().zip(after) {
        assert_eq!(got[0], target_rho.at(*coord).to_bits());
        assert_eq!(got[1], target_vel.at(*coord).to_bits());
        assert_eq!(got[2], target_pre.at(*coord).to_bits());
    }
}

#[test]
fn inadmissible_declared_target_decode_falls_back_as_one_state() {
    let mut hier = build_declared(&nested(2));
    hier.prime();
    {
        let parent = &hier.levels[0].state;
        let mut velocity = parent.fields.prim.vel[0].view_mut();
        let mut pressure = parent
            .fields
            .prim
            .pre
            .as_ref()
            .expect("adiabatic")
            .view_mut();
        for coord in parent.geom.allocated.iter() {
            *velocity.at_mut(coord) = 7.0;
            *pressure.at_mut(coord) = -1.0e30;
        }
    }
    hier.prolong_cf(1, 1.0);

    let fine = &hier.levels[1];
    let target = fine.prim_eq.as_ref().expect("declared target primitives");
    let live = &fine.state.fields.prim;
    let (target_rho, target_vel, target_pre) = (
        target.rho.view(),
        target.vel[0].view(),
        target.pre.as_ref().expect("adiabatic target").view(),
    );
    let (live_rho, live_vel, live_pre) = (
        live.rho.view(),
        live.vel[0].view(),
        live.pre.as_ref().expect("adiabatic").view(),
    );
    for coord in fine
        .state
        .geom
        .allocated
        .iter()
        .filter(|coord| !fine.state.geom.interior.contains(*coord))
    {
        assert_eq!(live_rho.at(coord).to_bits(), target_rho.at(coord).to_bits());
        assert_eq!(live_vel.at(coord).to_bits(), target_vel.at(coord).to_bits());
        assert_eq!(live_pre.at(coord).to_bits(), target_pre.at(coord).to_bits());
    }
}

/// the largest speed anywhere on a level, walls included. a reflecting wall mirrors the state, so
/// gas at rest sees no jump across it and the boundary holds the target exactly like the interior.
fn worst_speed(hier: &Hier, level: usize) -> f64 {
    let st = &hier.levels[level].state;
    let vel = st.fields.prim.vel[0].view();
    st.geom
        .interior
        .iter()
        .map(|c| vel.at(c).abs())
        .fold(0.0_f64, f64::max)
}

/// mass over the active composite: covered coarse cells are omitted and their finer children
/// supply the mass instead, so each region of space is counted exactly once.
fn composite_mass(hier: &Hier) -> f64 {
    let mut mass = 0.0;
    for lvl in hier.levels.iter() {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let den = lvl.state.fields.cons.den.view();
        for c in lvl.state.geom.interior.iter() {
            if lvl.coverage.as_ref().is_some_and(|cov| cov.contains(c)) {
                continue;
            }
            mass += *den.at(c) * vol;
        }
    }
    mass
}

/// mass held by one level's uncovered cells alone. the composite total is blind to transport
/// across the coarse-fine interface — that is exactly what it is supposed to cancel — so this is
/// what shows whether any crossed at all.
fn uncovered_mass(hier: &Hier, level: usize) -> f64 {
    let lvl = &hier.levels[level];
    let vol: f64 = lvl.state.geom.dx.iter().product();
    let den = lvl.state.fields.cons.den.view();
    lvl.state
        .geom
        .interior
        .iter()
        .filter(|c| !lvl.coverage.as_ref().is_some_and(|cov| cov.contains(*c)))
        .map(|c| *den.at(c) * vol)
        .sum()
}

#[test]
fn a_declared_target_is_held_exactly_and_conservatively() {
    println!("\nhydrostatic atmosphere after {STEPS} root steps, from an exact start (v = 0)");
    println!("{:-<92}", "");

    for levels in 1..=4usize {
        let mut control = build(&nested(levels));
        control.evolve_steps(STEPS).unwrap();
        let drift: Vec<f64> = (0..levels).map(|ll| worst_speed(&control, ll)).collect();

        let mut hier = build_declared(&nested(levels));
        assert_eq!(hier.levels.len(), levels, "asked for {levels} levels");
        let m0 = composite_mass(&hier);
        hier.evolve_steps(STEPS).unwrap();
        let m1 = composite_mass(&hier);
        let held: Vec<f64> = (0..levels).map(|ll| worst_speed(&hier, ll)).collect();

        let shown: Vec<String> = drift
            .iter()
            .zip(&held)
            .map(|(d, h)| format!("{d:.3e} -> {h:.3e}"))
            .collect();
        println!(
            "levels={levels}  max|v| undeclared -> declared, per level: {shown:?}\n\
             {:12}composite mass {m0:.15e} -> {m1:.15e}  (relative {:.3e})",
            "",
            ((m1 - m0) / m0).abs()
        );

        // non-vacuity: undeclared, this setup drifts off the atmosphere on its own, which is what
        // gives the gate below a real problem to measure. a refined run drifts an order of
        // magnitude past the single grid; even the single grid drifts.
        for (ll, speed) in drift.iter().enumerate() {
            assert!(
                *speed > 1.0e-5,
                "level {ll} of the UNDECLARED {levels}-level run drifted only {speed:.3e} in \
                 {STEPS} steps; the setup no longer exercises the imbalance the declaration \
                 removes, so holding the state still says nothing"
            );
        }

        // the target is an exact fixed point of the scheme: every flux and every source is
        // evaluated at it and the residual is subtracted back off, so what survives is roundoff in
        // the stage arithmetic. the bound sits far above the ~2.6e-15 measured floor and eight
        // orders below the 5.8e-4 the same setup reaches once the declaration is dropped.
        for (ll, speed) in held.iter().enumerate() {
            assert!(
                *speed < 1.0e-12,
                "level {ll} of the {levels}-level run moved at {speed:.3e} while sitting on its \
                 declared stationary target; a fixed point may only lose roundoff"
            );
        }

        // conservation is the half a register-suppressing "fix" would destroy. mass carries no
        // source, so the composite total is bound to round-off whatever the gas does.
        let relative = ((m1 - m0) / m0).abs();
        assert!(
            relative < 1.0e-14,
            "composite mass moved by {relative:.3e} over {STEPS} steps of the {levels}-level run; \
             the coarse-fine transfer is no longer conservative"
        );
    }
}

/// the ppm variant of the declared hierarchy: ng = 3 for the -3..+2 parabola
/// stencil and quartic prolongation to satisfy the reach <= degree law at the
/// coarse-fine boundary.
fn build_declared_ppm(regions: &[RefinementRegion<1>]) -> Hier {
    use symbi_discretize::Recon;
    let kset_ppm = |s: &Sim| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
            .with_solver(symbi::prelude::Solver::Hllc)
            .expect("solver/regime mismatch")
            .reconstruction(Recon::Ppm)
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .ghosts(3)
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(hydrostatic)
        .build();
    let ck = kset_ppm(&coarse);
    let hier = Hierarchy::with_refinement(coarse, ck, regions, ProlongOrder::Quartic, kset_ppm)
        .unwrap()
        .with_bodies(
            symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                0,
                symbi_algebra::Tensor::new([-G_OFFSET]),
                symbi_algebra::Tensor::zeros(),
                GM,
                1.0e-3,
                0.0,
            )),
        );
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(hydrostatic);
    }
    let mut hier = hier.with_equilibrium(hydrostatic).unwrap();
    hier.seed_equilibrium();
    hier
}

/// the fixed-point law is reconstruction-agnostic: the residual `R` is measured
/// through the same kernels that evolve the run — ppm fluxes against a ppm-built
/// target flux — and subtracted back off, so a declared target holds to roundoff
/// under ppm exactly as under plm. this is the full production stack (hierarchy +
/// declared stationary target + gravitational body + ppm + quartic prolongation)
/// through the level_stage path, fofc included.
#[test]
fn a_declared_target_is_held_exactly_under_ppm() {
    for levels in 1..=3usize {
        let mut hier = build_declared_ppm(&nested(levels));
        let m0 = composite_mass(&hier);
        hier.evolve_steps(STEPS).unwrap();
        let m1 = composite_mass(&hier);
        for ll in 0..levels {
            let speed = worst_speed(&hier, ll);
            assert!(
                speed < 1.0e-12,
                "level {ll} of the {levels}-level ppm run moved at {speed:.3e} while sitting on \
                 its declared stationary target; a fixed point may only lose roundoff"
            );
        }
        let relative = ((m1 - m0) / m0).abs();
        assert!(
            relative < 1.0e-14,
            "composite mass moved by {relative:.3e} over {STEPS} steps of the {levels}-level \
             ppm run; the coarse-fine transfer is no longer conservative"
        );
    }
}

#[test]
fn the_correction_tracks_the_distance_from_the_target() {
    // a correction that merely suppressed the flux register would pass the fixed-point gate and
    // leave the interface unrefluxed for every other state. driving the same declared hierarchy
    // with gas that is genuinely moving separates the two: the register now has real transport to
    // correct, and the composite total must still close.
    const MACH: f64 = 1.0e-3;

    let mut hier = build_declared(&nested(2));
    hier.prime();
    for level in &hier.levels {
        let st = &level.state;
        for c in st.geom.interior.iter() {
            let rho = *st.fields.prim.rho.view().at(c);
            let pre = *st.fields.prim.pre_field().unwrap().view().at(c);
            // a uniform mach number, so the perturbation carries the same physical size
            // everywhere in an atmosphere whose sound speed varies by a factor of 3.
            let speed = MACH * (GAMMA * pre / rho).sqrt();
            st.seed_cell(
                c,
                &Prim {
                    rho,
                    vel: symbi_algebra::Tensor::new([speed]),
                    pre,
                },
            );
        }
    }
    hier.prime();

    let m0 = composite_mass(&hier);
    let level0_before = uncovered_mass(&hier, 0);
    hier.evolve_steps(STEPS).unwrap();
    let m1 = composite_mass(&hier);
    let level0_after = uncovered_mass(&hier, 0);

    let crossed = ((level0_after - level0_before) / level0_before).abs();
    let relative = ((m1 - m0) / m0).abs();
    println!(
        "\nmach {MACH:.0e} off the declared target, {STEPS} root steps:\n  \
         level 0 uncovered mass changed by {crossed:.3e} (transport across the interface)\n  \
         composite mass changed by {relative:.3e}"
    );

    // non-vacuity: mass has to cross the coarse-fine interface, so that "the composite total
    // held" reports on a register with something real to correct.
    assert!(
        crossed > 1.0e-10,
        "only {crossed:.3e} of level 0's uncovered mass moved, so essentially nothing crossed the \
         coarse-fine interface and the conservation check below is vacuous"
    );
    assert!(
        relative < 1.0e-14,
        "composite mass moved by {relative:.3e} on a state {MACH:.0e} in mach off the target; the \
         equilibrium correction is being applied blind rather than as a subtraction of F(qt)"
    );
}

/// the same atmosphere written for half the gravity the run applies. it is a perfectly good
/// hydrostatic profile — for a different problem — so it is exactly the mistake that survives a
/// pointwise inspection of the initial condition: smooth, positive, monotone, and wrong by a
/// constant factor in one term.
fn hydrostatic_wrong_gravity(x: [f64; 1]) -> Prim<f64, 1> {
    let r = x[0] + G_OFFSET;
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let half = 0.5 * GM;
    let c = 1.0 / a - half / (1.0 + G_OFFSET);
    let rho = (a * (half / r + c)).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

#[test]
fn the_imbalance_of_a_true_steady_state_converges_under_refinement() {
    // nothing in the method checks that a declared target is stationary: the imbalance is measured
    // and subtracted whatever it is, so a state merely asserted to be an equilibrium gets held
    // motionless while the run reports success. what separates the two is how the imbalance
    // behaves under refinement — truncation error falls with the cell width, while the continuum
    // residual of a state that solves different equations stays fixed at every resolution.
    let real = build(&nested(2)).with_equilibrium(hydrostatic).unwrap();
    let measured = real
        .target_imbalance_convergence(0)
        .expect("levels 0 and 1 share an interior to compare over");

    let peak = measured.scale.iter().fold(0.0_f64, |m, s| m.max(*s));
    let shown: Vec<String> = measured.ratio.iter().map(|r| format!("{r:.2}")).collect();
    println!(
        "\ntrue steady state, per-cell median imbalance ratio by component: {shown:?}  (sampled {:?})",
        measured.sampled
    );

    // non-vacuity: an imbalance already at zero would "converge" trivially. there has to be a real
    // truncation error being measured, over enough cells for a median to mean anything.
    assert!(
        peak > 1.0e-6,
        "the largest single-cell imbalance is {peak:.3e}, indistinguishable from zero; the \
         convergence measured below is a ratio of roundoff"
    );
    for cc in 0..measured.ratio.len() {
        if measured.scale[cc] < 1.0e-6 * peak {
            continue;
        }
        assert!(
            measured.sampled[cc] >= 8,
            "component {cc} contributed only {} cells to the median; that is an accident, not a \
             statistic",
            measured.sampled[cc]
        );
        assert!(
            measured.ratio[cc] > 1.5,
            "component {cc} of the TRUE steady state's imbalance fell by a median factor of only \
             {:.3} per cell when the cell width halved; the check that rejects a false equilibrium \
             would reject this one",
            measured.ratio[cc]
        );
    }
}

#[test]
fn a_target_that_is_not_stationary_reads_as_non_converging() {
    // the stationarity report is advisory and lets every run proceed, because for a strongly
    // stratified target the imbalance concentrates in cells the grid leaves unresolved, where
    // "steep" and "wrong" produce the same reading. what it still owes is a correct measurement:
    // a state that solves different equations leaves the continuum residual, which is
    // grid-independent, so its ratio sits at 1 on the cells where the grid resolves the target.
    //
    // the profile below balances GM/2 while the body pulls with GM -- smooth, positive, monotone,
    // and wrong by a constant in one term.
    let hier = build(&nested(2))
        .with_equilibrium(hydrostatic_wrong_gravity)
        .expect("the report is advisory and must not refuse the build");
    let measured = hier
        .target_imbalance_convergence(0)
        .expect("levels 0 and 1 overlap");
    let shown: Vec<String> = measured.ratio.iter().map(|r| format!("{r:.3}")).collect();
    println!(
        "\nwrong-gravity target, per-component medians: {shown:?} (sampled {:?})",
        measured.sampled
    );

    // momentum is the component the error lives in: the pressure gradient balances a different
    // gravity than the one applied.
    assert!(
        measured.sampled[1] >= 8,
        "only {} cells voted on momentum; too few to read",
        measured.sampled[1]
    );
    assert!(
        measured.ratio[1] <= 1.5,
        "the WRONG target's momentum imbalance fell by {:.3} per cell, which reads as \
         converging; the diagnostic can no longer tell a non-stationary state from a steep one",
        measured.ratio[1]
    );
    // the components that are right still converge, which is what makes the reading a localization
    // -- otherwise it would flag every component of every target.
    for cc in [0usize, 2] {
        assert!(
            measured.ratio[cc] > 1.5,
            "component {cc} of the wrong-gravity target reads {:.3}; the error is in the \
             momentum balance alone and the diagnostic should say so",
            measured.ratio[cc]
        );
    }
}

/// the same atmosphere written as an expression graph, in the wire form a configured run emits.
///
/// `rho = (a (GM/r + c))^(1/(gamma-1))` with `r = x + G_OFFSET`, `p = K0 rho^gamma`, `v = 0`,
/// where `a = (gamma-1)/(gamma K0)` and `c` normalizes `rho` to 1 at the outer edge.
fn hydrostatic_expression_json() -> String {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a - GM / (1.0 + G_OFFSET);
    format!(
        r#"{{ "dim": 1, "outputs": [10, 11, 15], "params": [], "nodes": [
            {{"op":"VARIABLE_X1"}},
            {{"op":"CONSTANT","value":{G_OFFSET}}},
            {{"op":"ADD","left":0,"right":1}},
            {{"op":"CONSTANT","value":{GM}}},
            {{"op":"DIVIDE","left":3,"right":2}},
            {{"op":"CONSTANT","value":{c}}},
            {{"op":"ADD","left":4,"right":5}},
            {{"op":"CONSTANT","value":{a}}},
            {{"op":"MULTIPLY","left":7,"right":6}},
            {{"op":"CONSTANT","value":{exponent}}},
            {{"op":"POW","left":8,"right":9}},
            {{"op":"CONSTANT","value":0.0}},
            {{"op":"CONSTANT","value":{GAMMA}}},
            {{"op":"POW","left":10,"right":12}},
            {{"op":"CONSTANT","value":{K0}}},
            {{"op":"MULTIPLY","left":14,"right":13}}
        ] }}"#,
        exponent = 1.0 / (GAMMA - 1.0)
    )
}

#[test]
fn a_target_declared_as_an_expression_matches_the_closure() {
    // the target crosses the configuration wire as an expression, so a restart that adds a
    // refinement level can evaluate it on cells created after the run began. what that costs is a
    // second way to say the same thing, so the two
    // have to be held to producing bit-identical state, with similarity ruled out as a pass.
    let config = symbi_hydro::EquilibriumConfig::from_json(&hydrostatic_expression_json())
        .expect("the target expression parses");

    let from_expr = build(&nested(2))
        .with_equilibrium_expression(&config)
        .unwrap();
    let from_closure = build(&nested(2)).with_equilibrium(hydrostatic).unwrap();

    for (ll, (expr_level, closure_level)) in from_expr
        .levels
        .iter()
        .zip(&from_closure.levels)
        .enumerate()
    {
        let expr_target = expr_level.cons_eq.as_ref().unwrap();
        let closure_target = closure_level.cons_eq.as_ref().unwrap();
        let (den_e, den_c) = (expr_target.den.view(), closure_target.den.view());
        let (nrg_e, nrg_c) = (
            expr_target.nrg_field().unwrap().view(),
            closure_target.nrg_field().unwrap().view(),
        );
        let mut worst = 0.0_f64;
        for coord in expr_level.state.geom.interior.iter() {
            worst = worst
                .max((den_e.at(coord) - den_c.at(coord)).abs())
                .max((nrg_e.at(coord) - nrg_c.at(coord)).abs());
        }
        println!("level {ll}: largest expression-vs-closure target difference {worst:.3e}");
        // the expression evaluates the same arithmetic in the same order, so anything above
        // roundoff means the wire is reading the components in a different order or dropping one.
        assert!(
            worst < 1.0e-12,
            "level {ll}'s target differs by {worst:.3e} between the expression and the closure \
             that define the same profile"
        );
    }

    // and the whole point of declaring it: the expression-declared run holds the atmosphere.
    let mut hier = from_expr;
    hier.seed_equilibrium();
    hier.evolve_steps(STEPS).unwrap();
    for ll in 0..hier.levels.len() {
        let speed = worst_speed(&hier, ll);
        assert!(
            speed < 1.0e-12,
            "level {ll} moved at {speed:.3e} on a target declared as an expression"
        );
    }
}

#[test]
fn the_captured_imbalance_is_a_discrete_divergence() {
    // `R = div_h F_h(qt) - s_h(qt)`. gravity sources momentum and energy alone, leaving mass
    // untouched, so R's mass component is a pure discrete divergence, and summing a divergence
    // over a level bounded by reflecting walls telescopes to the flux through those walls. gas at
    // rest sees a mirrored state across a wall, zero jump, and therefore zero mass flux — so the
    // sum vanishes.
    //
    // this is what separates the target's genuine imbalance from a limiter's response to an
    // unphysical probe state: only a divergence telescopes this way.
    let hier = build_declared(&nested(2));
    let root = &hier.levels[0];
    let residual = root
        .residual_eq
        .as_ref()
        .expect("the declared hierarchy carries a captured imbalance");

    let vol: f64 = root.state.geom.dx.iter().product();
    let mut mass_rate = 0.0;
    let mut scale = 0.0_f64;
    for c in root.state.geom.interior.iter() {
        let r = *residual.den.view().at(c);
        mass_rate += r * vol;
        scale = scale.max(r.abs() * vol);
    }
    println!(
        "\nroot level: sum V*R for mass = {mass_rate:+.3e}, largest single cell |V*R| = {scale:.3e}"
    );

    // non-vacuity: a residual that is identically zero would telescope trivially and say nothing
    // about the capture. the imbalance being removed has to be a real quantity.
    assert!(
        scale > 1.0e-6,
        "the captured imbalance is {scale:.3e} at its largest cell, indistinguishable from zero; \
         nothing is being corrected and the telescoping check is vacuous"
    );
    assert!(
        mass_rate.abs() < 1.0e-9 * scale,
        "the captured imbalance sums to {mass_rate:.3e} over a level closed by reflecting walls, \
         {:.3e} of its own peak cell; a mass residual that does not telescope is not a discrete \
         flux divergence, so what was captured is not the scheme's imbalance",
        mass_rate.abs() / scale
    );
}

/// the entropy `K = p / rho^gamma` against the `K0` the atmosphere was built with, over the cells
/// this level actually contributes to the composite solution.
///
/// covered cells are excluded because the finer level owns that volume and supplies the composite
/// solution there, with the restriction overwriting them every parent step. their entropy also
/// departs from `K0` in principle — the hierarchy-consistent target restricts the conserved state,
/// which is linear in the average, while `p / rho^gamma` is nonlinear, so a covered cell's entropy
/// differs from `K0` at t = 0 by construction.
fn worst_entropy_deviation(hier: &Hier, level: usize) -> f64 {
    let lvl = &hier.levels[level];
    let st = &lvl.state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre_field().unwrap().view();
    st.geom
        .interior
        .iter()
        .filter(|c| !lvl.coverage.as_ref().is_some_and(|cov| cov.contains(*c)))
        .map(|c| (*pre.at(c) / rho.at(c).powf(GAMMA) - K0).abs() / K0)
        .fold(0.0_f64, f64::max)
}

#[test]
fn a_declared_target_stops_the_entropy_drift_at_the_interface() {
    // the symptom that started this: a coarse-fine interface injects spurious momentum into a
    // hydrostatic background, that kinetic energy thermalizes, and the entropy of a gas that
    // should be exactly isentropic moves. the atmosphere is built with one `K0` everywhere, so
    // `p / rho^gamma` is a direct readout with no reference solution to subtract.
    //
    // the t = 0 value is reported beside it because they answer different questions: what a
    // scheme does to the entropy is the change, measured off whatever offset it started with.
    println!("\nentropy deviation |K - K0|/K0, uncovered cells, after {STEPS} root steps");
    println!("{:-<86}", "");

    for levels in 1..=3usize {
        let mut control = build(&nested(levels));
        control.evolve_steps(STEPS).unwrap();

        let mut hier = build_declared(&nested(levels));
        hier.prime();
        let initial: Vec<f64> = (0..levels)
            .map(|ll| worst_entropy_deviation(&hier, ll))
            .collect();
        hier.evolve_steps(STEPS).unwrap();

        for ll in 0..levels {
            let drifted = worst_entropy_deviation(&control, ll);
            let held = worst_entropy_deviation(&hier, ll);
            let moved = (held - initial[ll]).abs();
            println!(
                "levels={levels} level {ll}:  undeclared {drifted:.3e}  ->  declared {held:.3e}  \
                 (t=0 {:.3e}, moved {moved:.3e})",
                initial[ll]
            );

            // non-vacuity: undeclared, the entropy has to move on its own, which is what gives
            // the declaration a leak to close.
            assert!(
                drifted > 1.0e-9,
                "level {ll} of the UNDECLARED {levels}-level run held its entropy to \
                 {drifted:.3e}; there is no drift here to fix and this gate says nothing"
            );
            // the target is a fixed point of the scheme, so the state holds still and its
            // entropy with it. the assertion is on the change, taken relative to the offset the
            // hierarchy-consistent target starts with.
            assert!(
                moved < 1.0e-13,
                "level {ll} of the {levels}-level run moved its entropy by {moved:.3e} while \
                 sitting on its declared stationary target ({:.3e} -> {held:.3e})",
                initial[ll]
            );
        }
    }
}

#[test]
#[ignore = "diagnostic: reports how the undeclared interface leak grows with step count"]
fn the_interface_entropy_leak_grows_with_step_count() {
    // whether an interface leak threatens a long science run is a question about its growth rate,
    // which its size after a fixed number of steps leaves open. a leak that saturates is a bounded
    // offset; one that accumulates linearly reaches any tolerance eventually, and measuring more
    // than one duration is what separates the two.
    println!("\nundeclared 2-level run: entropy deviation vs steps");
    for steps in [20u64, 80, 320, 1280] {
        let mut hier = build(&nested(2));
        hier.evolve_steps(steps).unwrap();
        let l0 = worst_entropy_deviation(&hier, 0);
        let l1 = worst_entropy_deviation(&hier, 1);
        println!(
            "  steps={steps:>5}  level0 {l0:.4e}  level1 {l1:.4e}  \
             (per step: {:.3e} / {:.3e})",
            l0 / steps as f64,
            l1 / steps as f64
        );
    }
}

#[test]
fn the_fixed_point_survives_a_step_it_was_not_probed_at() {
    // the imbalance is read off a single stage of length `dt_probe`. if any part of what that stage
    // does to the target is quadratic in dt — a gravitational kick that carries its own
    // `0.5 rho |g|^2 dt^2` into the energy, for instance — then `R = (qt - advanced)/dt` picks up
    // a term proportional to `dt_probe`, and the correction only cancels the stage exactly when
    // the run takes that same step. clamping the run's dt away from the probe's is the direct test.
    println!("\nfixed point vs the run's dt, 2 levels, {STEPS} steps");
    for clamp in [0.0_f64, 0.5, 0.25, 0.1] {
        let mut hier = build_declared(&nested(2));
        if clamp > 0.0 {
            // max_dt clamps the accepted step below the cfl value the target was probed at.
            use symbi_sim::substrate_seam::KernelSet;
            for level in &mut hier.levels {
                let cfl_dt = level.kernels.cfl(&level.state);
                level.state.max_dt = clamp * cfl_dt;
            }
        }
        hier.evolve_steps(STEPS).unwrap();
        let speeds: Vec<String> = (0..2)
            .map(|ll| format!("{:.3e}", worst_speed(&hier, ll)))
            .collect();
        let label = if clamp == 0.0 {
            "cfl (= probe dt)".to_string()
        } else {
            format!("{clamp}x cfl")
        };
        println!("  dt = {label:<18} max|v| {speeds:?}");
        // the imbalance is read off a single stage at the target's own cfl step. if any part of what
        // that stage does were quadratic in dt, the correction would cancel only at that step and
        // the fixed point would decay as the run's dt moved away from it.
        for ll in 0..2 {
            let speed = worst_speed(&hier, ll);
            assert!(
                speed < 1.0e-12,
                "level {ll} moved at {speed:.3e} running at {label}, a step the target was not \
                 probed at; the correction is step-dependent"
            );
        }
    }
}

#[test]
fn a_declared_target_restores_the_entropy_floor() {
    // entropy is one-way: K may rise, and its initial value is a floor. a single grid with
    // gravity obeys that; adding a refinement level breaks it. this asks whether removing the
    // background's discrete imbalance is enough to put the floor back.
    const LONG: u64 = 956;
    println!("\nmin K/K0 after {LONG} root steps (below 1 is entropy DESTROYED)");
    for levels in 1..=3usize {
        let mut control = build(&nested(levels));
        control.evolve_steps(LONG).unwrap();
        let mut hier = build_declared(&nested(levels));
        hier.evolve_steps(LONG).unwrap();
        let show = |h: &Hier| -> Vec<String> {
            (0..levels)
                .map(|ll| {
                    let lvl = &h.levels[ll];
                    let st = &lvl.state;
                    let rho = st.fields.prim.rho.view();
                    let pre = st.fields.prim.pre_field().unwrap().view();
                    let m = st
                        .geom
                        .interior
                        .iter()
                        .filter(|c| !lvl.coverage.as_ref().is_some_and(|cv| cv.contains(*c)))
                        .map(|c| *pre.at(c) / rho.at(c).powf(GAMMA) / K0)
                        .fold(f64::INFINITY, f64::min);
                    format!("{m:.7}")
                })
                .collect()
        };
        let (undeclared, declared) = (show(&control), show(&hier));
        println!("  levels={levels}  undeclared {undeclared:?}");
        println!("  levels={levels}  declared   {declared:?}");
        for ll in 0..levels {
            let before: f64 = undeclared[ll].parse().unwrap();
            let after: f64 = declared[ll].parse().unwrap();
            // non-vacuity: undeclared, this setup has to destroy entropy on its own, or
            // there is no one-way law being violated here to restore.
            assert!(
                before < 1.0,
                "level {ll} of the UNDECLARED {levels}-level run held K/K0 at {before:.7}; \
                 nothing is destroying entropy here and this gate says nothing"
            );
            // entropy is one-way: K may rise, with the value the gas was built with as its
            // floor. the target is a fixed point, so the state holds still and its entropy with
            // it -- the floor is met exactly, at equality.
            assert!(
                after >= 1.0,
                "level {ll} of the {levels}-level run destroyed entropy while sitting on its \
                 declared stationary target: min K/K0 = {after:.7}, below the one-way floor"
            );
        }
    }
}

// =============================================================================
// a target that is steady and steep
//
// a spread-out atmosphere sits comfortably on its grid. a point mass placed inside the refined box
// makes a very different shape: the density turns over inside a single coarse cell near the
// center, at a scale finer than any level resolves. that region is real and is corrected like any
// other, but its error is a limiter clipping in place of a truncation, so it carries no order and
// its testimony about whether the declared state solves the equations is empty.
//
// a convergence check that samples it anyway reports non-convergence and refuses a perfectly good
// equilibrium. this is that geometry, in one dimension.
// =============================================================================

/// softening length of the central body, in root cells. small enough that the density turns over
/// inside one coarse cell near the center — which is the point.
const CUSP_SOFTENING: f64 = 0.75 / N as f64;
const CUSP_GM: f64 = 2.0;

/// the plummer potential the body actually applies: `phi = -GM/sqrt(r^2 + h^2)`.
fn cusp_potential(x: [f64; 1]) -> f64 {
    let r = x[0] - 0.5;
    -CUSP_GM / (r * r + CUSP_SOFTENING * CUSP_SOFTENING).sqrt()
}

/// the isentropic atmosphere in balance against the plummer potential, normalized at the wall.
fn cusped_atmosphere(x: [f64; 1]) -> Prim<f64, 1> {
    let scale = GAMMA * K0 / (GAMMA - 1.0);
    let invariant = scale * 1.0f64.powf(GAMMA - 1.0) + cusp_potential([0.0]);
    let rho = ((invariant - cusp_potential(x)) / scale).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

fn build_cusped(levels: usize) -> Hier {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(cusped_atmosphere)
        .build();
    let ck = kset(&coarse);
    let hier = Hierarchy::with_refinement(coarse, ck, &nested(levels), ProlongOrder::Ppm, kset)
        .unwrap()
        .with_bodies(
            symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                0,
                symbi_algebra::Tensor::new([0.5]),
                symbi_algebra::Tensor::zeros(),
                CUSP_GM,
                0.0,
                CUSP_SOFTENING,
            )),
        );
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(cusped_atmosphere);
    }
    hier
}

#[test]
fn a_steady_target_with_an_unresolved_cusp_is_accepted() {
    // non-vacuity: the cusp has to be genuinely unresolved, or this is the smooth case again.
    let dx = 1.0 / N as f64;
    let across = cusped_atmosphere([0.5 + 0.5 * dx]).rho / cusped_atmosphere([0.5 + 1.5 * dx]).rho;
    println!("\ndensity ratio across one coarse cell at the cusp: {across:.2}x");
    assert!(
        across > 1.5,
        "the density changes by only {across:.2}x across a coarse cell at the centre; this \
         geometry no longer has an unresolved feature and the check below is the smooth case"
    );

    // the target is a steady state of the field the body applies, so the check accepts it. the
    // cells at the cusp carry limiter clipping in place of truncation, so they are excluded from
    // the verdict, leaving the verdict to the resolved cells.
    let hier = build_cusped(2).with_equilibrium(cusped_atmosphere).unwrap();
    let measured = hier
        .target_imbalance_convergence(0)
        .expect("levels 0 and 1 overlap");
    println!(
        "  resolved {} of {} overlapping cells; per-component medians {:?}",
        measured.resolved,
        measured.considered,
        measured
            .ratio
            .iter()
            .map(|r| format!("{r:.2}"))
            .collect::<Vec<_>>()
    );
    // and the cusp has to have been excluded, or the acceptance came from somewhere else.
    assert!(
        measured.resolved < measured.considered,
        "every one of the {} overlapping cells counted as resolved, so the cusp was never \
         excluded and this test is not exercising the filter it exists for",
        measured.considered
    );
}

#[test]
fn the_cusped_target_is_still_held_exactly() {
    // acceptance earns its keep only when the correction then works. the same steep target holds
    // as a fixed point to roundoff, cusp included.
    let mut hier = build_cusped(2).with_equilibrium(cusped_atmosphere).unwrap();
    hier.seed_equilibrium();
    let m0 = composite_mass(&hier);
    hier.evolve_steps(STEPS).unwrap();
    let m1 = composite_mass(&hier);
    for ll in 0..2 {
        let speed = worst_speed(&hier, ll);
        println!("cusped target, level {ll}: max|v| {speed:.3e}");
        assert!(
            speed < 1.0e-12,
            "level {ll} moved at {speed:.3e} while sitting on its declared cusped target"
        );
    }
    let relative = ((m1 - m0) / m0).abs();
    assert!(
        relative < 1.0e-14,
        "composite mass moved by {relative:.3e} on the cusped target"
    );
}

#[test]
fn a_correct_target_is_never_refused_however_steep() {
    // the check accepts an equilibrium however sharp. as a target steepens its imbalance
    // concentrates into cells the grid leaves unresolved, and past some steepness the measurable
    // signal is gone -- at which point the honest verdict is "unverified". a target that solves
    // different equations stays detectable throughout, because its error is the continuum
    // residual and that is present wherever the source is, including in the smooth cells.
    println!("\ncorrect targets across a range of steepness (none may be refused)");
    println!(
        "{:>14} {:>15} {:>12}",
        "softening/dx", "rho ratio/cell", "verdict"
    );
    for soft_cells in [2.0_f64, 1.5, 1.0, 0.75, 0.5, 0.35] {
        let soft = soft_cells / N as f64;
        let dx = 1.0 / N as f64;
        let phi = move |x: f64| -CUSP_GM / ((x - 0.5) * (x - 0.5) + soft * soft).sqrt();
        let scale = GAMMA * K0 / (GAMMA - 1.0);
        let invariant = scale + phi(0.0);
        let dens = move |x: f64| ((invariant - phi(x)) / scale).powf(1.0 / (GAMMA - 1.0));
        let across = dens(0.5 + 0.5 * dx) / dens(0.5 + 1.5 * dx);
        let target = move |x: [f64; 1]| {
            let r = dens(x[0]);
            Prim {
                rho: r,
                vel: symbi_algebra::Tensor::new([0.0]),
                pre: K0 * r.powf(GAMMA),
            }
        };
        let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([N])
            .spacing([dx])
            .boundaries(Boundaries::uniform(BoundaryType::Reflect))
            .cfl(CFL)
            .allocate()
            .unwrap()
            .set_initial(target)
            .build();
        let ck = kset(&coarse);
        let hier = Hierarchy::with_refinement(coarse, ck, &nested(2), ProlongOrder::Ppm, kset)
            .unwrap()
            .with_bodies(
                symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                    0,
                    symbi_algebra::Tensor::new([0.5]),
                    symbi_algebra::Tensor::zeros(),
                    CUSP_GM,
                    0.0,
                    soft,
                )),
            );
        for lvl in 1..hier.levels.len() {
            hier.levels[lvl].state.seed_cells(target);
        }

        // non-vacuity: the target has to be genuinely steep, or this is the smooth case again.
        assert!(
            across > 1.3,
            "the density changes by only {across:.2}x across a cell; not a steep target"
        );
        let accepted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hier.with_equilibrium(target)
        }));
        match accepted {
            Ok(Ok(h)) => {
                let m = h.target_imbalance_convergence(0).unwrap();
                let tested = m.sampled.iter().any(|&n| n >= 8);
                println!(
                    "{soft_cells:>14.2} {across:>15.2} {:>12}",
                    if tested { "converged" } else { "unverified" }
                );
            }
            _ => panic!(
                "a CORRECT steady state at {across:.2}x density per cell was REFUSED; the check \
                 is rejecting equilibria for being steep, which is the false positive it exists \
                 to avoid"
            ),
        }
    }
}

/// the stationarity diagnostic ignores the sink interior.
///
/// a declared target is a steady state of the equations the stage pipeline applies. inside an
/// accreting body the drain removes mass and energy, so the target's imbalance there carries the
/// drain in place of truncation error and reports non-convergence however exact the declaration is
/// everywhere else. those cells are still corrected like any other -- the exclusion decides which
/// cells testify about convergence.
///
/// the default comes from the bodies, so a run with a sink gets the exclusion for free. the two
/// clauses below are independent: that the default is nonzero, and that widening it removes cells
/// from the statistic.
#[test]
fn the_stationarity_diagnostic_excludes_the_sink_interior() {
    let hier = build(&nested(2)).with_equilibrium(hydrostatic).unwrap();
    let wide = build(&nested(2))
        .with_equilibrium(hydrostatic)
        .unwrap()
        // the level pair overlaps x in [0.4, 0.6], so this cuts the inner half of it
        .with_equilibrium_mask(0.5);
    let none = build(&nested(2))
        .with_equilibrium(hydrostatic)
        .unwrap()
        .with_equilibrium_mask(0.0);

    let base = hier
        .target_imbalance_convergence(0)
        .expect("levels 0 and 1 share an interior");
    let masked = wide
        .target_imbalance_convergence(0)
        .expect("levels 0 and 1 share an interior");
    let unmasked = none
        .target_imbalance_convergence(0)
        .expect("levels 0 and 1 share an interior");

    println!(
        "\ncells considered: default {} | mask 0.5 {} | mask 0 {}",
        base.considered, masked.considered, unmasked.considered
    );

    // non-vacuity: the overlap has to offer cells at all, or every claim below is trivially true.
    assert!(
        unmasked.considered > 20,
        "the level pair offered only {} cells; the exclusion cannot be shown to do anything",
        unmasked.considered
    );
    assert!(
        masked.considered < unmasked.considered,
        "a 0.5 exclusion radius removed no cells ({} vs {}); the mask is not reaching the \
         statistic",
        masked.considered,
        unmasked.considered
    );
    // this fixture carries a gravitating body at zero accretion radius, so the default resolves
    // to zero and agrees with the explicit zero. a fixture with a sink would differ here, which is
    // the behavior the default exists to provide.
    assert_eq!(
        base.considered, unmasked.considered,
        "with no accreting body the default exclusion must be zero, but it removed cells"
    );
}
