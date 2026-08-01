// =============================================================================
// refinement_well_balanced.rs
//
// a declared stationary target must be an EXACT fixed point of the refined scheme, and declaring
// one must not cost conservation.
//
// a hydrostatic atmosphere solves the continuum equations, not the discrete ones: the scheme
// leaves a residual `R = div_h F_h(qt) - s_h(qt)` at truncation order, so gas seeded on the exact
// profile starts moving. `R` is also GRID-DEPENDENT, so the coarse-fine flux register differences
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
// VELOCITY is the probe: the target has `v = 0` identically, so any speed after a step is the
// imbalance itself, with no reference state to subtract.
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
        .with_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new([-G_OFFSET]),
            symbi_algebra::Tensor::zeros(),
            GM,
            1.0e-3,
            0.0,
        )));
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

/// mass over the ACTIVE composite: covered coarse cells are omitted and their finer children
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
/// ACROSS the coarse-fine interface — that is exactly what it is supposed to cancel — so this is
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

        // NON-VACUITY: without the declaration this setup must genuinely fail to hold the
        // atmosphere, or the gate below is measuring a problem that is not there. a refined run
        // drifts an order of magnitude past the single grid; even the single grid drifts.
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
        // orders below the 5.8e-4 the same setup reaches without the declaration.
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

#[test]
fn the_correction_tracks_the_distance_from_the_target() {
    // a correction that merely suppressed the flux register would pass the fixed-point gate and
    // leave the interface unrefluxed for every OTHER state. driving the same declared hierarchy
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
            // a uniform mach number rather than a uniform speed, so the perturbation is the same
            // physical size everywhere in an atmosphere whose sound speed varies by a factor of 3.
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

    // NON-VACUITY: mass has to actually cross the coarse-fine interface, or "the composite total
    // held" is a statement about a register that never had anything to correct.
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

/// the same atmosphere written for HALF the gravity the run actually applies. it is a perfectly
/// good hydrostatic profile — for a different problem — so it is exactly the mistake that a
/// pointwise inspection of the initial condition does not catch: smooth, positive, monotone, and
/// wrong by a constant factor in one term.
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
    // and subtracted whatever it is, so a state merely ASSERTED to be an equilibrium gets held
    // motionless while the run reports no error. what separates the two is how the imbalance
    // behaves under refinement — truncation error falls with the cell width, the continuum
    // residual of a state that does not solve the equations does not fall at all.
    let real = build(&nested(2)).with_equilibrium(hydrostatic).unwrap();
    let (coarse, fine) = real
        .target_imbalance_norms(0)
        .expect("levels 0 and 1 share an interior to compare over");

    let peak = coarse.iter().fold(0.0_f64, |m, n| m.max(*n));
    let ratios: Vec<String> = coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| format!("{:.2}", c / f.max(f64::MIN_POSITIVE)))
        .collect();
    println!("\ntrue steady state, imbalance L1 coarse/fine by component: {ratios:?}");

    // NON-VACUITY: an imbalance already at zero would "converge" trivially. there has to be a real
    // truncation error being measured.
    assert!(
        peak > 1.0e-6,
        "the largest component of the imbalance is {peak:.3e}, indistinguishable from zero; the \
         convergence measured below is a ratio of roundoff"
    );
    for (cc, (c, f)) in coarse.iter().zip(&fine).enumerate() {
        if *c < 1.0e-6 * peak {
            continue;
        }
        let ratio = c / f.max(f64::MIN_POSITIVE);
        assert!(
            ratio > 1.5,
            "component {cc} of the TRUE steady state's imbalance fell by only {ratio:.3} when the \
             cell width halved; the check that rejects a false equilibrium would reject this one"
        );
    }
}

#[test]
#[should_panic(expected = "not a steady state")]
fn a_declared_target_that_is_not_stationary_is_rejected() {
    // the profile balances GM/2 while the body pulls with GM, so the continuum residual is
    // rho*GM/2 — grid-independent, and therefore visible as an imbalance that does not converge.
    // without this check the run would hold this state motionless and report nothing.
    let _ = build(&nested(2)).with_equilibrium(hydrostatic_wrong_gravity);
}

/// the same atmosphere written as an expression DAG, in the wire form a configured run emits.
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
    // the target crosses the configuration wire as an expression rather than as sampled data,
    // because a restart that adds a refinement level needs it evaluated on cells that did not
    // exist when the run began. what that costs is a second way to say the same thing, so the two
    // have to be held to producing the SAME state — not merely a similar one.
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
    // `R = div_h F_h(qt) - s_h(qt)`. gravity sources momentum and energy but never mass, so R's
    // mass component is a pure discrete divergence, and summing a divergence over a level bounded
    // by reflecting walls telescopes to the flux through those walls. gas at rest sees a mirrored
    // state across a wall, no jump, and therefore no mass flux — so the sum must vanish.
    //
    // this is what separates the target's genuine imbalance from a limiter's response to an
    // unphysical probe state: the latter is not a divergence and does not telescope.
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

    // NON-VACUITY: a residual that is identically zero would telescope trivially and prove nothing
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
