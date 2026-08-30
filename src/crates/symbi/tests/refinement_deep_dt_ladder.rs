// =============================================================================
// refinement_deep_dt_ladder.rs
//
// how the root timestep behaves on a deep hierarchy over a gravitational sound-speed profile —
// the regime a nested accretion ladder runs in, and one no existing gate reaches.
//
// the root step is the minimum over levels of `cfl(level_l) * 2^l`. which level attains that
// minimum is not a detail: it decides both the cost of a run and whether it is stable at all, and
// it depends entirely on how the sound speed varies across the levels.
//
// over a Bondi-like profile `c^2 = 1 + (gamma - 1) R_B / r`, an accretor at the box center means
// each level's innermost resolved radius is its own cell width. far outside the Bondi radius the
// sound speed is flat, so `cfl_l ~ dx_l ~ 2^{-l}` and the product `cfl_l * 2^l` is level
// independent — every level limits the root equally and the minimum is degenerate. well inside it
// the sound speed goes as `r^{-1/2}`, so `cfl_l ~ 2^{-3l/2}` and the product falls as `2^{-l/2}`:
// the minimum moves decisively onto the finest level, and the root is driven far below its own
// stability limit.
//
// both regimes are asserted here, and so is the crossover between them. the failure mode is cost,
// not instability: the scheme stays inside every level's cfl either way, but the root step — and
// therefore the wall time of the whole run — is set by whichever level attains the minimum.
//
// usage:
//  cargo test -p symbi --test refinement_deep_dt_ladder
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
/// cells per level. every level carries the same count: a level covers half its parent's extent at
/// twice the resolution, which is the nesting a centered accretor ladder uses.
const N: usize = 32;
/// the root half-width. with `N` cells the root's innermost resolved radius is `L0 / N`, which must
/// sit well outside the Bondi radius for the ladder to descend through the crossover — that is what
/// puts the flat and the gravitational regimes in one hierarchy.
const L0: f64 = 8.0;
/// the Bondi radius, small enough that the root resolves only the flat part of the profile
/// (`L0 / N = 0.25`, some twelve Bondi radii out) while the finest level reaches far inside it.
const R_B: f64 = 0.02;
/// levels in the deep ladder. the finest resolves `~1.5e-3 R_B`, so six levels sit well inside the
/// Bondi radius — enough to measure the asymptotic slope over.
const LEVELS: usize = 14;

/// the isothermal-limit Bondi sound speed: `c^2 = c_inf^2 + (gamma - 1) R_B / r`, flat far out and
/// going as `r^{-1/2}` well inside `R_B`. imposed as a state rather than solved for — what is under
/// test is how the timestep ladder responds to a sound-speed profile, not the profile itself.
fn sound_speed_sq(x: f64) -> f64 {
    1.0 + (GAMMA - 1.0) * R_B / x.abs().max(1.0e-30)
}

fn prim_at(x: [f64; 1]) -> Prim<f64, 1> {
    // c^2 = gamma p / rho at rho = 1.
    Prim {
        rho: 1.0,
        vel: Tensor::new([0.0]),
        pre: sound_speed_sq(x[0]) / GAMMA,
    }
}

/// a hierarchy of `levels` levels, each covering the inner half of its parent about the origin.
fn ladder(
    levels: usize,
) -> Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset> {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([-L0])
        .spacing([2.0 * L0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("root allocates")
        .set_initial(prim_at)
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);

    let regions: Vec<RefinementRegion<1>> = (1..levels)
        .map(|ll| {
            let half = L0 / (1u64 << ll) as f64;
            RefinementRegion {
                x_lo: [-half],
                x_hi: [half],
            }
        })
        .collect();

    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .expect("the ladder builds");

    // a fine level is allocated as zeros — `with_refinement` does not prolong from the parent — so
    // every level is seeded from the same profile, sampled at its own resolution.
    for level in hier.levels.iter_mut().skip(1) {
        level.state.seed_cells(prim_at);
    }
    for level in hier.levels.iter() {
        level.kernels.c2p(&level.state.store);
    }
    hier
}

/// each level's own cfl-limited step, and that step scaled by the subcycle count — the quantity the
/// root minimum is taken over.
fn ladder_rungs(
    hier: &Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>,
) -> (Vec<f64>, Vec<f64>) {
    let own: Vec<f64> = hier
        .levels
        .iter()
        .map(|l| l.kernels.cfl(&l.state.store))
        .collect();
    let scaled: Vec<f64> = own
        .iter()
        .enumerate()
        .map(|(ll, dt)| dt * (1u64 << ll) as f64)
        .collect();
    (own, scaled)
}

/// the innermost radius a level's timestep is actually set by. a level does not evolve the cells
/// its child covers -- the cfl reduction runs over the evolution regions, so those cells are
/// excluded -- and the accretor sits at the box center, so the innermost cell a level owns lies
/// just outside its child's extent. the finest level has no child and owns down to its own half
/// cell width, which anchors it some sixteen times deeper in radius than the level above and is
/// why its rung stands apart from the ladder.
fn controlling_radius(
    hier: &Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>,
    ll: usize,
) -> f64 {
    match hier.levels.get(ll + 1) {
        Some(child) => {
            let g = &child.state.geom;
            0.5 * g.interior.spaces[0].size() as f64 * g.dx[0]
        }
        None => 0.5 * hier.levels[ll].state.geom.dx[0],
    }
}

/// the rung ratio the sound-speed profile predicts between a level and the one below it. with
/// `c^2 = 1 + A/r` and the controlling radius halving from level to level, the rungs stand in the
/// ratio `c(r/2) / c(r) = sqrt((1 + 2u) / (1 + u))` for `u = A/r`. far outside the Bondi radius
/// `u -> 0` and the rungs are level independent; well inside it `u -> infinity` and the ratio
/// approaches `sqrt(2)`, the `c ~ r^(-1/2)` law.
fn predicted_ratio(r_control: f64) -> f64 {
    let u = (GAMMA - 1.0) * R_B / r_control;
    ((1.0 + 2.0 * u) / (1.0 + u)).sqrt()
}

#[test]
fn the_root_step_is_set_by_the_finest_level_once_the_ladder_reaches_inside_the_bondi_radius() {
    let hier = ladder(LEVELS);
    let (own, scaled) = ladder_rungs(&hier);

    for (ll, dt) in own.iter().enumerate() {
        assert!(
            dt.is_finite() && *dt > 0.0,
            "level {ll} reports a non-physical cfl step {dt}"
        );
    }

    // the premise: the ladder must actually straddle the Bondi radius, or only one of the two
    // regimes below is being exercised and the crossover assertion is vacuous.
    let outermost = controlling_radius(&hier, 0);
    let innermost = controlling_radius(&hier, LEVELS - 1);
    assert!(
        outermost > 10.0 * R_B && innermost < R_B / 100.0,
        "the ladder spans r = {innermost:e} to {outermost:e} against a Bondi radius of {R_B}; it \
         does not straddle the crossover, so the flat and gravitational regimes are not both here"
    );

    // the ladder itself, reported: a cost estimate for a deep run is built on these numbers.
    println!("[{LEVELS} levels, R_B = {R_B}] level : r_control/R_B : own dt : dt * 2^l : rung ratio");
    for ll in 0..LEVELS {
        let ratio = if ll == 0 {
            f64::NAN
        } else {
            scaled[ll - 1] / scaled[ll]
        };
        println!(
            "  {ll:2} : {:10.3e} : {:10.3e} : {:10.3e} : {ratio:.4}",
            controlling_radius(&hier, ll) / R_B,
            own[ll],
            scaled[ll]
        );
    }
    println!(
        "root step is {:.4} of the root's own cfl limit (naive 2^(-L/2) would be {:.4})",
        scaled[LEVELS - 1] / own[0],
        (2.0f64).powf(-((LEVELS - 1) as f64) / 2.0)
    );

    // the production selection, tied to the ladder above. everything else in this file reads the
    // per-level cfl and forms `dt * 2^l` itself, which tests the sound-speed profile but not the
    // code that consumes it — a driver that took the unscaled minimum, or ignored the finer levels
    // entirely, would leave every assertion here intact.
    let want = scaled.iter().copied().fold(f64::INFINITY, f64::min);
    let got = hier.root_cfl_dt();
    assert!(
        (got / want - 1.0).abs() < 1.0e-12,
        "the driver's root step is {got:e}, but the minimum over levels of dt_l * 2^l is {want:e}. \
         the rungs are {scaled:?}"
    );

    let argmin = scaled
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(ll, _)| ll)
        .expect("levels exist");
    assert_eq!(
        argmin,
        LEVELS - 1,
        "the root step is limited by level {argmin} of {LEVELS}, not the finest. the rungs are \
         {scaled:?} — which level attains the minimum decides the cost of every run on this \
         ladder, so it must not be a surprise"
    );

    // the collapse itself: the root runs far below its own stability limit, because the finest
    // level's requirement is what it inherits.
    let collapse = scaled[LEVELS - 1] / own[0];
    assert!(
        collapse < 0.1,
        "the root step is {collapse:.3} of the root's own cfl limit; on a ladder reaching \
         {innermost:e} inside a Bondi radius of {R_B} it must be driven well below it"
    );
}

#[test]
fn far_outside_the_bondi_radius_every_level_limits_the_root_equally() {
    // the flat regime. where the sound speed does not vary across the ladder, `cfl_l * 2^l` is
    // level independent: halving the cell width halves the step, and the level takes twice as many
    // of them. no level is preferred, and a hierarchy built entirely out here costs what naive
    // Berger-Oliger says it does.
    //
    // this is the control for the test above. without it, "the minimum lands on the finest level"
    // could be a property of the scheme rather than of the sound-speed profile driving it.
    const FLAT_LEVELS: usize = 5;
    let hier = shallow_flat_ladder(FLAT_LEVELS);
    let (_, scaled) = ladder_rungs(&hier);

    let (lo, hi) = scaled.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    assert!(
        hi / lo - 1.0 < 0.05,
        "the rungs spread over {:.1}% out where the sound speed is flat ({scaled:?}); with no \
         variation across the ladder every level must limit the root equally",
        100.0 * (hi / lo - 1.0)
    );
}

/// a ladder entirely outside the Bondi radius: the same nesting, shifted out so the profile is
/// flat across every level.
fn shallow_flat_ladder(
    levels: usize,
) -> Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset> {
    // uniform pressure: the sound speed is identical everywhere, which is the flat limit of the
    // Bondi profile at r >> R_B without depending on how far out "far" has to be.
    let flat = |_x: [f64; 1]| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0]),
        pre: 1.0 / GAMMA,
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([-L0])
        .spacing([2.0 * L0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("root allocates")
        .set_initial(flat)
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions: Vec<RefinementRegion<1>> = (1..levels)
        .map(|ll| {
            let half = L0 / (1u64 << ll) as f64;
            RefinementRegion {
                x_lo: [-half],
                x_hi: [half],
            }
        })
        .collect();
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .expect("the flat ladder builds");
    for level in hier.levels.iter_mut().skip(1) {
        level.state.seed_cells(flat);
    }
    for level in hier.levels.iter() {
        level.kernels.c2p(&level.state.store);
    }
    hier
}

#[test]
fn the_rung_decay_follows_the_sound_speed_profile_it_is_driven_by() {
    // the shape of the collapse, level by level, against the profile that drives it. each level's
    // step is set by the innermost cell it owns, and that radius halves from level to level, so
    // the rungs stand in the ratio `c(r/2) / c(r)` the profile dictates -- tending to `sqrt(2)`
    // deep inside the Bondi radius and to 1 far outside it.
    //
    // the exponent is what a cost estimate is built on, so it is checked against the profile
    // rather than against a recorded number: a ladder that decayed as `2^{-l}` or not at all
    // would price a deep run wrongly by orders of magnitude while still producing a stable,
    // plausible run.
    let hier = ladder(LEVELS);
    let (_, scaled) = ladder_rungs(&hier);

    // the finest level owns down to its own half cell, some sixteen times deeper in radius than
    // the level above owns, so its rung answers to a different anchor and is measured separately
    // below. every pair up to it is anchored the same way and follows the profile.
    //
    // the 5 percent band covers a known one-sided offset: the controlling cell's CENTER sits half
    // a parent width outside the child's edge, so the sound speed there is a little below the one
    // at the edge radius the prediction uses, and every measured ratio runs slightly under. the
    // offset shrinks with depth as the cell width falls against the radius; it peaks at 3.2
    // percent around level 7, where the profile is steepening fastest relative to the cell size.
    let mut worst = 0.0_f64;
    let mut worst_level = 0;
    for ll in 0..LEVELS - 2 {
        let predicted = predicted_ratio(controlling_radius(&hier, ll));
        let measured = scaled[ll] / scaled[ll + 1];
        let deviation = (measured / predicted - 1.0).abs();
        if deviation > worst {
            worst = deviation;
            worst_level = ll;
        }
        assert!(
            deviation < 0.05,
            "the rung ratio between levels {ll} and {} is {measured:.4}, against the \
             {predicted:.4} the sound-speed profile predicts at the controlling radius \
             {:.3e}. the timestep ladder is not tracking the profile driving it, so any cost \
             estimate built on that scaling is wrong",
            ll + 1,
            controlling_radius(&hier, ll),
        );
    }
    println!(
        "rung ratios track the profile to {:.2}% (worst, between levels {worst_level} and {})",
        100.0 * worst,
        worst_level + 1
    );

    // the premise: the ladder has to reach deep enough for the asymptotic law to be visible at
    // all, or the agreement above only ever exercises the flat end of the profile where every
    // predicted ratio is ~1 and the test is vacuous.
    let sqrt2 = std::f64::consts::SQRT_2;
    let deepest = scaled[LEVELS - 3] / scaled[LEVELS - 2];
    assert!(
        (deepest / sqrt2 - 1.0).abs() < 0.05,
        "the deepest child-anchored rung ratio is {deepest:.4}, not yet within reach of the \
         {sqrt2:.4} that `c ~ r^(-1/2)` demands; the ladder does not descend far enough inside \
         the Bondi radius for the asymptotic law to be under test"
    );

    // and the approach is monotone: each rung ratio sits above the one before, which is the
    // profile steepening as the ladder descends rather than a coincidence at one depth.
    for ll in 1..LEVELS - 2 {
        let (before, after) = (scaled[ll - 1] / scaled[ll], scaled[ll] / scaled[ll + 1]);
        assert!(
            after > before,
            "the rung ratio fell from {before:.4} to {after:.4} between levels {ll} and {}; \
             the profile steepens monotonically inward, so the ladder must too",
            ll + 1
        );
    }

    // the finest level, whose own half cell reaches far inside the radius its parent owns down
    // to: its rung drops by much more than the asymptotic factor, and that gap is the reason the
    // root step ends up far below the root's own stability limit.
    let finest = scaled[LEVELS - 2] / scaled[LEVELS - 1];
    assert!(
        finest > 2.0 * sqrt2 / 2.0_f64.sqrt(),
        "the finest level's rung ratio is {finest:.4}; owning down to its own half cell rather \
         than a child's edge should drop it by far more than the {sqrt2:.4} of a level that \
         shares the ladder's anchor"
    );
}

#[test]
fn the_step_the_driver_takes_is_the_step_it_reports() {
    // the root-dt formula exists twice — once in `root_cfl_dt`, which the decomposed driver calls
    // to take a global minimum across tiles, and once inside the root step itself. two copies of a
    // rule that decides every run's timestep is a drift hazard with no symptom: the monolithic and
    // decomposed drivers would simply take different steps, each internally consistent.
    //
    // the deep ladder is where a drift would matter most, since the two differ only in how they
    // weight levels and the weighting spans four orders of magnitude here.
    let mut hier = ladder(LEVELS);
    let reported = hier.root_cfl_dt();
    assert!(
        reported.is_finite() && reported > 0.0,
        "the reported root step is not physical: {reported}"
    );

    hier.evolve_steps(1).expect("one root step");
    let taken = hier.prev_dt_cfl;
    assert!(
        (taken / reported - 1.0).abs() < 1.0e-12,
        "the driver took a root step of {taken:e} after reporting {reported:e}. the two copies of \
         the per-level minimum have drifted, so a decomposed run and a monolithic one would \
         advance on different timesteps"
    );
}
