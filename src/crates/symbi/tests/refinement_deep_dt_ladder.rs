// =============================================================================
// refinement_deep_dt_ladder.rs
//
// how the root timestep behaves on a DEEP hierarchy over a gravitational sound-speed profile —
// the regime a nested accretion ladder runs in, and one no existing gate reaches.
//
// the root step is the minimum over levels of `cfl(level_l) * 2^l`. which level attains that
// minimum is not a detail: it decides both the cost of a run and whether it is stable at all, and
// it depends entirely on how the sound speed varies across the levels.
//
// over a Bondi-like profile `c^2 = 1 + (gamma - 1) R_B / r`, an accretor at the box center means
// each level's innermost resolved radius is its own cell width. far outside the Bondi radius the
// sound speed is flat, so `cfl_l ~ dx_l ~ 2^{-l}` and the product `cfl_l * 2^l` is level
// INDEPENDENT — every level limits the root equally and the minimum is degenerate. well inside it
// the sound speed goes as `r^{-1/2}`, so `cfl_l ~ 2^{-3l/2}` and the product falls as `2^{-l/2}`:
// the minimum moves decisively onto the FINEST level, and the root is driven far below its own
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
/// sit well OUTSIDE the Bondi radius for the ladder to descend through the crossover — that is what
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

/// the innermost radius a level resolves: half its own cell width, since the accretor sits on a
/// cell boundary at the box center.
fn inner_radius(
    hier: &Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>,
    ll: usize,
) -> f64 {
    0.5 * hier.levels[ll].state.geom.dx[0]
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
    let outermost = inner_radius(&hier, 0);
    let innermost = inner_radius(&hier, LEVELS - 1);
    assert!(
        outermost > 10.0 * R_B && innermost < R_B / 100.0,
        "the ladder spans r = {innermost:e} to {outermost:e} against a Bondi radius of {R_B}; it \
         does not straddle the crossover, so the flat and gravitational regimes are not both here"
    );

    // the ladder itself, reported: a cost estimate for a deep run is built on these numbers.
    println!("[{LEVELS} levels, R_B = {R_B}] level : r_inner/R_B : own dt : dt * 2^l : rung ratio");
    for ll in 0..LEVELS {
        let ratio = if ll == 0 {
            f64::NAN
        } else {
            scaled[ll - 1] / scaled[ll]
        };
        println!(
            "  {ll:2} : {:10.3e} : {:10.3e} : {:10.3e} : {ratio:.4}",
            inner_radius(&hier, ll) / R_B,
            own[ll],
            scaled[ll]
        );
    }
    println!(
        "root step is {:.4} of the root's own cfl limit (naive 2^(-L/2) would be {:.4})",
        scaled[LEVELS - 1] / own[0],
        (2.0f64).powf(-((LEVELS - 1) as f64) / 2.0)
    );

    // the PRODUCTION selection, tied to the ladder above. everything else in this file reads the
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

    // the collapse itself: the root runs far below its OWN stability limit, because the finest
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

/// a ladder entirely OUTSIDE the Bondi radius: the same nesting, shifted out so the profile is
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
    // the SHAPE of the collapse, not just its direction. with `c ~ r^{-1/2}` and each level's
    // innermost radius equal to its own cell width, `cfl_l ~ dx_l / c(dx_l) ~ 2^{-3l/2}`, so the
    // rung `cfl_l * 2^l` falls as `2^{-l/2}` — a factor of `sqrt(2)` per level.
    //
    // the exponent is what a cost estimate is built on, so it is checked against the profile rather
    // than against a recorded number: a ladder that decayed as `2^{-l}` or not at all would price a
    // deep run wrongly by orders of magnitude while still producing a stable, plausible run.
    let hier = ladder(LEVELS);
    let (_, scaled) = ladder_rungs(&hier);

    // only the levels well inside the Bondi radius follow the asymptotic scaling; outside it the
    // profile is flat and the rungs are level independent by construction.
    let deep: Vec<usize> = (0..LEVELS)
        .filter(|&ll| inner_radius(&hier, ll) < R_B / 20.0)
        .collect();
    assert!(
        deep.len() >= 4,
        "only {} level(s) sit well inside the Bondi radius; the asymptotic slope cannot be \
         measured over that few",
        deep.len()
    );

    let sqrt2 = std::f64::consts::SQRT_2;
    for pair in deep.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let ratio = scaled[a] / scaled[b];
        assert!(
            (ratio / sqrt2 - 1.0).abs() < 0.1,
            "the rung ratio between levels {a} and {b} is {ratio:.4}, not the {sqrt2:.4} that \
             c ~ r^(-1/2) demands. the timestep ladder is not tracking the sound-speed profile \
             driving it, so any cost estimate built on that scaling is wrong"
        );
    }
}

#[test]
fn the_step_the_driver_takes_is_the_step_it_reports() {
    // the root-dt formula exists TWICE — once in `root_cfl_dt`, which the decomposed driver calls
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
