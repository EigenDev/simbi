// =============================================================================
// census_refined.rs
//
// a census on a refined hierarchy, where the reduction is a sum over levels rather than over one
// grid.
//
// this is where a census is most exposed. a cell a finer level resolves is excluded from its
// coarse parent — counting both would inflate every extensive total by the refined volume — so the
// refined region enters the sum only through the fine level's own contribution. if that
// contribution is absent for any reason, the total is short by exactly the refined volume, and
// there is nothing in the number to say so: it is smooth, positive, of the right order, and
// drifts in a way that reads as a boundary loss.
//
// the composite total is therefore checked against the one quantity that cannot lie about it — the
// conserved mass the hierarchy itself tracks over the same leaf cells.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_refinement::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::census::{CensusEvaluator, RegisteredCensus};
use symbi_source_compile::CensusConfig;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Hier = Hierarchy<
    Newtonian,
    1,
    1,
    Cartesian,
    IdealGas<f64>,
    CpuSpace,
    HostMemory,
    AdiabaticSubstrateKernelSet<HostMemory, f64, 1>,
>;

const N: usize = 128;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;

fn kset(sim: &Sim) -> AdiabaticSubstrateKernelSet<HostMemory, f64, 1> {
    AdiabaticSubstrateKernelSet::new(GAMMA, CFL, &sim.geom.allocated)
}

/// a global mass census: no bins, so the sample is the composite total and any shortfall is
/// arithmetic rather than binning.
fn mass_census() -> CensusConfig {
    mass_census_at("root_step")
}

fn mass_census_at(cadence: &str) -> CensusConfig {
    CensusConfig::from_json(&format!(
        r#"{{
            "name": "mass", "op": "add", "axes": [],
            "value_names": ["mass"], "values": [2], "params": [],
            "cadence": "{cadence}",
            "nodes": [
                {{"op": "VARIABLE_RHO"}},
                {{"op": "VARIABLE_DV"}},
                {{"op": "MULTIPLY", "left": 0, "right": 1}}
            ]
        }}"#
    ))
    .expect("census parses")
}

/// a two-level hierarchy with the refined patch over the density peak, and one census registered
/// on the root (which owns the history).
fn refined_with(cfg: &CensusConfig) -> Hier {
    let coarse = coarse_sim();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [0.375],
        x_hi: [0.625],
    }];
    let mut hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).expect("refine");
    hier.levels[1].state.seed_cells(ic);
    hier.levels[0]
        .state
        .store
        .censuses
        .push(RegisteredCensus::new(
            CensusEvaluator::new(cfg).expect("census compiles"),
        ));
    hier
}

/// smooth, strongly structured density so the refined region holds a substantial and easily
/// identified share of the mass — a flat profile would make an undercount look like roundoff.
fn ic(x: [f64; 1]) -> Prim<f64, 1> {
    Prim::adiabatic(
        Density(1.0 + 3.0 * (-((x[0] - 0.5) * (x[0] - 0.5)) / 0.005).exp()),
        Tensor::new([0.0]),
        Pressure(1.0),
    )
}

fn coarse_sim() -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("sim")
        .set_initial(ic)
        .build()
}

/// the conserved mass over leaf cells, summed across levels — the same predicate a census uses,
/// computed directly from the conserved density so it shares no code with the census path.
fn leaf_mass(hier: &Hier) -> f64 {
    let mut total = 0.0;
    for level in &hier.levels {
        let sim = &level.state;
        let bg = sim.geom.block_geometry(sim.physics.metric);
        for c in sim.geom.interior.iter() {
            if level
                .coverage
                .as_ref()
                .is_some_and(|r: &symbi_algebra::Domain<1>| r.contains(c))
            {
                continue;
            }
            total += *sim.fields.cons.den.view().at(c) * bg.labframe_volume(c, sim.motion.a);
        }
    }
    total
}

#[test]
fn a_refined_hierarchy_censuses_the_whole_composite_domain() {
    let coarse = coarse_sim();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [0.375],
        x_hi: [0.625],
    }];
    let mut hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).expect("refine");
    hier.levels[1].state.seed_cells(ic);
    hier.levels[0]
        .state
        .store
        .censuses
        .push(RegisteredCensus::new(
            CensusEvaluator::new(&mass_census()).expect("census compiles"),
        ));

    let want = leaf_mass(&hier);

    // the premise: the refined region must hold a substantial share, or a total that omits it
    // would be within roundoff of the right answer and this gate would prove nothing.
    let root_only: f64 = {
        let sim = &hier.levels[0].state;
        let bg = sim.geom.block_geometry(sim.physics.metric);
        sim.geom
            .interior
            .iter()
            .filter(|c| {
                !hier.levels[0]
                    .coverage
                    .as_ref()
                    .is_some_and(|r: &symbi_algebra::Domain<1>| r.contains(*c))
            })
            .map(|c| *sim.fields.cons.den.view().at(c) * bg.labframe_volume(c, sim.motion.a))
            .sum()
    };
    assert!(
        root_only < 0.6 * want,
        "the refined region holds only {:.1}% of the mass; a census that dropped it entirely \
         would still land near the right total",
        100.0 * (1.0 - root_only / want)
    );

    hier.evolve(0.02).expect("evolve");
    assert!(
        hier.levels[0].state.iteration > 0,
        "the hierarchy took no step"
    );

    let history = &hier.levels[0].state.store.censuses[0].history;
    assert!(!history.is_empty(), "no census sample was recorded");
    let got = history.values()[0];
    let want = leaf_mass_at_first_sample(&hier, want);

    assert!(
        (got / want - 1.0).abs() < 1.0e-10,
        "the composite census total is {got} against a leaf mass of {want} (short by {:.2}%). \
         a covered coarse cell is excluded from its parent, so the refined volume enters the sum \
         only through the fine level's own contribution — a level that contributes nothing leaves \
         the total short by exactly that volume.",
        100.0 * (1.0 - got / want)
    );
}

/// the mass at the time of the first sample. on a periodic domain with no sources the leaf mass is
/// conserved, so the initial value is the reference throughout — asserted rather than assumed.
fn leaf_mass_at_first_sample(hier: &Hier, initial: f64) -> f64 {
    let now = leaf_mass(hier);
    assert!(
        (now / initial - 1.0).abs() < 1.0e-10,
        "the leaf mass drifted from {initial} to {now} over the run, so it is not a fixed \
         reference for the census total"
    );
    initial
}

#[test]
fn a_per_level_census_records_each_level_on_its_own_subcycle() {
    // levels are time-aligned only at root-step boundaries, and a level subcycles once per parent
    // step. sampling every level at the root boundary therefore under-resolves exactly the
    // innermost, fastest-decorrelating region — the one refinement was added to resolve. a
    // per-level census instead samples each level on its own step, so the finer level contributes
    // proportionally more rows over the same span.
    let mut hier = refined_with(&mass_census_at("per_level_step"));
    hier.evolve(0.02).expect("evolve");

    let history = &hier.levels[0].state.store.censuses[0].history;
    let levels = history.level();
    let root_rows = levels.iter().filter(|&&l| l == 0).count();
    let fine_rows = levels.iter().filter(|&&l| l == 1).count();

    assert!(
        root_rows > 0 && fine_rows > 0,
        "a per-level census recorded {root_rows} root row(s) and {fine_rows} fine row(s); both \
         levels must sample, or the cadence is only reaching one of them"
    );
    // the refinement ratio is two, so the fine level takes two subcycle steps per root step.
    assert_eq!(
        fine_rows,
        2 * root_rows,
        "the fine level recorded {fine_rows} rows against {root_rows} root rows; at a refinement \
         ratio of two it subcycles twice per root step, so anything else means it is being \
         sampled at the root's cadence rather than its own"
    );

    // each row carries its own level's clock, not the root's. a fine row taken mid-step sits
    // strictly between two root rows, which is the time skew a consumer needs to see.
    let fine_times: Vec<f64> = levels
        .iter()
        .zip(history.time())
        .filter(|&(&l, _)| l == 1)
        .map(|(_, &t)| t)
        .collect();
    let root_times: Vec<f64> = levels
        .iter()
        .zip(history.time())
        .filter(|&(&l, _)| l == 0)
        .map(|(_, &t)| t)
        .collect();
    assert!(
        fine_times.iter().any(|t| root_times.iter().all(|r| r != t)),
        "every fine row landed on a root row's time; the rows are tagged by level but carry the \
         root clock, so the finer sampling is not actually resolving anything between root steps"
    );

    // and the totals are still physical: a fine row covers that level's leaf cells alone, so it is
    // a fraction of the composite mass rather than a whole or a zero.
    let composite = leaf_mass(&hier);
    let fine_value = history
        .level()
        .iter()
        .zip(history.values())
        .find(|&(&l, _)| l == 1)
        .map(|(_, &v)| v)
        .expect("a fine row");
    assert!(
        fine_value > 0.0 && fine_value < composite,
        "the fine row holds {fine_value} against a composite mass of {composite}; a per-level row \
         is that level's own leaf cells, neither the whole domain nor nothing"
    );
}

#[test]
fn a_root_step_census_is_not_also_sampled_per_level() {
    // the two cadences are exclusive. a census recorded by both paths would enter its history
    // twice per root step under two different level tags — the same physical state counted as two
    // independent samples, which biases every time average toward whichever level sampled more.
    let mut hier = refined_with(&mass_census());
    hier.evolve(0.02).expect("evolve");

    let history = &hier.levels[0].state.store.censuses[0].history;
    assert!(!history.is_empty(), "no sample was recorded at all");
    assert!(
        history.level().iter().all(|&l| l == 0),
        "a root-step census recorded rows tagged {:?}; every level's partial is combined into one \
         row before it is recorded, so a row from a finer level means it was also sampled inside \
         the subcycle",
        history.level()
    );
    assert_eq!(
        history.len(),
        hier.levels[0].state.iteration as usize,
        "a root-step census must record exactly one row per accepted root step"
    );
}
