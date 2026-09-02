// =============================================================================
// census_hierarchy_driver.rs
//
// a census through the refinement hierarchy driver, which is the driver the configuration front
// end actually runs. every other census test drives the uni-grid evolve loop.
//
// that asymmetry matters because the hierarchy has its own sampling hook and its own checkpoint
// writer, and either one going missing is silent: a hierarchy that never takes the samples, or a
// writer that never emits the census group, produces no crash and no missing-kernel panic. a
// checkpoint with no census group reads exactly like a run that registered none, so the whole
// feature reports success while recording nothing.
//
// the hierarchy is otherwise reachable only through the python extension, where observing any of
// this means a multi-minute optimized rebuild; from here it takes milliseconds.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_amr::refinement::Hierarchy;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::census::{CensusEvaluator, RegisteredCensus};
use symbi_source_compile::CensusConfig;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 1, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 32;
const R_LO: f64 = 1.0;
const DR: f64 = 0.1;
const GAMMA: f64 = 5.0 / 3.0;

/// a mass census with no bin axes: one global bucket, so the sample is the total mass and any
/// drift in it is arithmetic rather than binning.
fn mass_census() -> CensusConfig {
    CensusConfig::from_json(
        r#"{
            "name": "mass",
            "op": "add",
            "axes": [],
            "value_names": ["mass"],
            "values": [2],
            "params": [],
            "nodes": [
                {"op": "VARIABLE_RHO"},
                {"op": "VARIABLE_DV"},
                {"op": "MULTIPLY", "left": 0, "right": 1}
            ]
        }"#,
    )
    .expect("census config parses")
}

fn build() -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.4)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            Prim::adiabatic(
                Density(1.0 + 0.25 * (x[0] - R_LO).sin()),
                Tensor::new([0.0]),
                Pressure(0.6),
            )
        })
        .build()
}

#[test]
fn the_hierarchy_driver_records_a_census_sample_per_step() {
    let mut sim = build();
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(&mass_census()).expect("census compiles"),
    ));

    let kset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.evolve(0.05).expect("hierarchy evolve");

    let recorded = &hier.levels[0].state.store.censuses[0].history;
    assert!(
        hier.levels[0].state.iteration > 0,
        "the hierarchy took no step, so recording nothing proves nothing"
    );
    assert!(
        !recorded.is_empty(),
        "the hierarchy driver took {} step(s) and recorded NO census sample. the sampling call \
         is missing from this driver — the uni-grid loop having one does not put it here.",
        hier.levels[0].state.iteration
    );

    // the sample must be the physical total, not a zero-shaped placeholder: an empty accumulator
    // and a working one both produce a row.
    let values = recorded.values();
    let mass = values[0];
    assert!(
        mass.is_finite() && mass > 0.0,
        "the recorded mass is not a physical total: {mass}"
    );
    assert_eq!(
        recorded.dropped()[0],
        0,
        "an axis-free census drops nothing"
    );
}

#[test]
fn a_hierarchy_checkpoint_carries_the_census_group() {
    // sampling into a history that no writer emits is indistinguishable, from the outside, from
    // never sampling at all.
    let mut sim = build();
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(&mass_census()).expect("census compiles"),
    ));
    let kset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, kset);
    hier.evolve(0.05).expect("hierarchy evolve");

    let dir = std::env::temp_dir().join(format!("census_hier_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("chkpt.h5");
    let levels: Vec<&Sim> = hier.levels.iter().map(|l| &l.state).collect();
    symbi_sim::checkpoint::write_hierarchy_checkpoint(
        &levels,
        path.to_str().expect("utf-8 path"),
        &Default::default(),
    )
    .expect("hierarchy checkpoint written");

    let bytes = std::fs::read(&path).expect("checkpoint readable");
    let _ = std::fs::remove_dir_all(&dir);
    // the group name travels as a plain string in the hdf5 link table, so its presence is
    // checkable without a reader. what is asserted is that the writer emitted the group at all;
    // the failure mode under test is its complete absence.
    let needle = b"census";
    assert!(
        bytes.windows(needle.len()).any(|w| w == needle),
        "the hierarchy checkpoint contains no census group. this driver's writer builds its tree \
         independently of the uni-grid one, so a group emitted only there never reaches a run \
         launched from a configuration."
    );
}
