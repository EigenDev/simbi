// =============================================================================
// census_restart.rs
//
// a census continues its schedule across a restart. the checkpoint records each
// census's latest sample per level, and a restart through the hierarchy driver
// (the driver a configuration runs) takes it up, so with the configuration
// unchanged:
// - the records written before the restart, followed by the records of the
//   resumed run, are the uninterrupted run's records bit for bit: the same
//   times, the same values, no row repeated and no extra sample at the restart;
// - an accumulating census goes on folding into the row the file holds and
//   ends on the uninterrupted run's row;
// - a census the file records no schedule for (a file written before the
//   schedule existed) takes the restart time as its latest sample, so its first
//   resumed sample falls one interval later.
// the interval spans several steps and the restart falls between samples, so a
// resumed run that ignored the schedule would sample on its first step.
//
// run: cargo test -p symbi --test census_restart
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_refinement::refinement::Hierarchy;
use symbi_sim::census::{CensusEvaluator, RegisteredCensus};
use symbi_source_compile::CensusConfig;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 1, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const INTERVAL: f64 = 0.13;
const STEPS: u64 = 40;
const SPLIT: u64 = 17;

fn census(name: &str, accumulate: bool) -> RegisteredCensus {
    let cfg = CensusConfig::from_json(&format!(
        r#"{{
            "name": "{name}", "op": "add", "axes": [], "sample_interval": {INTERVAL}, "accumulate": {accumulate},
            "value_names": ["mass"], "values": [2], "params": [],
            "nodes": [
                {{"op": "VARIABLE_RHO"}},
                {{"op": "VARIABLE_DV"}},
                {{"op": "MULTIPLY", "left": 0, "right": 1}}
            ]
        }}"#
    ))
    .expect("census parses");
    RegisteredCensus::new(CensusEvaluator::new(&cfg).expect("census compiles"))
}

fn build(names: &[(&str, bool)]) -> Hier {
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([1.0])
        .spacing([0.1])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.4)
        .allocate()
        .expect("sim")
        .set_initial(|x| Prim::adiabatic(Density(1.0 + 0.25 * (3.0 * x[0]).sin()), Tensor::new([0.0]), Pressure(0.6)))
        .build();
    for (name, accumulate) in names {
        sim.store.censuses.push(census(name, *accumulate));
    }
    let k = Kern::new(GAMMA, 0.4, &sim.geom.allocated);
    Hierarchy::single(sim, k)
}

/// (time, values) of every row of the census at `index`
fn rows(h: &Hier, index: usize) -> Vec<(u64, Vec<u64>)> {
    let history = &h.levels[0].state.store.censuses[index].history;
    let stride = history.values().len() / history.len().max(1);
    (0..history.len())
        .map(|r| (history.time()[r].to_bits(), history.values()[r * stride..(r + 1) * stride].iter().map(|v| v.to_bits()).collect()))
        .collect()
}

/// the run split at `SPLIT` steps: the first segment, then a fresh hierarchy carrying
/// `resumed_names` restored from the first segment's checkpoint and run to `STEPS`
fn split_run(first_names: &[(&str, bool)], resumed_names: &[(&str, bool)], tag: &str) -> (Hier, Hier) {
    let mut first = build(first_names);
    first.evolve_steps(SPLIT).expect("first segment");
    let dir = std::env::temp_dir().join(format!("census_restart_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("chkpt.h5");
    let path = path.to_str().expect("utf-8 path");
    let levels: Vec<&Sim> = first.levels.iter().map(|l| &l.state).collect();
    symbi_sim::checkpoint::write_hierarchy_checkpoint(&levels, path, &Default::default()).expect("checkpoint");
    let mut resumed = build(resumed_names);
    resumed.restore_from_checkpoint(path).expect("restore");
    resumed.evolve_steps(STEPS - SPLIT).expect("resumed segment");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(resumed.levels[0].state.iteration, STEPS, "the resumed run stopped at {}", resumed.levels[0].state.iteration);
    (first, resumed)
}

#[test]
fn the_records_before_and_after_a_restart_are_the_uninterrupted_record() {
    let mut whole = build(&[("mass", false)]);
    whole.evolve_steps(STEPS).expect("uninterrupted run");
    let want = rows(&whole, 0);
    assert!(want.len() >= 6 && (want.len() as u64) < STEPS / 2, "{} samples over {STEPS} steps: the interval spans no steps", want.len());
    let (first, resumed) = split_run(&[("mass", false)], &[("mass", false)], "plain");
    let before = rows(&first, 0);
    // the restart falls between samples: the step after it is not due
    let restart = f64::from_bits(before.last().expect("a sample before the restart").0);
    assert!(first.levels[0].state.time - restart < INTERVAL, "premise: the restart is due at once");
    let after = rows(&resumed, 0);
    let joined: Vec<_> = before.into_iter().chain(after).collect();
    assert_eq!(joined.len(), want.len(), "the joined record holds {} rows, the uninterrupted {}", joined.len(), want.len());
    assert_eq!(joined, want, "the joined record differs from the uninterrupted one");
}

#[test]
fn an_accumulating_census_ends_on_the_uninterrupted_row() {
    let mut whole = build(&[("acc", true)]);
    whole.evolve_steps(STEPS).expect("uninterrupted run");
    let (_, resumed) = split_run(&[("acc", true)], &[("acc", true)], "acc");
    let (w, r) = (&whole.levels[0].state.store.censuses[0].history, &resumed.levels[0].state.store.censuses[0].history);
    assert_eq!(w.len(), 1, "an accumulating census holds one row");
    assert!(w.n_samples()[0] >= 6, "the row folds {} samples", w.n_samples()[0]);
    assert_eq!(rows(&resumed, 0), rows(&whole, 0), "the resumed row differs from the uninterrupted one");
    assert_eq!((r.n_samples(), r.t_start().iter().map(|t| t.to_bits()).collect::<Vec<_>>()), (w.n_samples(), w.t_start().iter().map(|t| t.to_bits()).collect::<Vec<_>>()));
}

#[test]
fn a_census_without_a_recorded_schedule_samples_one_interval_after_the_restart() {
    let (first, resumed) = split_run(&[("mass", false)], &[("mass", false), ("late", false)], "fallback");
    let restart = first.levels[0].state.time;
    let late = &resumed.levels[0].state.store.censuses[1].history;
    assert!(!late.is_empty(), "the late census never sampled");
    assert!(late.time()[0] - restart >= INTERVAL, "the late census sampled at {} after a restart at {restart}", late.time()[0]);
}
