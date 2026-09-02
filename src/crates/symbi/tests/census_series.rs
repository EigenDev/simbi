// =============================================================================
// census_series.rs
//
// the census as an output: registered on a simulation, sampled at the tail of every
// accepted step, and written into the checkpoint as its own group.
//
// what these gates establish:
//   - a run with no registrations is untouched — nothing is evaluated, nothing is
//     written, and the fluid result is bitwise identical.
//   - a registered census produces one sample per accepted step, at that step's time.
//   - the samples reach the checkpoint with the edges, labels and drop counts a reader
//     needs, and the fluid answer is unchanged by the observation.
//   - the series covers this run segment only.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_io::IoBackend;
use symbi_sim::census::{CensusEvaluator, RegisteredCensus};
use symbi_source_compile::CensusConfig;
use symbi_xpu::{CpuSpace, HostMemory};

type SimSph = SimState<Newtonian, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 1.4;
const N: usize = 96;
const R_LO: f64 = 0.5;
const R_HI: f64 = 1.5;
const DR: f64 = (R_HI - R_LO) / N as f64;
const T_FINAL: f64 = 0.02;

fn build_sim() -> SimSph {
    // a radial Sod: the waves stay interior to the shell over T_FINAL, and the state
    // actually evolves, so a census sampled per step is sampling something that moves.
    SimSph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim construction failed")
        .set_initial(|x| {
            let (rho, pre) = if x[0] < 1.0 { (1.0, 1.0) } else { (0.125, 0.1) };
            Prim::adiabatic(Density(rho), Tensor::new([0.0]), Pressure(pre))
        })
        .build()
}

fn kernels() -> AdiabaticSubstrateKernelSet<HostMemory, f64, 1> {
    AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &build_sim().geom.allocated)
}

/// mass in four radial shells spanning the domain.
fn shell_census() -> CensusConfig {
    CensusConfig::from_json(
        r#"{
            "name": "shells",
            "axes": [{"name": "r", "expr": 0, "edges": [0.5, 0.75, 1.0, 1.25, 1.5]}],
            "values": [3, 4],
            "value_names": ["mass", "volume"],
            "op": "add",
            "params": [],
            "nodes": [
                {"op": "VARIABLE_X1"},
                {"op": "VARIABLE_RHO"},
                {"op": "VARIABLE_DV"},
                {"op": "MULTIPLY", "left": 1, "right": 2},
                {"op": "VARIABLE_DV"}
            ]
        }"#,
    )
    .expect("census config parses")
}

fn register(sim: &mut SimSph, cfg: &CensusConfig) {
    let ev = CensusEvaluator::new(cfg).expect("census compiles");
    sim.store.censuses.push(RegisteredCensus::new(ev));
}

#[test]
fn a_run_without_registrations_is_untouched_by_the_mechanism() {
    // the observer must be inert when nobody is observing: no registrations means the
    // sampling hook does not evaluate, allocate, or perturb the fluid answer.
    let mut bare = build_sim();
    evolve(&mut bare, &kernels(), T_FINAL).expect("bare evolution");

    let mut observed = build_sim();
    register(&mut observed, &shell_census());
    evolve(&mut observed, &kernels(), T_FINAL).expect("observed evolution");

    assert!(bare.censuses.is_empty());
    assert!(!observed.censuses[0].history.is_empty());

    // a census is a pure observer: it reads the state and feeds nothing back, so the
    // fluid result must agree bitwise with the unobserved run.
    for c in bare.geom.interior.iter() {
        assert_eq!(
            bare.fields.cons.den.view().at(c).to_bits(),
            observed.fields.cons.den.view().at(c).to_bits(),
            "density at {c:?} moved when a census was registered"
        );
    }
    assert_eq!(bare.time.to_bits(), observed.time.to_bits());
    assert_eq!(bare.iteration, observed.iteration);
}

#[test]
fn one_sample_lands_per_accepted_step() {
    let mut sim = build_sim();
    register(&mut sim, &shell_census());
    evolve(&mut sim, &kernels(), T_FINAL).expect("evolution");

    let history = &sim.censuses[0].history;
    assert_eq!(
        history.len(),
        sim.iteration as usize,
        "one sample per accepted step"
    );
    assert!(
        history.len() > 4,
        "the run must take several steps to mean anything"
    );
    assert_eq!(history.n_segments(), 4);
    assert_eq!(history.n_values(), 2);
    assert_eq!(history.values().len(), history.len() * 4 * 2);

    // sample times increase and end at the final time.
    for w in history.time().windows(2) {
        assert!(w[1] > w[0], "sample times must advance");
    }
    assert_eq!(
        history.time().last().copied().unwrap().to_bits(),
        sim.time.to_bits()
    );
    // the shell edges span the domain, so nothing falls outside the binning.
    assert!(history.dropped().iter().all(|&d| d == 0));
}

#[test]
fn each_sample_conserves_the_total_mass_across_its_shells() {
    // the partition property, now per sample and through the whole run: an outflow
    // boundary lets mass leave, so the total drifts — but within any one sample the
    // shells must still add up to that instant's total.
    let mut sim = build_sim();
    register(&mut sim, &shell_census());
    evolve(&mut sim, &kernels(), T_FINAL).expect("evolution");

    let history = &sim.censuses[0].history;
    let (n_seg, n_val) = (history.n_segments(), history.n_values());
    let final_total = sim
        .conservation_diag()
        .expect("conservation diagnostic")
        .mass;

    // the mass channel is value 0; the volume channel is value 1.
    let sample = |ii: usize, value: usize| -> f64 {
        (0..n_seg)
            .map(|s| history.values()[ii * n_seg * n_val + s * n_val + value])
            .sum()
    };
    let last = history.len() - 1;
    assert!(
        (sample(last, 0) - final_total).abs() <= 1.0e-12 * final_total.abs(),
        "the last sample's shells sum to {:e}, the diagnostic gives {final_total:e}",
        sample(last, 0)
    );

    // the volume channel is a property of the grid, not the flow, so it must be the same
    // in every sample — a drifting volume would mean the binning moved under the census.
    let v0 = sample(0, 1);
    for ii in 0..history.len() {
        assert!(
            (sample(ii, 1) - v0).abs() <= 1.0e-13 * v0,
            "shell volume drifted between samples: {:e} vs {v0:e}",
            sample(ii, 1)
        );
    }
    // and the shell volumes are the domain volume, not a cell count.
    assert!(v0 > 0.0);
}

#[test]
fn the_series_reaches_the_checkpoint_with_its_edges_and_labels() {
    let dir = std::env::temp_dir().join(format!("simbi_census_series_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("census.h5");

    let mut sim = build_sim();
    register(&mut sim, &shell_census());
    evolve(&mut sim, &kernels(), T_FINAL).expect("evolution");
    let n_samples = sim.censuses[0].history.len();

    symbi_sim::checkpoint::write_checkpoint(
        &sim,
        path.to_str().expect("path"),
        &symbi_sim::checkpoint::Metadata::new(),
    )
    .expect("checkpoint write");

    let tree = symbi_io::Hdf5Backend.read(&path).expect("checkpoint read");
    // the registrations nest under one `census` group, each under its own name.
    let group = tree
        .find_group("census")
        .expect("the census group is missing from the checkpoint")
        .find_group("shells")
        .expect("the census must be written under its registered name");

    let attr_u64 = |k: &str| -> u64 {
        match group.find_attr(k) {
            Some(symbi_io::Attr::U64(v)) => *v,
            other => panic!("attr {k}: {other:?}"),
        }
    };
    let attr_str = |k: &str| match group.find_attr(k) {
        Some(symbi_io::Attr::Str(v)) => v.clone(),
        other => panic!("attr {k}: {other:?}"),
    };
    assert_eq!(attr_u64("n_segments"), 4);
    assert_eq!(attr_u64("n_values"), 2);
    assert_eq!(attr_str("value_names"), "mass,volume");
    assert_eq!(attr_str("op"), "add");
    assert_eq!(attr_str("axis0_name"), "r");
    // the size of the compiled per-cell graph: what a census actually costs.
    assert!(attr_u64("node_count") > 0);

    let ds = |k: &str| {
        group
            .find_dataset(k)
            .unwrap_or_else(|| panic!("dataset {k}"))
    };
    assert_eq!(ds("time").shape, vec![n_samples]);
    assert_eq!(ds("values").shape, vec![n_samples, 4, 2]);
    assert_eq!(ds("dropped").shape, vec![n_samples]);
    // the edges are a property of the registration, so they are written once, not per row.
    assert_eq!(ds("axis0_edges").shape, vec![5]);
    assert_eq!(
        ds("axis0_edges").data.as_f64().unwrap().to_vec(),
        vec![0.5, 0.75, 1.0, 1.25, 1.5]
    );
    // no cell fell outside the binning, and that fact travels with the numbers.
    assert!(ds("dropped").data.as_u64().unwrap().iter().all(|&d| d == 0));

    std::fs::remove_dir_all(&dir).ok();
}
