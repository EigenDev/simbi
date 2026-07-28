// =============================================================================
// census_expressions.rs
//
// the user-facing half of a binned reduction: a census whose bin axes and accumulators
// are EXPRESSIONS, lowered from the serialized wire form the python front door emits and
// compiled through the same path a source term takes.
//
// what these gates establish:
//   - the cell-volume leaf resolves to the block geometry's lab-frame measure, so
//     `density * cell_volume` summed over a spherical grid is the conservation
//     diagnostic's total mass. this is the dV leaf's only binding, and the spherical
//     r^2 dr measure is what makes the check non-vacuous.
//   - an accumulator can RECONSTRUCT a conserved quantity from primitives — the
//     energy census below is p/(gamma-1) + rho v^2/2, written as a dag.
//   - the bin axes and the values share one graph, so a coordinate used by both is
//     evaluated once.
//   - a cell outside the bin edges, and a ghost cell, are excluded distinctly.
// =============================================================================

use symbi::regimes::substrate_gpu::field_segmented_reduce;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::CensusConfig;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_sim::census::{CensusEvaluator, SEGMENT_EXCLUDED};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type SimSph = SimState<Newtonian, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 1.4;
const N: usize = 256;
const R_LO: f64 = 0.5;
const R_HI: f64 = 4.5;
const DR: f64 = (R_HI - R_LO) / N as f64;

/// a steeply falling atmosphere: the density spans orders of magnitude across the shell
/// while the cell volume grows as r^2, so the two weightings pull in opposite directions
/// and a census using the wrong measure cannot land on the right total by coincidence.
fn density_at(r: f64) -> f64 {
    r.powi(-3)
}

fn pressure_at(r: f64) -> f64 {
    0.1 * density_at(r)
}

fn build_sim() -> SimSph {
    let sim = SimSph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim construction failed")
        .set_initial(|x| Prim {
            rho: density_at(x[0]),
            vel: Tensor::new([0.0]),
            pre: pressure_at(x[0]),
        })
        .build();
    // seeding writes the CONSERVED state; the primitives a census reads are produced by
    // the conserved-to-primitive recovery, which the evolve loop runs each stage. recover
    // once here so the sampled state is the live one a census sees in production.
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.c2p(&sim.store);
    sim
}

/// the mass census: one accumulator, `rho * dv`, no bin axes.
fn mass_census() -> CensusConfig {
    CensusConfig::from_json(
        r#"{
            "name": "conservation",
            "axes": [],
            "values": [2],
            "value_names": ["mass"],
            "op": "add",
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

#[test]
fn a_mass_expression_census_reproduces_the_conservation_diagnostic() {
    // the dV leaf's binding, end to end: `density * cell_volume` accumulated over a
    // spherical grid IS the conservation diagnostic's total mass.
    let sim = build_sim();
    let diag = sim.conservation_diag().expect("conservation diagnostic");

    let ev = CensusEvaluator::new(&mass_census()).expect("census compiles");
    assert_eq!(ev.spec().n_segments(), 1, "no axes means one bucket");
    assert!(
        ev.params_for().iter().any(|p| p == "dv"),
        "the census must declare the cell-volume leaf, else it is not weighting by the measure"
    );

    let fields = sim.census_fields(&ev).expect("host-resident sim");
    let refs: Vec<_> = fields.values.iter().collect();
    let census = field_segmented_reduce(
        &refs,
        &fields.segment,
        &sim.geom.interior,
        ev.spec().n_segments(),
        ev.spec().op(),
    );

    assert_eq!(census.dropped, 0, "a census with no axes drops nothing");
    let tol = 1.0e-12 * diag.mass.abs();
    assert!(
        (census.values[0] - diag.mass).abs() <= tol,
        "expression census mass {:e} != conservation diagnostic {:e}",
        census.values[0],
        diag.mass
    );

    // non-vacuity: on this r^2 grid an unweighted sum is a different number, so the
    // agreement above is a statement about the measure and not about a flat geometry.
    let unweighted: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.prim.rho.view().at(c))
        .sum();
    assert!(
        (unweighted - diag.mass).abs() > 0.1 * diag.mass.abs(),
        "the r^2 measure must matter here (unweighted {unweighted:e} vs weighted {:e})",
        diag.mass
    );
}

#[test]
fn an_accumulator_can_reconstruct_the_conserved_energy_from_primitives() {
    // the census leaves are the primitives, so an extensive conserved total is written as
    // the expression that defines it: E = p/(gamma-1) + rho v^2 / 2, times the measure.
    // this is what makes the mechanism a language rather than a fixed set of diagnostics.
    let sim = build_sim();
    let diag = sim.conservation_diag().expect("conservation diagnostic");
    let diag_energy = diag.energy.expect("Newtonian carries an energy equation");

    let cfg = CensusConfig::from_json(
        r#"{
            "name": "energy",
            "axes": [],
            "values": [11],
            "value_names": ["energy"],
            "op": "add",
            "params": [0.4],
            "nodes": [
                {"op": "VARIABLE_PRESSURE"},
                {"op": "PARAMETER", "param_idx": 0},
                {"op": "DIVIDE", "left": 0, "right": 1},
                {"op": "VARIABLE_RHO"},
                {"op": "VARIABLE_VEL1"},
                {"op": "MULTIPLY", "left": 4, "right": 4},
                {"op": "MULTIPLY", "left": 3, "right": 5},
                {"op": "CONSTANT", "value": 0.5},
                {"op": "MULTIPLY", "left": 7, "right": 6},
                {"op": "ADD", "left": 2, "right": 8},
                {"op": "VARIABLE_DV"},
                {"op": "MULTIPLY", "left": 9, "right": 10}
            ]
        }"#,
    )
    .expect("census config parses");

    let ev = CensusEvaluator::new(&cfg).expect("census compiles");
    let fields = sim.census_fields(&ev).expect("host-resident sim");
    let refs: Vec<_> = fields.values.iter().collect();
    let census = field_segmented_reduce(
        &refs,
        &fields.segment,
        &sim.geom.interior,
        ev.spec().n_segments(),
        ev.spec().op(),
    );

    let tol = 1.0e-12 * diag_energy.abs();
    assert!(
        (census.values[0] - diag_energy).abs() <= tol,
        "reconstructed energy {:e} != conservation diagnostic {:e}",
        census.values[0],
        diag_energy
    );
}

#[test]
fn a_binned_expression_census_partitions_the_total() {
    // a radial axis and a mass accumulator sharing one dag. the shells must sum back to
    // the global total, which is the check that catches a bucket assignment that
    // double-counts or loses cells.
    let sim = build_sim();
    let diag = sim.conservation_diag().expect("conservation diagnostic");

    let cfg = CensusConfig::from_json(
        r#"{
            "name": "shells",
            "axes": [{"name": "r", "expr": 0, "edges": [0.5, 1.0, 2.0, 3.0, 4.5]}],
            "values": [3],
            "value_names": ["mass"],
            "op": "add",
            "params": [],
            "nodes": [
                {"op": "VARIABLE_X1"},
                {"op": "VARIABLE_RHO"},
                {"op": "VARIABLE_DV"},
                {"op": "MULTIPLY", "left": 1, "right": 2}
            ]
        }"#,
    )
    .expect("census config parses");

    let ev = CensusEvaluator::new(&cfg).expect("census compiles");
    assert_eq!(ev.spec().n_segments(), 4, "four shells from five edges");

    let fields = sim.census_fields(&ev).expect("host-resident sim");
    let refs: Vec<_> = fields.values.iter().collect();
    let census = field_segmented_reduce(
        &refs,
        &fields.segment,
        &sim.geom.interior,
        ev.spec().n_segments(),
        ev.spec().op(),
    );

    assert_eq!(
        census.dropped, 0,
        "the shell edges span the domain, so no interior cell is outside the binning"
    );
    for (s, v) in census.values.iter().enumerate() {
        assert!(
            *v > 0.0,
            "shell {s} is empty; the binning does not tile the gas"
        );
    }
    // the shells fall off steeply, so the inner one must dominate — a binning that
    // scattered cells into the wrong shells could still sum correctly.
    assert!(
        census.values[0] > census.values[3],
        "an r^-3 atmosphere puts more mass in the inner shell than the outer one"
    );

    let binned: f64 = census.values.iter().sum();
    let tol = 1.0e-12 * diag.mass.abs();
    assert!(
        (binned - diag.mass).abs() <= tol,
        "shell masses sum to {binned:e}, global total is {:e}",
        diag.mass
    );
}

#[test]
fn ghost_cells_are_excluded_rather_than_binned() {
    // a ghost cell is not physical gas. it must carry the excluded marker, distinct from
    // the marker for a cell that was to be reduced and fell outside the edges — otherwise
    // the halo would be reported as a shortfall of the binning.
    let sim = build_sim();
    let ev = CensusEvaluator::new(&mass_census()).expect("census compiles");
    let fields = sim.census_fields(&ev).expect("host-resident sim");

    let interior: std::collections::HashSet<[isize; 1]> = sim.geom.interior.iter().collect();
    let mut n_ghost = 0usize;
    for c in sim.geom.allocated.iter() {
        let seg = *fields.segment.view().at(c);
        if interior.contains(&c) {
            assert_eq!(
                seg, 0,
                "an interior cell bins into the single bucket at {c:?}"
            );
        } else {
            n_ghost += 1;
            assert_eq!(seg, SEGMENT_EXCLUDED, "ghost cell {c:?} must be excluded");
        }
    }
    assert!(
        n_ghost > 0,
        "the grid must have a halo for this to mean anything"
    );
}

#[test]
fn a_census_sampled_before_the_recovery_fails_loudly() {
    // seeding writes the CONSERVED state; the primitives a census reads come from the
    // conserved-to-primitive recovery. sampling before that ran would report a total
    // mass of zero with no complaint, which reads as physics rather than as a mistake.
    let sim = SimSph::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim construction failed")
        .set_initial(|x| Prim {
            rho: density_at(x[0]),
            vel: Tensor::new([0.0]),
            pre: pressure_at(x[0]),
        })
        .build();
    assert!(
        !sim.store.has_recovered_primitives(),
        "a freshly seeded store has not recovered its primitives"
    );

    let ev = CensusEvaluator::new(&mass_census()).expect("census compiles");
    let panicked =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.census_fields(&ev)))
            .is_err();
    assert!(
        panicked,
        "sampling before the recovery must fail loudly, not report zeros"
    );

    // and after the recovery the same census samples normally.
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.c2p(&sim.store);
    assert!(sim.store.has_recovered_primitives());
    assert!(sim.census_fields(&ev).is_some());
}

#[test]
fn a_forward_referencing_dag_is_refused_with_a_readable_error() {
    // the wire format is a topologically ordered dag. a node whose operand comes later is
    // malformed, and it must be reported against the config rather than surfacing as an
    // out-of-bounds inside a graph pass, where nothing points back at the cause.
    let cfg = CensusConfig::from_json(
        r#"{
            "name": "bad", "axes": [], "values": [0], "value_names": ["v"],
            "op": "add", "params": [],
            "nodes": [
                {"op": "MULTIPLY", "left": 1, "right": 2},
                {"op": "VARIABLE_RHO"},
                {"op": "VARIABLE_DV"}
            ]
        }"#,
    )
    .expect("census config parses as json");
    // CensusEvaluator holds compiled kernels and is not Debug, so unwrap_err won't compile.
    let err = match CensusEvaluator::new(&cfg) {
        Err(e) => e,
        Ok(_) => panic!("a forward reference must be refused"),
    };
    assert!(
        err.contains("topologically ordered"),
        "the error must name the ordering rule, got: {err}"
    );
}

#[test]
fn a_census_reading_pressure_is_refused_on_a_regime_without_one() {
    // an isothermal regime carries no pressure field. reading `pre` there would silently
    // accumulate zeros, which is a wrong answer rather than a missing one.
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::isothermal::IsoNewtonian;
    use symbi_hydro::state::PrimG;

    type SimIso = SimState<IsoNewtonian, 1, Spherical, Isothermal<f64>, CpuSpace, HostMemory>;
    let sim = SimIso::build(IsoNewtonian, Isothermal { cs: 1.0 }, Spherical)
        .cells([N])
        .origin([R_LO])
        .spacing([DR])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("iso sim construction failed")
        .set_initial(|x| PrimG::<f64, 1, IsoModel> {
            rho: density_at(x[0]),
            vel: Tensor::new([0.0]),
            pre: Default::default(),
        })
        .build();

    let cfg = CensusConfig::from_json(
        r#"{
            "name": "thermal", "axes": [], "values": [2], "value_names": ["p_dv"],
            "op": "add", "params": [],
            "nodes": [
                {"op": "VARIABLE_PRESSURE"},
                {"op": "VARIABLE_DV"},
                {"op": "MULTIPLY", "left": 0, "right": 1}
            ]
        }"#,
    )
    .expect("census config parses");
    let ev = CensusEvaluator::new(&cfg).expect("census compiles");
    assert!(ev.reads_pressure());

    let panicked =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.census_fields(&ev))).is_err();
    assert!(
        panicked,
        "a census reading pressure on an isothermal regime must fail loudly, not read zeros"
    );
}
