// =============================================================================
// single_decomp_identity.rs
//
// exact driver-equivalence gate for a single grid and a one-tile decomposition.
// both drivers start from independently constructed identical states, use the
// same substrate kernels and timestep policy, and advance to the same final
// time. every conserved value, the clock, and the final timestep must match
// bitwise.
// =============================================================================

use symbi::prelude::*;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
use symbi_algebra::Domain;
use symbi_geometry::MotionState;
use symbi_grid::Field;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_ib::{Body, BodyCollection, BodyKind};
use symbi_sim::tracers::seed_mass_weighted;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_user_sources;

const GAMMA: f64 = 1.4;
const T_FINAL: f64 = 0.03;

type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
type Kern = symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn make(timestepping: Timestepping) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32, 24])
        .bounds([0.0, 0.0], [1.0, 0.75])
        .boundaries(BoundaryType::Periodic)
        .cfl(0.35)
        .timestepping(timestepping)
        .finish()
        .expect("simulation construction");
    sim.seed_cells(|x| {
        let phase = 2.0 * std::f64::consts::PI * x[0];
        Prim::adiabatic(
            Density(1.0 + 0.1 * phase.sin()),
            Tensor::new([0.15, -0.08]),
            Pressure(0.8 + 0.05 * phase.cos()),
        )
    });
    let kernels = sim.substrate();
    (sim, kernels)
}

fn assert_field_bits(
    single: &Field<f64, 2, HostMemory>,
    decomposed: &Field<f64, 2, HostMemory>,
    interior: &Domain<2>,
    name: &str,
) {
    for cell in interior.iter() {
        assert_eq!(
            single.view().at(cell).to_bits(),
            decomposed.view().at(cell).to_bits(),
            "{name} differs at {cell:?}: {} vs {}",
            single.view().at(cell),
            decomposed.view().at(cell),
        );
    }
}

fn assert_driver_identity(timestepping: Timestepping) {
    let (mut single, single_kernels) = make(timestepping);
    let (mut decomposed, decomposed_kernels) = make(timestepping);
    run_and_assert_identity(
        &mut single,
        &single_kernels,
        &mut decomposed,
        &decomposed_kernels,
        timestepping,
    );
}

fn run_and_assert_identity(
    single: &mut Sim,
    single_kernels: &Kern,
    decomposed: &mut Sim,
    decomposed_kernels: &Kern,
    timestepping: Timestepping,
) {
    evolve(single, single_kernels, T_FINAL).expect("single-grid evolve");
    evolve_decomposed(
        &mut [&mut *decomposed],
        &[decomposed_kernels],
        [1, 1],
        &[0],
        timestepping,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );

    assert_eq!(
        single.time.to_bits(),
        decomposed.time.to_bits(),
        "time differs"
    );
    assert_eq!(single.dt.to_bits(), decomposed.dt.to_bits(), "dt differs");
    let interior = &single.geom.interior;
    assert_field_bits(
        &single.fields.cons.den,
        &decomposed.fields.cons.den,
        interior,
        "cons.den",
    );
    for component in 0..2 {
        assert_field_bits(
            &single.fields.cons.mom[component],
            &decomposed.fields.cons.mom[component],
            interior,
            &format!("cons.mom[{component}]"),
        );
    }
    assert_field_bits(
        single.fields.cons.nrg_field().expect("single energy"),
        decomposed
            .fields
            .cons
            .nrg_field()
            .expect("decomposed energy"),
        interior,
        "cons.nrg",
    );
}

fn make_with_sources(timestepping: Timestepping) -> (Sim, Kern) {
    let (sim, _) = make(timestepping);
    let configs = [
        r#"{
            "kind":"force", "dim":2, "outputs":[2,3], "params":[0.25],
            "vocabulary":{"reads":["x_0"],"params":[0]},
            "nodes":[{"op":"PARAMETER","param_idx":0},
                     {"op":"VARIABLE_X1"},
                     {"op":"MULTIPLY","left":0,"right":1},
                     {"op":"CONSTANT","value":0.0}]
        }"#,
        r#"{
            "kind":"force", "dim":2, "outputs":[2,3], "params":[0.75],
            "vocabulary":{"reads":["x_0"],"params":[0]},
            "nodes":[{"op":"PARAMETER","param_idx":0},
                     {"op":"VARIABLE_X1"},
                     {"op":"MULTIPLY","left":0,"right":1},
                     {"op":"CONSTANT","value":0.0}]
        }"#,
        r#"{
            "kind":"force", "dim":2, "outputs":[23,24], "params":[],
            "vocabulary":{"reads":["x_0"],"params":[]},
            "nodes":[
                {"op":"VARIABLE_X1"},{"op":"CONSTANT","value":0.0},
                {"op":"CONSTANT","value":0.0},{"op":"CONSTANT","value":1.0},
                {"op":"SUBTRACT","left":0,"right":1},
                {"op":"MULTIPLY","left":3,"right":4},{"op":"ADD","left":2,"right":5},
                {"op":"CONSTANT","value":0.5},{"op":"CONSTANT","value":0.5},
                {"op":"CONSTANT","value":-1.0},{"op":"SUBTRACT","left":0,"right":7},
                {"op":"MULTIPLY","left":9,"right":10},{"op":"ADD","left":8,"right":11},
                {"op":"CONSTANT","value":0.5},{"op":"LT","left":0,"right":13},
                {"op":"IF_THEN_ELSE","condition":14,"true_case":6,"false_case":12},
                {"op":"CONSTANT","value":0.0},{"op":"CONSTANT","value":0.0},
                {"op":"CONSTANT","value":0.0},{"op":"LT","left":0,"right":18},
                {"op":"CONSTANT","value":1.0},{"op":"GT","left":0,"right":20},
                {"op":"IF_THEN_ELSE","condition":21,"true_case":17,"false_case":15},
                {"op":"IF_THEN_ELSE","condition":19,"true_case":16,"false_case":22},
                {"op":"CONSTANT","value":0.0}
            ]
        }"#,
    ]
    .map(|json| SourceConfig::from_json(json).expect("source parse"));
    let (built, params) =
        build_user_sources(&configs, &NEWTONIAN_SPEC).expect("source composition");
    let kernels = sim.substrate().with_fused_runtime_source(built, params);
    (sim, kernels)
}

fn assert_source_driver_identity(timestepping: Timestepping) {
    let (mut single, single_kernels) = make_with_sources(timestepping);
    let (mut decomposed, decomposed_kernels) = make_with_sources(timestepping);
    let initial_momentum: f64 = single
        .geom
        .interior
        .iter()
        .map(|cell| *single.fields.cons.mom[0].view().at(cell))
        .sum();

    run_and_assert_identity(
        &mut single,
        &single_kernels,
        &mut decomposed,
        &decomposed_kernels,
        timestepping,
    );

    let final_momentum: f64 = single
        .geom
        .interior
        .iter()
        .map(|cell| *single.fields.cons.mom[0].view().at(cell))
        .sum();
    assert!(
        (final_momentum - initial_momentum).abs() > 1.0e-3,
        "source produced no momentum; driver identity is vacuous",
    );
}

fn make_with_body(timestepping: Timestepping) -> (Sim, Kern) {
    let (sim, _) = make(timestepping);
    let bodies = BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.5, 0.375]),
        Tensor::zeros(),
        1.0,
        0.05,
        0.12,
        10.0,
        1.0e-3,
        0.18,
    ));
    let sim = sim.with_bodies(bodies);
    let kernels = sim.substrate().with_source_fusion();
    (sim, kernels)
}

fn accreted_mass(sim: &Sim) -> f64 {
    let body = sim
        .immersed
        .as_ref()
        .expect("body collection")
        .bodies
        .get(0);
    match body.kind {
        BodyKind::BlackHole {
            total_accreted_mass,
            ..
        } => total_accreted_mass,
        _ => panic!("test body is not a black hole"),
    }
}

fn assert_body_driver_identity(timestepping: Timestepping) {
    let (mut single, single_kernels) = make_with_body(timestepping);
    let (mut decomposed, decomposed_kernels) = make_with_body(timestepping);
    let (mut body_free, body_free_kernels) = make(timestepping);

    run_and_assert_identity(
        &mut single,
        &single_kernels,
        &mut decomposed,
        &decomposed_kernels,
        timestepping,
    );
    evolve(&mut body_free, &body_free_kernels, T_FINAL).expect("body-free evolve");

    let single_mass = accreted_mass(&single);
    let decomposed_mass = accreted_mass(&decomposed);
    assert_eq!(
        single_mass.to_bits(),
        decomposed_mass.to_bits(),
        "body accretion ledger differs",
    );
    assert!(
        single_mass > 1.0e-6,
        "body recorded no accretion; driver identity is vacuous",
    );
    let minimum_density = single
        .geom
        .interior
        .iter()
        .map(|cell| *single.fields.cons.den.view().at(cell))
        .fold(f64::INFINITY, f64::min);
    assert!(
        minimum_density < 0.99,
        "body sink removed no mass; driver identity is vacuous",
    );
    let gravity_changed_far_field = single.geom.interior.iter().any(|cell| {
        let position = single.geom.cell_coord(cell);
        let radius = ((position[0] - 0.5).powi(2) + (position[1] - 0.375).powi(2)).sqrt();
        radius > 0.25
            && single.fields.cons.mom[0].view().at(cell).to_bits()
                != body_free.fields.cons.mom[0].view().at(cell).to_bits()
    });
    assert!(
        gravity_changed_far_field,
        "body gravity left the far-field momentum unchanged; driver identity is vacuous",
    );
}

fn make_with_motion(timestepping: Timestepping) -> (Sim, Kern) {
    let (mut sim, _) = make(timestepping);
    sim.motion = MotionState::homologous(1.0, 0.5);
    let kernels = sim.substrate();
    (sim, kernels)
}

fn assert_motion_driver_identity(timestepping: Timestepping) {
    let (mut single, single_kernels) = make_with_motion(timestepping);
    let (mut decomposed, decomposed_kernels) = make_with_motion(timestepping);

    run_and_assert_identity(
        &mut single,
        &single_kernels,
        &mut decomposed,
        &decomposed_kernels,
        timestepping,
    );

    assert_eq!(
        single.motion.a.to_bits(),
        decomposed.motion.a.to_bits(),
        "mesh scale factor differs",
    );
    assert_eq!(
        single.motion.a_dot.to_bits(),
        decomposed.motion.a_dot.to_bits(),
        "mesh expansion rate differs",
    );
    assert!(
        single.motion.a > 1.0 + 0.25 * T_FINAL,
        "mesh scale factor never advanced; driver identity is vacuous",
    );
}

fn make_with_tracers(timestepping: Timestepping) -> (Sim, Kern) {
    let (mut sim, _) = make(timestepping);
    sim.tracers = Some(seed_mass_weighted(&sim, 128));
    let kernels = sim.substrate();
    (sim, kernels)
}

fn assert_tracer_driver_identity(timestepping: Timestepping) {
    let (mut single, single_kernels) = make_with_tracers(timestepping);
    let (mut decomposed, decomposed_kernels) = make_with_tracers(timestepping);
    let initial_owners = single
        .tracers
        .as_ref()
        .expect("single tracers")
        .owner
        .clone();

    run_and_assert_identity(
        &mut single,
        &single_kernels,
        &mut decomposed,
        &decomposed_kernels,
        timestepping,
    );

    let single_tracers = single.tracers.as_ref().expect("single tracers");
    let decomposed_tracers = decomposed.tracers.as_ref().expect("decomposed tracers");
    assert_eq!(
        single_tracers.id, decomposed_tracers.id,
        "tracer ids differ"
    );
    assert_eq!(
        single_tracers.weight.to_bits(),
        decomposed_tracers.weight.to_bits(),
        "tracer weights differ",
    );
    assert_eq!(single_tracers.len(), decomposed_tracers.len());
    assert_eq!(
        single_tracers.owner, decomposed_tracers.owner,
        "tracer owners differ"
    );
    let mut moved = 0usize;
    for index in 0..single_tracers.len() {
        let single_flags = single_tracers.flags[index];
        let decomposed_flags = decomposed_tracers.flags[index];
        assert_eq!(
            single_flags.escaped, decomposed_flags.escaped,
            "tracer {index} escaped flag differs",
        );
        assert_eq!(
            single_flags.crossed_sink, decomposed_flags.crossed_sink,
            "tracer {index} sink flag differs",
        );
        assert_eq!(
            single_flags.crossing_time.to_bits(),
            decomposed_flags.crossing_time.to_bits(),
            "tracer {index} crossing time differs",
        );
        if single_tracers.owner[index] != initial_owners[index] {
            moved += 1;
        }
    }
    assert!(
        moved > 0,
        "no tracer crossed a cell face; driver identity is vacuous",
    );
}

#[test]
fn euler_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_driver_identity(Timestepping::Euler);
}

#[test]
fn rk2_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk3_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_driver_identity(Timestepping::Rk3);
}

#[test]
fn euler_source_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_source_driver_identity(Timestepping::Euler);
}

#[test]
fn rk2_source_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_source_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk3_source_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_source_driver_identity(Timestepping::Rk3);
}

#[test]
fn euler_body_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_body_driver_identity(Timestepping::Euler);
}

#[test]
fn rk2_body_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_body_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk3_body_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_body_driver_identity(Timestepping::Rk3);
}

#[test]
fn euler_motion_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_motion_driver_identity(Timestepping::Euler);
}

#[test]
fn rk2_motion_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_motion_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk3_motion_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_motion_driver_identity(Timestepping::Rk3);
}

#[test]
fn euler_tracer_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_tracer_driver_identity(Timestepping::Euler);
}

#[test]
fn rk2_tracer_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_tracer_driver_identity(Timestepping::Rk2);
}

#[test]
fn rk3_tracer_single_grid_equals_one_tile_decomposed_bitwise() {
    assert_tracer_driver_identity(Timestepping::Rk3);
}
