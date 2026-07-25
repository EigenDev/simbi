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
use symbi::sim::decomp::{evolve_decomposed, LocalCopy};
use symbi_algebra::Domain;
use symbi_grid::Field;
use symbi_hydro::expr_bridge::build_user_sources;
use symbi_hydro::{SourceConfig, NEWTONIAN_SPEC};

const GAMMA: f64 = 1.4;
const T_FINAL: f64 = 0.03;

type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;
type Kern = symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet<
    HostMemory,
    f64,
    2,
>;

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
        Prim {
            rho: 1.0 + 0.1 * phase.sin(),
            vel: Tensor::new([0.15, -0.08]),
            pre: 0.8 + 0.05 * phase.cos(),
        }
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

    assert_eq!(single.time.to_bits(), decomposed.time.to_bits(), "time differs");
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
        decomposed.fields.cons.nrg_field().expect("decomposed energy"),
        interior,
        "cons.nrg",
    );
}

fn make_with_sources(timestepping: Timestepping) -> (Sim, Kern) {
    let (sim, _) = make(timestepping);
    let configs = [
        r#"{
            "kind":"force", "dim":2, "outputs":[2,3], "params":[0.25],
            "nodes":[{"op":"PARAMETER","param_idx":0},
                     {"op":"VARIABLE_X1"},
                     {"op":"MULTIPLY","left":0,"right":1},
                     {"op":"CONSTANT","value":0.0}]
        }"#,
        r#"{
            "kind":"force", "dim":2, "outputs":[2,3], "params":[0.75],
            "nodes":[{"op":"PARAMETER","param_idx":0},
                     {"op":"VARIABLE_X1"},
                     {"op":"MULTIPLY","left":0,"right":1},
                     {"op":"CONSTANT","value":0.0}]
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
