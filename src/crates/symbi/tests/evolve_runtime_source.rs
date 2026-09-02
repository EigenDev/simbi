// =============================================================================
// evolve_runtime_source.rs
//
// end to end: a user source loaded at runtime (python -> json -> SourceConfig, no recompile)
// actually drives the live evolve loop. a force config `a = [p0, 0]` (p0 = 0.5) is bridged +
// wrapped (build_user_source) into a SourceEvaluator, attached to the adiabatic kernel-set via
// `with_runtime_source`, and `source_apply` interprets it per cell each SSP stage. a uniform
// periodic box under a uniform +x force is a clean galilean boost: density stays uniform (zero
// flux divergence), so total x-momentum grows as rho*g*t*ncells exactly. the runtime twin of
// `evolve_fused_source_routing` (which uses an AOT-baked source).
// =============================================================================

use symbi::prelude::*;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::{build_user_source, build_user_sources};

type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;

fn mom_x_total(sim: &Sim) -> f64 {
    sim.geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.mom[0].view().at(c))
        .sum()
}

#[test]
fn runtime_loaded_force_accelerates_gas() {
    // exactly what the python `Dag.force_source([p0, const(0.0)], dim=2)` emits, p0 = 0.5.
    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.5],
        "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "CONSTANT", "value": 0.0} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse config");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("wrap source");

    let mut sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .finish()
        .unwrap();
    sim.seed_cells(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)));

    // attach the runtime-loaded source — no recompile, no AOT-baked kernel.
    let sub = sim
        .substrate()
        .with_runtime_source(built, cfg.params.clone());

    assert!(
        mom_x_total(&sim).abs() < 1e-12,
        "x-momentum should start at zero"
    );

    let t_final = 0.05;
    evolve(&mut sim, &sub, t_final).expect("evolve under runtime source");

    // d(mom_x)/dt = rho * g = 1 * 0.5 per cell; uniform box stays rho=1, so
    //   total mom_x = 0.5 * t_final * n_interior_cells = 0.5 * 0.05 * 256 = 6.4.
    let got = mom_x_total(&sim);
    let expected = 0.5 * t_final * (16.0 * 16.0);
    assert!(got > 0.0, "gas did not accelerate: mom_x = {got}");
    assert!(
        (got - expected).abs() / expected < 0.02,
        "runtime force wrong magnitude: mom_x = {got}, expected ~{expected}",
    );

    // density stays uniform (the boost is galilean) — a sanity check the source didn't corrupt mass.
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        assert!(
            (rho - 1.0).abs() < 1e-6,
            "density drifted at {c:?}: rho = {rho}"
        );
    }
}

#[test]
fn runtime_source_collection_sums_independent_parameters() {
    let first = SourceConfig::from_json(
        r#"{
            "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.2],
            "nodes": [ {"op": "PARAMETER", "param_idx": 0},
                       {"op": "CONSTANT", "value": 0.0} ]
        }"#,
    )
    .expect("parse first source");
    let second = SourceConfig::from_json(
        r#"{
            "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.3],
            "nodes": [ {"op": "PARAMETER", "param_idx": 0},
                       {"op": "CONSTANT", "value": 0.0} ]
        }"#,
    )
    .expect("parse second source");
    let (built, params) =
        build_user_sources(&[first, second], &NEWTONIAN_SPEC).expect("compose sources");

    let mut sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .finish()
        .unwrap();
    sim.seed_cells(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)));
    let sub = sim.substrate().with_runtime_source(built, params);

    let t_final = 0.05;
    evolve(&mut sim, &sub, t_final).expect("evolve under source collection");

    let got = mom_x_total(&sim);
    let expected = (0.2 + 0.3) * t_final * (16.0 * 16.0);
    assert!(
        (got - expected).abs() / expected < 0.02,
        "composed force wrong magnitude: mom_x = {got}, expected {expected}",
    );
}
