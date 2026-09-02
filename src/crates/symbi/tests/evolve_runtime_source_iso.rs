// =============================================================================
// evolve_runtime_source_iso.rs
//
// the runtime user-source mechanism is regime-agnostic: the same mechanism that drives the
// adiabatic set (evolve_runtime_source.rs) drives the isothermal set with no energy equation. a
// force `a = [p0, 0]` (p0 = 0.5) loaded at runtime (python -> json -> SourceConfig) accelerates the
// gas; iso momentum is rho*v, so total x-momentum grows as rho*g*t*ncells exactly — and crucially
// no `nrg` overlay is emitted (iso has no energy field), exercising the `has_energy = false` path.
// cooling / nrg-targeted sources are rejected at `build_user_source(&cfg, &ISO_NEWTONIAN_SPEC)`,
// covered by the expr_bridge validation unit tests.
// =============================================================================

use symbi::prelude::*;
use symbi_hydro::ISO_NEWTONIAN_SPEC;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::state::PrimG;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_user_source;

type Sim = SimCpu<IsoNewtonian, 2, Cartesian, Isothermal<f64>>;

fn mom_x_total(sim: &Sim) -> f64 {
    sim.geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.mom[0].view().at(c))
        .sum()
}

#[test]
fn runtime_loaded_force_accelerates_iso_gas() {
    // force a = [p0, 0], p0 = 0.5 — same config shape as the adiabatic twin.
    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.5],
        "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "CONSTANT", "value": 0.0} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse config");
    // validated against the iso spec: the nrg overlay is dropped (no energy), mom survives.
    let built = build_user_source(&cfg, &ISO_NEWTONIAN_SPEC).expect("wrap source");
    assert_eq!(
        built.len(),
        1,
        "iso force must yield ONLY the mom overlay (no nrg)"
    );
    assert_eq!(built[0].0, "mom");

    let mut sim = Sim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .finish()
        .unwrap();
    sim.seed_cells(|_| {
        PrimG::<f64, 2, IsoModel>::isothermal(Density(1.0), Tensor::new([0.0, 0.0]))
    });

    let sub = sim
        .substrate()
        .with_runtime_source(built, cfg.params.clone());
    assert!(
        mom_x_total(&sim).abs() < 1e-12,
        "x-momentum should start at zero"
    );

    let t_final = 0.05;
    evolve(&mut sim, &sub, t_final).expect("evolve under runtime source");

    // d(mom_x)/dt = rho*g = 0.5 per cell; uniform box stays rho=1 -> total = 0.5*t*256.
    let got = mom_x_total(&sim);
    let expected = 0.5 * t_final * (16.0 * 16.0);
    assert!(got > 0.0, "iso gas did not accelerate: mom_x = {got}");
    assert!(
        (got - expected).abs() / expected < 0.02,
        "iso runtime force wrong magnitude: mom_x = {got}, expected ~{expected}",
    );
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        assert!(
            (rho - 1.0).abs() < 1e-6,
            "density drifted at {c:?}: rho = {rho}"
        );
    }
}
