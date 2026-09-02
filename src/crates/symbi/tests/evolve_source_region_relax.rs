// =============================================================================
// evolve_source_region_relax.rs
//
// region + relax axes, end-to-end through the live evolve loop (the runtime user-source path:
// json -> build_user_source -> SourceEvaluator -> source_apply). proves the region and relax
// axes change the physics of the solution:
//
//   region: a force masked to chi(x) = [x_0 < 0.5] accelerates the left half of the box alone; the
//           right half stays at rest. the mask is a property of the source, applied per cell.
//   relax:  a velocity sponge (S_mom = kappa*rho*(0 - v), kappa > 0) damps a uniform flow toward
//           rest — momentum decays monotonically and stays on one side of the target (the
//           kappa >= 0 stability axiom).
// =============================================================================

use symbi::prelude::*;
use symbi_source_compile::expr_bridge::build_user_source;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_source_compile::SourceConfig;

type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;

fn build_box() -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .finish()
        .unwrap();
    sim.seed_cells(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0, 0.0]),
        pre: 1.0,
    });
    sim
}

#[test]
fn region_masked_force_acts_only_in_the_left_half() {
    // force a = [0.5, 0], region chi = (x_0 < 0.5) ? 1 : 0. masked S_mom_0 = chi * rho * 0.5.
    // nodes: 0=param p0 (=accel), 1=const 0, 2=VAR_X1, 3=const 0.5, 4=LT(2,3), 5=const 1,
    //        6=IF_THEN_ELSE(cond=4, then=5, else=1). outputs=[0,1], region=6.
    let json = r#"{
        "kind": "force", "dim": 2, "outputs": [0, 1], "region": 6, "params": [0.5],
        "nodes": [
            {"op": "PARAMETER", "param_idx": 0},
            {"op": "CONSTANT", "value": 0.0},
            {"op": "VARIABLE_X1"},
            {"op": "CONSTANT", "value": 0.5},
            {"op": "LT", "left": 2, "right": 3},
            {"op": "CONSTANT", "value": 1.0},
            {"op": "IF_THEN_ELSE", "condition": 4, "true_case": 5, "false_case": 1}
        ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("force+region");

    let mut sim = build_box();
    let sub = sim
        .substrate()
        .with_runtime_source(built, cfg.params.clone());
    evolve(&mut sim, &sub, 0.05).expect("evolve");

    // the source acts in the left half alone. the hydro then leaks a little momentum across the
    // x=0.5 interface (a real pressure/advection response), so the test asserts the asymmetry
    // between the halves and allows the right half a small residual: every left cell is
    // accelerated, and the total right-half momentum stays a small fraction of the left's. an
    // unmasked source would make the two halves identical.
    let mut left_total = 0.0;
    let mut right_total = 0.0;
    for c in sim.geom.interior.iter() {
        let x = sim.geom.cell_coord(c);
        let mom = *sim.fields.cons.mom[0].view().at(c);
        if x[0] < 0.5 {
            assert!(
                mom > 1e-6,
                "left cell at x={x:?} should be accelerated, mom = {mom}"
            );
            left_total += mom;
        } else {
            right_total += mom;
        }
    }
    assert!(left_total > 0.0, "left half must be populated");
    assert!(
        right_total < 0.15 * left_total,
        "region must concentrate the force in the left half: left = {left_total}, right = {right_total}",
    );
}

#[test]
fn relax_sponge_damps_uniform_flow() {
    // relax velocity toward v_ref = 0 with rate kappa = p0 = 2. outputs = [kappa, v_ref_0, v_ref_1].
    // on a uniform periodic flow: d(mom)/dt = -kappa*mom -> mom(t) = mom0 * exp(-kappa*t).
    let json = r#"{
        "kind": "relax", "dim": 2, "outputs": [0, 1, 1], "params": [2.0],
        "nodes": [ {"op": "PARAMETER", "param_idx": 0}, {"op": "CONSTANT", "value": 0.0} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("relax");

    let mut sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([16, 16])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(BoundaryType::Periodic)
        .finish()
        .unwrap();
    // uniform rightward flow v_x = 1 (the perturbation the sponge absorbs).
    sim.seed_cells(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([1.0, 0.0]),
        pre: 1.0,
    });

    let mom0: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.mom[0].view().at(c))
        .sum();
    let t_final = 0.05;
    let sub = sim
        .substrate()
        .with_runtime_source(built, cfg.params.clone());
    evolve(&mut sim, &sub, t_final).expect("evolve");
    let mom1: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.mom[0].view().at(c))
        .sum();

    // damped, monotone, no overshoot: 0 < mom1 < mom0, near exp(-kappa*t) = exp(-0.1) ~ 0.905.
    assert!(
        mom1 > 0.0,
        "relaxation must not overshoot past rest: mom1 = {mom1}"
    );
    assert!(
        mom1 < mom0,
        "relaxation must damp the flow: mom0 = {mom0}, mom1 = {mom1}"
    );
    let ratio = mom1 / mom0;
    assert!(
        (0.80..0.95).contains(&ratio),
        "decay should track exp(-kappa*t) ~ 0.905: ratio = {ratio}",
    );
    // density stays uniform (a uniform decelerating flow develops no gradients).
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        assert!(
            (rho - 1.0).abs() < 1e-6,
            "density drifted at {c:?}: rho = {rho}"
        );
    }
}
