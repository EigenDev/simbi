// =============================================================================
// driven_boundary_cpu.rs
//
// docs/design/33 task 24 (CPU): a DRIVEN boundary prescribes its ghost prim state from a user DAG.
// the standard ghost-fill SKIPS the driven face (Driven -> BcType::Skip); the driven pass then
// evaluates the boundary DAG over that face's ghost band and ASSIGNS prim.{rho,vel,pre}.
//
// a steady inflow on x_lo (rho=2, v=[1,0], p=3) prescribed by constants; after ghost_fill the x_lo
// ghost band must hold exactly that state, independent of the interior — the (Coord, Assign)
// instance of the unified DAG operator, end to end through the live kernel-set.
// =============================================================================

use symbi::prelude::*;
use symbi::sim::evolve::KernelSet;
use symbi_hydro::expr_bridge::build_boundary_dag;
use symbi_hydro::{SourceConfig, NEWTONIAN_SPEC};

type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;

#[test]
fn driven_inflow_prescribes_the_ghost_state() {
    // x_lo driven, x_hi outflow, y periodic.
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]);
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap();
    // interior at rest, density 1 — DELIBERATELY different from the inflow, so the test proves the
    // ghost state comes from the DAG, not pulled from the interior.
    sim.seed_cells(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 });

    // prescribe [rho=2, vel_0=1, vel_1=0, pre=3] (constants).
    let json = r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":3.0} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary");
    let (sub, id) = sim.substrate().with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0, "first registration is id 0 (matches Driven(0))");

    sub.ghost_fill(&sim);

    // the x_lo ghost band: physical x_0 < 0 (the domain starts at 0), transverse y in the interior
    // (0, 1). those cells must hold the prescribed inflow state.
    let mut checked = 0usize;
    for c in sim.geom.allocated.iter() {
        let x = sim.geom.cell_coord(c);
        let is_xlo_ghost = x[0] < 0.0;
        let y_interior = x[1] > 0.0 && x[1] < 1.0;
        if is_xlo_ghost && y_interior {
            let rho = *sim.fields.prim.rho.view().at(c);
            let v0 = *sim.fields.prim.vel[0].view().at(c);
            let v1 = *sim.fields.prim.vel[1].view().at(c);
            let p = *sim.fields.prim.pre_field().unwrap().view().at(c);
            assert!((rho - 2.0).abs() < 1e-12, "x_lo ghost rho at {x:?} = {rho}, want 2");
            assert!((v0 - 1.0).abs() < 1e-12, "x_lo ghost vel_0 at {x:?} = {v0}, want 1");
            assert!(v1.abs() < 1e-12, "x_lo ghost vel_1 at {x:?} = {v1}, want 0");
            assert!((p - 3.0).abs() < 1e-12, "x_lo ghost pre at {x:?} = {p}, want 3");
            checked += 1;
        }
    }
    assert!(checked > 0, "no x_lo ghost-band cells found to check");
}
