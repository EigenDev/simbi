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

#[test]
fn all_driven_faces_fill_the_corner_ghosts() {
    // every face driven with the same constant prescription. the standard pullback skips a
    // ghost region whose contacting faces are ALL driven (they are Skip to it), so the
    // edge/corner ghost blocks are written only if each driven slab spans the full allocation
    // on its transverse axes; an interior-clamped band leaves them at their allocation zeros,
    // and a rho = 0 corner ghost is read as gas by any multi-dimensional stencil (a viscous
    // 3x3, a CT corner EMF). after ghost_fill EVERY ghost cell must hold the prescription.
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Driven(1)],
        [BoundaryType::Driven(2), BoundaryType::Driven(3)],
    ]);
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap();
    sim.seed_cells(|_| Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0]), pre: 1.0 });

    let json = r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":3.0} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let mut sub = sim.substrate();
    for face in 0..4u16 {
        let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary");
        let (s, id) = sub.with_driven_boundary(built, cfg.params.clone());
        sub = s;
        assert_eq!(id, face, "registration order matches Driven(id)");
    }

    sub.ghost_fill(&sim);

    let (mut ghosts, mut corners) = (0usize, 0usize);
    for c in sim.geom.allocated.iter() {
        if sim.geom.interior.contains(c) {
            continue;
        }
        let x = sim.geom.cell_coord(c);
        let x_out = x[0] < 0.0 || x[0] > 1.0;
        let y_out = x[1] < 0.0 || x[1] > 1.0;
        if x_out && y_out {
            corners += 1;
        }
        ghosts += 1;
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *sim.fields.prim.pre_field().unwrap().view().at(c);
        assert!((rho - 2.0).abs() < 1e-12, "ghost rho at {x:?} = {rho}, want 2 (corner left unwritten?)");
        assert!((p - 3.0).abs() < 1e-12, "ghost pre at {x:?} = {p}, want 3");
    }
    assert!(ghosts > 0 && corners > 0, "expected face and corner ghost cells to be checked");
}

#[test]
fn iso_mhd_driven_inflow_prescribes_the_ghost_state() {
    // the isothermal-MHD driven prescription is [rho, v1, v2, v3, B1, B2, B3] (no pressure
    // slot; the eos closure p = cs^2 rho covers the ghosts). a purely out-of-plane B_z is
    // div-free by construction, so the cell-B prescription needs no CT face sub-problem.
    // x_lo driven, everything else outflow: after ghost_fill the ENTIRE x_lo ghost slab —
    // corners included — holds the prescribed inflow.
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::isothermal_mhd::IsothermalMhd;
    use symbi_hydro::mhd_state::MhdPrimG;
    use symbi_hydro::state::PrimG;
    use symbi_hydro::ISO_MHD_SPEC;

    type SimI = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::Outflow],
    ]);
    let sim = SimI::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap();
    sim.seed_cells(|_| MhdPrimG {
        hydro: PrimG { rho: 1.0, vel: Tensor::new([0.0, 0.0, 0.0]), pre: Default::default() },
        mag: Tensor::new([0.0, 0.0, 0.0]),
    });

    // [rho, v1, v2, v3, B1, B2, B3] = [2, 1, 0, 0, 0, 0, 0.5].
    let json = r#"{
        "kind": "dirichlet", "dim": 3, "outputs": [0, 1, 2, 2, 2, 2, 3], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":0.5} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_boundary_dag(&cfg, &ISO_MHD_SPEC).expect("iso-mhd driven boundary");
    let (sub, id) = sim.substrate().with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0);

    sub.ghost_fill(&sim);

    let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
    let mut checked = 0usize;
    for c in sim.geom.allocated.iter() {
        let x = sim.geom.cell_coord(c);
        if x[0] >= 0.0 {
            continue; // interior or non-x_lo ghost
        }
        checked += 1;
        let rho = *sim.fields.prim.rho.view().at(c);
        let v0 = *sim.fields.prim.vel[0].view().at(c);
        let bz = *mhd.bcell[2].view().at(c);
        assert!((rho - 2.0).abs() < 1e-12, "x_lo ghost rho at {x:?} = {rho}, want 2");
        assert!((v0 - 1.0).abs() < 1e-12, "x_lo ghost vel_0 at {x:?} = {v0}, want 1");
        assert!((bz - 0.5).abs() < 1e-12, "x_lo ghost B_z at {x:?} = {bz}, want 0.5");
    }
    assert!(checked > 0, "no x_lo ghost cells found to check");
}
