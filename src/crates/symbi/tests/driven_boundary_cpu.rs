// =============================================================================
// driven_boundary_cpu.rs
//
// a driven boundary prescribes its ghost prim state from a user DAG (CPU).
// the standard ghost-fill skips the driven face (Driven -> BcType::Skip); the driven pass then
// evaluates the boundary DAG over that face's ghost band and assigns prim.{rho,vel,pre}.
//
// a steady inflow on x_lo (rho=2, v=[1,0], p=3) prescribed by constants; after ghost_fill the x_lo
// ghost band must hold exactly that state, independent of the interior — the (Coord, Assign)
// instance of the unified DAG operator, end to end through the live kernel-set.
// =============================================================================

use symbi::prelude::*;
use symbi::sim::evolve::KernelSet;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_boundary_dag;

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
    // interior at rest, density 1 — deliberately different from the inflow, so the test proves the
    // ghost state comes from the DAG; had it been copied from the interior it would carry density 1.
    sim.seed_cells(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)));

    // prescribe [rho=2, vel_0=1, vel_1=0, pre=3] (constants).
    let json = r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":3.0} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary");
    let (sub, id) = sim
        .substrate()
        .with_driven_boundary(built, cfg.params.clone());
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
            assert!(
                (rho - 2.0).abs() < 1e-12,
                "x_lo ghost rho at {x:?} = {rho}, want 2"
            );
            assert!(
                (v0 - 1.0).abs() < 1e-12,
                "x_lo ghost vel_0 at {x:?} = {v0}, want 1"
            );
            assert!(v1.abs() < 1e-12, "x_lo ghost vel_1 at {x:?} = {v1}, want 0");
            assert!(
                (p - 3.0).abs() < 1e-12,
                "x_lo ghost pre at {x:?} = {p}, want 3"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no x_lo ghost-band cells found to check");
}

#[test]
fn all_driven_faces_fill_the_corner_ghosts() {
    // every face driven with the same constant prescription. the standard pullback skips a
    // ghost region whose contacting faces are all driven (they are Skip to it), so the
    // edge/corner ghost blocks are written only if each driven slab spans the full allocation
    // on its transverse axes; an interior-clamped band leaves them at their allocation zeros,
    // and a rho = 0 corner ghost is read as gas by any multi-dimensional stencil (a viscous
    // 3x3, a CT corner EMF). after ghost_fill every ghost cell must hold the prescription.
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
    sim.seed_cells(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)));

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
        assert!(
            (rho - 2.0).abs() < 1e-12,
            "ghost rho at {x:?} = {rho}, want 2 (corner left unwritten?)"
        );
        assert!((p - 3.0).abs() < 1e-12, "ghost pre at {x:?} = {p}, want 3");
    }
    assert!(
        ghosts > 0 && corners > 0,
        "expected face and corner ghost cells to be checked"
    );
}

#[test]
fn iso_mhd_driven_inflow_prescribes_the_ghost_state() {
    // the isothermal-MHD driven prescription is [rho, v1, v2, v3, B1, B2, B3] (no pressure
    // slot; the eos closure p = cs^2 rho covers the ghosts). a purely out-of-plane B_z is
    // div-free by construction, so the cell-B prescription needs no CT face sub-problem.
    // x_lo driven, everything else outflow: after ghost_fill the entire x_lo ghost slab —
    // corners included — holds the prescribed inflow.
    use symbi_hydro::ISO_MHD_SPEC;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::isothermal_mhd::IsothermalMhd;
    use symbi_hydro::mhd_state::MhdPrimG;
    use symbi_hydro::state::PrimG;

    type SimI =
        SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
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
    sim.seed_cells(|_| {
        MhdPrimG::new(
            PrimG::isothermal(Density(1.0), Tensor::new([0.0, 0.0, 0.0])),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    });

    // [rho, v1, v2, v3, B1, B2, B3] = [2, 1, 0, 0, 0, 0, 0.5].
    let json = r#"{
        "kind": "dirichlet", "dim": 3, "outputs": [0, 1, 2, 2, 2, 2, 3], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":0.5} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_boundary_dag(&cfg, &ISO_MHD_SPEC).expect("iso-mhd driven boundary");
    let (sub, id) = sim
        .substrate()
        .with_driven_boundary(built, cfg.params.clone());
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
        assert!(
            (rho - 2.0).abs() < 1e-12,
            "x_lo ghost rho at {x:?} = {rho}, want 2"
        );
        assert!(
            (v0 - 1.0).abs() < 1e-12,
            "x_lo ghost vel_0 at {x:?} = {v0}, want 1"
        );
        assert!(
            (bz - 0.5).abs() < 1e-12,
            "x_lo ghost B_z at {x:?} = {bz}, want 0.5"
        );
    }
    assert!(checked > 0, "no x_lo ghost cells found to check");
}

// a driven face prescribes the dye of the fluid it injects, as a trailing output after the prim
// state. the interior is undyed and the ghost band is poisoned beforehand, so the prescribed
// concentration can come from neither a copy of the interior nor a leftover value.
#[test]
fn driven_inflow_prescribes_the_injected_dye() {
    const CHI_IN: f64 = 0.6;
    const POISON: f64 = -7.0;
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]);
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap()
        .with_passive_scalar()
        .expect("chi alloc");
    sim.seed_cells(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)));
    let chi_f = sim.fields.prim.chi_field().expect("prim chi");
    // interior undyed; everything outside it poisoned.
    for c in sim.geom.allocated.iter() {
        let v = if sim.geom.interior.contains(c) {
            0.0
        } else {
            POISON
        };
        chi_f.view_mut().set(c, v);
    }

    // [rho=2, vel_0=1, vel_1=0, pre=3, chi=CHI_IN] — the prim state plus the trailing dye.
    let json = r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3, 4], "params": [],
        "nodes": [ {"op":"CONSTANT","value":2.0}, {"op":"CONSTANT","value":1.0},
                   {"op":"CONSTANT","value":0.0}, {"op":"CONSTANT","value":3.0},
                   {"op":"CONSTANT","value":0.6} ]
    }"#;
    let cfg = SourceConfig::from_json(json).expect("parse");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary with dye");
    assert!(
        built.iter().any(|(slot, _)| slot == "chi"),
        "the trailing output must lower to a chi prescription, got slots {:?}",
        built.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>()
    );
    let (sub, _) = sim
        .substrate()
        .with_driven_boundary(built, cfg.params.clone());

    sub.ghost_fill(&sim);

    let mut checked = 0usize;
    for c in sim.geom.allocated.iter() {
        let x = sim.geom.cell_coord(c);
        if x[0] >= 0.0 || x[1] <= 0.0 || x[1] >= 1.0 {
            continue;
        }
        let got = *chi_f.view().at(c);
        assert!(
            (got - POISON).abs() > 1e-12,
            "x_lo dye ghost at {c:?} was never written (still poisoned)"
        );
        assert!(
            (got - CHI_IN).abs() < 1e-12,
            "x_lo dye ghost at {c:?}: {got} != prescribed {CHI_IN}"
        );
        checked += 1;
    }
    assert!(checked > 0, "no x_lo ghost-band cells found to check");
    // the premise: the prescribed dye differs from the undyed interior, so a copy of the interior
    // would fail the assertion above rather than sneak through.
    assert!(
        CHI_IN.abs() > 1e-12,
        "prescribed dye equals the interior; the gate cannot separate prescription from copy"
    );
}
