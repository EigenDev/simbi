// =============================================================================
// refinement_driven_mhd.rs
//
// driven (dirichlet) boundaries on a refined MHD hierarchy. the coarse-fine machinery
// (EMF registers, staggered restriction/prolongation) operates on CF interfaces only,
// never physical faces, so a driven face rides the same ghost path as the uni-grid MHD
// driven fill: prims + cell B prescribed by the coordinate DAG, staggered face B left to
// the CT ghost fill. two oracles: exact uniform preservation (uniform B_x, all faces
// driven at the uniform state, both levels, div(B) at machine zero), and the prescription
// landing verbatim — including the cell B — in the FINE level's ghost slab on a flush
// driven face while div(B) stays clean under the resulting inflow.
// =============================================================================

use std::sync::atomic::Ordering;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_boundary_dag;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::{NEWTONIAN_MHD_SPEC, SourceConfig};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 16;
const B0: f64 = 0.2;
const STEPS: u64 = 4;

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3>;

// [rho, v1, v2, v3, pre, B1, B2, B3]: the full newtonian-mhd prescription. B is the
// uniform in-plane B_x of the interior so the prescription is div-consistent.
fn mhd_dag(rho: f64, vx: f64, pre: f64) -> String {
    format!(
        r#"{{
        "kind": "dirichlet", "dim": 3, "outputs": [0, 1, 2, 2, 3, 4, 2, 2], "params": [],
        "nodes": [ {{"op":"CONSTANT","value":{rho}}}, {{"op":"CONSTANT","value":{vx}}},
                   {{"op":"CONSTANT","value":0.0}}, {{"op":"CONSTANT","value":{pre}}},
                   {{"op":"CONSTANT","value":{B0}}} ]
    }}"#
    )
}

fn with_driven(sim: &Sim, json: &str) -> Kset {
    let cfg = SourceConfig::from_json(json).expect("parse boundary config");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_MHD_SPEC).expect("lower mhd boundary dag");
    let (k, id) = Kset::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
        .with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0);
    k
}

// uniform gas + uniform staggered/cell B_x over a level's interior (div-free, coarse-fine
// consistent); the same absolute state on both levels.
fn fill(sim: &Sim, rho: f64, pre: f64) {
    let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
    for c in &sim.geom.interior.extend(0, 0, 1) {
        mhd.bface[0].view_mut().set(c, B0);
    }
    for aa in 1..3 {
        for c in &sim.geom.interior.extend(aa, 0, 1) {
            mhd.bface[aa].view_mut().set(c, 0.0);
        }
    }
    mhd.bface_initialized.store(true, Ordering::Relaxed);
    let nrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        mhd.bcell[0].view_mut().set(c, B0);
        mhd.bcell[1].view_mut().set(c, 0.0);
        mhd.bcell[2].view_mut().set(c, 0.0);
        sim.fields.cons.den.view_mut().set(c, rho);
        for dd in 0..3 {
            sim.fields.cons.mom[dd].view_mut().set(c, 0.0);
        }
        nrg.view_mut().set(c, pre / (GAMMA - 1.0) + 0.5 * B0 * B0);
    }
}

fn build_hier(
    boundaries: Boundaries<3>,
    region: RefinementRegion<3>,
    json: String,
    rho: f64,
    pre: f64,
) -> Hierarchy<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset> {
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(boundaries)
        .cfl(CFL)
        .finish()
        .unwrap();
    fill(&coarse, rho, pre);
    let ck = with_driven(&coarse, &json);
    let hier = Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, |s| {
        with_driven(s, &json)
    })
    .unwrap();
    fill(&hier.levels[1].state, rho, pre);
    hier
}

fn div_b_max(sim: &Sim) -> f64 {
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let dx = sim.geom.dx[0];
    let mut worst = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let mut div = 0.0;
        for d in 0..3 {
            let mut chi = c;
            chi[d] += 1;
            div += (*mhd.bface[d].view().at(chi) - *mhd.bface[d].view().at(c)) / dx;
        }
        worst = worst.max(div.abs());
    }
    worst
}

#[test]
fn uniform_mhd_stays_uniform_with_all_faces_driven() {
    let (rho0, pre0) = (1.0, 1.0);
    let boundaries = Boundaries::<3>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Driven(0)],
        [BoundaryType::Driven(0), BoundaryType::Driven(0)],
        [BoundaryType::Driven(0), BoundaryType::Driven(0)],
    ]);
    let region = RefinementRegion {
        x_lo: [0.25; 3],
        x_hi: [0.75; 3],
    };
    let mut hier = build_hier(boundaries, region, mhd_dag(rho0, 0.0, pre0), rho0, pre0);
    hier.evolve_steps(STEPS).unwrap();
    for (ll, level) in hier.levels.iter().enumerate() {
        let sim = &level.state;
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for c in sim.geom.interior.iter() {
            let den = *sim.fields.cons.den.view().at(c);
            let bx = *mhd.bcell[0].view().at(c);
            assert!(
                (den - rho0).abs() < 1e-12,
                "level {ll}: den drifted to {den} at {c:?}"
            );
            assert!(
                (bx - B0).abs() < 1e-12,
                "level {ll}: bcell_x drifted to {bx} at {c:?}"
            );
        }
        let db = div_b_max(sim);
        assert!(
            db < 1e-12,
            "level {ll}: div(B) = {db:e} under all-driven faces"
        );
    }
}

#[test]
fn fine_mhd_level_flush_against_a_driven_face_holds_the_prescription() {
    let (rho_in, vx_in, pre_in) = (2.0, 0.4, 3.0);
    let boundaries = Boundaries::<3>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::Outflow],
    ]);
    let region = RefinementRegion {
        x_lo: [0.0, 0.25, 0.25],
        x_hi: [0.5, 0.75, 0.75],
    };
    let mut hier = build_hier(boundaries, region, mhd_dag(rho_in, vx_in, pre_in), 1.0, 1.0);
    hier.evolve_steps(STEPS).unwrap();

    let fine = &hier.levels[1].state;
    assert_eq!(
        fine.boundaries.lo(0),
        BoundaryType::Driven(0),
        "fine face did not inherit driven"
    );
    let mhd = fine.fields.mhd.as_ref().unwrap();
    let mut checked = 0usize;
    for c in fine.geom.allocated.iter() {
        let x = fine.geom.cell_coord(c);
        if x[0] >= 0.0 {
            continue;
        }
        checked += 1;
        let rho = *fine.fields.prim.rho.view().at(c);
        let bx = *mhd.bcell[0].view().at(c);
        assert!(
            (rho - rho_in).abs() < 1e-12,
            "fine ghost rho at {x:?} = {rho}, want {rho_in}"
        );
        assert!(
            (bx - B0).abs() < 1e-12,
            "fine ghost bcell_x at {x:?} = {bx}, want {B0}"
        );
    }
    assert!(checked > 0, "no fine x_lo ghost cells found");

    // the inflow entered the fine level, and neither level grew a monopole.
    let mut momx = 0.0;
    for c in fine.geom.interior.iter() {
        momx += (*fine.fields.cons.mom[0].view().at(c)).abs();
    }
    assert!(
        momx > 1e-3,
        "driven inflow never entered the fine level ({momx:e})"
    );
    for (ll, level) in hier.levels.iter().enumerate() {
        let db = div_b_max(&level.state);
        assert!(
            db < 1e-12,
            "level {ll}: div(B) = {db:e} under a driven mhd inflow"
        );
    }
}
