// =============================================================================
// refinement_driven_boundary.rs
//
// driven (dirichlet) boundaries on a refined hierarchy. the mechanics under test:
// - a fine level flush against a driven physical face INHERITS `Driven(id)` there and
//   evaluates the same coordinate graph at its own finer ghost coordinates (the graphs are
//   registered on every level's kernel set);
// - an interior fine level carries CoarseFine faces throughout and leaves the graphs untouched;
// - the fill ordering (prolong_cf, then ghost_fill whose tail is the driven pass) gives the
//   driven prescription deterministic ownership of the driven/coarse-fine corner overlap.
// three oracles: exact uniform preservation with all faces driven at the uniform state, the
// prescription landing verbatim in the FINE level's ghost band on a flush face, and the
// covered-coarse == restriction sync surviving a driven inflow.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::transfer::restrict_cell_field;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi::symbi_grid::Field;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_boundary_dag;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{NEWTONIAN_SPEC, SourceConfig};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const STEPS: u64 = 6;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a constant prescription [rho, v1, v2, p]; registered on every level so a fine level flush
// against a driven face resolves its inherited Driven(0).
fn const_dag(rho: f64, vx: f64, pre: f64) -> String {
    format!(
        r#"{{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
        "nodes": [ {{"op":"CONSTANT","value":{rho}}}, {{"op":"CONSTANT","value":{vx}}},
                   {{"op":"CONSTANT","value":0.0}}, {{"op":"CONSTANT","value":{pre}}} ]
    }}"#
    )
}

fn with_driven(sim: &Sim, json: &str) -> Kset {
    let cfg = SourceConfig::from_json(json).expect("parse boundary config");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("lower boundary dag");
    let (k, id) =
        Kset::new(GAMMA, CFL, &sim.geom.allocated).with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0);
    k
}

fn build_hier(
    boundaries: Boundaries<2>,
    region: RefinementRegion<2>,
    json: String,
    ic: impl Fn([f64; 2]) -> Prim<f64, 2> + Copy,
) -> Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset> {
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 2])
        .origin([0.0; 2])
        .spacing([dx; 2])
        .boundaries(boundaries)
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(ic)
        .build();
    let ck = with_driven(&coarse, &json);
    Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, |s| {
        with_driven(s, &json)
    })
    .unwrap()
}

// seed the fine level's conserved state directly (harness convention:
// the fine state comes from this seeding, leaving with_refinement's IC prolongation out of it).
fn seed_fine(
    hier: &Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>,
    rho: f64,
    pre: f64,
) {
    let fine = &hier.levels[1].state;
    let nrg = fine.fields.cons.nrg_field().unwrap();
    for c in fine.geom.interior.iter() {
        fine.fields.cons.den.view_mut().set(c, rho);
        nrg.view_mut().set(c, pre / (GAMMA - 1.0));
        for dd in 0..2 {
            fine.fields.cons.mom[dd].view_mut().set(c, 0.0);
        }
    }
}

#[test]
fn uniform_gas_stays_uniform_with_all_faces_driven() {
    // every face driven at exactly the uniform interior state: any seam/corner/ordering
    // corruption between the driven fill and the coarse-fine prolongation breaks uniformity.
    let (rho0, pre0) = (2.0, 1.0);
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Driven(0)],
        [BoundaryType::Driven(0), BoundaryType::Driven(0)],
    ]);
    let region = RefinementRegion {
        x_lo: [0.25; 2],
        x_hi: [0.75; 2],
    };
    let mut hier = build_hier(boundaries, region, const_dag(rho0, 0.0, pre0), |_| Prim {
        rho: rho0,
        vel: Tensor::new([0.0; 2]),
        pre: pre0,
    });
    seed_fine(&hier, rho0, pre0);
    hier.evolve_steps(STEPS).unwrap();
    for (lvl, level) in hier.levels.iter().enumerate() {
        let sim = &level.state;
        for c in sim.geom.interior.iter() {
            let den = *sim.fields.cons.den.view().at(c);
            assert!(
                (den - rho0).abs() < 1e-12,
                "level {lvl}: den drifted to {den} at {c:?} under all-driven faces"
            );
        }
    }
}

#[test]
fn fine_level_flush_against_a_driven_face_holds_the_prescription() {
    // the refined region touches x_lo, so the fine level INHERITS Driven(0) there; after
    // evolution its x_lo ghost slab must hold the DAG values evaluated at the FINE ghost
    // coordinates — proving the inheritance, the per-level registration, and the fill.
    let (rho_in, vx_in, pre_in) = (2.0, 0.5, 3.0);
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::Outflow],
    ]);
    let region = RefinementRegion {
        x_lo: [0.0, 0.25],
        x_hi: [0.5, 0.75],
    };
    let mut hier = build_hier(boundaries, region, const_dag(rho_in, vx_in, pre_in), |_| {
        Prim {
            rho: 1.0,
            vel: Tensor::new([0.0; 2]),
            pre: 1.0,
        }
    });
    seed_fine(&hier, 1.0, 1.0);
    hier.evolve_steps(STEPS).unwrap();

    let fine = &hier.levels[1].state;
    assert_eq!(
        fine.boundaries.lo(0),
        BoundaryType::Driven(0),
        "flush fine face did not inherit the driven boundary"
    );
    let mut checked = 0usize;
    for c in fine.geom.allocated.iter() {
        let x = fine.geom.cell_coord(c);
        if x[0] >= 0.0 {
            continue;
        }
        checked += 1;
        let rho = *fine.fields.prim.rho.view().at(c);
        let v0 = *fine.fields.prim.vel[0].view().at(c);
        assert!(
            (rho - rho_in).abs() < 1e-12,
            "fine x_lo ghost rho at {x:?} = {rho}, want {rho_in}"
        );
        assert!(
            (v0 - vx_in).abs() < 1e-12,
            "fine x_lo ghost vel at {x:?} = {v0}, want {vx_in}"
        );
    }
    assert!(checked > 0, "no fine x_lo ghost cells found");
    // the inflow genuinely entered the fine region (non-vacuous).
    let mut momx = 0.0;
    for c in fine.geom.interior.iter() {
        momx += (*fine.fields.cons.mom[0].view().at(c)).abs();
    }
    assert!(
        momx > 1e-3,
        "driven inflow never entered the fine level ({momx:e})"
    );
}

#[test]
fn covered_restriction_sync_survives_a_driven_inflow() {
    // an interior fine box with a driven inflow washing over it: the covered coarse
    // conserved cells must still BE the restriction of the fine children after every step.
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::Outflow],
    ]);
    let region = RefinementRegion {
        x_lo: [0.25; 2],
        x_hi: [0.75; 2],
    };
    let mut hier = build_hier(boundaries, region, const_dag(2.0, 0.5, 3.0), |_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0; 2]),
        pre: 1.0,
    });
    seed_fine(&hier, 1.0, 1.0);
    hier.evolve_steps(2 * STEPS).unwrap();

    let (lo, hi) = hier.levels.split_at(1);
    let (c, f) = (&lo[0].state, &hi[0].state);
    let cov = lo[0].coverage.as_ref().unwrap();
    for (nm, fine, coarse) in [
        ("den", &f.fields.cons.den, &c.fields.cons.den),
        ("m1", &f.fields.cons.mom[0], &c.fields.cons.mom[0]),
    ] {
        let scratch = Field::<f64, 2, HostMemory>::zeros(&c.geom.allocated).unwrap();
        restrict_cell_field(fine, &scratch, cov);
        let mut worst = 0.0_f64;
        for cc in cov.iter() {
            let live = *coarse.view().at(cc);
            let want = *scratch.view().at(cc);
            worst = worst.max((live - want).abs() / want.abs().max(1e-300));
        }
        assert!(
            worst < 1e-13,
            "covered '{nm}' out of sync under driven inflow: {worst:e}"
        );
    }
}
