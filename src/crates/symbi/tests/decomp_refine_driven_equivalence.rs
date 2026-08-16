// =============================================================================
// decomp_refine_driven_equivalence.rs
//
// driven boundaries x refinement x decomposition: per-tile 2-level hierarchies whose root is
// split into tiles, with a driven inflow on x_lo and the refined patch flush against that face
// — so the owning tile's fine level inherits `Driven(id)` and evaluates the coordinate DAG at
// its own fine ghost coordinates while the root halo exchange couples the tiles. must
// reproduce the monolithic refined run (itself gated by refinement_driven_boundary.rs) to
// round-off through the production `evolve_hierarchy_decomposed` loop.
//
// the prescription is position-dependent (rho = 2 + 0.2*y, the cell's global y): with the cut
// along the driven face, both edge tiles own a piece of it, so a tile evaluating the DAG at a
// local coordinate diverges; with the cut perpendicular, the injected gas crosses the cut via
// the root exchange.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, unflatten};
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
};
use symbi::sim::state::*;
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
const N: usize = 32; // root cells per axis
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

// [rho, v1, v2, p] = [2 + 0.2*y, 0.5, 0, 3]: a global-coordinate-decisive x_lo inflow.
fn boundary_json() -> String {
    r#"{
        "kind": "dirichlet", "dim": 2, "outputs": [4, 5, 6, 7], "params": [],
        "nodes": [ {"op": "VARIABLE_X2"}, {"op": "CONSTANT", "value": 0.2},
                   {"op": "MULTIPLY", "left": 0, "right": 1}, {"op": "CONSTANT", "value": 2.0},
                   {"op": "ADD", "left": 3, "right": 2},
                   {"op": "CONSTANT", "value": 0.5}, {"op": "CONSTANT", "value": 0.0},
                   {"op": "CONSTANT", "value": 3.0} ]
    }"#
    .to_string()
}

fn kset(sim: &Sim) -> Kern {
    let cfg = SourceConfig::from_json(&boundary_json()).expect("parse boundary config");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("lower boundary dag");
    let (k, id) =
        Kern::new(GAMMA, CFL, &sim.geom.allocated).with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0);
    k
}

// the refined patch flush against the driven x_lo face, inside the bottom-left quadrant so a
// single tile owns it under every tested topology (cuts at x = 0.5 and/or y = 0.5).
fn patch() -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [0.0, 0.125],
        x_hi: [0.375, 0.375],
    }
}

fn build_root(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build()
}

fn phys_boundaries() -> [[BoundaryType; 2]; 2] {
    [
        [BoundaryType::Driven(0), BoundaryType::Outflow],
        [BoundaryType::Outflow, BoundaryType::Outflow],
    ]
}

fn build_mono() -> Hier {
    let root = build_root([N, N], [0.0, 0.0], Boundaries(phys_boundaries()));
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, &[patch()], ProlongOrder::Ppm, kset)
        .expect("mono hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

fn build_tiles(counts: [usize; 2]) -> Vec<Hier> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let phys = phys_boundaries();
    let total: usize = counts.iter().product();
    let mut tiles = Vec::with_capacity(total);
    for flat in 0..total {
        let tc = unflatten(flat, counts);
        let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let lo = if tc[a] == 0 {
                phys[a][0]
            } else {
                BoundaryType::CoarseFine
            };
            let hi = if tc[a] == counts[a] - 1 {
                phys[a][1]
            } else {
                BoundaryType::CoarseFine
            };
            [lo, hi]
        }));
        let root = build_root(m, origin, bnd);
        let owns_patch = (0..2).all(|a| {
            origin[a] <= patch().x_lo[a] && origin[a] + m[a] as f64 * DX >= patch().x_hi[a]
        });
        let mut h = if owns_patch {
            let k = kset(&root);
            let h = Hier::with_refinement(root, k, &[patch()], ProlongOrder::Ppm, kset)
                .expect("tile hierarchy");
            h.seed_fine_from_coarse().expect("seed fine");
            h
        } else {
            let k = kset(&root);
            Hier::single(root, k)
        };
        h.prime();
        tiles.push(h);
    }
    tiles
}

fn run_decomposed(tiles: &mut [Hier], counts: [usize; 2]) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    evolve_hierarchy_decomposed(
        tiles,
        counts,
        &devices,
        &LocalCopy,
        Timestepping::Euler,
        0.0,
        T_FINAL,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

// composite the density at the finest covering resolution, keyed by global fine index.
fn composite_fine(tiles: &[Hier], counts: [usize; 2]) -> Vec<f64> {
    let fn_n = 2 * N;
    let mut out = vec![f64::NAN; fn_n * fn_n];
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    for (flat, h) in tiles.iter().enumerate() {
        let tc = unflatten(flat, counts);
        let root = &h.levels[0].state;
        let cov = h.levels[0].coverage.as_ref();
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        for c in root.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - rlo[a]) as usize);
            let d = *root.fields.cons.den.view().at(c);
            for sy in 0..2 {
                for sx in 0..2 {
                    out[(2 * g[1] + sy) * fn_n + (2 * g[0] + sx)] = d;
                }
            }
        }
        if h.levels.len() > 1 {
            let fine = &h.levels[1].state;
            let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
            for c in fine.geom.interior.iter() {
                let fx = (c[0] - flo[0]) as usize + (flo[0] as usize);
                let fy = (c[1] - flo[1]) as usize + (flo[1] as usize);
                out[fy * fn_n + fx] = *fine.fields.cons.den.view().at(c);
            }
        }
    }
    out
}

fn assert_matches(counts: [usize; 2]) {
    let mut mono = build_mono();
    mono.evolve(T_FINAL).expect("mono evolve");
    let mono_den = composite_fine(std::slice::from_ref(&mono), [1, 1]);

    let mut tiles = build_tiles(counts);
    run_decomposed(&mut tiles, counts);
    let dec_den = composite_fine(&tiles, counts);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some composite cells were never written"
    );
    // the inflow actually raised the density near x_lo (non-vacuous): the fine patch sits on
    // the driven face, so its edge cells feel the rho = 2+ inflow within T_FINAL.
    let inflow_gain = mono_den.iter().cloned().fold(0.0_f64, f64::max) - 1.0;
    assert!(
        inflow_gain > 0.05,
        "driven inflow never registered (max den gain {inflow_gain:e})"
    );

    let err = mono_den
        .iter()
        .zip(&dec_den)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err < 1e-12,
        "{counts:?}: refined+decomposed driven diverged from mono: {err:e}"
    );
}

#[test]
fn driven_refined_cut_perpendicular_to_the_face() {
    assert_matches([2, 1]);
}

#[test]
fn driven_refined_cut_along_the_face() {
    // both edge tiles own a piece of the driven face; tile 0 additionally owns the flush
    // refined patch — the fine level inherits Driven(0) while the cut rides the exchange.
    assert_matches([1, 2]);
}

#[test]
fn driven_refined_four_tiles() {
    assert_matches([2, 2]);
}
