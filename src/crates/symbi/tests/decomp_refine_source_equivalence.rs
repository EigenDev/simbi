// =============================================================================
// decomp_refine_source_equivalence.rs
//
// USER SOURCES x REFINEMENT x DECOMPOSITION: per-tile 2-level hierarchies, each level of
// each tile carrying the SAME runtime source, must reproduce the monolithic refined run to
// round-off through the PRODUCTION `evolve_hierarchy_decomposed` loop. the source is the
// POSITION-DEPENDENT force a = [x, 0] (VARIABLE_X1 -> the cell's global x): a tile or fine
// level evaluating it at a local coordinate diverges at every cut and level seam; a constant
// force could not catch either bug. the canonical per-level stage drives the additive
// source_apply on every level of every tile.
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
use symbi_hydro::expr_bridge::build_user_source;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{NEWTONIAN_SPEC, SourceConfig};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const SOURCE_JSON: &str = r#"{
    "kind": "force", "dim": 2, "outputs": [0, 1], "params": [],
    "nodes": [ {"op": "VARIABLE_X1"}, {"op": "CONSTANT", "value": 0.0} ]
}"#;

fn kset(sim: &Sim) -> Kern {
    let cfg = SourceConfig::from_json(SOURCE_JSON).expect("parse source config");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("lower source");
    Kern::new(GAMMA, CFL, &sim.geom.allocated).with_runtime_source(built, cfg.params.clone())
}

// a smooth bump in the bottom-left quadrant; the refined patch covers it and stays inside
// tile 0 for every tested topology (cuts at x = 0.5 and/or y = 0.5).
fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.25) / 0.08).powi(2) + ((y - 0.25) / 0.08).powi(2))).exp()
}

fn patch() -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [0.125, 0.125],
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
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim {
                rho: 1.0 + b,
                vel: Tensor::new([0.0, 0.0]),
                pre: 1.0 + b,
            }
        })
        .build()
}

fn build_mono() -> Hier {
    let root = build_root(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
    );
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, &[patch()], ProlongOrder::Ppm, kset)
        .expect("mono hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

fn build_tiles(counts: [usize; 2]) -> Vec<Hier> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let total: usize = counts.iter().product();
    let mut tiles = Vec::with_capacity(total);
    for flat in 0..total {
        let tc = unflatten(flat, counts);
        let origin = std::array::from_fn(|a| tc[a] as f64 * m[a] as f64 * DX);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let lo = if tc[a] == 0 {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            };
            let hi = if tc[a] == counts[a] - 1 {
                BoundaryType::Outflow
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

fn composite_fine(
    tiles: &[Hier],
    counts: [usize; 2],
    pick: impl Fn(&Sim, [isize; 2]) -> f64,
) -> Vec<f64> {
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
            let d = pick(root, c);
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
                out[fy * fn_n + fx] = pick(fine, c);
            }
        }
    }
    out
}

fn assert_matches(counts: [usize; 2]) {
    let den = |s: &Sim, c: [isize; 2]| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c: [isize; 2]| *s.fields.cons.mom[0].view().at(c);

    let mut mono = build_mono();
    mono.evolve(T_FINAL).expect("mono evolve");
    let mono_den = composite_fine(std::slice::from_ref(&mono), [1, 1], den);
    let mono_momx = composite_fine(std::slice::from_ref(&mono), [1, 1], momx);

    let mut tiles = build_tiles(counts);
    run_decomposed(&mut tiles, counts);
    let dec_den = composite_fine(&tiles, counts, den);
    let dec_momx = composite_fine(&tiles, counts, momx);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some composite cells were never written"
    );
    // the position force accelerates every cell: total |mom_x| well above zero, else vacuous.
    let total: f64 = mono_momx.iter().map(|v| v.abs()).sum();
    assert!(
        total > 1e-3,
        "source produced no momentum ({total:e}); test is vacuous"
    );

    let de = mono_den
        .iter()
        .zip(&dec_den)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let me = mono_momx
        .iter()
        .zip(&dec_momx)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        de < 1e-12,
        "{counts:?}: density diverged under refined+decomposed source: {de:e}"
    );
    assert!(
        me < 1e-12,
        "{counts:?}: mom_x diverged under refined+decomposed source: {me:e}"
    );
}

#[test]
fn source_refined_two_tiles_x() {
    assert_matches([2, 1]);
}

#[test]
fn source_refined_two_tiles_y() {
    assert_matches([1, 2]);
}

#[test]
fn source_refined_four_tiles() {
    assert_matches([2, 2]);
}
