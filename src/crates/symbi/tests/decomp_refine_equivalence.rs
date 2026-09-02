// =============================================================================
// decomp_refine_equivalence.rs
//
// refinement x decomposition (tile-local static refinement, euler root). a 2-level
// hierarchy whose root is split into tiles must reproduce the monolithic hierarchy run, when each
// refined patch lives entirely inside one root tile's interior. each tile then owns a complete
// sub-hierarchy that the existing recursive advance drives unchanged (all prolong / restrict /
// reflux stay local to the owning tile); the only cross-tile coupling is the root-level halo
// exchange, done between root steps (euler root = single stage, so a between-step exchange is
// correct -- the rk2-root between-stage case is covered separately).
//
// the decomposed driver lockstep-advances N per-tile hierarchies: global dt = min over tiles of
// `root_cfl_dt()`, then `evolve(t + dt)` drives each tile by exactly that dt (the global min
// collapses each tile's internal cfl clamp), then exchange root halos. cpu-only + 2d hydro.
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
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32; // root cells per axis
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

// a smooth density+pressure bump centered in the bottom-left quadrant (x ~ y ~ 0.25) so its
// dynamics + the refined patch stay inside tile 0 for every topology (x-cut, y-cut, 2x2 -- the cuts
// are at x = 0.5 and/or y = 0.5). waves stay interior over T_FINAL.
fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.25) / 0.08).powi(2) + ((y - 0.25) / 0.08).powi(2))).exp()
}

// the refined patch: physical box well inside the bottom-left quadrant [0,0.5]^2, interior to it,
// so its coarse-fine prolongation reads only tile-0-local root cells (no cross-tile coupling) under
// any of the tested decompositions.
fn patch() -> RefinementRegion<2> {
    RefinementRegion {
        x_lo: [0.125, 0.125],
        x_hi: [0.375, 0.375],
    }
}

fn build_root(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>, ts: Timestepping) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(ts)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
        })
        .build()
}

// the monolithic reference: one full-grid root + the refined patch, run via the canonical
// `hier.evolve()` (so the test validates both the decomposition and that the staged driver
// reproduces the canonical recursive advance).
fn build_mono(ts: Timestepping) -> Hier {
    let root = build_root(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
        ts,
    );
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, &[patch()], ProlongOrder::Ppm, kset)
        .expect("mono hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

// the decomposed build: `counts` root tiles. tile 0 (containing the patch) is a 2-level hierarchy;
// every other tile is single-level. each tile's root carries CoarseFine on internal faces + the
// physical boundary outside, so the root halo exchange owns the cut.
fn build_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<Hier> {
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
        let root = build_root(m, origin, bnd, ts);
        // the patch lives in the single tile whose physical extent contains it (the bottom-left
        // quadrant -> tile 0 for every tested topology). check both axes.
        let owns_patch =
            (0..2).all(|a| origin[a] <= 0.125 && origin[a] + m[a] as f64 * DX >= 0.375);
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

// drive the tiles through the production decomposed-hierarchy loop (symbi-amr): exercises the
// same `evolve_hierarchy_decomposed` a multi-gpu refinement run would call, so a divergence
// between test and production is impossible. host: all tiles on "device 0".
fn run_decomposed(tiles: &mut [Hier], counts: [usize; 2], t_final: f64, ts: Timestepping) {
    let devices: Vec<i32> = vec![0; tiles.len()];
    evolve_hierarchy_decomposed(
        tiles,
        counts,
        &devices,
        &LocalCopy,
        ts,
        0.0,
        t_final,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

// scatter every level's density into one global composite grid at the finest resolution that
// covers each point: a covered coarse cell is skipped in favor of its fine children. keyed by
// absolute fine-index (fine levels use the global index space), so mono and decomposed line up.
fn composite_fine(tiles: &[Hier], counts: [usize; 2]) -> Vec<f64> {
    // the fine grid is the whole domain at 2x: 2N x 2N.
    let fn_n = 2 * N;
    let mut out = vec![f64::NAN; fn_n * fn_n];
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    for (flat, h) in tiles.iter().enumerate() {
        let tc = unflatten(flat, counts);
        // root cells -> their 2x2 fine block (so every output cell is written once, from the finest
        // level that covers it). the covered root cells are overwritten by the actual fine level.
        let root = &h.levels[0].state;
        let cov = h.levels[0].coverage.as_ref();
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        for c in root.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue; // covered: the fine level writes these
                }
            }
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - rlo[a]) as usize);
            let d = *root.fields.cons.den.view().at(c);
            for sy in 0..2 {
                for sx in 0..2 {
                    let fx = 2 * g[0] + sx;
                    let fy = 2 * g[1] + sy;
                    out[fy * fn_n + fx] = d;
                }
            }
        }
        // fine level (if any): absolute fine index == global fine index (tile shares global origin).
        if h.levels.len() > 1 {
            let fine = &h.levels[1].state;
            let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
            for c in fine.geom.interior.iter() {
                // absolute fine index: interior_lo is coverage.lo*ratio in the tile's index space;
                // tile shares the global origin so this is the global fine index.
                let fx = (c[0] - flo[0]) as usize + (flo[0] as usize);
                let fy = (c[1] - flo[1]) as usize + (flo[1] as usize);
                out[fy * fn_n + fx] = *fine.fields.cons.den.view().at(c);
            }
        }
    }
    out
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_refine_matches(counts: [usize; 2], ts: Timestepping) {
    let mut mono = build_mono(ts);
    mono.evolve(T_FINAL).expect("mono evolve"); // canonical recursive advance
    let mono_c = composite_fine(std::slice::from_ref(&mono), [1, 1]);

    let mut dec = build_tiles(counts, ts);
    run_decomposed(&mut dec, counts, T_FINAL, ts);
    let dec_c = composite_fine(&dec, counts);

    assert!(
        mono_c.iter().all(|v| v.is_finite()) && dec_c.iter().all(|v| v.is_finite()),
        "some composite cells were never written"
    );
    // non-vacuous: the fine patch must hold non-trivial structure (the bump).
    let spread = mono_c.iter().cloned().fold(0.0_f64, f64::max)
        - mono_c.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        spread > 1e-2,
        "composite is flat ({spread:e}); test is vacuous"
    );

    let e = max_err(&mono_c, &dec_c);
    assert!(
        e < 1e-12,
        "{counts:?} refined decomposed vs monolithic composite err {e:e}"
    );
}

#[test]
fn refine_euler_two_tile_x_cut() {
    assert_refine_matches([2, 1], Timestepping::Euler);
}

#[test]
fn refine_euler_two_tile_y_cut() {
    assert_refine_matches([1, 2], Timestepping::Euler);
}

// the 2x2 grid: the patch sits in tile 0 (bottom-left quadrant); the other three tiles are
// single-level. exercises the root halo exchange across both cuts with a refined tile present.
#[test]
fn refine_euler_quad_tile_2d_grid() {
    assert_refine_matches([2, 2], Timestepping::Euler);
}

// RK2 root: the root halo must be exchanged between the two root stages -- the corrector
// reads each neighbor's stage-1 update. a between-step-only exchange would diverge here. driven by
// the level_step_begin / level_stage / level_step_tail interface.
#[test]
fn refine_rk2_two_tile_x_cut() {
    assert_refine_matches([2, 1], Timestepping::Rk2);
}

#[test]
fn refine_rk2_two_tile_y_cut() {
    assert_refine_matches([1, 2], Timestepping::Rk2);
}

#[test]
fn refine_rk2_quad_tile_2d_grid() {
    assert_refine_matches([2, 2], Timestepping::Rk2);
}
