// =============================================================================
// decomp_refine_p3_equivalence.rs
//
// refinement x decomposition: a refined patch that spans a tile cut. the fine level is
// itself split across tiles, so on top of the root halo exchange the driver must exchange
// the fine-level halos between the fine tiles at the shared cut. the flux/emf reflux registers stay
// tile-local (a coarse cell + the fine cells at its face are co-located), so the only new coupling
// is the fine-level exchange.
//
// the patch [0.375,0.625]^2 straddles the x-cut at x=0.5: tile 0 refines its left half, tile 1 its
// right half; the two fine grids share the cut (each inherits a CoarseFine boundary there from its
// root tile, so fine ghost_fill leaves it for the fine exchange). a smooth bump centered on the cut
// makes the fine dynamics cross it. decomposed == canonical monolithic (hier.evolve) to round-off.
// cpu-only + 2d hydro; euler + rk2 root (rk2 also exchanges between the fine stages).
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
    seed_decomposed_fine_from_coarse,
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
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.03;
const RATIO: usize = 2;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

// a smooth bump centered on the cut (0.5, 0.5), inside the patch -> the fine dynamics straddle the
// fine cut, exercising the fine-level exchange. waves stay interior over T_FINAL.
fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp()
}

// the global refined patch, straddling the x-cut at 0.5.
const PX_LO: f64 = 0.375;
const PX_HI: f64 = 0.625;
const PY_LO: f64 = 0.375;
const PY_HI: f64 = 0.625;

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

// the canonical monolithic reference: full-grid root + the spanning patch, run via hier.evolve().
fn build_mono(ts: Timestepping) -> Hier {
    let root = build_root(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
        ts,
    );
    let k = kset(&root);
    let region = RefinementRegion {
        x_lo: [PX_LO, PY_LO],
        x_hi: [PX_HI, PY_HI],
    };
    let mut h =
        Hier::with_refinement(root, k, &[region], ProlongOrder::Ppm, kset).expect("mono hier");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

// root tiles from the partition's own extents; each refines by clipping the global patch to its
// physical range, so the fine grids meet at every cut the patch crosses. unequal root tiles give
// unequal fine grids, which is the case the fine-level exchange has to handle.
fn build_tiles(partition: &Partition<2>, ts: Timestepping) -> Vec<Hier> {
    let counts = partition.counts();
    let total = partition.n_tiles();
    let mut tiles = Vec::with_capacity(total);
    for flat in 0..total {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m: [usize; 2] = [ext[0].1, ext[1].1];
        let origin: [f64; 2] = std::array::from_fn(|a| ext[a].0 as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| origin[a] + m[a] as f64 * DX);
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
        // clip the global patch to this tile's physical extent.
        let cx = [PX_LO.max(origin[0]), PX_HI.min(hi[0])];
        let cy = [PY_LO.max(origin[1]), PY_HI.min(hi[1])];
        let owns_patch = cx[0] < cx[1] && cy[0] < cy[1];
        let h = if owns_patch {
            let k = kset(&root);
            let region = RefinementRegion {
                x_lo: [cx[0], cy[0]],
                x_hi: [cx[1], cy[1]],
            };
            let h = Hier::with_refinement(root, k, &[region], ProlongOrder::Ppm, kset)
                .expect("tile hier");
            h
        } else {
            let k = kset(&root);
            Hier::single(root, k)
        };
        tiles.push(h);
    }
    // seeding is decomposition-aware: the fine interiors are prolonged only after the root
    // conserved cut halos carry the neighbor tiles' data. seeding per tile inside the loop
    // reads each tile's standalone boundary fill through the cut and the composite differs
    // from the monolithic hierarchy before any evolution. `prime` runs a c2p audit on every
    // level, so it must see the seeded fine level and runs after.
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy)
        .expect("decomposed fine seed");
    for h in tiles.iter_mut() {
        h.prime();
    }
    tiles
}

// drive the tiles through the production decomposed-hierarchy loop (symbi-amr). the same lib fn
// `evolve_hierarchy_decomposed` handles the tile-local case (fine sub-grid 1x1) and the spanning case
// (the patch spans the cut, so the fine sub-grid has an internal cut and the fine halos are exchanged)
// -- this test proves the spanning-patch path. host: all tiles on "device 0".
fn run_p3(tiles: &mut [Hier], counts: [usize; 2], t_final: f64, ts: Timestepping) {
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

// scatter every tile's root (outside coverage) + fine density into one global 2N x 2N composite,
// keyed by global fine index (root cell g -> its 2x2 fine block; fine cell -> its global fine index
// = tile_root_offset*ratio + local fine offset).
fn composite_fine(tiles: &[Hier], partition: &Partition<2>) -> Vec<f64> {
    let fn_n = RATIO * N;
    let counts = partition.counts();
    let mut out = vec![f64::NAN; fn_n * fn_n];
    for (flat, h) in tiles.iter().enumerate() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let root = &h.levels[0].state;
        let cov = h.levels[0].coverage.as_ref();
        let rlo: [isize; 2] = std::array::from_fn(|a| root.geom.interior.spaces[a].lo);
        for c in root.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            let g: [usize; 2] = std::array::from_fn(|a| ext[a].0 + (c[a] - rlo[a]) as usize);
            let d = *root.fields.cons.den.view().at(c);
            for sy in 0..RATIO {
                for sx in 0..RATIO {
                    out[(RATIO * g[1] + sy) * fn_n + (RATIO * g[0] + sx)] = d;
                }
            }
        }
        if h.levels.len() > 1 {
            let fine = &h.levels[1].state;
            let flo: [isize; 2] = std::array::from_fn(|a| fine.geom.interior.spaces[a].lo);
            // the coverage lo in this tile's global root cells (tile offset + its interior offset).
            let cov = h.levels[0].coverage.as_ref().unwrap();
            let clo: [isize; 2] = std::array::from_fn(|a| cov.spaces[a].lo);
            for c in fine.geom.interior.iter() {
                // global fine index = (global root cell of the coverage start)*ratio + the fine
                // cell's offset within the fine interior. this is consistent between the monolithic
                // (one tile, full patch) and decomposed (per-tile clipped patch) runs.
                let gf: [usize; 2] = std::array::from_fn(|a| {
                    let cov_global_root = ext[a].0 + (clo[a] - rlo[a]) as usize;
                    cov_global_root * RATIO + (c[a] - flo[a]) as usize
                });
                out[gf[1] * fn_n + gf[0]] = *fine.fields.cons.den.view().at(c);
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

fn assert_p3_matches(counts: [usize; 2], ts: Timestepping) {
    let partition =
        Partition::uniform([N, N], counts).expect("even tile counts divide the root grid");
    assert_p3_matches_partition(&partition, ts);
}

fn assert_p3_matches_partition(partition: &Partition<2>, ts: Timestepping) {
    let counts = partition.counts();
    let whole = Partition::explicit([N, N], [Vec::new(), Vec::new()])
        .expect("the uncut partition is one tile");
    let mut mono = build_mono(ts);
    // construction gate, before either side evolves: the seeded composites must already agree.
    // this is what localizes a future regression to seeding rather than to the driver -- the
    // divergence this test caught lived here, was smoothed by evolution, and read as a
    // dynamics bug for six hypotheses straight.
    {
        let mono_c0 = composite_fine(std::slice::from_ref(&mono), &whole);
        let dec_c0 = composite_fine(&build_tiles(partition, ts), partition);
        let e0 = max_err(&mono_c0, &dec_c0);
        assert!(
            e0 < 1e-12,
            "{counts:?} {ts:?}: decomposed and monolithic hierarchies differ at CONSTRUCTION \
             (composite err {e0:e}); the fine seeding read unexchanged data"
        );
    }
    mono.evolve(T_FINAL).expect("mono evolve");
    let mono_c = composite_fine(std::slice::from_ref(&mono), &whole);

    let mut dec = build_tiles(partition, ts);
    run_p3(&mut dec, counts, T_FINAL, ts);
    let dec_c = composite_fine(&dec, partition);

    assert!(
        mono_c.iter().all(|v| v.is_finite()) && dec_c.iter().all(|v| v.is_finite()),
        "some composite cells were never written"
    );
    let spread = mono_c.iter().cloned().fold(0.0_f64, f64::max)
        - mono_c.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        spread > 1e-2,
        "composite is flat ({spread:e}); test is vacuous"
    );

    let e = max_err(&mono_c, &dec_c);
    assert!(
        e < 1e-12,
        "{counts:?} {ts:?} P3 (patch across cut) composite err {e:e}"
    );
}

// the python output path: gather each decomposed level into a global hierarchy (root over `counts`,
// fine over the `fine_subgrid` sub-grid) via `gather_interiors`, and confirm it equals the
// monolithic hierarchy level-for-level. this is exactly what `run_refined_decomposed_loop` does to
// write a multi-level checkpoint; the per-tile evolve is the same path the assert_p3 tests prove.
#[test]
fn refine_p3_gather_reassembles_hierarchy() {
    use symbi::sim::decomp::gather_interiors;
    use symbi::sim::refinement::fine_subgrid;
    let ts = Timestepping::Rk2;
    let counts = [2, 1];

    let mut mono = build_mono(ts);
    mono.evolve(T_FINAL).expect("mono evolve");

    let partition = Partition::uniform([N, N], counts).expect("even counts divide the root grid");
    let mut dec = build_tiles(&partition, ts);
    run_p3(&mut dec, counts, T_FINAL, ts);

    // a fresh full hierarchy as the gather target (root + the full patch), like the binding's global.
    let global = build_mono(ts);
    let devices: Vec<i32> = vec![0; dec.len()];
    let fg = fine_subgrid(&dec, counts, &devices).expect("refined tiles -> fine sub-grid");
    {
        let roots: Vec<_> = dec.iter().map(|h| &*h.levels[0].state).collect();
        gather_interiors(&*global.levels[0].state, &roots, counts);
        let fines: Vec<_> = fg.order.iter().map(|&i| &*dec[i].levels[1].state).collect();
        gather_interiors(&*global.levels[1].state, &fines, fg.counts);
    }

    let level_err = |g: &Sim, mo: &Sim| -> f64 {
        let mut e = 0.0_f64;
        for c in g.geom.interior.iter() {
            e = e.max((g.fields.cons.den.view().at(c) - mo.fields.cons.den.view().at(c)).abs());
        }
        e
    };
    let er = level_err(&global.levels[0].state, &mono.levels[0].state);
    let ef = level_err(&global.levels[1].state, &mono.levels[1].state);
    assert!(er < 1e-12, "gathered root level err {er:e}");
    assert!(ef < 1e-12, "gathered fine level err {ef:e}");
}

#[test]
fn refine_p3_euler_x_cut() {
    assert_p3_matches([2, 1], Timestepping::Euler);
}

#[test]
fn refine_p3_rk2_x_cut() {
    assert_p3_matches([2, 1], Timestepping::Rk2);
}

// the patch spans cuts on BOTH axes: four root tiles each own a different corner of it,
// and every fine tile's ghosts on its two cut sides sit beyond what its own parent can
// feed, so those cells are carried by the fine halo exchange instead of prolongation.
#[test]
fn refine_p3_rk2_uniform_quad() {
    assert_p3_matches([2, 2], Timestepping::Rk2);
}

// the same two-axis span with unequal cuts: root tiles of four different shapes.
#[test]
fn refine_p3_rk2_ragged_quad() {
    let ragged = Partition::explicit([N, N], [vec![17], vec![15]])
        .expect("interior cuts are strictly increasing and inside the grid");
    assert!(!ragged.is_uniform(), "the gate must exercise unequal tiles");
    assert_p3_matches_partition(&ragged, Timestepping::Rk2);
}

// an unequal x-cut at cell 17: root tiles of 17 and 15 cells, and the patch splits into fine
// grids of 10 and 6 cells that still share the fine seam. tile extents come from the partition,
// so the fine patch clip, the fine exchange and the composite all follow the actual cut.
#[test]
fn refine_p3_rk2_ragged_x_cut() {
    let ragged = Partition::explicit([N, N], [vec![17], Vec::new()])
        .expect("interior cuts are strictly increasing and inside the grid");
    assert!(!ragged.is_uniform(), "the gate must exercise unequal tiles");
    assert_p3_matches_partition(&ragged, Timestepping::Rk2);
}
