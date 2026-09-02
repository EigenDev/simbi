// =============================================================================
// decomp_refine_body_equivalence.rs
//
// immersed bodies x refinement x decomposition: per-tile 2-level hierarchies, each carrying
// the same bodies at their global positions (finest-owns-bodies per tile: the fine level
// holds the full accreting body, the root a gravity-only proxy), driven through the
// production `evolve_hierarchy_decomposed` loop, reproduce the monolithic refined run
// to round-off — fields and the accreted-mass bookkeeping alike. the decomposed body phase sums
// each tile's finest-level feedback partials (the tile fine interiors partition the sink
// region) and applies the identical global delta everywhere, the same lockstep contract the
// flat decomposed loop proves in decomp_body_equivalence.
//
// two placements: the sink inside one tile's fine patch, and the sink straddling a root cut
// with the refined patch spanning both tiles (each tile's clipped fine level drains its own
// share; the clipped containment invariant holds on both).
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
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

// an accreting sink at `pos`: gravitating mass fixed, drain + accretion bookkeeping active.
// a fresh collection per attach (with_bodies takes ownership); same global position everywhere.
fn sink_at(pos: [f64; 2]) -> BodyCollection<f64, 2> {
    BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new(pos),
        Tensor::zeros(),
        1.0,  // gravitating mass (held fixed)
        0.04, // radius
        0.08, // softening
        10.0, // sink_rate
        0.5,  // sink_delta
        0.08, // accretion_radius
    ))
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
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
}

// clip a region to a tile's physical slab; None when the overlap is empty or degenerate.
fn clip(
    region: &RefinementRegion<2>,
    origin: [f64; 2],
    m: [usize; 2],
) -> Option<RefinementRegion<2>> {
    let mut lo = [0.0; 2];
    let mut hi = [0.0; 2];
    for a in 0..2 {
        let tlo = origin[a];
        let thi = origin[a] + m[a] as f64 * DX;
        lo[a] = region.x_lo[a].max(tlo);
        hi[a] = region.x_hi[a].min(thi);
        if hi[a] - lo[a] < DX {
            return None;
        }
    }
    Some(RefinementRegion { x_lo: lo, x_hi: hi })
}

fn build_mono(region: &RefinementRegion<2>, sink_pos: [f64; 2]) -> Hier {
    let root = build_root(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
    );
    let k = kset(&root);
    let h = Hier::with_refinement(
        root,
        k,
        std::slice::from_ref(region),
        ProlongOrder::Ppm,
        kset,
    )
    .expect("mono hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    let mut h = h.with_bodies(sink_at(sink_pos));
    h.prime();
    h
}

fn build_tiles(counts: [usize; 2], region: &RefinementRegion<2>, sink_pos: [f64; 2]) -> Vec<Hier> {
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
        let h = match clip(region, origin, m) {
            Some(r) => {
                let k = kset(&root);
                let h = Hier::with_refinement(root, k, &[r], ProlongOrder::Ppm, kset)
                    .expect("tile hierarchy");
                h.seed_fine_from_coarse().expect("seed fine");
                h
            }
            None => {
                let k = kset(&root);
                Hier::single(root, k)
            }
        };
        let mut h = h.with_bodies(sink_at(sink_pos));
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

fn composite_den(tiles: &[Hier], counts: [usize; 2]) -> Vec<f64> {
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
            // the fine interior index is tile-local (coverage.lo * ratio in the tile's root
            // index space); the global fine index adds the tile's root offset at 2x.
            let fine = &h.levels[1].state;
            for c in fine.geom.interior.iter() {
                let fx = 2 * tc[0] * m[0] + c[0] as usize;
                let fy = 2 * tc[1] * m[1] + c[1] as usize;
                out[fy * fn_n + fx] = *fine.fields.cons.den.view().at(c);
            }
        }
    }
    out
}

fn accreted(h: &Hier) -> f64 {
    let finest = &h.levels[h.levels.len() - 1].state;
    let im = finest.immersed.as_ref().expect("bodies attached");
    let mut total = 0.0;
    im.bodies.visit_accretion(|b| {
        if let symbi_ib::BodyKind::BlackHole {
            total_accreted_mass,
            ..
        } = &b.kind
        {
            total += *total_accreted_mass;
        }
    });
    total
}

fn assert_matches(counts: [usize; 2], region: RefinementRegion<2>, sink_pos: [f64; 2]) {
    let mut mono = build_mono(&region, sink_pos);
    mono.evolve(T_FINAL).expect("mono evolve");
    let mono_den = composite_den(std::slice::from_ref(&mono), [1, 1]);
    let mono_acc = accreted(&mono);

    let mut tiles = build_tiles(counts, &region, sink_pos);
    run_decomposed(&mut tiles, counts);
    let dec_den = composite_den(&tiles, counts);

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some composite cells were never written"
    );
    // the sink actually removed mass (non-vacuous).
    assert!(
        mono_acc > 1e-6,
        "the sink accreted nothing ({mono_acc:e}); test is vacuous"
    );
    // every tile carries the identical global accreted-mass tally, equal to the monolithic one.
    for (i, h) in tiles.iter().enumerate() {
        let a = accreted(h);
        assert!(
            (a - mono_acc).abs() < 1e-12 * mono_acc.max(1.0),
            "{counts:?} tile {i}: accreted mass {a:e} != mono {mono_acc:e}"
        );
    }
    let err = mono_den
        .iter()
        .zip(&dec_den)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err < 1e-12,
        "{counts:?}: refined+decomposed body run diverged from mono: {err:e}"
    );
}

#[test]
fn sink_inside_one_tiles_patch() {
    // the patch and the sink live in the bottom-left quadrant: one tile owns both.
    let region = RefinementRegion {
        x_lo: [0.125, 0.125],
        x_hi: [0.375, 0.375],
    };
    let region2 = RefinementRegion {
        x_lo: region.x_lo,
        x_hi: region.x_hi,
    };
    assert_matches([2, 1], region2, [0.25, 0.25]);
    assert_matches([2, 2], region, [0.25, 0.25]);
}

#[test]
fn sink_straddling_a_cut_with_the_patch_spanning_tiles() {
    // the patch spans the x = 0.5 cut in a [2, 1] tiling; the sink sits on the cut, so each
    // tile's clipped fine level drains its own share and the cross-tile sum restores the
    // global reaction.
    let region = RefinementRegion {
        x_lo: [0.25, 0.25],
        x_hi: [0.75, 0.75],
    };
    assert_matches([2, 1], region, [0.5, 0.5]);
}
