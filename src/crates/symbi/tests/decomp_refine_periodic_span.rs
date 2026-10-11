// =============================================================================
// decomp_refine_periodic_span.rs
//
// a refined patch spanning a periodic axis that is cut into host tiles. the root
// wraps along x and the patch covers all of x over a band in y, so it has no
// face along x: its fine level wraps across the tiles at the seam as its root
// does, and the covered core continues across the seam. a flow along x carries
// gas through the seam on both levels. two tiles cut at root cell 16 and three
// uneven tiles reach the single hierarchy's conserved state on every level bit
// for bit.
//
// run: cargo test -p symbi --test decomp_refine_periodic_span
// =============================================================================
use std::collections::HashMap;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;
// the patch spans all of x over root cells 12..20 in y
const PY_LO: f64 = 0.375;
const PY_HI: f64 = 0.625;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

/// a density bump straddling the seam at x = 0 carried along x through it
fn state(x: [f64; 2]) -> Prim<f64, 2> {
    let dx = (x[0] - 0.05).rem_euclid(1.0).min((0.05 - x[0]).rem_euclid(1.0));
    let bump = 0.5 * (-((dx / 0.08).powi(2) + ((x[1] - 0.5) / 0.08).powi(2))).exp();
    Prim::adiabatic(Density(1.0 + bump), Tensor::new([0.6, 0.1 * (2.0 * std::f64::consts::PI * x[0]).sin()]), Pressure(1.0))
}

fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(state)
        .build()
}

fn whole() -> RefinementRegion<2> {
    RefinementRegion { x_lo: [0.0, PY_LO], x_hi: [1.0, PY_HI] }
}

fn build_single() -> Hier {
    let bnd = Boundaries([[BoundaryType::Periodic; 2], [BoundaryType::Outflow; 2]]);
    let root = build_root([N, N], [0, 0], bnd);
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, &[whole()], ProlongOrder::Ppm, kset).expect("single hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

/// root tiles cut along x alone: the seam and the cuts are coarse-fine sides to each tile, the
/// root wraps along x, and every tile refines its share of the patch
fn build_tiles(partition: &Partition<2>) -> Vec<Hier> {
    let counts = partition.counts();
    let mut tiles: Vec<Hier> = (0..partition.n_tiles())
        .map(|flat| {
            let ext = partition.tile_extents(unflatten(flat, counts));
            let bnd = Boundaries([[BoundaryType::CoarseFine; 2], [BoundaryType::Outflow; 2]]);
            let root = build_root([ext[0].1, ext[1].1], [ext[0].0, ext[1].0], bnd);
            let k = kset(&root);
            let (lo, hi) = (ext[0].0 as f64 * DX, (ext[0].0 + ext[0].1) as f64 * DX);
            let region = RefinementRegion { x_lo: [lo, PY_LO], x_hi: [hi, PY_HI] };
            Hier::with_refinement_placed(root, k, &[region], &[whole()], ProlongOrder::Ppm, kset, [N, N], [true, false])
                .expect("tile hierarchy")
        })
        .collect();
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("decomposed fine seed");
    for h in tiles.iter_mut() {
        h.prime();
    }
    tiles
}

/// (component, level, centroid on twice the level's lattice) -> bits, over every tile's interiors
fn interior_state(tiles: &[Hier]) -> HashMap<(usize, usize, [i64; 2]), u64> {
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let c = &s.fields.cons;
            let scale = (N << ll) as f64;
            for x in s.geom.interior.iter() {
                let at = s.geom.centroid(x);
                let key: [i64; 2] = std::array::from_fn(|a| (at[a] * scale * 2.0).round() as i64);
                let values = [*c.den.view().at(x), *c.mom[0].view().at(x), *c.mom[1].view().at(x), *c.nrg_field().unwrap().view().at(x)];
                for (k, v) in values.iter().enumerate() {
                    let previous = out.insert((k, ll, key), v.to_bits());
                    assert!(previous.is_none(), "component {k} on level {ll} at {key:?} is held by two tiles");
                }
            }
        }
    }
    out
}

fn assert_tiles_match(cuts: Vec<usize>) {
    let mut single = build_single();
    single.evolve(T_FINAL).expect("single evolve");
    let partition = Partition::explicit([N, N], [cuts.clone(), vec![]]).expect("cuts inside the grid");
    let mut tiles = build_tiles(&partition);
    let devices = vec![0; tiles.len()];
    evolve_hierarchy_decomposed(&mut tiles, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, |_, _, _| {
        std::ops::ControlFlow::Continue(())
    });
    for (k, tile) in tiles.iter().enumerate() {
        assert!(tile.crash.is_none(), "cuts {cuts:?}: tile {k} crashed: {:?}", tile.crash);
        assert_eq!(tile.levels[0].state.iteration, single.levels[0].state.iteration, "cuts {cuts:?}: tile {k} stopped short");
    }
    let (want, got) = (interior_state(std::slice::from_ref(&single)), interior_state(&tiles));
    assert_eq!(got.len(), want.len(), "cuts {cuts:?}: the tiles hold a different set of cells");
    let mut differing: Vec<_> = want.iter().filter(|(k, v)| got.get(k) != Some(v)).map(|(k, _)| *k).collect();
    differing.sort();
    assert!(differing.is_empty(), "cuts {cuts:?}: {} values differ, first {:?}", differing.len(), differing.first());
}

#[test]
fn a_patch_spanning_a_cut_periodic_axis_matches_the_single_hierarchy_on_two_tiles() {
    // premise: the gas moves through the seam on the fine level within the run
    let mut single = build_single();
    let fine0 = single.density_snapshot(1);
    single.evolve(T_FINAL).expect("single evolve");
    let moved = single.density_snapshot(1).iter().zip(&fine0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(moved > 1.0e-2, "the fine density moved by {moved:e}; the seam carries nothing");
    assert!(single.levels[0].state.iteration >= 4, "the run took {} root steps", single.levels[0].state.iteration);
    assert_tiles_match(vec![16]);
}

#[test]
fn a_patch_spanning_a_cut_periodic_axis_matches_the_single_hierarchy_on_three_uneven_tiles() {
    assert_tiles_match(vec![10, 21]);
}
