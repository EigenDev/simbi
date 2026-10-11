// =============================================================================
// decomp_refine_balanced_nested.rs
//
// the balanced reconstruction on nested refined host tiles. an isentropic
// atmosphere at rest in a point mass's potential under two nested boxes (level
// 1 over root cells 8..24, level 2 over root cells 12..20), on the balanced
// reconstruction. each level's restriction band rewrites the parent cells
// within four parent cells of its seams from the uncovered parent cell beyond:
// level 1's bands lie on root cells 8..12 and 20..24, level 2's on level-1
// cells 24..28 and 36..40 (root cells 12..14 and 18..20). the tiles equal the
// single hierarchy in every density of every level over about seventy-five
// root steps on every layout keeping each band on one tile: one cut through
// both boxes, both axes cut, ragged cuts between level 2's bands, and cuts
// outside level 1. the gate asserts its premises: level 2 lives on at least
// two tiles and the balance changes level 2.
//
// run: cargo test -p symbi --test decomp_refine_balanced_nested
// =============================================================================
use std::collections::HashMap;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GM: f64 = 100.0;
const C: [f64; 2] = [0.5, -1.0];
// about seventy-five root steps: every band is rewritten every step of its parent level and the
// coarse-fine transfer every substep of each refined level
const T_FINAL: f64 = 0.16;

fn regions() -> [RefinementRegion<2>; 2] {
    [
        RefinementRegion { x_lo: [0.25, 0.25], x_hi: [0.75, 0.75] },
        RefinementRegion { x_lo: [0.375, 0.375], x_hi: [0.625, 0.625] },
    ]
}

fn atmosphere([x, y]: [f64; 2]) -> Prim<f64, 2> {
    // isentropic, p = rho^gamma at rest: gamma/(gamma-1) rho^(gamma-1) - GM/r is uniform, rho = 1
    // at (0.5, 1)
    let r = ((x - C[0]).powi(2) + (y - C[1]).powi(2)).sqrt();
    let k = GAMMA / (GAMMA - 1.0);
    let cst = k - GM / 2.0;
    let rho = ((cst + GM / r) / k).powf(1.0 / (GAMMA - 1.0));
    Prim::adiabatic(Density(rho), Tensor::new([0.0, 0.0]), Pressure(rho.powf(GAMMA)))
}

fn balanced(s: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &s.geom.allocated).reconstruction(Recon::Plm).well_balanced_reconstruction(true)
}

fn plain(s: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &s.geom.allocated).reconstruction(Recon::Plm).well_balanced_reconstruction(false)
}

fn root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells).spacing([DX; 2]).origin([0.0; 2])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd).cfl(CFL).ghosts(2).timestepping(Timestepping::Rk2)
        .allocate().unwrap().set_initial(atmosphere).build()
}

fn bodies() -> symbi_ib::BodyCollection<f64, 2> {
    symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(0, Tensor::new(C), Tensor::zeros(), GM, 1.0e-3, 0.0))
}

fn single(make: fn(&Sim) -> Kern) -> Hier {
    let r = root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow));
    let k = make(&r);
    let h = Hier::with_refinement(r, k, &regions(), ProlongOrder::Ppm, make).unwrap();
    h.seed_fine_from_coarse().unwrap();
    let mut h = h.with_bodies(bodies());
    h.prime();
    h
}

/// each tile refines every region clipped to its physical range, down to the first region it
/// misses; the refined levels are seeded after each parent level's cut halos are exchanged
fn tiles(partition: &Partition<2>) -> Vec<Hier> {
    let counts = partition.counts();
    let whole = regions();
    let mut tiles = Vec::new();
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m = [ext[0].1, ext[1].1];
        let off = [ext[0].0, ext[1].0];
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let r = root(m, off, bnd);
        let lo: [f64; 2] = std::array::from_fn(|a| off[a] as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| (off[a] + m[a]) as f64 * DX);
        let clipped: Vec<RefinementRegion<2>> = whole
            .iter()
            .map(|w| RefinementRegion {
                x_lo: std::array::from_fn(|a| w.x_lo[a].max(lo[a])),
                x_hi: std::array::from_fn(|a| w.x_hi[a].min(hi[a])),
            })
            .take_while(|w| (0..2).all(|a| w.x_lo[a] < w.x_hi[a]))
            .collect();
        let k = balanced(&r);
        let h = if clipped.is_empty() {
            Hier::single(r, k)
        } else {
            Hier::with_refinement_of(r, k, &clipped, &whole[..clipped.len()], ProlongOrder::Ppm, balanced).unwrap()
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).unwrap();
    tiles.into_iter().map(|h| { let mut h = h.with_bodies(bodies()); h.prime(); h }).collect()
}

/// (level, centroid on that level's lattice) -> interior density bits, over every tile
fn densities(tiles: &[Hier]) -> HashMap<(usize, [i64; 2]), u64> {
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                let x = s.geom.centroid(c);
                let key = (ll, std::array::from_fn(|a| (x[a] * scale * 2.0).round() as i64));
                let previous = out.insert(key, s.fields.cons.den.view().at(c).to_bits());
                assert!(previous.is_none(), "level {ll} cell {key:?} is held by two tiles");
            }
        }
    }
    out
}

fn assert_tiles_match(cuts: [Vec<usize>; 2], want: &HashMap<(usize, [i64; 2]), u64>, steps: u64) {
    let partition = Partition::explicit([N, N], cuts.clone()).unwrap();
    let mut t = tiles(&partition);
    let on_level_2 = t.iter().filter(|h| h.levels.len() > 2).count();
    if cuts.iter().any(|c| c.iter().any(|&x| (12..20).contains(&x))) {
        assert!(on_level_2 >= 2, "cuts {cuts:?}: level 2 lives on {on_level_2} tile(s); no cut crosses it");
    }
    let devices = vec![0; t.len()];
    evolve_hierarchy_decomposed(&mut t, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, |_, _, _| std::ops::ControlFlow::Continue(()));
    for (flat, h) in t.iter().enumerate() {
        assert!(h.crash.is_none(), "cuts {cuts:?}: tile {flat} crashed: {:?}", h.crash);
        assert_eq!(h.levels[0].state.iteration, steps, "cuts {cuts:?}: tile {flat} stopped short");
    }
    let got = densities(&t);
    let mut keys: Vec<_> = got.keys().collect();
    keys.sort();
    let mut want_keys: Vec<_> = want.keys().collect();
    want_keys.sort();
    assert_eq!(keys, want_keys, "cuts {cuts:?}: the tiles hold a different set of level cells");
    let differing: Vec<_> = want.iter().filter(|(k, v)| got[k] != **v).map(|(k, _)| *k).collect();
    if let Some(&(ll, cell)) = differing.iter().min() {
        panic!(
            "cuts {cuts:?}: {} cells differ; first on level {ll} at {cell:?}: {:.12e} against {:.12e}",
            differing.len(),
            f64::from_bits(got[&(ll, cell)]),
            f64::from_bits(want[&(ll, cell)])
        );
    }
}

#[test]
fn balanced_nested_tiles_match_the_single_hierarchy_with_every_band_on_one_tile() {
    let unbalanced = { let mut h = single(plain); h.evolve(T_FINAL).unwrap(); h };
    let reference = { let mut h = single(balanced); h.evolve(T_FINAL).unwrap(); h };
    let steps = reference.levels[0].state.iteration;
    assert!(steps > 50, "the run took {steps} root steps");
    let want = densities(std::slice::from_ref(&reference));
    let off = densities(std::slice::from_ref(&unbalanced));
    let differs = want.iter().any(|(k, v)| k.0 == 2 && off[k] != *v);
    assert!(differs, "the balance left level 2 unchanged; the gate is vacuous");
    // level 1 spans root cells 8..24 and level 2 root cells 12..20; a root cut c falls on level-1
    // cell 2c, so the bands admit cuts at 14..=18 inside the boxes and anything outside 8..=24
    for cuts in [[vec![16], vec![]], [vec![16], vec![16]], [vec![14], vec![18]], [vec![15], vec![17]], [vec![4, 16], vec![28]]] {
        assert_tiles_match(cuts, &want, steps);
    }
}
