// =============================================================================
// decomp_refine_nested.rs
//
// a three-level hierarchy (root, level 1 over [0.25, 0.75]^2, level 2 over
// [0.375, 0.625]^2) whose root is cut at the domain center, so both refined
// levels span the cut. the decomposed hierarchy equals the single hierarchy on
// every level's interior, cell for cell:
// - at step zero, after the decomposed seeding prolongs each level from its
//   parent with the parent's cut halos current;
// - after several root steps, each refined level subcycled with its own halo
//   exchange across the cut and restricted into its parent.
// cells are matched by their physical centroids. the gate asserts its own
// premises: level 2 lives on at least two tiles and carries structure. under
// a viscosity, with a cut that leaves level 2 on one tile, the tile holding
// levels 0 and 1 alone applies the viscous transport where the single
// hierarchy does, on level 2 only. an accreting sink couples to the run's
// finest level alone: a tile holding levels 0 and 1 of a three-level run, and a
// tile holding the root alone of a two-level run, add no reaction, so the
// accreted mass and the force history equal the single hierarchy's bit for bit
// (both arms failed before: the short tile added its own gas's pull). a level-2
// patch face lying on a cut is refused by name.
//
// run: cargo test -p symbi --test decomp_refine_nested
// =============================================================================
use std::collections::HashMap;

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
use symbi_sim::substrate_seam::WithViscosity;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
// about three root steps (the root cfl step is about 0.0057)
const T_FINAL: f64 = 0.015;

fn regions() -> [RefinementRegion<2>; 2] {
    [
        RefinementRegion { x_lo: [0.25, 0.25], x_hi: [0.75, 0.75] },
        RefinementRegion { x_lo: [0.375, 0.375], x_hi: [0.625, 0.625] },
    ]
}

/// a bump centered on the cut crossing, off the diagonal so no symmetry hides a swapped index
fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.07).powi(2) + ((y - 0.48) / 0.09).powi(2))).exp()
}

/// a root tile of `cells` placed as production places it: the whole grid's origin and the index
/// offset of its first cell
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
        .set_initial(|[x, y]| {
            let b = bump(x, y);
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
        })
        .build()
}

const NU: f64 = 2e-3;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

fn kset_viscous(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated).with_viscosity(NU)
}

fn build_single(make: fn(&Sim) -> Kern) -> Hier {
    let root = build_root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow));
    let k = make(&root);
    let mut h = Hier::with_refinement(root, k, &regions(), ProlongOrder::Ppm, make).expect("single hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

/// each tile refines every region clipped to its physical range, down to the first region it
/// misses; the refined levels are seeded after each parent level's cut halos are exchanged
fn build_tiles(partition: &Partition<2>, whole: &[RefinementRegion<2>], make: fn(&Sim) -> Kern) -> symbi_xpu::Result<Vec<Hier>> {
    let counts = partition.counts();
    let mut tiles = Vec::with_capacity(partition.n_tiles());
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m: [usize; 2] = [ext[0].1, ext[1].1];
        let offset: [usize; 2] = [ext[0].0, ext[1].0];
        let lo: [f64; 2] = std::array::from_fn(|a| offset[a] as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| lo[a] + m[a] as f64 * DX);
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let root = build_root(m, offset, bnd);
        let clipped: Vec<RefinementRegion<2>> = whole
            .iter()
            .map(|r| RefinementRegion {
                x_lo: std::array::from_fn(|a| r.x_lo[a].max(lo[a])),
                x_hi: std::array::from_fn(|a| r.x_hi[a].min(hi[a])),
            })
            .take_while(|r| (0..2).all(|a| r.x_lo[a] < r.x_hi[a]))
            .collect();
        let k = make(&root);
        let h = if clipped.is_empty() {
            Hier::single(root, k)
        } else {
            Hier::with_refinement_of(root, k, &clipped, &whole[..clipped.len()], ProlongOrder::Ppm, make)?
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy)?;
    for h in tiles.iter_mut() {
        h.prime();
    }
    Ok(tiles)
}

/// (level, centroid on that level's lattice) -> interior density, over every tile
fn interior_density(tiles: &[Hier]) -> HashMap<(usize, [i64; 2]), f64> {
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                let x = s.geom.centroid(c);
                let key = (ll, std::array::from_fn(|a| (x[a] * scale * 2.0).round() as i64));
                let previous = out.insert(key, *s.fields.cons.den.view().at(c));
                assert!(previous.is_none(), "level {ll} cell {key:?} is held by two tiles");
            }
        }
    }
    out
}

fn assert_levels_match(single: &[Hier], tiles: &[Hier], label: &str) {
    let want = interior_density(single);
    let got = interior_density(tiles);
    assert_eq!(
        {
            let mut k: Vec<_> = got.keys().collect();
            k.sort();
            k
        },
        {
            let mut k: Vec<_> = want.keys().collect();
            k.sort();
            k
        },
        "{label}: the tiles hold a different set of level cells than the single hierarchy"
    );
    let differing: Vec<_> = want.iter().filter(|(k, v)| got[k].to_bits() != v.to_bits()).collect();
    if let Some(((ll, cell), v)) = differing.iter().max_by(|a, b| (got[a.0] - a.1).abs().total_cmp(&(got[b.0] - b.1).abs())) {
        panic!(
            "{label}: {} cells differ; worst on level {ll} at {cell:?}: {:.12e} against {v:.12e}",
            differing.len(),
            got[&(*ll, *cell)]
        );
    }
}

fn assert_premises(tiles: &[Hier], min_level_2_tiles: usize) {
    let on_level_2 = tiles.iter().filter(|h| h.levels.len() > 2).count();
    assert!(on_level_2 >= min_level_2_tiles, "level 2 lives on {on_level_2} tile(s), fewer than {min_level_2_tiles}");
    let level_2: Vec<f64> = interior_density(tiles).into_iter().filter(|((ll, _), _)| *ll == 2).map(|(_, v)| v).collect();
    let spread = level_2.iter().cloned().fold(f64::MIN, f64::max) - level_2.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread > 1e-2, "level 2 carries a density spread of {spread:.3e}; the gate is blind");
}

fn evolve_tiles(tiles: &mut [Hier], partition: &Partition<2>) {
    evolve_hierarchy_decomposed(
        tiles,
        partition.counts(),
        &vec![0; partition.n_tiles()],
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

fn check(cuts: [Vec<usize>; 2]) {
    check_with(cuts, kset, 2);
}

fn check_with(cuts: [Vec<usize>; 2], make: fn(&Sim) -> Kern, min_level_2_tiles: usize) {
    let partition = Partition::explicit([N, N], cuts.clone()).expect("cuts inside the grid");
    let mut single = build_single(make);
    let mut tiles = build_tiles(&partition, &regions(), make).expect("tile hierarchies");
    assert_premises(&tiles, min_level_2_tiles);
    assert_levels_match(std::slice::from_ref(&single), &tiles, &format!("cuts {cuts:?}, seeded"));
    single.evolve(T_FINAL).expect("single evolve");
    evolve_tiles(&mut tiles, &partition);
    assert_eq!(tiles[0].levels[0].state.iteration, single.levels[0].state.iteration, "root step counts differ");
    assert!(single.levels[0].state.iteration >= 2, "fewer than two root steps; the subcycle is barely exercised");
    assert_levels_match(std::slice::from_ref(&single), &tiles, &format!("cuts {cuts:?}, evolved"));
}

#[test]
fn nested_levels_across_an_x_cut_match_the_single_hierarchy() {
    check([vec![16], vec![]]);
}

#[test]
fn nested_levels_across_a_y_cut_match_the_single_hierarchy() {
    check([vec![], vec![16]]);
}

#[test]
fn nested_levels_across_both_cuts_match_the_single_hierarchy() {
    check([vec![16], vec![16]]);
}

#[test]
fn nested_levels_across_uneven_cuts_match_the_single_hierarchy() {
    check([vec![15], vec![17]]);
}

/// the cut at root cell 21 crosses level 1 and leaves level 2 on the first tile, so the second
/// tile holds levels 0 and 1 of a three-level run
#[test]
fn a_tile_short_of_the_finest_level_leaves_the_viscous_transport_to_it() {
    check_with([vec![21], vec![]], kset_viscous, 1);
    let partition = Partition::explicit([N, N], [vec![21], vec![]]).expect("cuts inside the grid");
    let tiles = build_tiles(&partition, &regions(), kset_viscous).expect("tile hierarchies");
    assert_eq!(tiles[1].levels.len(), 2, "the second tile holds a different set of levels");
    let mut inviscid = build_single(kset);
    let mut viscous = build_single(kset_viscous);
    inviscid.evolve(T_FINAL).expect("inviscid evolve");
    viscous.evolve(T_FINAL).expect("viscous evolve");
    let (a, b) = (interior_density(std::slice::from_ref(&inviscid)), interior_density(std::slice::from_ref(&viscous)));
    assert!(
        a.iter().any(|(k, v)| k.0 == 2 && b[k].to_bits() != v.to_bits()),
        "the viscosity left level 2 unchanged; the arm is vacuous"
    );
}

/// a level-2 face on a cut puts the fine faces of its reflux and the level-1 cell they correct
/// on different tiles, and no register contribution crosses a cut on a refined level
#[test]
fn a_level_2_face_on_a_cut_is_refused_by_name() {
    let partition = Partition::explicit([N, N], [vec![16], vec![]]).expect("cuts inside the grid");
    let whole = [
        RefinementRegion { x_lo: [0.25, 0.25], x_hi: [0.75, 0.75] },
        RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] },
    ];
    let err = build_tiles(&partition, &whole, kset).err().expect("a level-2 face on a cut was admitted");
    assert!(err.detail.contains("lies on a decomposition cut"), "the refusal names another reason: {}", err.detail);
}

// an accreting sink at the centre of a three-level run. the single hierarchy holds the full body on
// level 2 and couples it there; every tile holds it on its own deepest level, so a tile short of
// level 2 couples it on level 1. the cut at root cell 21 leaves level 2 on the first tile alone
fn sink() -> symbi_ib::BodyCollection<f64, 2> {
    symbi_ib::BodyCollection::new().add(symbi_ib::Body::black_hole(0, Tensor::new([0.5, 0.48]), Tensor::zeros(), 1.0, 0.03, 0.06, 10.0, 0.5, 0.06))
}

/// the body's accreted mass and its force history, read from the deepest level `h` holds
fn body_record(h: &Hier) -> (f64, Vec<f64>) {
    let im = h.levels.last().unwrap().state.immersed.as_ref().expect("bodies attached");
    let mut accreted = 0.0;
    im.bodies.visit_accretion(|b| {
        if let symbi_ib::BodyKind::BlackHole { total_accreted_mass, .. } = &b.kind {
            accreted += *total_accreted_mass;
        }
    });
    (accreted, im.history.force().to_vec())
}

#[test]
fn a_tile_short_of_the_finest_level_couples_a_nested_sink_as_the_single_hierarchy_does() {
    let partition = Partition::explicit([N, N], [vec![21], vec![]]).expect("cuts inside the grid");
    let mut single = build_single(kset).with_bodies(sink());
    single.prime();
    let mut tiles: Vec<Hier> = build_tiles(&partition, &regions(), kset)
        .expect("tile hierarchies")
        .into_iter()
        .map(|h| {
            let mut h = h.with_bodies(sink());
            h.prime();
            h
        })
        .collect();
    assert_eq!((tiles[0].levels.len(), tiles[1].levels.len()), (3, 2), "premise: the second tile holds levels 0 and 1 alone");
    single.evolve(T_FINAL).expect("single evolve");
    evolve_tiles(&mut tiles, &partition);
    assert_levels_match(std::slice::from_ref(&single), &tiles, "a nested sink, cut at 21");
    let (want_mass, want_force) = body_record(&single);
    assert!(want_mass > 1e-8, "the sink accreted nothing; the gate is vacuous");
    for (k, tile) in tiles.iter().enumerate() {
        let (mass, force) = body_record(tile);
        assert_eq!(mass.to_bits(), want_mass.to_bits(), "tile {k}: accreted {mass:e} against {want_mass:e}");
        assert_eq!(force.len(), want_force.len(), "tile {k}: force history length");
        for (ii, (a, b)) in force.iter().zip(&want_force).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "tile {k}: force entry {ii} is {a:e} against {b:e}");
        }
    }
}

// the same sink on a two-level run whose first tile, cut at root cell 6, holds the root alone
#[test]
fn an_unrefined_tile_adds_no_reaction_to_a_refined_sink() {
    let partition = Partition::explicit([N, N], [vec![6], vec![]]).expect("cuts inside the grid");
    let one = [regions()[0]];
    let root = build_root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow));
    let k = kset(&root);
    let single = Hier::with_refinement(root, k, &one, ProlongOrder::Ppm, kset).expect("single hierarchy");
    single.seed_fine_from_coarse().expect("seed fine");
    let mut single = single.with_bodies(sink());
    single.prime();
    let mut tiles: Vec<Hier> = build_tiles(&partition, &one, kset)
        .expect("tile hierarchies")
        .into_iter()
        .map(|h| {
            let mut h = h.with_bodies(sink());
            h.prime();
            h
        })
        .collect();
    assert_eq!((tiles[0].levels.len(), tiles[1].levels.len()), (1, 2), "premise: the first tile holds the root alone");
    single.evolve(T_FINAL).expect("single evolve");
    evolve_tiles(&mut tiles, &partition);
    assert_levels_match(std::slice::from_ref(&single), &tiles, "a refined sink, cut at 6");
    let (want_mass, want_force) = body_record(&single);
    assert!(want_mass > 1e-8 && want_force.iter().any(|f| f.abs() > 1e-3), "premise: the sink accreted and felt a force");
    for (k, tile) in tiles.iter().enumerate() {
        let (mass, force) = body_record(tile);
        assert_eq!(mass.to_bits(), want_mass.to_bits(), "tile {k}: accreted {mass:e} against {want_mass:e}");
        assert_eq!(force.len(), want_force.len(), "tile {k}: force history length");
        for (ii, (a, b)) in force.iter().zip(&want_force).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "tile {k}: force entry {ii} is {a:e} against {b:e}");
        }
    }
}
