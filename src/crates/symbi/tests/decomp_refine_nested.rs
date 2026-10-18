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
// patch face lying on a cut is refused by name on a build carrying no register
// unit across cuts; on one that does, under a root halo deep enough for the
// nested prolongation reach, the face's units cross the level-1 cut to the tile
// holding the cells they correct, which refluxes and recovers them before the
// level's halos cross, and the tiles equal the single hierarchy in every
// conserved component, also after a rejected level-2 stage. under the
// production halo the same face is refused by name on a build gathering no
// parent stencil; on one that does, the face's fine ghosts prolong from a
// level-1 prim stencil gathered across the cut, the neighbor's coarse-fine
// ghosts included, and the primed face ghosts and every conserved component
// after several root steps equal the single hierarchy's.
//
// run: cargo test -p symbi --test decomp_refine_nested
// =============================================================================
use std::collections::HashMap;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::launched::{nested_partitions, nested_receiving_faces_of_tile, receiving_faces_of_tile};
use symbi_sim::atlas::TileId;
use symbi::sim::refinement::{
    CutFaces, Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, prime_decomposed_levels,
    seed_decomposed_fine_from_coarse,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_discretize::Recon;
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
    build_root_with(cells, offset, bnd, 2)
}

/// `build_root` with `ng` ghost cells
fn build_root_with(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, ng: usize) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .ghosts(ng)
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
    let devices = vec![0; tiles.len()];
    prime_decomposed_levels(&mut tiles, counts, &devices, &LocalCopy).expect("decomposed prime");
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

/// a root halo of five cells: a nested level's ghosts (six) reach five parent cells past a face, inside
/// the parent level's halo of six
const NG_DEEP: usize = 5;

/// the numerics of a run: the root halo, the kernel set every level carries and the coarse-fine
/// prolongation, as production pairs them (plm: ng 2, ppm prolongation; ppm: ng 3, quartic)
#[derive(Clone, Copy)]
struct Scheme {
    ng: usize,
    make: fn(&Sim) -> Kern,
    prolong: ProlongOrder,
}

/// plm reconstruction under a root halo of `ng`
fn plm(ng: usize) -> Scheme {
    Scheme { ng, make: kset, prolong: ProlongOrder::Ppm }
}

fn kset_ppm(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated).reconstruction(Recon::Ppm)
}

/// ppm reconstruction under its production root halo of three, prolonged by the quartic transfer
const PPM: Scheme = Scheme { ng: 3, make: kset_ppm, prolong: ProlongOrder::Quartic };

/// every refined level's patch in its parent's cells: level 1 in root cells, then each deeper patch
/// from its parent patch's origin, region `k` refining level `k` of cells `DX / 2^k`
fn nested_patches(whole: &[RefinementRegion<2>]) -> Vec<[(usize, usize); 2]> {
    let cell = |x: f64, level: usize| (x / DX * (1usize << level) as f64).round() as usize;
    (0..whole.len())
        .map(|k| {
            std::array::from_fn(|a| {
                let origin = if k == 0 { 0 } else { cell(whole[k - 1].x_lo[a], k) };
                (cell(whole[k].x_lo[a], k) - origin, cell(whole[k].x_hi[a], k) - origin)
            })
        })
        .collect()
}

/// the single hierarchy over `whole` under `scheme`
fn build_single_over(whole: &[RefinementRegion<2>], scheme: Scheme) -> Hier {
    let root = build_root_with([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow), scheme.ng);
    let k = (scheme.make)(&root);
    let mut h = Hier::with_refinement(root, k, whole, scheme.prolong, scheme.make).expect("single hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

/// tiles carrying register units across cuts under `scheme`: a nested patch face on a cut of its
/// parent level contributes its units to the parent-level tile across the cut, which holds no finer
/// level and carries the receiving register of the cells they correct; the root register carries
/// the level-1 faces on root cuts the same way. a `donate` build gathers the parent cells such a
/// face prolongs from past its parent's halo into a prim stencil, primed once every tile primed
fn build_tiles_contributing(partition: &Partition<2>, whole: &[RefinementRegion<2>], scheme: Scheme, donate: bool) -> symbi_xpu::Result<Vec<Hier>> {
    let counts = partition.counts();
    let patches = nested_patches(whole);
    let p1 = patches[0];
    let fine = nested_partitions(partition, &patches).expect("level partitions");
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
        let root = build_root_with(m, offset, bnd, scheme.ng);
        let clipped: Vec<RefinementRegion<2>> = whole
            .iter()
            .map(|r| RefinementRegion {
                x_lo: std::array::from_fn(|a| r.x_lo[a].max(lo[a])),
                x_hi: std::array::from_fn(|a| r.x_hi[a].min(hi[a])),
            })
            .take_while(|r| (0..2).all(|a| r.x_lo[a] < r.x_hi[a]))
            .collect();
        let k = (scheme.make)(&root);
        let mut h = if clipped.is_empty() {
            match receiving_faces_of_tile(partition, p1, TileId(flat as u32)) {
                Some((p, f)) => Hier::single(root, k).with_receiving_register(&p, f).expect("root receiving register"),
                None => Hier::single(root, k),
            }
        } else if donate {
            Hier::with_refinement_of_donating(root, k, &clipped, &whole[..clipped.len()], scheme.prolong, scheme.make)?
        } else {
            Hier::with_refinement_of_contributing(root, k, &clipped, &whole[..clipped.len()], scheme.prolong, scheme.make)?
        };
        // a tile whose finest level lies beside a finer patch face on a cut of that level
        let finest = h.levels.len() - 1;
        if finest >= 1 && finest < patches.len() {
            if let Some((p, f)) = nested_receiving_faces_of_tile(partition, &fine, &patches, TileId(flat as u32), finest) {
                h = h.with_receiving_register_at(finest, &p, f).expect("nested receiving register");
            }
        }
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("decomposed fine seed");
    for h in tiles.iter_mut() {
        h.prime();
    }
    prime_decomposed_levels(&mut tiles, counts, &vec![0; partition.n_tiles()], &LocalCopy).expect("decomposed prime");
    Ok(tiles)
}

/// (level, centroid, component) -> bits of every conserved component of every interior cell
fn interior_conserved(tiles: &[Hier]) -> HashMap<(usize, [i64; 2], usize), u64> {
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let cons = &s.fields.cons;
            let fields = [&cons.den, &cons.mom[0], &cons.mom[1], cons.nrg_field().expect("an energy")];
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                let x = s.geom.centroid(c);
                let at: [i64; 2] = std::array::from_fn(|a| (x[a] * scale * 2.0).round() as i64);
                for (q, f) in fields.iter().enumerate() {
                    let previous = out.insert((ll, at, q), f.view().at(c).to_bits());
                    assert!(previous.is_none(), "level {ll} cell {at:?} is held by two tiles");
                }
            }
        }
    }
    out
}

/// (centroid, component) -> bits of the level-2 primitives in the ghost layers across the faces of the
/// whole level-2 patch `whole`: the cells outside it along one axis and beside it along the other
fn face_ghost_prims(tiles: &[Hier], whole: &RefinementRegion<2>) -> HashMap<([i64; 2], usize), u64> {
    face_ghost_prims_of(tiles, whole, 2)
}

/// `face_ghost_prims` of level `level` around its whole patch `whole`
fn face_ghost_prims_of(tiles: &[Hier], whole: &RefinementRegion<2>, level: usize) -> HashMap<([i64; 2], usize), u64> {
    let mut out = HashMap::new();
    for h in tiles.iter().filter(|h| h.levels.len() > level) {
        let s = &h.levels[level].state;
        let prim = &s.fields.prim;
        let fields = [&prim.rho, &prim.vel[0], &prim.vel[1], prim.pre.as_ref().expect("a pressure")];
        for c in s.geom.allocated.iter() {
            let x = s.geom.centroid(c);
            let inside: [bool; 2] = std::array::from_fn(|a| whole.x_lo[a] < x[a] && x[a] < whole.x_hi[a]);
            if inside[0] == inside[1] {
                continue;
            }
            let at: [i64; 2] = std::array::from_fn(|a| (x[a] * (N << level) as f64 * 2.0).round() as i64);
            for (q, f) in fields.iter().enumerate() {
                out.insert((at, q), f.view().at(c).to_bits());
            }
        }
    }
    out
}

/// a level-2 patch face on a level-1 cut: the single hierarchy and the tiles carrying the units
/// across the cut, the tiles rejecting the level-2 stage `reject` once as the single hierarchy does,
/// equal in every conserved component of every level after several root steps. under a root halo
/// of `ng` short of the nested prolongation reach the face prolongs from a gathered prim stencil
fn assert_nested_face_on_cut_matches(cuts: [Vec<usize>; 2], whole: [RefinementRegion<2>; 2], reject: Option<(u64, usize)>, ng: usize) {
    let partition = Partition::explicit([N, N], cuts.clone()).expect("cuts inside the grid");
    let mut single = build_single_over(&whole, plm(ng));
    let donate = ng < NG_DEEP;
    let mut tiles = build_tiles_contributing(&partition, &whole, plm(ng), donate).expect("tile hierarchies");
    let stencils = tiles.iter().filter(|h| h.prim_stencil_at(2).is_some()).count();
    assert_eq!(stencils > 0, donate, "cuts {cuts:?}, ng {ng}: {stencils} level-2 prim stencils");
    // the primed ghosts across the level-2 faces, those on the cut prolonged from the stencil or
    // from level 1's crossed cut halo: every one equals the single hierarchy's bit for bit
    let (want, got) = (face_ghost_prims(std::slice::from_ref(&single), &whole[1]), face_ghost_prims(&tiles, &whole[1]));
    assert!(!got.is_empty(), "cuts {cuts:?}: no level-2 face ghost compared");
    let differing: Vec<_> = got.iter().filter(|(k, v)| want.get(k) != Some(v)).map(|(k, _)| *k).collect();
    assert!(differing.is_empty(), "cuts {cuts:?}: {} primed level-2 face ghosts differ, first {:?}", differing.len(), differing.iter().min());
    let contributing = tiles.iter().any(|h| h.flux_registers.get(1).is_some_and(|r| r.has_contributing()));
    let receiving = tiles.iter().any(|h| h.levels.len() == 2 && h.flux_registers.get(1).is_some_and(|r| r.has_receiving()));
    assert!(contributing && receiving, "cuts {cuts:?}: no level-2 face contributes across a level-1 cut; the gate is vacuous");
    if let Some((substep, stage)) = reject {
        single.reject_level_once_at(0, 2, substep, stage);
        for h in tiles.iter_mut().filter(|h| h.levels.len() > 2) {
            h.reject_level_once_at(0, 2, substep, stage);
        }
    }
    single.evolve(T_FINAL).expect("single evolve");
    evolve_tiles(&mut tiles, &partition);
    assert!(!single.forced_fine_rejection_pending(), "the single hierarchy never reached the forced rejection");
    for (i, h) in tiles.iter().enumerate() {
        assert!(h.crash.is_none(), "cuts {cuts:?}: tile {i} crashed: {:?}", h.crash);
        assert!(!h.forced_fine_rejection_pending(), "cuts {cuts:?}: tile {i} never reached the forced rejection");
    }
    assert!(single.levels[0].state.iteration >= 2, "fewer than two root steps");
    let (want, got) = (interior_conserved(std::slice::from_ref(&single)), interior_conserved(&tiles));
    assert_eq!(got.len(), want.len(), "cuts {cuts:?}: the tiles hold a different set of cells");
    let differing: Vec<_> = want.iter().filter(|(k, v)| got.get(k) != Some(v)).map(|(k, _)| *k).collect();
    assert!(differing.is_empty(), "cuts {cuts:?}: {} values differ, first {:?}", differing.len(), differing.iter().min());
}

/// level 2 over [0.5, 0.625] x [0.375, 0.625]: its low x face lies on the cut at x = 0.5, inside level 1
#[test]
fn a_level_2_face_on_a_level_1_cut_matches_the_single_hierarchy() {
    let low = [regions()[0].clone(), RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] }];
    assert_nested_face_on_cut_matches([vec![16], vec![]], low, None, NG_DEEP);
    // the high x face on the cut
    let high = [regions()[0].clone(), RefinementRegion { x_lo: [0.375, 0.375], x_hi: [0.5, 0.625] }];
    assert_nested_face_on_cut_matches([vec![16], vec![]], high, None, NG_DEEP);
    // the low x face and the high y face on the cuts of both axes
    let corner = [regions()[0].clone(), RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.5] }];
    assert_nested_face_on_cut_matches([vec![16], vec![16]], corner, None, NG_DEEP);
}

/// the same faces under the production halo (two root ghost cells, three on level 1), the level-2
/// fine ghosts prolonged from a level-1 prim stencil gathered across the cut
#[test]
fn a_level_2_face_on_a_level_1_cut_matches_through_the_prim_stencil_at_the_production_halo() {
    let low = [regions()[0].clone(), RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] }];
    assert_nested_face_on_cut_matches([vec![16], vec![]], low, None, 2);
    let high = [regions()[0].clone(), RefinementRegion { x_lo: [0.375, 0.375], x_hi: [0.5, 0.625] }];
    assert_nested_face_on_cut_matches([vec![16], vec![]], high, None, 2);
    let corner = [regions()[0].clone(), RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.5] }];
    assert_nested_face_on_cut_matches([vec![16], vec![16]], corner, None, 2);
    // level 1 ending two of its cells before the cut: the stencil's four level-1 cells past the face
    // reach two of the neighbor's coarse-fine ghosts
    let shallow = [
        RefinementRegion { x_lo: [0.46875, 0.25], x_hi: [0.75, 0.75] },
        RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] },
    ];
    assert_nested_face_on_cut_matches([vec![16], vec![]], shallow, None, 2);
}

/// a level-2 stage rejected in its second substep after the first substep's units folded across the
/// level-1 cut: the replay starts from clean registers and matches the single hierarchy rejected alike
#[test]
fn a_level_2_face_on_a_level_1_cut_matches_after_a_rejected_level_2_stage() {
    let low = [regions()[0].clone(), RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] }];
    assert_nested_face_on_cut_matches([vec![16], vec![]], low.clone(), Some((1, 1)), NG_DEEP);
    assert_nested_face_on_cut_matches([vec![16], vec![]], low, Some((1, 1)), 2);
}

/// under the production halo (two root ghost cells, three on level 1) a level-2 face on a level-1 cut
/// prolongs its fine ghosts four level-1 cells past the face, one past level 1's halo: refused by name
#[test]
fn a_level_2_face_on_a_level_1_cut_under_the_production_halo_is_refused_by_name() {
    let partition = Partition::explicit([N, N], [vec![16], vec![]]).expect("cuts inside the grid");
    let low = [regions()[0].clone(), RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] }];
    let err = build_tiles_contributing(&partition, &low, plm(2), false).err().expect("a level-2 face on a level-1 cut was admitted");
    assert!(
        err.detail.contains("lies on a cut of level 1 and its prolongation stencil reaches 1 beyond that level's halo of 3 cells"),
        "the refusal names another reason: {}",
        err.detail
    );
}

/// level 1 spanning the wrapping x axis of a two-tile ring, level 2's low x face on level 1's cut at
/// x = 0.5: under the production halo the face's stencil would gather level 1 across the seam,
/// refused by name; under a halo holding the reach the same layout builds
#[test]
fn a_level_2_stencil_under_a_level_spanning_a_wrapping_axis_is_refused_by_name() {
    let whole = [
        RefinementRegion { x_lo: [0.0, 0.25], x_hi: [1.0, 0.75] },
        RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] },
    ];
    let clipped = [RefinementRegion { x_lo: [0.5, 0.25], x_hi: [1.0, 0.75] }, whole[1].clone()];
    let build = |ng: usize| {
        let bnd = Boundaries([[BoundaryType::CoarseFine; 2], [BoundaryType::Outflow; 2]]);
        let root = build_root_with([N / 2, N], [N / 2, 0], bnd, ng);
        let k = kset(&root);
        Hier::with_refinement_placed(root, k, &clipped, &whole, ProlongOrder::Ppm, kset, [N, N], [true, false], CutFaces::Donate)
    };
    let deep = build(NG_DEEP).expect("the layout under a halo holding the reach");
    assert!(deep.prim_stencil_at(2).is_none() && deep.levels.len() == 3, "the deep-halo control gathers a stencil");
    let err = build(2).err().expect("a level-2 stencil under a level spanning a wrapping axis was admitted");
    assert!(
        err.detail.contains("a level-2 refinement patch face lies on a cut of level 1") && err.detail.contains("spans a wrapping axis"),
        "the refusal names another reason: {}",
        err.detail
    );
}

/// the cells two maps disagree on: every key of `got` absent from `want` or holding other bits, and
/// every key of `want` absent from `got` when `whole` is set
fn disagreeing<K: std::hash::Hash + Eq + Copy + Ord + std::fmt::Debug>(want: &HashMap<K, u64>, got: &HashMap<K, u64>, whole: bool) -> Vec<K> {
    let mut out: Vec<K> = got.iter().filter(|(k, v)| want.get(k) != Some(v)).map(|(k, _)| *k).collect();
    if whole {
        out.extend(want.keys().filter(|k| !got.contains_key(k)).copied());
    }
    out.sort();
    out
}

/// a nested patch face on a cut of its parent level under `scheme`, the deepest region `whole`'s
/// last: the single hierarchy and the tiles carrying the units across the cut through a prim
/// stencil, both rejecting the deepest level's stage `reject` (substep, stage) once. returns the
/// cells that differ in each comparison: the interior conserved state at step zero, the deepest
/// level's face ghosts at step zero, the interior conserved state after several root steps, and
/// the face ghosts then. the premises are asserted: a stencil and a contributing face on the
/// deepest level, and a receiving tile holding the parent level and none of the deepest
fn characterize(cuts: [Vec<usize>; 2], whole: &[RefinementRegion<2>], scheme: Scheme, reject: Option<(u64, usize)>) -> Vec<(&'static str, usize, String)> {
    let deepest = whole.len();
    let partition = Partition::explicit([N, N], cuts.clone()).expect("cuts inside the grid");
    let mut single = build_single_over(whole, scheme);
    let mut tiles = build_tiles_contributing(&partition, whole, scheme, true).expect("tile hierarchies");
    // the deepest level's face reaches ceil((ng + 1) / 2) + the stencil half-width parent cells past
    // it (ng = the refined levels' halo, one above the root's); past the parent's halo of ng it reads
    // a gathered prim stencil, within it the parent's exchanged halo
    let refined_ng = scheme.ng + 1;
    let reach = refined_ng.div_ceil(2) + scheme.prolong.ghost_width();
    let stencil = tiles.iter().any(|h| h.prim_stencil_at(deepest).is_some());
    assert_eq!(stencil, reach > refined_ng, "cuts {cuts:?}: reach {reach} past a parent halo of {refined_ng}, stencil {stencil}");
    let contributing = tiles.iter().any(|h| h.flux_registers.get(deepest - 1).is_some_and(|r| r.has_contributing()));
    let receiving = tiles.iter().any(|h| h.levels.len() == deepest && h.flux_registers.get(deepest - 1).is_some_and(|r| r.has_receiving()));
    assert!(contributing && receiving, "cuts {cuts:?}: no level-{deepest} face contributes to a level-{} tile holding none of level {deepest}", deepest - 1);
    let mut report = Vec::new();
    let mut compare = |label: &'static str, cells: Vec<String>| {
        report.push((label, cells.len(), cells.first().cloned().unwrap_or_default()));
    };
    compare("initial interior", fmt(disagreeing(&interior_conserved(std::slice::from_ref(&single)), &interior_conserved(&tiles), true)));
    let ghosts = |t: &[Hier]| face_ghost_prims_of(t, &whole[deepest - 1], deepest);
    let tile_ghosts = ghosts(&tiles);
    assert!(!tile_ghosts.is_empty(), "cuts {cuts:?}: no level-{deepest} face ghost compared");
    compare("initial face ghosts", fmt(disagreeing(&ghosts(std::slice::from_ref(&single)), &tile_ghosts, false)));
    if let Some((substep, stage)) = reject {
        single.reject_level_once_at(0, deepest, substep, stage);
        for h in tiles.iter_mut().filter(|h| h.levels.len() > deepest) {
            h.reject_level_once_at(0, deepest, substep, stage);
        }
    }
    single.evolve(T_FINAL).expect("single evolve");
    evolve_tiles(&mut tiles, &partition);
    assert!(!single.forced_fine_rejection_pending(), "the single hierarchy never reached the forced rejection");
    for (i, h) in tiles.iter().enumerate() {
        assert!(h.crash.is_none(), "cuts {cuts:?}: tile {i} crashed: {:?}", h.crash);
        assert!(!h.forced_fine_rejection_pending(), "cuts {cuts:?}: tile {i} never reached the forced rejection");
    }
    assert!(single.levels[0].state.iteration >= 2, "fewer than two root steps");
    compare("evolved interior", fmt(disagreeing(&interior_conserved(std::slice::from_ref(&single)), &interior_conserved(&tiles), true)));
    compare("evolved face ghosts", fmt(disagreeing(&ghosts(std::slice::from_ref(&single)), &ghosts(&tiles), false)));
    report
}

fn fmt<K: std::fmt::Debug>(v: Vec<K>) -> Vec<String> {
    v.into_iter().map(|k| format!("{k:?}")).collect()
}

fn assert_characterized(label: &str, report: Vec<(&'static str, usize, String)>) {
    assert!(report.iter().all(|(_, n, _)| *n == 0), "{label}: {report:#?}");
}

/// ppm under its production halo (root ng 3, quartic prolongation): a level-2 face on a level-1 cut
/// reaches four level-1 cells past the face, inside level 1's halo of four, so its fine ghosts
/// prolong from level 1's exchanged halo
#[test]
fn a_ppm_level_2_face_on_a_level_1_cut_characterized() {
    let level_1 = regions()[0].clone();
    for (label, cuts, level_2) in [
        ("low", [vec![16], vec![]], RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] }),
        ("high", [vec![16], vec![]], RefinementRegion { x_lo: [0.375, 0.375], x_hi: [0.5, 0.625] }),
        ("corner", [vec![16], vec![16]], RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.5] }),
    ] {
        assert_characterized(&format!("ppm {label}"), characterize(cuts, &[level_1.clone(), level_2], PPM, None));
    }
    let low = RefinementRegion { x_lo: [0.5, 0.375], x_hi: [0.625, 0.625] };
    assert_characterized("ppm low, rejected level-2 stage", characterize([vec![16], vec![]], &[level_1, low], PPM, Some((1, 1))));
}

/// a level-3 face on a level-2 cut under the plm production halo: level 2 over [0.375, 0.625]^2
/// spans the root cut at x = 0.5, level 3 has a face on it, and the level-2 tile across the cut holds
/// no level-3 tile and receives the face's units
#[test]
fn a_level_3_face_on_a_level_2_cut_characterized() {
    let level_1 = regions()[0].clone();
    let level_2 = RefinementRegion { x_lo: [0.375, 0.375], x_hi: [0.625, 0.625] };
    for (label, cuts, level_3) in [
        ("low", [vec![16], vec![]], RefinementRegion { x_lo: [0.5, 0.4375], x_hi: [0.5625, 0.5625] }),
        ("high", [vec![16], vec![]], RefinementRegion { x_lo: [0.4375, 0.4375], x_hi: [0.5, 0.5625] }),
        ("corner", [vec![16], vec![16]], RefinementRegion { x_lo: [0.5, 0.4375], x_hi: [0.5625, 0.5] }),
    ] {
        assert_characterized(&format!("level 3 {label}"), characterize(cuts, &[level_1.clone(), level_2.clone(), level_3], plm(2), None));
    }
    let low = RefinementRegion { x_lo: [0.5, 0.4375], x_hi: [0.5625, 0.5625] };
    assert_characterized("level 3 low, rejected level-3 stage", characterize([vec![16], vec![]], &[level_1, level_2, low], plm(2), Some((1, 1))));
}

/// (level, centroid) -> (density bits, pressure bits) of every cell a tile holds of each refined
/// level, its interior when `interior`, its ghosts otherwise
fn level_cells(tiles: &[Hier], interior: bool) -> Vec<((usize, [i64; 2]), (u64, u64))> {
    let mut out = Vec::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate().skip(1) {
            let s = &level.state;
            for c in s.geom.allocated.iter().filter(|c| s.geom.interior.contains(*c) == interior) {
                let x = s.geom.centroid(c);
                let at: [i64; 2] = std::array::from_fn(|a| (x[a] * (N << ll) as f64 * 2.0).round() as i64);
                let pre = s.fields.prim.pre.as_ref().expect("a pressure");
                out.push(((ll, at), (s.fields.prim.rho.view().at(c).to_bits(), pre.view().at(c).to_bits())));
            }
        }
    }
    out
}

/// after the decomposed prime every refined level's cut ghosts hold the neighbor's interior, bit for
/// bit, under every root halo: a level primes from its parent's crossed cut halo. under a root halo
/// of three or more a tile's own prime prolongs the parent cells it holds into the cut-side ghosts,
/// whose pressure differs from the neighbor's conservatively seeded interior in the last places
#[test]
fn primed_cut_ghosts_hold_the_neighbors_interior() {
    let partition = Partition::explicit([N, N], [vec![16], vec![16]]).expect("cuts inside the grid");
    let three = [regions()[0].clone(), RefinementRegion { x_lo: [0.375, 0.375], x_hi: [0.625, 0.625] }, RefinementRegion { x_lo: [0.4375, 0.4375], x_hi: [0.5625, 0.5625] }];
    for ng in [2, 3, 4, 5] {
        let tiles = build_tiles_contributing(&partition, &three, plm(ng), true).expect("tile hierarchies");
        let interior: HashMap<_, _> = level_cells(&tiles, true).into_iter().collect();
        let ghosts: Vec<_> = level_cells(&tiles, false).into_iter().filter(|(k, _)| interior.contains_key(k)).collect();
        assert!(ghosts.iter().any(|((ll, _), _)| *ll == 3), "ng {ng}: no level-3 cut ghost compared; the gate is vacuous");
        let differing: Vec<_> = ghosts.iter().filter(|(k, v)| interior[k] != *v).map(|(k, _)| *k).collect();
        assert!(differing.is_empty(), "ng {ng}: {} of {} cut ghosts differ from the neighbor's interior, first {:?}", differing.len(), ghosts.len(), differing.first());
    }
}
