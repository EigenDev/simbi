// =============================================================================
// decomp_refine_mhd_periodic_span.rs
//
// a refined patch spanning a periodic axis that is cut into magnetized host
// tiles. the root wraps along x and the patch covers all of x over a band in y,
// so it has no face along x: its fine level, its staggered faces included,
// wraps across the tiles at the seam as its root does. a 2.5d newtonian MHD
// flow along x carries gas and a divergence-free field periodic in x through
// the seam on both levels. two tiles cut at root cell 16 and three uneven
// tiles reach the single hierarchy's conserved state and cell field on every
// level bit for bit, and every staggered face. the gate asserts its premises:
// the fine density moves, the fine field evolves, and the same run without a
// field leaves a measurably different gas. a second level nested in the
// spanning patch, clear of the periodic faces with the cuts through it,
// reaches the single hierarchy on the same tiles, and so does a second level
// spanning the periodic axis too, wrapping across the tiles at the seam. the
// launched driver's parent face stencil plan for that second level lands
// inside the stencil fields the builder holds, and a plan whose box widens the
// spanned side by a plane is refused by the pre-exchange check, before any
// value moves.
//
// run: cargo test -p symbi --test decomp_refine_mhd_periodic_span
// =============================================================================
use std::collections::HashMap;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::decomp::{FinePartition, LocalCopy, Partition, Topology, flatten, unflatten};
use symbi::sim::refinement::launched::face_stencil_boxes_of_level_tile;
use symbi_sim::atlas::{ExchangePlan, Placement, Span, TileId};
use symbi_sim::phase_table::PhaseTable;
use symbi_sim::plan_exchange::PlanExchange;
use symbi_fabric::WorkerId;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;
type Store = FieldStore<2, 3, HostMemory, f64>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.04;
/// the field strength against the unit gas pressure: beta ~ 5
const B_SCALE: f64 = 0.6;
// the patch spans all of x over root cells 12..20 in y
const PY_LO: f64 = 0.375;
const PY_HI: f64 = 0.625;

// face-normal components each independent of their own axis: discretely divergence-free, and
// periodic in x
fn bx(scale: f64, y: f64) -> f64 {
    scale * (0.4 + 0.2 * (2.0 * std::f64::consts::PI * y + 0.3).sin())
}

fn by(scale: f64, x: f64) -> f64 {
    scale * (0.2 + 0.3 * (2.0 * std::f64::consts::PI * x + 0.7).cos())
}

/// a density bump straddling the seam at x = 0 carried along x through it
fn gas(x: [f64; 2]) -> Prim<f64, 3> {
    let dx = (x[0] - 0.05).rem_euclid(1.0).min((0.05 - x[0]).rem_euclid(1.0));
    let bump = 0.5 * (-((dx / 0.08).powi(2) + ((x[1] - 0.5) / 0.08).powi(2))).exp();
    Prim::adiabatic(Density(1.0 + bump), Tensor::new([0.6, 0.1 * (2.0 * std::f64::consts::PI * x[0]).sin(), 0.0]), Pressure(1.0))
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
}

/// a root tile placed as production places it, seeding the faces of its own lattice
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, scale: f64) -> Sim {
    build_root_shifted(cells, offset, bnd, scale, 0.0)
}

/// `build_root` of the state translated by `shift` along the periodic x
fn build_root_shifted(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, scale: f64, shift: f64) -> Sim {
    let moved = |x: [f64; 2]| [(x[0] - shift).rem_euclid(1.0), x[1]];
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|x| {
            let x = moved(x);
            MhdPrim::new(gas(x), Tensor::new([bx(scale, x[1]), by(scale, x[0]), 0.1 * scale]))
        })
        .seed_faces(|d, x| {
            let inside = (0..2).all(|a| {
                let (lo, hi) = (offset[a] as f64 * DX, (offset[a] + cells[a]) as f64 * DX);
                let eps = 1e-9 * DX;
                if a == d { x[a] >= lo - eps && x[a] <= hi + eps } else { x[a] > lo && x[a] < hi }
            });
            let x = moved(x);
            match (inside, d) {
                (false, _) => 0.0,
                (true, 0) => bx(scale, x[1]),
                (true, _) => by(scale, x[0]),
            }
        })
        .build()
}

fn whole() -> RefinementRegion<2> {
    RefinementRegion { x_lo: [0.0, PY_LO], x_hi: [1.0, PY_HI] }
}

fn build_single(scale: f64) -> Hier {
    build_single_shifted(scale, 0.0)
}

/// the nested level clear of the periodic faces over root cells 6..26 in x
fn nested_clear() -> RefinementRegion<2> {
    RefinementRegion { x_lo: [0.1875, 0.4375], x_hi: [0.8125, 0.5625] }
}

fn build_single_shifted(scale: f64, shift: f64) -> Hier {
    build_single_over(scale, shift, &[whole()])
}

fn build_single_over(scale: f64, shift: f64, regions: &[RefinementRegion<2>]) -> Hier {
    let bnd = Boundaries([[BoundaryType::Periodic; 2], [BoundaryType::Outflow; 2]]);
    let root = build_root_shifted([N, N], [0, 0], bnd, scale, shift);
    let k = kset(&root);
    let mut h = Hier::with_refinement(root, k, regions, ProlongOrder::Ppm, kset).expect("single hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    h.prime();
    h
}

/// root tiles cut along x alone: the seam and the cuts are coarse-fine sides to each tile, the
/// root wraps along x, and every tile refines its share of each region down to the first region
/// it holds no cell of, as the production tiles clip them
fn build_tiles(partition: &Partition<2>, regions: &[RefinementRegion<2>]) -> Vec<Hier> {
    let counts = partition.counts();
    let mut tiles: Vec<Hier> = (0..partition.n_tiles())
        .map(|flat| {
            let ext = partition.tile_extents(unflatten(flat, counts));
            let bnd = Boundaries([[BoundaryType::CoarseFine; 2], [BoundaryType::Outflow; 2]]);
            let root = build_root([ext[0].1, ext[1].1], [ext[0].0, ext[1].0], bnd, B_SCALE);
            let k = kset(&root);
            let (lo, hi) = (ext[0].0 as f64 * DX, (ext[0].0 + ext[0].1) as f64 * DX);
            let mut clipped = Vec::new();
            for (k, r) in regions.iter().enumerate() {
                let (c_lo, c_hi) = (r.x_lo[0].max(lo), r.x_hi[0].min(hi));
                if c_hi - c_lo <= 0.5 * DX * 0.5f64.powi(k as i32) {
                    break;
                }
                clipped.push(RefinementRegion { x_lo: [c_lo, r.x_lo[1]], x_hi: [c_hi, r.x_hi[1]] });
            }
            let whole_patches = &regions[..clipped.len()];
            Hier::with_refinement_placed(root, k, &clipped, whole_patches, ProlongOrder::Ppm, kset, [N, N], [true, false], symbi::sim::refinement::CutFaces::Contribute)
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
fn interior_state(tiles: &[Hier]) -> HashMap<(&'static str, usize, [i64; 2]), u64> {
    let picks: [(&'static str, fn(&Store, [isize; 2]) -> f64); 8] = [
        ("den", |s, c| *s.fields.cons.den.view().at(c)),
        ("mom0", |s, c| *s.fields.cons.mom[0].view().at(c)),
        ("mom1", |s, c| *s.fields.cons.mom[1].view().at(c)),
        ("mom2", |s, c| *s.fields.cons.mom[2].view().at(c)),
        ("nrg", |s, c| *s.fields.cons.nrg_field().unwrap().view().at(c)),
        ("bcell0", |s, c| *s.fields.mhd.as_ref().unwrap().bcell[0].view().at(c)),
        ("bcell1", |s, c| *s.fields.mhd.as_ref().unwrap().bcell[1].view().at(c)),
        ("bcell2", |s, c| *s.fields.mhd.as_ref().unwrap().bcell[2].view().at(c)),
    ];
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                let x = s.geom.centroid(c);
                let at: [i64; 2] = std::array::from_fn(|a| (x[a] * scale * 2.0).round() as i64);
                for (name, pick) in &picks {
                    let previous = out.insert((*name, ll, at), pick(s, [c[0], c[1]]).to_bits());
                    assert!(previous.is_none(), "{name} on level {ll} at {at:?} is held by two tiles");
                }
            }
        }
    }
    out
}

/// (axis, level, face position on twice that level's lattice) -> bits, over every face bounding an
/// interior cell of every tile; a face two tiles hold carries the same bits on both
fn face_state(tiles: &[Hier]) -> HashMap<(usize, usize, [i64; 2]), u64> {
    let mut out = HashMap::new();
    for h in tiles {
        for (ll, level) in h.levels.iter().enumerate() {
            let s = &level.state;
            let mhd = s.fields.mhd.as_ref().unwrap();
            let scale = (N << ll) as f64;
            for c in s.geom.interior.iter() {
                for d in 0..2 {
                    let mut hi = [c[0], c[1]];
                    hi[d] += 1;
                    for f in [[c[0], c[1]], hi] {
                        let x = s.geom.face_coord(f, d);
                        let at: [i64; 2] = std::array::from_fn(|a| (x[a] * scale * 2.0).round() as i64);
                        let bits = mhd.bface[d].view().at(f).to_bits();
                        if let Some(previous) = out.insert((d, ll, at), bits) {
                            assert_eq!(previous, bits, "face {d} on level {ll} at {at:?} differs between the tiles holding it");
                        }
                    }
                }
            }
        }
    }
    out
}

/// the largest magnitude of `pick` over the fine level's interior
fn fine_max(h: &Hier, pick: impl Fn(&Store, [isize; 2]) -> f64) -> f64 {
    let s = &h.levels[1].state;
    s.geom.interior.iter().map(|c| pick(s, [c[0], c[1]]).abs()).fold(0.0, f64::max)
}

fn assert_same(label: &str, single: &Hier, tiles: &[Hier]) {
    let (want, got) = (interior_state(std::slice::from_ref(single)), interior_state(tiles));
    assert_eq!(got.len(), want.len(), "{label}: the tiles hold a different set of cells");
    let mut differing: Vec<_> = want.iter().filter(|(k, v)| got.get(k) != Some(v)).map(|(k, _)| *k).collect();
    differing.sort();
    assert!(differing.is_empty(), "{label}: {} values differ, first {:?}", differing.len(), differing.first());
    let (want_faces, got_faces) = (face_state(std::slice::from_ref(single)), face_state(tiles));
    assert_eq!(got_faces.len(), want_faces.len(), "{label}: the tiles hold a different set of faces");
    let mut differing_faces: Vec<_> = want_faces.iter().filter(|(k, v)| got_faces.get(k) != Some(v)).map(|(k, _)| *k).collect();
    differing_faces.sort();
    assert!(differing_faces.is_empty(), "{label}: {} face values differ, first {:?}", differing_faces.len(), differing_faces.first());
}

fn assert_tiles_match(cuts: Vec<usize>, single: &Hier) {
    assert_tiles_match_over(cuts, single, &[whole()]);
}

fn assert_tiles_match_over(cuts: Vec<usize>, single: &Hier, regions: &[RefinementRegion<2>]) {
    let partition = Partition::explicit([N, N], [cuts.clone(), vec![]]).expect("cuts inside the grid");
    let mut tiles = build_tiles(&partition, regions);
    assert_same(&format!("cuts {cuts:?}, seeded"), &build_single_over(B_SCALE, 0.0, regions), &tiles);
    let devices = vec![0; tiles.len()];
    evolve_hierarchy_decomposed(&mut tiles, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, |_, _, _| {
        std::ops::ControlFlow::Continue(())
    });
    for (k, tile) in tiles.iter().enumerate() {
        assert!(tile.crash.is_none(), "cuts {cuts:?}: tile {k} crashed: {:?}", tile.crash);
        assert_eq!(tile.levels[0].state.iteration, single.levels[0].state.iteration, "cuts {cuts:?}: tile {k} stopped short");
    }
    assert_same(&format!("cuts {cuts:?}, evolved"), single, &tiles);
}

fn evolved_single(scale: f64) -> Hier {
    let mut single = build_single(scale);
    single.evolve(T_FINAL).expect("single evolve");
    assert!(single.crash.is_none(), "the single hierarchy crashed: {:?}", single.crash);
    single
}

#[test]
fn a_magnetized_patch_spanning_a_cut_periodic_axis_matches_the_single_hierarchy_on_two_tiles() {
    let initial = build_single(B_SCALE);
    let single = evolved_single(B_SCALE);
    assert!(single.levels[0].state.iteration >= 4, "the run took {} root steps", single.levels[0].state.iteration);
    // measured: the fine density moves by 0.14, the fine y-face field by 0.033, and the field
    // changes the fine density by 3.0e-3 against the zero-field run
    let moved = fine_max(&single, |s, c| {
        *s.fields.cons.den.view().at(c) - *initial.levels[1].state.fields.cons.den.view().at(c)
    });
    assert!(moved > 1.0e-2, "the fine density moved by {moved:e}; the seam carries nothing");
    let field_change = fine_max(&single, |s, c| {
        *s.fields.mhd.as_ref().unwrap().bface[1].view().at(c) - *initial.levels[1].state.fields.mhd.as_ref().unwrap().bface[1].view().at(c)
    });
    assert!(field_change > 1.0e-3, "the fine y-face field moved by {field_change:e}; the field is static");
    let unmagnetized = evolved_single(0.0);
    let gas_change = fine_max(&single, |s, c| *s.fields.cons.den.view().at(c) - *unmagnetized.levels[1].state.fields.cons.den.view().at(c));
    assert!(gas_change > 1.0e-3, "the field changes the fine density by at most {gas_change:e}; it couples nothing");
    assert_tiles_match(vec![16], &single);
}

#[test]
fn a_magnetized_patch_spanning_a_cut_periodic_axis_matches_the_single_hierarchy_on_three_uneven_tiles() {
    assert_tiles_match(vec![10, 21], &evolved_single(B_SCALE));
}

/// the single hierarchy alone: the problem translated by eight root cells along the periodic x
/// evolves to the same fine state translated by sixteen fine cells, bit for bit, every conserved
/// and cell field component. the cells beside the seam where it meets the coarse-fine faces read
/// the ghost cells at that meeting, so the translation holds only where those cells carry the
/// periodic image of the prolonged rows
#[test]
fn a_magnetized_patch_spanning_a_periodic_axis_evolves_the_same_wherever_the_seam_falls() {
    const SHIFT: usize = 8;
    let mut at_seam = build_single_shifted(B_SCALE, 0.0);
    let mut shifted = build_single_shifted(B_SCALE, SHIFT as f64 * DX);
    at_seam.evolve(T_FINAL).expect("evolve");
    shifted.evolve(T_FINAL).expect("evolve");
    assert_eq!(at_seam.levels[0].state.iteration, shifted.levels[0].state.iteration, "the runs took different step counts");
    let (want, got) = (interior_state(std::slice::from_ref(&at_seam)), interior_state(std::slice::from_ref(&shifted)));
    let mut differing: Vec<_> = want
        .iter()
        .filter(|((name, ll, at), v)| {
            let period = 2 * (N << ll) as i64;
            let moved = [(at[0] + 2 * (SHIFT << ll) as i64).rem_euclid(period), at[1]];
            got.get(&(*name, *ll, moved)) != Some(v)
        })
        .map(|(k, _)| *k)
        .collect();
    differing.sort();
    assert!(differing.is_empty(), "{} values differ under the translation, first {:?}", differing.len(), differing.first());
}

fn evolved_nested(regions: &[RefinementRegion<2>]) -> Hier {
    let initial = build_single_over(B_SCALE, 0.0, regions);
    let mut single = build_single_over(B_SCALE, 0.0, regions);
    single.evolve(T_FINAL).expect("single evolve");
    assert!(single.crash.is_none(), "the single hierarchy crashed: {:?}", single.crash);
    assert_eq!(single.levels.len(), 3, "the hierarchy holds {} levels", single.levels.len());
    // the nested level evolves its gas and its field
    let nested = &single.levels[2].state;
    let start = &initial.levels[2].state;
    let moved = nested
        .geom
        .interior
        .iter()
        .map(|c| (*nested.fields.cons.den.view().at([c[0], c[1]]) - *start.fields.cons.den.view().at([c[0], c[1]])).abs())
        .fold(0.0, f64::max);
    assert!(moved > 1.0e-2, "the nested density moved by {moved:e}");
    let field = nested
        .geom
        .interior
        .iter()
        .map(|c| {
            let (now, then) = (nested.fields.mhd.as_ref().unwrap(), start.fields.mhd.as_ref().unwrap());
            (*now.bface[1].view().at([c[0], c[1]]) - *then.bface[1].view().at([c[0], c[1]])).abs()
        })
        .fold(0.0, f64::max);
    assert!(field > 1.0e-3, "the nested y-face field moved by {field:e}");
    single
}

#[test]
fn a_nested_magnetized_patch_clear_of_the_periodic_faces_matches_the_single_hierarchy_across_a_cut() {
    let regions = [whole(), nested_clear()];
    let single = evolved_nested(&regions);
    assert_tiles_match_over(vec![16], &single, &regions);
    assert_tiles_match_over(vec![10, 21], &single, &regions);
}

/// the nested level spanning the whole periodic x axis over root cells 14..18 in y, inside the
/// spanning patch
fn nested_spanning() -> RefinementRegion<2> {
    RefinementRegion { x_lo: [0.0, 0.4375], x_hi: [1.0, 0.5625] }
}

#[test]
fn a_nested_magnetized_patch_spanning_a_cut_periodic_axis_matches_the_single_hierarchy() {
    let regions = [whole(), nested_spanning()];
    let single = evolved_nested(&regions);
    assert_tiles_match_over(vec![16], &single, &regions);
    assert_tiles_match_over(vec![10, 21], &single, &regions);
}

/// the launched driver's parent face stencil plan of the nested level of `regions` over the tiles
/// of `partition`, its boxes derived as the launch derives them with the spanned axes `spans`,
/// checked against the stencil fields the builder gives every tile: Ok when every transfer lands
/// inside its held field
fn nested_stencil_check(partition: &Partition<2>, tiles: &[Hier], spans: [bool; 2]) -> Result<(), String> {
    // the first patch spans x (root cells 0..32) over y 12..20; the nested box spans x over root
    // y 14..18, so level-1 cells 4..12 of the first patch in y
    let patch = [(0usize, N), (12usize, 20usize)];
    let rel = [(0usize, 2 * N), (4usize, 12usize)];
    let fine = FinePartition::of(partition, patch, 2).expect("level-1 partition");
    let parent = &fine.partition;
    let (counts, fine_counts) = (partition.counts(), parent.counts());
    let root_of = |t: usize| flatten(fine.root_tile_of(unflatten(t, fine_counts)), counts);
    let frame = |t: TileId| -> [isize; 2] {
        let ext = partition.tile_extents(fine.root_tile_of(unflatten(t.0 as usize, fine_counts)));
        std::array::from_fn(|a| (2 * patch[a].0.saturating_sub(ext[a].0)) as isize)
    };
    let boxes: Vec<Option<Vec<[Span; 2]>>> = (0..parent.n_tiles())
        .map(|t| {
            let ext = parent.tile_extents(unflatten(t, fine_counts));
            face_stencil_boxes_of_level_tile(parent, rel, TileId(t as u32), false, spans).map(|components| {
                components
                    .iter()
                    .map(|d| {
                        let d = d.as_ref().expect("a face component with a stencil box");
                        std::array::from_fn(|a| Span { lo: d.spaces[a].lo + ext[a].0 as isize, hi: d.spaces[a].hi + ext[a].0 as isize })
                    })
                    .collect()
            })
        })
        .collect();
    let ng = tiles[0].levels[0].state.geom.ng;
    let plan = ExchangePlan::face_stencil_shifted(parent, &Topology::wrapping([true, false]), &boxes, ng + 1, &frame)
        .map_err(|e| e.to_string())?;
    assert!(plan.transfers().count() > 0, "the nested stencil plan moves nothing; the check is vacuous");
    let placement = Placement::new(1, vec![WorkerId(0); parent.n_tiles()]).expect("placement");
    let table = PhaseTable::refined(2, 2, 2).expect("phase table");
    let exchange = PlanExchange::new(&plan, &placement, WorkerId(0), table);
    let held: Vec<Vec<&symbi_grid::Field<f64, 2, HostMemory>>> = (0..parent.n_tiles())
        .map(|t| tiles[root_of(t)].face_stencil_of(2).map(|st| st.old.iter().collect()).unwrap_or_default())
        .collect();
    exchange.check_destinations(&held)
}

/// the nested level spanning the periodic x axis takes its face halo along x from its own wrap,
/// so its parent face stencil stops at the spanned sides; the plan built that way lands inside the
/// builder's stencil fields on every tile, and the plan widening those sides by a plane is refused
/// before any value moves, naming the transfer that would land past its field
#[test]
fn a_nested_face_stencil_wider_than_its_held_field_is_refused_before_any_write() {
    for cuts in [vec![16], vec![10, 21]] {
        let partition = Partition::explicit([N, N], [cuts.clone(), vec![]]).expect("cuts inside the grid");
        let mut tiles = build_tiles(&partition, &[whole(), nested_spanning()]);
        for h in tiles.iter_mut() {
            h.enable_nested_face_stencils(3).expect("nested face stencils");
        }
        assert!(tiles.iter().any(|h| h.face_stencil_of(2).is_some()), "cuts {cuts:?}: no tile holds a nested stencil; the gate is vacuous");
        nested_stencil_check(&partition, &tiles, [true, false]).unwrap_or_else(|e| panic!("cuts {cuts:?}: the spanned plan is refused: {e}"));
        let refused = nested_stencil_check(&partition, &tiles, [false, false]).expect_err("the widened plan passed the pre-exchange check");
        assert!(refused.contains("past the held field's domain"), "cuts {cuts:?}: the refusal names another cause: {refused}");
    }
}
