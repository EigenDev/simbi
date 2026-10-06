// =============================================================================
// decomp_refine_iso_closure.rs
//
// the isothermal closure field cs^2 of a refined level seeded on tiles whose
// cuts cross the refined patch. the seed prolongs cs^2 over the fine interior
// and a two-cell ghost band; where the patch runs to a cut, the band's
// prolongation stencil reaches past the parent tile's allocation, so the seed
// clips the band to what the parent supplies and continues cs^2 into the rest.
// for the globally isothermal closure (one sound speed per run):
// - every fine cell a stencil can read, interior, cut halos and the corner
//   where four tiles meet, holds exactly the configured cs^2 and is finite,
//   on ragged cuts on both axes and on a single cut;
// - the patch meets a cut on every layout; with cuts on both axes the unclipped
//   band read one past the end of the parent tile's buffer (index 374 of 374
//   on the ragged layout), and those layouts now seed to completion.
// a spatially varying cs^2 at a clipped edge stops the seed by name: the
// continuation equals the prolongation for a uniform closure alone.
// restart: the tiles are written through the partitioned checkpoint writer and
// restored onto a different ragged partition through the production loader;
// before any step every stencil-readable fine cs^2 cell holds the recorded
// value (a non-default 0.49), bitwise and finite, cut and corner halos
// included. a restart whose configured sound speed differs from the recorded
// one restores the recorded value in the interior and the ghosts alike.
//
// run: cargo test -p symbi --test decomp_refine_iso_closure
// =============================================================================
use symbi::prelude::SimSubstrate;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::launched::RefinedGrids;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, seed_decomposed_fine_from_coarse};
use symbi_hydro::isothermal::IsoNewtonian as Iso;
use symbi_io::attr::Metadata;
use symbi_sim::checkpoint::{LevelTiles, PhysicsIdentity, TileView, load_partitioned_level, write_partitioned_checkpoint};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<IsoNewtonian, 2, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
type Kern = <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet;
type Hier = Hierarchy<IsoNewtonian, 2, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, Kern>;

const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const CS: f64 = 0.7;
// the refined patch covers root cells 8..24 on both axes
const PATCH: RefinementRegion<2> = RefinementRegion { x_lo: [0.25, 0.25], x_hi: [0.75, 0.75] };

fn kernels(s: &Sim) -> Kern {
    s.substrate()
}

/// a root tile placed as production places it: the whole grid's origin and the index offset of
/// its first cell
fn root(ext: [(usize, usize); 2], bnd: Boundaries<2>, cs: f64) -> Sim {
    Sim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([ext[0].0 as isize, ext[1].0 as isize])
        .boundaries(bnd)
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root allocation")
        .set_initial(|[x, y]| {
            let bump = 0.3 * (-(((x - 0.5) / 0.1).powi(2) + ((y - 0.5) / 0.1).powi(2))).exp();
            PrimG::<f64, 2, IsoModel>::isothermal(Density(1.0 + bump), Tensor::new([0.1, -0.05]))
        })
        .build()
}

/// the tiles of `partition`, each refining the patch clipped to it, before any seed
fn tiles(partition: &Partition<2>, perturb_parent_cs2: bool) -> Vec<Hier> {
    tiles_at(partition, perturb_parent_cs2, CS)
}

fn tiles_at(partition: &Partition<2>, perturb_parent_cs2: bool, cs: f64) -> Vec<Hier> {
    let counts = partition.counts();
    (0..partition.n_tiles())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let ext = partition.tile_extents(tc);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
                [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
            }));
            let sim = root(ext, bnd, cs);
            if perturb_parent_cs2 {
                let cs2 = sim.fields.cs2.as_ref().expect("an isothermal closure field");
                let c = sim.geom.interior.iter().next().expect("an interior cell");
                *cs2.view_mut().at_mut(c) *= 1.5;
            }
            let lo: [f64; 2] = std::array::from_fn(|a| (ext[a].0 as f64 * DX).max(PATCH.x_lo[a]));
            let hi: [f64; 2] = std::array::from_fn(|a| ((ext[a].0 + ext[a].1) as f64 * DX).min(PATCH.x_hi[a]));
            let k = kernels(&sim);
            if (0..2).all(|a| lo[a] < hi[a]) {
                let clipped = RefinementRegion { x_lo: lo, x_hi: hi };
                Hier::with_refinement_of(sim, k, &[clipped], &[PATCH], ProlongOrder::Ppm, kernels).expect("tile hierarchy")
            } else {
                Hier::single(sim, k)
            }
        })
        .collect()
}

/// the refined tiles whose fine interior runs to an edge of the parent's interior, where the seed
/// clips its band
fn clipped_tiles(tiles: &[Hier]) -> usize {
    tiles
        .iter()
        .filter(|h| h.levels.len() > 1)
        .filter(|h| {
            let cov = h.levels[0].coverage.as_ref().expect("a coverage");
            let int = &h.levels[0].state.geom.interior;
            (0..2).any(|a| cov.spaces[a].lo == int.spaces[a].lo || cov.spaces[a].hi == int.spaces[a].hi)
        })
        .count()
}

/// every allocated fine cs^2 cell of every refined tile holds `want`, bitwise and finite
fn assert_fine_closure(tiles: &[Hier], want: f64, label: &str) {
    let mut checked = 0usize;
    for (t, h) in tiles.iter().enumerate().filter(|(_, h)| h.levels.len() > 1) {
        let fine = &h.levels[1].state;
        let cs2 = fine.fields.cs2.as_ref().expect("a fine closure field").view();
        for c in fine.geom.allocated.iter() {
            let v = *cs2.at(c);
            assert!(v.is_finite() && v.to_bits() == want.to_bits(), "{label}: tile {t} fine cell {c:?} holds cs^2 = {v}, expected {want}");
            checked += 1;
        }
    }
    assert!(checked > 0, "{label}: no fine cell was checked");
}

/// the whole grid's patch in root cells: 8..24 on both axes
const PATCH_CELLS: [(usize, usize); 2] = [(8, 24), (8, 24)];

/// write the seeded tiles of `partition` through the partitioned writer, both levels
fn write(tiles: &[Hier], partition: &Partition<2>, path: &str) {
    let counts = partition.counts();
    let grids = RefinedGrids::of([N, N], [0.0, 0.0], [DX, DX], tiles[0].levels[0].state.geom.ng, PATCH_CELLS);
    let root_offset = |flat: usize| -> [isize; 2] {
        let ext = partition.tile_extents(unflatten(flat, counts));
        [ext[0].0 as isize, ext[1].0 as isize]
    };
    let fine_offset = |flat: usize, h: &Hier| -> [isize; 2] {
        let f = &h.levels[1].state.geom.interior;
        let r = root_offset(flat);
        std::array::from_fn(|a| 2 * r[a] + f.spaces[a].lo)
    };
    let levels = vec![
        LevelTiles {
            tiles: tiles.iter().enumerate().map(|(i, h)| TileView { state: &h.levels[0].state.store, offset: root_offset(i) }).collect(),
            grid: grids.root.clone(),
            header: None,
        },
        LevelTiles {
            tiles: tiles
                .iter()
                .enumerate()
                .filter(|(_, h)| h.levels.len() > 1)
                .map(|(i, h)| TileView { state: &h.levels[1].state.store, offset: fine_offset(i, h) })
                .collect(),
            grid: grids.fine.clone(),
            header: None,
        },
    ];
    let identity = PhysicsIdentity::of(&tiles[0].levels[0].state);
    write_partitioned_checkpoint::<Iso, 2, 2, HostMemory>(&identity, &levels, path, &Metadata::new()).expect("checkpoint write");
}

/// the tiles of `partition` built at sound speed `cs`, seeded, then restored from `path` through the
/// production loader, both levels
fn restore(partition: &Partition<2>, path: &str, cs: f64) -> Vec<Hier> {
    let counts = partition.counts();
    let mut tiles = tiles_at(partition, false, cs);
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("seed");
    // the restored value is the loader's alone: before the load every fine cell holds the
    // configured closure
    assert_fine_closure(&tiles, cs * cs, &format!("seeded at a configured cs = {cs}, before the load"));
    let grids = RefinedGrids::of([N, N], [0.0, 0.0], [DX, DX], tiles[0].levels[0].state.geom.ng, PATCH_CELLS);
    for (flat, h) in tiles.iter_mut().enumerate() {
        let ext = partition.tile_extents(unflatten(flat, counts));
        let r = [ext[0].0 as isize, ext[1].0 as isize];
        load_partitioned_level(&mut h.levels[0].state, path, 0, r, &grids.root).expect("root restore");
        if h.levels.len() > 1 {
            let f = h.levels[1].state.geom.interior.clone();
            let offset = std::array::from_fn(|a| 2 * r[a] + f.spaces[a].lo);
            load_partitioned_level(&mut h.levels[1].state, path, 1, offset, &grids.fine).expect("fine restore");
        }
    }
    tiles
}

fn restart_arm(restart_cs: f64) {
    let written = Partition::explicit([N, N], [vec![13], vec![18]]).expect("cuts inside the grid");
    let tiles = tiles(&written, false);
    seed_decomposed_fine_from_coarse(&tiles, written.counts(), &vec![0; tiles.len()], &LocalCopy).expect("seed");
    let dir = std::env::temp_dir().join(format!("iso_closure_restart_{}_{}", std::process::id(), restart_cs.to_bits()));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    let path = dir.join("tiles.h5");
    write(&tiles, &written, path.to_str().expect("utf-8 path"));
    let resumed = Partition::explicit([N, N], [vec![17], vec![11]]).expect("cuts inside the grid");
    let restored = restore(&resumed, path.to_str().expect("utf-8 path"), restart_cs);
    assert!(restored.iter().filter(|h| h.levels.len() > 1).count() >= 2, "the resumed partition cuts no refined tile");
    assert_fine_closure(&restored, CS * CS, &format!("restored at a configured cs = {restart_cs}"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_restart_onto_another_ragged_partition_restores_every_readable_closure_cell() {
    restart_arm(CS);
}

#[test]
fn a_restart_configured_with_another_sound_speed_restores_the_recorded_closure_everywhere() {
    restart_arm(0.9);
}

fn assert_closure_complete(cuts: [Vec<usize>; 2]) {
    let partition = Partition::explicit([N, N], cuts.clone()).expect("cuts inside the grid");
    let tiles = tiles(&partition, false);
    let refined = tiles.iter().filter(|h| h.levels.len() > 1).count();
    assert!(clipped_tiles(&tiles) == refined && refined >= 2, "cuts {cuts:?}: the patch never meets a cut; the clip is never exercised");
    seed_decomposed_fine_from_coarse(&tiles, partition.counts(), &vec![0; tiles.len()], &LocalCopy).expect("seed");
    let want = (CS * CS).to_bits();
    for (t, h) in tiles.iter().enumerate().filter(|(_, h)| h.levels.len() > 1) {
        let fine = &h.levels[1].state;
        let cs2 = fine.fields.cs2.as_ref().expect("a fine closure field").view();
        for c in fine.geom.allocated.iter() {
            let v = *cs2.at(c);
            assert!(v.is_finite() && v.to_bits() == want, "cuts {cuts:?}: tile {t} fine cell {c:?} holds cs^2 = {v}, expected {}", CS * CS);
        }
    }
}

/// the layout whose seed read past the parent tile's buffer: ragged cuts on both axes, four tiles
/// meeting at a corner inside the patch
#[test]
fn ragged_cuts_with_a_corner_inside_the_patch_seed_a_complete_closure() {
    assert_closure_complete([vec![13], vec![18]]);
}

#[test]
fn centered_cuts_seed_a_complete_closure() {
    assert_closure_complete([vec![16], vec![16]]);
}

#[test]
fn a_single_cut_seeds_a_complete_closure() {
    assert_closure_complete([vec![15], vec![]]);
}

#[test]
fn a_varying_closure_at_a_clipped_edge_stops_the_seed_by_name() {
    let partition = Partition::explicit([N, N], [vec![13], vec![18]]).expect("cuts inside the grid");
    let tiles = tiles(&partition, true);
    let counts = partition.counts();
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).expect("seed")
    }))
    .expect_err("a varying closure seeded across a clipped edge");
    let message = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(message.contains("is not one finite value"), "the seed stopped for another reason: {message}");
}
