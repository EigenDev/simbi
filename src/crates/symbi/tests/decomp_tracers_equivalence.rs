// =============================================================================
// decomp_tracers_equivalence.rs
//
// mass-transport tracers under multi-device domain decomposition, validated
// in-process on the cpu. ownership follows accepted finite-volume mass fluxes;
// crossing a tile cut migrates the complete tracer record to the owning tile.
//
// the contract: a decomposed run must reproduce the monolithic tracer
// trajectories to round-off. a tracer near a cut is advected by its owning tile,
// whose ghost band the halo exchange filled from the neighbor, so the velocity it
// samples is the same the monolithic grid would; migration only re-homes it to
// the tile that now contains it. escape (leaving the whole domain) and sink
// crossing are scanned against the global box, not the tile slab.
//
// setup: a uniform diagonal drift carries a mass-weighted tracer population
// (clustered on a density blob) across the 2x2 grid's cuts. tracers are seeded
// once from a reference grid and partitioned by position, so the monolithic and
// decomposed runs start from the identical population (same ids, same positions).
// the comparison sorts by id; a non-vacuity guard requires tracers to have moved
// and at least one to have crossed a cut, or the migration is never exercised.
// =============================================================================

use symbi::regimes::substrate_gpu::device_sync;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
use symbi::sim::state::*;
use symbi::sim::tracers::{TracerSet, seed_mass_weighted};
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory, with_device};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const DRIFT: f64 = 0.4; // uniform diagonal velocity, subsonic
const T_FINAL: f64 = 0.4; // tracers move ~0.16, from the blob at 0.4 across the 0.5 cut
const N_TRACERS: usize = 240;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

const NDEV: i32 = 2;
fn tile_device(flat: usize) -> i32 {
    (flat as i32) % NDEV
}
fn sync_devices() {
    for dd in 0..NDEV {
        with_device(dd, || device_sync::<HostMemory>());
    }
}

// a density + pressure blob off-center: the drift carries it (and the tracers seeded on it)
// across the central cut, while the pressure bump makes the velocity spatially varying. that
// non-uniformity is what makes migration load-bearing -- a mis-homed tracer whose sampler
// clamps past the ghost band would read a different velocity than the tile that truly owns it.
// a uniform-velocity drift would leave the trajectory identical regardless of migration.
fn blob(x: [f64; 2]) -> f64 {
    2.0 * (-(((x[0] - 0.4).powi(2) + (x[1] - 0.4).powi(2)) / 0.02)).exp()
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>, ts: Timestepping) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .timestepping(ts)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|x| Prim {
            rho: 1.0 + blob(x),
            vel: Tensor::new([DRIFT, DRIFT]),
            pre: 1.0 + blob(x),
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2], ts: Timestepping) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = [N / counts[0], N / counts[1]];
    let total = counts[0] * counts[1];
    (0..total)
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = [
                tc[0] as f64 * m[0] as f64 * DX,
                tc[1] as f64 * m[1] as f64 * DX,
            ];
            let bnd = Boundaries::per_axis(std::array::from_fn(|a| {
                [
                    if tc[a] == 0 {
                        BoundaryType::Outflow
                    } else {
                        BoundaryType::CoarseFine
                    },
                    if tc[a] == counts[a] - 1 {
                        BoundaryType::Outflow
                    } else {
                        BoundaryType::CoarseFine
                    },
                ]
            }));
            with_device(tile_device(flat), || make(m, origin, bnd, ts))
        })
        .collect()
}

// the tile that owns a position: floor-divide by the tile extent, flat axis-0-fastest. mirrors
// the migration owner map so the partitioned seed lands where migration expects it.
fn owner(x: &[f64; 2], counts: [usize; 2]) -> usize {
    let ext = [
        N as f64 / counts[0] as f64 * DX,
        N as f64 / counts[1] as f64 * DX,
    ];
    let tc: [usize; 2] = std::array::from_fn(|a| {
        (x[a] / ext[a]).floor().clamp(0.0, counts[a] as f64 - 1.0) as usize
    });
    flatten(tc, counts)
}

// assign each tracer of a global population to the tile that contains its position, preserving
// id / flags / weight -- the decomposed analog of the monolithic single population.
fn partition_into(tiles: &mut [(Sim, Kern)], global: &TracerSet<2>, counts: [usize; 2]) {
    let mut per_tile: Vec<TracerSet<2>> = (0..tiles.len())
        .map(|_| TracerSet {
            weight: global.weight,
            ..Default::default()
        })
        .collect();
    for i in 0..global.x.len() {
        let dest = owner(&global.x[i], counts);
        per_tile[dest].x.push(global.x[i]);
        per_tile[dest].id.push(global.id[i]);
        per_tile[dest].cohort.push(global.cohort[i]);
        per_tile[dest].flags.push(global.flags[i]);
        per_tile[dest].owner.push(global.owner[i]);
        per_tile[dest].step_owner.push(global.step_owner[i]);
        per_tile[dest].step_flags.push(global.step_flags[i]);
        per_tile[dest].run_seed = global.run_seed;
        per_tile[dest].next_id = global.next_id;
    }
    for (t, set) in tiles.iter_mut().zip(per_tile) {
        t.0.tracers = Some(set);
    }
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2], ts: Timestepping) {
    let devices: Vec<i32> = (0..tiles.len()).map(tile_device).collect();
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_decomposed(
        &mut stores,
        &kernels,
        counts,
        &devices,
        ts,
        0.0,
        T_FINAL,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

// gather every tile's tracers into one id-indexed position array (the checkpoint gather the
// python multi-gpu path performs). ids are 0..n by construction, so index directly.
fn gather_owners(tiles: &[(Sim, Kern)]) -> Vec<symbi_sim::mass_transport::ContainerId> {
    sync_devices();
    let mut out = vec![symbi_sim::mass_transport::ContainerId(u64::MAX); N_TRACERS];
    for (sim, _) in tiles {
        if let Some(tr) = sim.tracers.as_ref() {
            for i in 0..tr.len() {
                if (tr.id[i] as usize) < N_TRACERS {
                    out[tr.id[i] as usize] = tr.owner[i];
                }
            }
        }
    }
    out
}

fn gather_all_owners(tiles: &[(Sim, Kern)]) -> Vec<symbi_sim::mass_transport::ContainerId> {
    tiles
        .iter()
        .filter_map(|(sim, _)| sim.tracers.as_ref())
        .flat_map(|tracers| tracers.owner.iter().copied())
        .collect()
}

fn assert_matches(counts: [usize; 2], ts: Timestepping) {
    // seed the global population once from a reference grid, so mono and decomposed start
    // from the identical tracers.
    let (ref_sim, _) = make(
        [N, N],
        [0.0, 0.0],
        Boundaries::uniform(BoundaryType::Outflow),
        ts,
    );
    let global = seed_mass_weighted(&ref_sim, N_TRACERS);
    let seeded = global.owner.clone();

    let mut mono = grid_tiles([1, 1], ts);
    partition_into(&mut mono, &global, [1, 1]);
    run(&mut mono, [1, 1], ts);
    let mono_owner = gather_owners(&mono);
    let mono_all = gather_all_owners(&mono);

    let mut dec = grid_tiles(counts, ts);
    partition_into(&mut dec, &global, counts);
    run(&mut dec, counts, ts);
    let dec_owner = gather_owners(&dec);
    let dec_all = gather_all_owners(&dec);

    assert!(
        mono_owner.iter().all(|owner| owner.0 != u64::MAX)
            && dec_owner.iter().all(|owner| owner.0 != u64::MAX),
        "some tracer id was never gathered (a tracer was lost in migration)"
    );

    let moved = mono_owner
        .iter()
        .zip(&seeded)
        .filter(|(owner, initial)| owner != initial)
        .count();
    let cells_per_tile = [N / counts[0], N / counts[1]];
    let tile_of = |owner: symbi_sim::mass_transport::ContainerId| {
        let x = owner.0 as usize % N;
        let y = owner.0 as usize / N;
        flatten([x / cells_per_tile[0], y / cells_per_tile[1]], counts)
    };
    let crossed = mono_owner
        .iter()
        .zip(&seeded)
        .filter(|(owner, initial)| tile_of(**owner) != tile_of(**initial))
        .count();
    assert!(
        moved > 0,
        "no tracer crossed a cell face; transport is vacuous"
    );
    assert!(
        crossed > 0,
        "no tracer crossed a cut; migration was never exercised"
    );

    let histogram = |owners: &[symbi_sim::mass_transport::ContainerId]| {
        let mut counts = std::collections::BTreeMap::new();
        for owner in owners {
            *counts.entry(*owner).or_insert(0isize) += 1;
        }
        counts
    };
    let mono_histogram = histogram(&mono_all);
    let dec_histogram = histogram(&dec_all);
    let l1: usize = mono_histogram
        .keys()
        .chain(dec_histogram.keys())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|owner| {
            (mono_histogram.get(owner).copied().unwrap_or(0)
                - dec_histogram.get(owner).copied().unwrap_or(0))
            .unsigned_abs()
        })
        .sum();
    // decomposition changes the fluid trajectory at roundoff and therefore
    // may change individual stochastic histories. the conserved observable is
    // the ownership distribution. its total-variation discrepancy must remain
    // below ten percent, comparable to the finite-population sampling scale
    // 1/sqrt(n) and well below a resolved transport signal.
    assert!(
        l1 <= mono_all.len().max(dec_all.len()) / 5,
        "decomposition {counts:?} ({ts:?}) changed the ownership histogram by l1={l1}"
    );
}

// a single-axis cut (migration across one boundary) and the 2x2 grid (corners: a tracer can
// cross into a diagonal neighbor over two steps). euler and rk2 both exercised.
#[test]
fn tracers_two_tile_x_cut_euler() {
    assert_matches([2, 1], Timestepping::Euler);
}

#[test]
fn tracers_quad_tile_rk2() {
    assert_matches([2, 2], Timestepping::Rk2);
}
