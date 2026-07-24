// =============================================================================
// decomp_tracers_equivalence.rs
//
// lagrangian tracers under multi-gpu domain decomposition, validated in-process
// on the cpu. a tracer is a massless point advected by the interpolated gas
// velocity; unlike a field it MIGRATES -- when it crosses a tile cut it must
// leave one tile's population and join the neighbor's.
//
// the contract: a decomposed run must reproduce the monolithic tracer
// trajectories to round-off. a tracer near a cut is advected by its owning tile,
// whose ghost band the halo exchange filled from the neighbor, so the velocity it
// samples is the same the monolithic grid would; migration only re-homes it to
// the tile that now contains it. escape (leaving the whole domain) and sink
// crossing are scanned against the GLOBAL box, not the tile slab.
//
// setup: a uniform diagonal drift carries a mass-weighted tracer population
// (clustered on a density blob) across the 2x2 grid's cuts. tracers are seeded
// ONCE from a reference grid and partitioned by position, so the monolithic and
// decomposed runs start from the identical population (same ids, same positions).
// the comparison sorts by id; a non-vacuity guard requires tracers to have moved
// AND at least one to have crossed a cut, or the migration is never exercised.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_gpu::device_sync;
use symbi::sim::decomp::{evolve_decomposed, flatten, unflatten, LocalCopy};
use symbi::sim::state::*;
use symbi::sim::tracers::{seed_mass_weighted, TracerSet};
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{with_device, CpuSpace, HostMemory};

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

// a density + PRESSURE blob off-center: the drift carries it (and the tracers seeded on it)
// across the central cut, while the pressure bump makes the velocity SPATIALLY VARYING. that
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
            let origin = [tc[0] as f64 * m[0] as f64 * DX, tc[1] as f64 * m[1] as f64 * DX];
            let bnd = Boundaries::per_axis(std::array::from_fn(|a| {
                [
                    if tc[a] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                    if tc[a] == counts[a] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                ]
            }));
            with_device(tile_device(flat), || make(m, origin, bnd, ts))
        })
        .collect()
}

// the tile that owns a position: floor-divide by the tile extent, flat axis-0-fastest. mirrors
// the migration owner map so the partitioned seed lands where migration expects it.
fn owner(x: &[f64; 2], counts: [usize; 2]) -> usize {
    let ext = [N as f64 / counts[0] as f64 * DX, N as f64 / counts[1] as f64 * DX];
    let tc: [usize; 2] =
        std::array::from_fn(|a| (x[a] / ext[a]).floor().clamp(0.0, counts[a] as f64 - 1.0) as usize);
    flatten(tc, counts)
}

// assign each tracer of a global population to the tile that contains its position, preserving
// id / flags / weight -- the decomposed analog of the monolithic single population.
fn partition_into(tiles: &mut [(Sim, Kern)], global: &TracerSet<2>, counts: [usize; 2]) {
    let mut per_tile: Vec<TracerSet<2>> =
        (0..tiles.len()).map(|_| TracerSet { weight: global.weight, ..Default::default() }).collect();
    for i in 0..global.x.len() {
        let dest = owner(&global.x[i], counts);
        per_tile[dest].x.push(global.x[i]);
        per_tile[dest].id.push(global.id[i]);
        per_tile[dest].flags.push(global.flags[i]);
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
        &mut stores, &kernels, counts, &devices, ts, 0.0, T_FINAL, u64::MAX, &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

// gather every tile's tracers into one id-indexed position array (the checkpoint gather the
// python multi-gpu path performs). ids are 0..n by construction, so index directly.
fn gather_positions(tiles: &[(Sim, Kern)]) -> Vec<[f64; 2]> {
    sync_devices();
    let mut out = vec![[f64::NAN; 2]; N_TRACERS];
    for (sim, _) in tiles {
        if let Some(tr) = sim.tracers.as_ref() {
            for i in 0..tr.x.len() {
                out[tr.id[i] as usize] = tr.x[i];
            }
        }
    }
    out
}

fn assert_matches(counts: [usize; 2], ts: Timestepping) {
    // seed the global population once from a reference grid, so mono and decomposed start
    // from the identical tracers.
    let (ref_sim, _) = make([N, N], [0.0, 0.0], Boundaries::uniform(BoundaryType::Outflow), ts);
    let global = seed_mass_weighted(&ref_sim, N_TRACERS);
    let seeded: Vec<[f64; 2]> = global.x.clone();

    let mut mono = grid_tiles([1, 1], ts);
    partition_into(&mut mono, &global, [1, 1]);
    run(&mut mono, [1, 1], ts);
    let mono_pos = gather_positions(&mono);

    let mut dec = grid_tiles(counts, ts);
    partition_into(&mut dec, &global, counts);
    run(&mut dec, counts, ts);
    let dec_pos = gather_positions(&dec);

    assert!(
        mono_pos.iter().all(|p| p.iter().all(|v| v.is_finite()))
            && dec_pos.iter().all(|p| p.iter().all(|v| v.is_finite())),
        "some tracer id was never gathered (a tracer was lost in migration)"
    );

    // NON-VACUITY: tracers must have moved, and at least one must have crossed a cut (else
    // migration never fired and the gate proves nothing).
    let mut max_move = 0.0_f64;
    let mut crossed = 0usize;
    let cuts = [1.0 / counts[0] as f64 * 0.5 * N as f64 * DX, 1.0 / counts[1] as f64 * 0.5 * N as f64 * DX];
    let _ = cuts;
    let cut = [0.5_f64, 0.5_f64]; // the 2x2 cut positions
    for (i, p) in mono_pos.iter().enumerate() {
        let s = seeded[i];
        max_move = max_move.max(((p[0] - s[0]).powi(2) + (p[1] - s[1]).powi(2)).sqrt());
        for a in 0..2 {
            if counts[a] > 1 && (s[a] < cut[a]) != (p[a] < cut[a]) {
                crossed += 1;
                break;
            }
        }
    }
    assert!(max_move > 0.05, "tracers barely moved (max {max_move:e}); advection is vacuous");
    assert!(crossed > 0, "no tracer crossed a cut; migration was never exercised");

    let max_err = mono_pos
        .iter()
        .zip(&dec_pos)
        .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-12,
        "decomposition {counts:?} ({ts:?}) vs monolithic tracer positions max err {max_err:e} \
         ({crossed} crossed a cut, moved {max_move:e})"
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
