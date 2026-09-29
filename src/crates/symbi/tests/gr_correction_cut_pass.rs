// =============================================================================
// gr_correction_cut_pass.rs
//
// the first-order correction of curved GRMHD driven across a cut. the fixture
// is the magnetized cartesian kerr-schild box (mass 1, no excision, UCT) cut
// along x through the hole, with the cell (8, 8, 8) at r = 0.54 against the cut
// on the upper tile flagged troubled at the mark of one stage of step 0, its
// state physical, through the store's one-shot test hook. the lower tile's
// interior carries no flag; the trouble reaches its cut halo, and its pass is
// active: the redo on the shared faces acts, and curved GRMHD's conservative
// source replay runs tile-wide within the active substage, as the single grid's
// troubled pass runs it over the same cells.
// per stage, predictor and corrector:
// - the cut run matches the single-grid hierarchy run through its end, bit for
//   bit, with the same accepted times and troubled cells, and the flag changed
//   the run;
// - the clean tile's own report books no troubled cell and names the
//   conservative replay it ran.
// the replay clips no cell of the clean tile here, so it and the shared redo
// leave the same state; a clean-side cell whose source the replay must clip is
// not exercised.
//
// run: cargo test -p symbi --test gr_correction_cut_pass
// =============================================================================

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::decomp::{
    LocalCopy, Partition, Schedule, Topology, evolve_scheduled, exchange_grid, exchange_trouble,
    unflatten,
};
use symbi::sim::evolve::KernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{RetryExclusion, SourceReplayOutcome};
use symbi_algebra::Tensor;
use symbi_geometry::{Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_sim::stage::{
    SingleGridStep, StageArgs, fold_stage_excision, fold_stage_from_correction,
    fold_stage_through_mark,
};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;
type Store = FieldStore<3, 3, HostMemory, f64>;
type State = BTreeMap<(usize, [usize; 3]), u64>;

const N: usize = 16;
const L: f64 = 5.0;
const DX: f64 = 2.0 * L / N as f64;
const MASS: f64 = 1.0;
const T_FINAL: f64 = 1.0;
/// the flagged cell, the first column of the upper tile
const CELL: [usize; 3] = [N / 2; 3];
const CUT: usize = N / 2;
const POLICY: RetryExclusion = RetryExclusion::LegacyEuclideanSphere { radius: 2.0 };

/// the box over `ext` of the global grid, placed by the whole grid's origin and its first cell;
/// with `flag = Some((step, stage))` the cell is flagged on the tile whose interior holds it
fn tile(ext: [(usize, usize); 3], bnd: Boundaries<3>, flag: Option<(u64, usize)>) -> Sim {
    let m = SchwarzschildKSCartesian { mass: MASS };
    let bz = move |x: [f64; 3]| 0.05 / m.sqrt_det_gamma(Tensor::new(x));
    let mut s = Sim::build(Rmhd, IdealGas { gamma: 4.0 / 3.0 }, m)
        .cells([ext[0].1, ext[1].1, ext[2].1])
        .spacing([DX; 3])
        .origin([-L; 3])
        .index_offset([ext[0].0 as isize, ext[1].0 as isize, ext[2].0 as isize])
        .boundaries(bnd)
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.1)),
                Tensor::new([0.0, 0.0, bz(x)]),
            )
        })
        .seed_faces(|axis, x| if axis == 2 { bz(x) } else { 0.0 })
        .build()
        .with_rollback_snapshot()
        .expect("rollback snapshot");
    s.retry_exclusion = POLICY;
    let local: [isize; 3] = std::array::from_fn(|a| {
        s.geom.interior.spaces[a].lo + CELL[a] as isize - ext[a].0 as isize
    });
    if let Some((step, stage)) = flag {
        if s.geom.interior.contains(local) {
            *s.workspace.troubled_once.lock().unwrap() = Some((step, stage, local, false));
        }
    }
    s
}

fn kernels(s: &Sim) -> Kern {
    Kern::new(4.0 / 3.0, 0.3, 1.5, &s.geom.allocated)
        .with_solver(Solver::Hlle)
        .expect("solver")
        .ct_method(CtMethod::Uct)
}

/// the two tiles of the cut and their extents
fn tiles(flag: Option<(u64, usize)>) -> (Vec<Sim>, Vec<[(usize, usize); 3]>, [usize; 3]) {
    let partition = Partition::explicit([N; 3], [vec![CUT], Vec::new(), Vec::new()]).expect("cut");
    let counts = partition.counts();
    let extents: Vec<[(usize, usize); 3]> = (0..partition.n_tiles())
        .map(|f| partition.tile_extents(unflatten(f, counts)))
        .collect();
    let sims = extents
        .iter()
        .enumerate()
        .map(|(f, ext)| {
            let tc = unflatten(f, counts);
            let bnd = Boundaries(std::array::from_fn(|a| {
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
            tile(*ext, bnd, flag)
        })
        .collect();
    (sims, extents, counts)
}

/// the conserved fields per cell and the face fields per face over the global grid, a shared face
/// from the upper tile
fn record(state: &mut State, s: &Store, ext: [(usize, usize); 3]) {
    let lo: [isize; 3] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo);
    let c = &s.fields.cons;
    let cells = [
        &c.den,
        &c.mom[0],
        &c.mom[1],
        &c.mom[2],
        c.nrg_field().unwrap(),
    ];
    for x in s.geom.interior.iter() {
        let g: [usize; 3] = std::array::from_fn(|a| ext[a].0 + (x[a] - lo[a]) as usize);
        for (kind, f) in cells.iter().enumerate() {
            state.insert((kind, g), f.view().at(x).to_bits());
        }
    }
    for (d, f) in s.fields.mhd.as_ref().unwrap().bface.b.iter().enumerate() {
        let hi: [usize; 3] = std::array::from_fn(|a| ext[a].1 + usize::from(a == d));
        for i in 0..hi[0] {
            for j in 0..hi[1] {
                for k in 0..hi[2] {
                    let x = [lo[0] + i as isize, lo[1] + j as isize, lo[2] + k as isize];
                    state.insert(
                        (5 + d, [ext[0].0 + i, ext[1].0 + j, ext[2].0 + k]),
                        f.view().at(x).to_bits(),
                    );
                }
            }
        }
    }
}

/// a run's accepted times, accepted troubled cells, and final state
struct Run {
    times: Vec<u64>,
    troubled: u64,
    state: State,
}

fn single(flag: Option<(u64, usize)>) -> Run {
    let whole = [(0, N); 3];
    let sim = tile(whole, Boundaries::uniform(BoundaryType::Outflow), flag);
    let k = kernels(&sim);
    let mut hier = Hierarchy::single(sim, k);
    let mut times = Vec::new();
    let d = hier
        .evolve_with_callback(T_FINAL, 1, |h| {
            times.push(h.levels[0].state.time.to_bits());
            ControlFlow::Continue(())
        })
        .expect("single-grid run");
    let mut state = State::new();
    record(&mut state, &hier.levels[0].state, whole);
    Run {
        times,
        troubled: d.guards.troubled_cells.total,
        state,
    }
}

fn cut(flag: Option<(u64, usize)>) -> Run {
    let (mut sims, extents, counts) = tiles(flag);
    let ks: Vec<Kern> = sims.iter().map(kernels).collect();
    let mut times = Vec::new();
    let d = {
        let mut stores: Vec<&mut Store> = sims.iter_mut().map(|s| &mut **s).collect();
        let kernel_refs: Vec<&Kern> = ks.iter().collect();
        let schedule = Schedule::derive(counts, stores[0].geom.ng, &Topology::open());
        evolve_scheduled(
            &mut stores,
            &kernel_refs,
            &schedule,
            &[0, 0],
            Timestepping::Rk2,
            0.0,
            T_FINAL,
            1,
            &LocalCopy,
            |_, t, _| {
                times.push(t.to_bits());
                ControlFlow::Continue(())
            },
        )
    };
    let mut state = State::new();
    for (s, ext) in sims.iter().zip(&extents) {
        record(&mut state, s, *ext);
    }
    Run {
        times,
        troubled: d.guards.troubled_cells.total,
        state,
    }
}

/// the clean lower tile's own correction report in `stage` of step 0: (its troubled count, its
/// replay, whether the trouble reached its cut halo). the stage runs through the mark and the
/// trouble exchange as the scheduled driver runs it, then the clean tile's correction phase runs
/// with its cut-trouble mark set, exactly as its correction span runs it.
fn neighbor_report(stage: usize) -> (u64, SourceReplayOutcome, bool) {
    let (sims, _, counts) = tiles(Some((0, stage)));
    let ks: Vec<Kern> = sims.iter().map(kernels).collect();
    let sh: Vec<&Store> = sims.iter().map(|s| &**s).collect();
    let schedule = Schedule::derive(counts, sh[0].geom.ng, &Topology::open());
    let devices = [0i32, 0];
    for (s, k) in sh.iter().zip(&ks) {
        k.c2p(s);
        k.ghost_fill(s);
    }
    exchange_grid(&sh, &schedule, &devices, &LocalCopy);
    for (s, k) in sh.iter().zip(&ks) {
        k.ghost_fill(s);
        k.snapshot_retry(s);
        k.snapshot(s);
    }
    let dt = ks
        .iter()
        .zip(&sh)
        .map(|(k, s)| k.cfl(s))
        .fold(f64::INFINITY, f64::min);
    let stages = Timestepping::Rk2.stages();
    for ii in 0..=stage {
        let args = StageArgs {
            dt,
            a0: stages[ii].0,
            ac: stages[ii].1,
            stage: ii,
            n_stages: 2,
            injection_weight: symbi_sim::driver::downstream_injection_weight(stages, ii),
            allow_elision: false,
        };
        for (s, k) in sh.iter().zip(&ks) {
            let _ = fold_stage_through_mark(s, k, args, &mut |_| {});
        }
        let received = exchange_trouble(&sh, &schedule, &devices, &LocalCopy);
        let cut_trouble: Vec<bool> = (0..2)
            .map(|i| {
                received[i]
                    .iter()
                    .any(|region| ks[i].fofc_flags_in(sh[i], region) > 0)
            })
            .collect();
        if ii == stage {
            sh[0]
                .workspace
                .fofc_cut_trouble
                .store(cut_trouble[0], std::sync::atomic::Ordering::Relaxed);
            let report = ks[0].fofc(sh[0], dt, args.a0, args.ac, ii as u8 + 1);
            return (report.troubled(), report.replay(), cut_trouble[0]);
        }
        for i in 0..2 {
            let _ = fold_stage_from_correction(sh[i], &ks[i], args, cut_trouble[i], &mut |_| {});
            fold_stage_excision(&[sh[i]], &[&ks[i]], &mut SingleGridStep).expect("excision");
        }
        exchange_grid(&sh, &schedule, &devices, &LocalCopy);
        for (s, k) in sh.iter().zip(&ks) {
            k.ghost_fill(s);
        }
    }
    unreachable!("the stage loop returns at the named stage")
}

fn assert_flagged_cell_against_the_cut(stage: usize) {
    let clean = single(None);
    let one = single(Some((0, stage)));
    let two = cut(Some((0, stage)));
    let changed = one
        .state
        .iter()
        .filter(|(k, v)| clean.state.get(k) != Some(v))
        .count();
    assert!(
        changed > 0,
        "stage {stage}: the flag left the run unchanged; the gate is vacuous"
    );
    assert!(
        one.troubled >= 1,
        "stage {stage}: the single-grid run booked no troubled cell"
    );
    assert_eq!(
        two.times, one.times,
        "stage {stage}: the cut run accepted different times"
    );
    assert_eq!(
        two.troubled, one.troubled,
        "stage {stage}: the cut run booked a different troubled census"
    );
    let differing = one
        .state
        .iter()
        .filter(|(k, v)| two.state.get(k) != Some(v))
        .count();
    assert_eq!(
        differing, 0,
        "stage {stage}: {differing} cell and face values differ from the single-grid run"
    );
    let (troubled, replay, reached) = neighbor_report(stage);
    assert!(
        reached,
        "stage {stage}: the trouble never reached the clean tile's cut halo"
    );
    assert_eq!(
        troubled, 0,
        "stage {stage}: the clean tile booked a troubled cell of its own"
    );
    assert_eq!(
        replay,
        SourceReplayOutcome::ConservativeReplay,
        "stage {stage}: the clean tile's pass reports another replay"
    );
}

#[test]
fn a_flagged_cell_against_a_cut_corrects_alike_in_the_predictor() {
    assert_flagged_cell_against_the_cut(0);
}

#[test]
fn a_flagged_cell_against_a_cut_corrects_alike_in_the_corrector() {
    assert_flagged_cell_against_the_cut(1);
}
