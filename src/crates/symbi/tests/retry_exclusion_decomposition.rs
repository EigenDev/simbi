// =============================================================================
// retry_exclusion_decomposition.rs
//
// the retry exclusion decides alike on one grid and across a cut. the fixture is
// the magnetized cartesian kerr-schild box (mass 1, no excision, UCT) with one
// cell at r = 1.03, inside r_+ = 2, made unrecoverable in the predictor of step
// 1 through the store's one-shot test hook: the stage input and output carry a
// negative energy, the redo cannot land the cell and the correcting select
// freezes it. under no exclusion the freeze rejects the step, which replays at
// half the timestep with the hook spent; under the legacy euclidean sphere of
// radius 2 the step is accepted with the freeze. per policy, the run cut across
// two tiles matches the single-grid hierarchy run in its accepted times, its
// accepted troubled and frozen cells and its state, bit for bit, through the
// poisoned step; the two policies decide differently.
//
// the cut lies three cells from the poisoned cell. a stage input poisoned after
// the stage snapshot disagrees with the halo copies the neighbor received at the
// stage's start, a state no run reaches, so trouble against a cut is exercised
// by a physical troubled flag in gr_correction_cut_pass instead. the runs stop
// after the poisoned step: a freeze the policy accepts holds the cell at its
// poisoned stage input, which no later step recovers.
//
// run: cargo test -p symbi --test retry_exclusion_decomposition
// =============================================================================

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::decomp::{LocalCopy, Partition, Schedule, Topology, evolve_scheduled, unflatten};
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::RetryExclusion;
use symbi_algebra::Tensor;
use symbi_geometry::{Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
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
const POISON_STEP: u64 = 1;
/// the poisoned cell: centroid (0.9375, 0.3125, 0.3125), r = 1.03, inside r_+ = 2
const CELL: [usize; 3] = [N / 2 + 1, N / 2, N / 2];
/// the cut along x, three cells below the poisoned cell
const CUT: usize = N / 2 - 2;
const SPHERE: RetryExclusion = RetryExclusion::LegacyEuclideanSphere { radius: 2.0 };

/// the box over `ext` of the global grid, placed by the whole grid's origin and its first cell,
/// the poisoned cell armed on the tile whose interior holds it
fn tile(ext: [(usize, usize); 3], bnd: Boundaries<3>, policy: RetryExclusion) -> Sim {
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
    s.retry_exclusion = policy;
    let local: [isize; 3] = std::array::from_fn(|a| {
        s.geom.interior.spaces[a].lo + CELL[a] as isize - ext[a].0 as isize
    });
    if s.geom.interior.contains(local) {
        *s.workspace.unrecoverable_once.lock().unwrap() = vec![(POISON_STEP, 0, local, false)];
    }
    s
}

fn kernels(s: &Sim) -> Kern {
    Kern::new(4.0 / 3.0, 0.3, 1.5, &s.geom.allocated)
        .with_solver(Solver::Hlle)
        .expect("solver")
        .ct_method(CtMethod::Uct)
}

/// continue through the poisoned step, then stop
fn through_the_poisoned_step(accepted: usize) -> ControlFlow<()> {
    if accepted as u64 > POISON_STEP {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
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

/// a run's accepted times, accepted troubled and frozen cells, and state
struct Run {
    times: Vec<u64>,
    troubled: u64,
    frozen: u64,
    state: State,
}

fn single(policy: RetryExclusion) -> Run {
    let whole = [(0, N); 3];
    let sim = tile(whole, Boundaries::uniform(BoundaryType::Outflow), policy);
    let k = kernels(&sim);
    let mut hier = Hierarchy::single(sim, k);
    let mut times = Vec::new();
    let d = hier
        .evolve_with_callback(T_FINAL, 1, |h| {
            times.push(h.levels[0].state.time.to_bits());
            through_the_poisoned_step(times.len())
        })
        .expect("single-grid run");
    let mut state = State::new();
    record(&mut state, &hier.levels[0].state, whole);
    Run {
        times,
        troubled: d.guards.troubled_cells.total,
        frozen: d.guards.frozen_cells.total,
        state,
    }
}

fn cut(policy: RetryExclusion) -> Run {
    let partition = Partition::explicit([N; 3], [vec![CUT], Vec::new(), Vec::new()]).expect("cut");
    let counts = partition.counts();
    let extents: Vec<[(usize, usize); 3]> = (0..partition.n_tiles())
        .map(|f| partition.tile_extents(unflatten(f, counts)))
        .collect();
    let mut sims: Vec<Sim> = extents
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
            tile(*ext, bnd, policy)
        })
        .collect();
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
                through_the_poisoned_step(times.len())
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
        frozen: d.guards.frozen_cells.total,
        state,
    }
}

fn assert_cut_matches_single(policy: RetryExclusion, label: &str) -> Run {
    let one = single(policy);
    let two = cut(policy);
    assert_eq!(
        two.times, one.times,
        "{label}: the cut run accepted different times"
    );
    assert_eq!(
        two.troubled, one.troubled,
        "{label}: the cut run accepted a different number of troubled cells"
    );
    assert_eq!(
        two.frozen, one.frozen,
        "{label}: the cut run accepted a different number of frozen cells"
    );
    assert_eq!(
        two.state.len(),
        one.state.len(),
        "{label}: the states differ in extent"
    );
    let differing = one
        .state
        .iter()
        .filter(|(k, v)| two.state.get(k) != Some(v))
        .count();
    assert_eq!(
        differing, 0,
        "{label}: {differing} cell and face values differ from the single-grid run"
    );
    one
}

#[test]
fn each_policy_decides_alike_on_one_grid_and_across_a_cut() {
    let none = assert_cut_matches_single(RetryExclusion::None, "no exclusion");
    let sphere = assert_cut_matches_single(SPHERE, "the covering sphere");
    // the premise: the eligible freeze decides differently under the two policies
    assert!(
        sphere.frozen >= 1,
        "the covering sphere accepted no frozen cell; the freeze never happened"
    );
    assert_eq!(
        none.frozen, 0,
        "no exclusion accepted a frozen cell; its freeze never rejected the step"
    );
    assert_ne!(
        none.times, sphere.times,
        "the two policies accepted the same times; the gate is vacuous"
    );
}
