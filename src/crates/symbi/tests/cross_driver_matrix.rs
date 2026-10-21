// =============================================================================
// cross_driver_matrix.rs
//
// the single-node driver matrix: the first-order correction and its freeze,
// replay and rollback paths agree between the single-grid hierarchy
// (Hierarchy::single) and the in-process decomposition (evolve_scheduled) for
// every admitted regime. each regime runs the same events through the store's
// one-shot test hooks:
// - cut_predictor / cut_corrector: a physical cell against a cut flagged
//   troubled at the mark of the RK2 predictor, and of the corrector;
// - corner_predictor / corner_corrector: the same at the corner of four tiles;
// - freeze: a cell made unrecoverable in the predictor, at least a halo width
//   from every cut, so its redo cannot land it and the correcting select
//   freezes it. curved GRMHD accepts the freeze inside its retry exclusion;
// - inadmissible freeze: on every other regime, gas and magnetized, flat and
//   curved, the frozen cell holds the poisoned input outside the admissible
//   set, so the same event rejects the step, which replays at half the
//   timestep with the hook spent;
// - rollback: curved GRMHD under no retry exclusion rejects the same freeze and
//   replays the step likewise.
// per event the decomposed run matches the single-grid run bit for bit in its
// accepted times, its accepted troubled and frozen cells, every conserved field,
// and on a magnetized regime every face and cell field; both copies of every
// shared face agree; and a premise asserts the event changed the run. a
// poisoned stage input disagrees with the halo copies a neighbor received at the
// stage's start, so the freeze and rollback cells keep a halo width from every
// cut; trouble at a cut is a troubled flag on a physical state.
//
// run: cargo test -p symbi --test cross_driver_matrix
// =============================================================================

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::ops::ControlFlow;
use symbi::prelude::SimSubstrate;
use symbi::sim::decomp::{LocalCopy, Partition, Schedule, Topology, evolve_scheduled, unflatten};
use symbi::sim::evolve::KernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi_sim::run_diagnostics::RunDiagnostics;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::RetryExclusion;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

/// (field id, global index) -> bits: ids 0.. the conserved fields, 100 + d the face field of
/// component d, 200 + c the cell field of component c
type State = BTreeMap<(usize, Vec<usize>), u64>;
/// every copy of a face the tiles hold: (component, global face index) -> bits of each copy
type Copies = BTreeMap<(usize, Vec<usize>), Vec<u64>>;

/// a run's accepted times and census, its state and its face copies
struct Run {
    times: Vec<u64>,
    troubled: u64,
    frozen: u64,
    state: State,
    copies: Copies,
}

/// the complete state of one store over its owned interior, placed on the global grid by `lo`, the
/// global index of its first owned cell; every face it holds, closing faces included, to `copies`
fn record<const D: usize, const DOF: usize>(
    state: &mut State,
    copies: &mut Copies,
    s: &FieldStore<D, DOF, HostMemory, f64>,
    lo: [usize; D],
    extent: [usize; D],
) {
    let first: [isize; D] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo);
    let global = |x: [isize; D]| -> Vec<usize> { (0..D).map(|a| lo[a] + (x[a] - first[a]) as usize).collect() };
    let c = &s.fields.cons;
    let mut cells: Vec<&symbi_grid::Field<f64, D, HostMemory>> = vec![&c.den];
    cells.extend(c.mom.iter());
    cells.extend(c.nrg_field());
    for x in s.geom.interior.iter() {
        for (kind, f) in cells.iter().enumerate() {
            state.insert((kind, global(x)), f.view().at(x).to_bits());
        }
    }
    let Some(m) = s.fields.mhd.as_ref() else { return };
    for x in s.geom.interior.iter() {
        for (cc, f) in m.bcell.b.iter().enumerate() {
            state.insert((200 + cc, global(x)), f.view().at(x).to_bits());
        }
    }
    for (d, f) in m.bface.b.iter().enumerate() {
        let hi: [usize; D] = std::array::from_fn(|a| extent[a] + usize::from(a == d));
        let count: usize = hi.iter().product();
        for flat in 0..count {
            let mut rest = flat;
            let idx: [usize; D] = std::array::from_fn(|a| {
                let v = rest % hi[a];
                rest /= hi[a];
                v
            });
            let x: [isize; D] = std::array::from_fn(|a| first[a] + idx[a] as isize);
            let bits = f.view().at(x).to_bits();
            let g = global(x);
            copies.entry((d, g.clone())).or_default().push(bits);
            state.entry((100 + d, g)).or_insert(bits);
        }
    }
}

/// the decomposed run over `stores`, stopping after `stop` accepted steps
fn scheduled<const D: usize, const DOF: usize, K>(
    stores: &mut [&mut FieldStore<D, DOF, HostMemory, f64>],
    kernels: &[K],
    counts: [usize; D],
    topology: &Topology<D>,
    stop: usize,
) -> (Vec<u64>, RunDiagnostics)
where
    K: KernelSet<D, DOF, HostMemory, f64>,
    Cartesian: Metric<f64, D>,
{
    let schedule = Schedule::derive(counts, stores[0].geom.ng, topology);
    let kernel_refs: Vec<&K> = kernels.iter().collect();
    let devices = vec![0i32; stores.len()];
    let mut times = Vec::new();
    let d = evolve_scheduled(stores, &kernel_refs, &schedule, &devices, Timestepping::Rk2, 0.0, 100.0, 1, &LocalCopy, |_, t, _| {
        times.push(t.to_bits());
        if times.len() >= stop { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    });
    (times, d)
}

/// the decomposed run matches the single-grid run bit for bit, its shared faces agree, and the
/// event changed the run against `clean`
fn assert_match(label: &str, clean: &Run, one: &Run, many: &Run) {
    let changed = one.state.iter().filter(|(k, v)| clean.state.get(k) != Some(v)).count();
    assert!(changed > 0, "{label}: the event left the run unchanged; the gate is vacuous");
    assert_eq!(many.times, one.times, "{label}: the decomposed run accepted different times");
    assert_eq!(many.troubled, one.troubled, "{label}: the decomposed run booked a different troubled census");
    assert_eq!(many.frozen, one.frozen, "{label}: the decomposed run booked a different frozen census");
    assert_eq!(many.state.len(), one.state.len(), "{label}: the states differ in extent");
    let differing: Vec<_> = one.state.iter().filter(|(k, v)| many.state.get(k) != Some(v)).map(|(k, _)| k.clone()).collect();
    assert!(differing.is_empty(), "{label}: {} values differ from the single-grid run, first {:?}", differing.len(), differing.first());
    let split: Vec<_> = many.copies.iter().filter(|(_, v)| v.iter().any(|b| *b != v[0])).map(|(k, _)| k.clone()).collect();
    assert!(split.is_empty(), "{label}: {} shared faces disagree between their tiles, first {:?}", split.len(), split.first());
}

/// the events of one regime. the invoking module declares: `D`, `DOF`, `N` (cells per axis), `Sim`,
/// `build(ext, boundaries, policy) -> Sim`, `kernels(&Sim)`, `DOMAIN` (its boundary per side) and `WRAPS`, the cells
/// `CUT_CELL`, `CORNER_CELL`, `FREEZE_CELL`, the cut lists `CUTS`, `CORNER_CUTS`, `FREEZE_CUTS`,
/// and `FREEZE_POLICY`, the retry exclusion of its freeze event
macro_rules! matrix_tests {
    () => {
        #[derive(Clone, Copy)]
        enum Hook {
            Clean,
            Troubled(usize, [usize; D]),
            Unrecoverable([usize; D]),
        }

        /// the event's cell, flagged or poisoned in step 1
        fn arm(s: &Sim, ext: [(usize, usize); D], hook: Hook) {
            let local = |cell: [usize; D]| -> [isize; D] {
                std::array::from_fn(|a| s.geom.interior.spaces[a].lo + cell[a] as isize - ext[a].0 as isize)
            };
            match hook {
                Hook::Clean => {}
                Hook::Troubled(stage, cell) => {
                    if s.geom.interior.contains(local(cell)) {
                        *s.workspace.troubled_once.lock().unwrap() = Some((1, stage, local(cell), false));
                    }
                }
                Hook::Unrecoverable(cell) => {
                    if s.geom.interior.contains(local(cell)) {
                        *s.workspace.unrecoverable_once.lock().unwrap() = vec![(1, 0, local(cell), false)];
                    }
                }
            }
        }

        fn single(hook: Hook, policy: RetryExclusion, stop: usize) -> Run {
            let whole = [(0usize, N); D];
            let sim = build(whole, Boundaries(DOMAIN), policy);
            arm(&sim, whole, hook);
            let k = kernels(&sim);
            let mut hier = Hierarchy::single(sim, k);
            let mut times = Vec::new();
            let d = hier
                .evolve_with_callback(100.0, 1, |h| {
                    times.push(h.levels[0].state.time.to_bits());
                    if times.len() >= stop { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
                })
                .expect("single-grid run");
            let (mut state, mut copies) = (State::new(), Copies::new());
            record(&mut state, &mut copies, &hier.levels[0].state, [0; D], [N; D]);
            Run { times, troubled: d.guards.troubled_cells.total, frozen: d.guards.frozen_cells.total, state, copies }
        }

        fn decomposed(cuts: [Vec<usize>; D], hook: Hook, policy: RetryExclusion, stop: usize) -> Run {
            let partition = Partition::explicit([N; D], cuts).expect("cuts inside the grid");
            let counts = partition.counts();
            let extents: Vec<[(usize, usize); D]> =
                (0..partition.n_tiles()).map(|f| partition.tile_extents(unflatten(f, counts))).collect();
            let mut sims: Vec<Sim> = extents
                .iter()
                .enumerate()
                .map(|(f, ext)| {
                    let tc = unflatten(f, counts);
                    let bnd = Boundaries(std::array::from_fn(|a| {
                        if counts[a] == 1 {
                            DOMAIN[a]
                        } else {
                            [
                                if tc[a] == 0 && !WRAPS[a] { DOMAIN[a][0] } else { BoundaryType::CoarseFine },
                                if tc[a] == counts[a] - 1 && !WRAPS[a] { DOMAIN[a][1] } else { BoundaryType::CoarseFine },
                            ]
                        }
                    }));
                    let s = build(*ext, bnd, policy);
                    arm(&s, *ext, hook);
                    s
                })
                .collect();
            let ks: Vec<_> = sims.iter().map(kernels).collect();
            let (times, d) = {
                let mut stores: Vec<&mut FieldStore<D, { DOF }, HostMemory, f64>> = sims.iter_mut().map(|s| &mut **s).collect();
                scheduled(&mut stores, &ks, counts, &Topology::wrapping(WRAPS), stop)
            };
            let (mut state, mut copies) = (State::new(), Copies::new());
            for (s, ext) in sims.iter().zip(&extents) {
                record(&mut state, &mut copies, s, std::array::from_fn(|a| ext[a].0), std::array::from_fn(|a| ext[a].1));
            }
            Run { times, troubled: d.guards.troubled_cells.total, frozen: d.guards.frozen_cells.total, state, copies }
        }

        fn flagged(label: &str, cuts: [Vec<usize>; D], stage: usize, cell: [usize; D]) {
            let clean = single(Hook::Clean, FREEZE_POLICY, 3);
            let one = single(Hook::Troubled(stage, cell), FREEZE_POLICY, 3);
            let many = decomposed(cuts, Hook::Troubled(stage, cell), FREEZE_POLICY, 3);
            assert!(one.troubled >= 1, "{label}: the single-grid run booked no troubled cell");
            assert_match(label, &clean, &one, &many);
        }

        #[test]
        fn cut_predictor() {
            flagged("cut, predictor", CUTS(), 0, CUT_CELL);
        }

        #[test]
        fn cut_corrector() {
            flagged("cut, corrector", CUTS(), 1, CUT_CELL);
        }

        #[test]
        fn corner_predictor() {
            flagged("corner, predictor", CORNER_CUTS(), 0, CORNER_CELL);
        }

        #[test]
        fn corner_corrector() {
            flagged("corner, corrector", CORNER_CUTS(), 1, CORNER_CELL);
        }

    };
}

/// the freeze event of a regime whose correction accepts a freeze under `FREEZE_POLICY`
macro_rules! freeze_test {
    () => {
        #[test]
        fn freeze() {
            let clean = single(Hook::Clean, FREEZE_POLICY, 2);
            let one = single(Hook::Unrecoverable(FREEZE_CELL), FREEZE_POLICY, 2);
            let many = decomposed(FREEZE_CUTS(), Hook::Unrecoverable(FREEZE_CELL), FREEZE_POLICY, 2);
            assert!(one.frozen >= 1, "freeze: the single-grid run accepted no frozen cell; the freeze never happened");
            assert_match("freeze", &clean, &one, &many);
        }
    };
}

/// the freeze event of a regime outside a retry exclusion: the poisoned freeze leaves its cell
/// outside the admissible set, so the attempt is rejected and the step replays at half the timestep with the
/// hook spent
macro_rules! inadmissible_freeze_test {
    () => {
        #[test]
        fn inadmissible_freeze() {
            let clean = single(Hook::Clean, FREEZE_POLICY, 2);
            let one = single(Hook::Unrecoverable(FREEZE_CELL), FREEZE_POLICY, 2);
            let many = decomposed(FREEZE_CUTS(), Hook::Unrecoverable(FREEZE_CELL), FREEZE_POLICY, 2);
            assert_eq!(one.frozen, 0, "inadmissible freeze: the rejected freeze survived into the accepted state");
            assert_ne!(one.times, clean.times, "inadmissible freeze: the step was not replayed at a smaller timestep");
            assert_match("inadmissible freeze", &clean, &one, &many);
        }
    };
}

/// the rollback event of a regime that carries a retry contract: its freeze under
/// `ROLLBACK_POLICY` rejects the step, which replays at half the timestep with the hook spent
macro_rules! rollback_test {
    () => {
        #[test]
        fn rollback() {
            let policy = ROLLBACK_POLICY;
            let accepted = single(Hook::Unrecoverable(FREEZE_CELL), FREEZE_POLICY, 2);
            let one = single(Hook::Unrecoverable(FREEZE_CELL), policy, 2);
            let many = decomposed(FREEZE_CUTS(), Hook::Unrecoverable(FREEZE_CELL), policy, 2);
            assert_eq!(one.frozen, 0, "rollback: the rejected freeze survived into the accepted state");
            assert_ne!(one.times, accepted.times, "rollback: the step was not replayed at a smaller timestep");
            assert_match("rollback", &accepted, &one, &many);
        }
    };
}

/// a smooth periodic flow on the unit square: density and velocity waves at unit amplitude scale
fn wave([x, y]: [f64; 2]) -> (f64, [f64; 2]) {
    (1.0 + 0.2 * (TAU * x).sin() * (TAU * y).sin(), [0.3 * (TAU * y).sin(), 0.3 * (TAU * x).sin()])
}

/// the periodic 2D square of `N` cells a side, with the cut, corner and freeze layouts
macro_rules! periodic_square {
    () => {
        const D: usize = 2;
        const N: usize = 32;
        const H: f64 = 1.0 / N as f64;
        const DOMAIN: [[BoundaryType; 2]; 2] = [[BoundaryType::Periodic; 2]; 2];
        const WRAPS: [bool; 2] = [true; 2];
        const CUT_CELL: [usize; 2] = [16, 8];
        const CORNER_CELL: [usize; 2] = [16, 16];
        const FREEZE_CELL: [usize; 2] = [8, 24];
        const FREEZE_POLICY: RetryExclusion = RetryExclusion::None;
        #[allow(non_snake_case)]
        fn CUTS() -> [Vec<usize>; 2] {
            [vec![16], Vec::new()]
        }
        #[allow(non_snake_case)]
        fn CORNER_CUTS() -> [Vec<usize>; 2] {
            [vec![16], vec![16]]
        }
        #[allow(non_snake_case)]
        fn FREEZE_CUTS() -> [Vec<usize>; 2] {
            [vec![16], Vec::new()]
        }
    };
}

mod newtonian {
    use super::*;
    use symbi_hydro::Newtonian;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 2;
    periodic_square!();
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|x| {
                let (rho, v) = wave(x);
                Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(1.0))
            })
            .build();
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();
}

mod isothermal {
    use super::*;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::isothermal::IsoNewtonian;
    type Sim = SimStateGeneric<IsoNewtonian, 2, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 2;
    periodic_square!();
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|x| {
                let (rho, v) = wave(x);
                PrimG::<f64, 2, IsoModel>::isothermal(Density(rho), Tensor::new(v))
            })
            .build();
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();
}

mod rhd {
    use super::*;
    use symbi_hydro::Rhd;
    type Sim = SimState<Rhd, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 2;
    periodic_square!();
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(Rhd, IdealGas { gamma: 4.0 / 3.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|x| {
                let (rho, v) = wave(x);
                Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(1.0))
            })
            .build();
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();
}

/// the periodic magnetized flow of the square: the wave's gas and a face field B_x(y), B_y(x),
/// divergence-free on the staggered faces
fn field_x(y: f64) -> f64 {
    -0.2 * (TAU * y).sin()
}
fn field_y(x: f64) -> f64 {
    0.2 * (2.0 * TAU * x).sin()
}

mod nmhd {
    use super::*;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 3;
    periodic_square!();
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|[x, y]| {
                let (rho, v) = wave([x, y]);
                MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1], 0.0]), Pressure(1.0)), Tensor::new([field_x(y), field_y(x), 0.0]))
            })
            .seed_faces(|axis, [x, y]| if axis == 0 { field_x(y) } else { field_y(x) })
            .build();
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();
}

/// the magnetized gas of `nmhd` pulled by a softened point mass off every cut, so the body
/// source acts in the stage, in the first-order redo and in the replayed step of every event
mod nmhd_body {
    use super::*;
    use std::cell::Cell;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    use symbi_ib::{Body, BodyCollection};
    type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 3;
    periodic_square!();
    thread_local! {
        static WITH_BODY: Cell<bool> = const { Cell::new(true) };
    }
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|[x, y]| {
                let (rho, v) = wave([x, y]);
                MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1], 0.0]), Pressure(1.0)), Tensor::new([field_x(y), field_y(x), 0.0]))
            })
            .seed_faces(|axis, [x, y]| if axis == 0 { field_x(y) } else { field_y(x) })
            .build();
        if WITH_BODY.with(Cell::get) {
            let mass = Body::gravitational(0, Tensor::new([0.37, 0.61]), Tensor::zeros(), 0.05, 0.0, 0.1);
            s = s.with_bodies(BodyCollection::new().add(mass));
        }
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();

    #[test]
    fn the_body_acts() {
        let pulled = single(Hook::Clean, FREEZE_POLICY, 3);
        WITH_BODY.with(|w| w.set(false));
        let free = single(Hook::Clean, FREEZE_POLICY, 3);
        let changed = pulled.state.iter().filter(|(k, v)| (1..=2).contains(&k.0) && free.state.get(k) != Some(v)).count();
        assert!(changed > N * N, "the body changed {changed} momentum values; the matrix exercises no gravity");
    }
}

mod imhd {
    use super::*;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::isothermal_mhd::IsothermalMhd;
    use symbi_hydro::mhd_state::MhdPrimG;
    type Sim = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 3;
    periodic_square!();
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|[x, y]| {
                let (rho, v) = wave([x, y]);
                MhdPrimG::<f64, 3, IsoModel>::new(
                    PrimG::isothermal(Density(rho), Tensor::new([v[0], v[1], 0.0])),
                    Tensor::new([field_x(y), field_y(x), 0.0]),
                )
            })
            .seed_faces(|axis, [x, y]| if axis == 0 { field_x(y) } else { field_y(x) })
            .build();
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();
}

/// the isothermal magnetized gas of `imhd` pulled by the softened point mass of `nmhd_body`
mod imhd_body {
    use super::*;
    use std::cell::Cell;
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::isothermal_mhd::IsothermalMhd;
    use symbi_hydro::mhd_state::MhdPrimG;
    use symbi_ib::{Body, BodyCollection};
    type Sim = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 3;
    periodic_square!();
    thread_local! {
        static WITH_BODY: Cell<bool> = const { Cell::new(true) };
    }
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|[x, y]| {
                let (rho, v) = wave([x, y]);
                MhdPrimG::<f64, 3, IsoModel>::new(
                    PrimG::isothermal(Density(rho), Tensor::new([v[0], v[1], 0.0])),
                    Tensor::new([field_x(y), field_y(x), 0.0]),
                )
            })
            .seed_faces(|axis, [x, y]| if axis == 0 { field_x(y) } else { field_y(x) })
            .build();
        if WITH_BODY.with(Cell::get) {
            let mass = Body::gravitational(0, Tensor::new([0.37, 0.61]), Tensor::zeros(), 0.05, 0.0, 0.1);
            s = s.with_bodies(BodyCollection::new().add(mass));
        }
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();

    #[test]
    fn the_body_acts() {
        let pulled = single(Hook::Clean, FREEZE_POLICY, 3);
        WITH_BODY.with(|w| w.set(false));
        let free = single(Hook::Clean, FREEZE_POLICY, 3);
        let changed = pulled.state.iter().filter(|(k, v)| (1..=2).contains(&k.0) && free.state.get(k) != Some(v)).count();
        assert!(changed > N * N, "the body changed {changed} momentum values; the matrix exercises no gravity");
    }
}

mod rmhd {
    use super::*;
    use symbi_hydro::rmhd::Rmhd;
    type Sim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    const DOF: usize = 3;
    periodic_square!();
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(Rmhd, IdealGas { gamma: 4.0 / 3.0 }, Cartesian)
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([0.0; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|[x, y]| {
                let (rho, v) = wave([x, y]);
                MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1], 0.0]), Pressure(1.0)), Tensor::new([field_x(y), field_y(x), 0.0]))
            })
            .seed_faces(|axis, [x, y]| if axis == 0 { field_x(y) } else { field_y(x) })
            .build();
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet {
        s.substrate()
    }
    matrix_tests!();
    inadmissible_freeze_test!();
}

/// GR hydro: the cold atmosphere in mach-173 radial infall onto an excised cartesian
/// kerr-schild hole (mass 0.3, excised at r < 0.35) on [-1.2, 1.2]^2; it corrects at the
/// excision rim by itself, so the events add their cells on top of that
mod grhd {
    use super::*;
    use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
    use symbi::sim::substrate_seam::WithExcision;
    use symbi_hydro::Rhd;
    type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
    const D: usize = 2;
    const DOF: usize = 2;
    const N: usize = 32;
    const L: f64 = 1.2;
    const H: f64 = 2.0 * L / N as f64;
    const DOMAIN: [[BoundaryType; 2]; 2] = [[BoundaryType::Outflow; 2]; 2];
    const WRAPS: [bool; 2] = [false; 2];
    /// (16, 24): r = 0.64, outside the excision
    const CUT_CELL: [usize; 2] = [16, 24];
    const CORNER_CELL: [usize; 2] = [16, 24];
    /// (8, 24): r = 0.84, eight cells from the cut
    const FREEZE_CELL: [usize; 2] = [8, 24];
    const FREEZE_POLICY: RetryExclusion = RetryExclusion::None;
    #[allow(non_snake_case)]
    fn CUTS() -> [Vec<usize>; 2] {
        [vec![16], Vec::new()]
    }
    #[allow(non_snake_case)]
    fn CORNER_CUTS() -> [Vec<usize>; 2] {
        [vec![16], vec![24]]
    }
    #[allow(non_snake_case)]
    fn FREEZE_CUTS() -> [Vec<usize>; 2] {
        [vec![16], Vec::new()]
    }
    fn build(ext: [(usize, usize); 2], bnd: Boundaries<2>, policy: RetryExclusion) -> Sim {
        let mut s = Sim::build(Rhd, IdealGas { gamma: 4.0 / 3.0 }, SchwarzschildKSCartesian { mass: 0.3 })
            .cells([ext[0].1, ext[1].1])
            .spacing([H; 2])
            .origin([-L; 2])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|x| {
                let r = (x[0] * x[0] + x[1] * x[1]).sqrt().max(1.0e-12);
                Prim::adiabatic(Density(1.0), Tensor::new([-0.2 * x[0] / r, -0.2 * x[1] / r]), Pressure(1.0e-6))
            })
            .build();
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> RhdSubstrateKernelSet<HostMemory, f64, 2> {
        RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(4.0 / 3.0, 0.3, &s.geom.allocated).with_excision(0.35, 1.0, 1.0)
    }
    matrix_tests!();
    inadmissible_freeze_test!();
}

/// GRMHD: the magnetized cartesian kerr-schild box (mass 1, no excision, UCT, HLLE) on
/// [-5, 5]^3 at 16^3; curved GRMHD carries the retry contract, so its rollback event runs
mod grmhd {
    use super::*;
    use symbi::regimes::substrate_kernels::Solver;
    use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
    use symbi_hydro::rmhd::Rmhd;
    type Sim = SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
    const D: usize = 3;
    const DOF: usize = 3;
    const N: usize = 16;
    const L: f64 = 5.0;
    const H: f64 = 2.0 * L / N as f64;
    const DOMAIN: [[BoundaryType; 2]; 3] = [[BoundaryType::Outflow; 2]; 3];
    const WRAPS: [bool; 3] = [false; 3];
    /// (8, 8, 8): r = 0.54, beside the clamped center where the geometric source is stiffest
    const CUT_CELL: [usize; 3] = [8, 8, 8];
    const CORNER_CELL: [usize; 3] = [8, 8, 8];
    /// (9, 8, 8): r = 1.03, inside r_+ = 2, three cells above the freeze cut
    const FREEZE_CELL: [usize; 3] = [9, 8, 8];
    const FREEZE_POLICY: RetryExclusion = RetryExclusion::LegacyEuclideanSphere { radius: 2.0 };
    const ROLLBACK_POLICY: RetryExclusion = RetryExclusion::None;
    #[allow(non_snake_case)]
    fn CUTS() -> [Vec<usize>; 3] {
        [vec![8], Vec::new(), Vec::new()]
    }
    #[allow(non_snake_case)]
    fn CORNER_CUTS() -> [Vec<usize>; 3] {
        [vec![8], vec![8], Vec::new()]
    }
    #[allow(non_snake_case)]
    fn FREEZE_CUTS() -> [Vec<usize>; 3] {
        [vec![6], Vec::new(), Vec::new()]
    }
    fn build(ext: [(usize, usize); 3], bnd: Boundaries<3>, policy: RetryExclusion) -> Sim {
        let m = SchwarzschildKSCartesian { mass: 1.0 };
        let bz = move |x: [f64; 3]| 0.05 / m.sqrt_det_gamma(Tensor::new(x));
        let mut s = Sim::build(Rmhd, IdealGas { gamma: 4.0 / 3.0 }, m)
            .cells([ext[0].1, ext[1].1, ext[2].1])
            .spacing([H; 3])
            .origin([-L; 3])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize, ext[2].0 as isize])
            .boundaries(bnd)
            .cfl(0.3)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|x| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.1)), Tensor::new([0.0, 0.0, bz(x)])))
            .seed_faces(|axis, x| if axis == 2 { bz(x) } else { 0.0 })
            .build()
            .with_rollback_snapshot()
            .expect("rollback snapshot");
        s.retry_exclusion = policy;
        s
    }
    fn kernels(s: &Sim) -> RmhdSubstrateKernelSet3D<HostMemory, f64> {
        RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(4.0 / 3.0, 0.3, 1.5, &s.geom.allocated)
            .with_solver(Solver::Hlle)
            .expect("solver")
            .ct_method(CtMethod::Uct)
    }
    matrix_tests!();
    freeze_test!();
    rollback_test!();
}

/// GRMHD source clipping at a cut: a cold, strongly magnetized kerr-schild box (mass 1, no
/// excision, UCT, HLLE, pressure 0.01, B0 = 0.5, cfl 0.8) whose central cells, root indices 7
/// and 8 on every axis, leave the admissible set under the full geometric source by their own
/// physics in the first two steps. their conservative source replay clips the source to a
/// fraction below one, a state-changing correction with no hook involved. a cut at x = 8
/// splits the clipped block between two tiles; a cut at x = 9 holds it on one side, its x = 8
/// layer against the cut and the neighbor clean. each run matches the single grid bit for bit
/// with both copies of every shared face agreeing, and each asserts that a replay clipped.
mod grmhd_clipping {
    use super::*;
    use symbi::regimes::substrate_kernels::Solver;
    use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
    use symbi_hydro::rmhd::Rmhd;
    type Sim = SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
    type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;
    const N: usize = 16;
    const L: f64 = 5.0;
    const H: f64 = 2.0 * L / N as f64;
    const CFL: f64 = 0.8;
    const STEPS: usize = 2;

    fn build(ext: [(usize, usize); 3], bnd: Boundaries<3>) -> Sim {
        let m = SchwarzschildKSCartesian { mass: 1.0 };
        let bz = move |x: [f64; 3]| 0.5 / m.sqrt_det_gamma(Tensor::new(x));
        let mut s = Sim::build(Rmhd, IdealGas { gamma: 4.0 / 3.0 }, m)
            .cells([ext[0].1, ext[1].1, ext[2].1])
            .spacing([H; 3])
            .origin([-L; 3])
            .index_offset([ext[0].0 as isize, ext[1].0 as isize, ext[2].0 as isize])
            .boundaries(bnd)
            .cfl(CFL)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|x| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.01)), Tensor::new([0.0, 0.0, bz(x)])))
            .seed_faces(|axis, x| if axis == 2 { bz(x) } else { 0.0 })
            .build()
            .with_rollback_snapshot()
            .expect("rollback snapshot");
        // the clipping sits inside the horizon; the legacy sphere keeps a freeze there accepted
        s.retry_exclusion = RetryExclusion::LegacyEuclideanSphere { radius: 2.0 };
        s
    }

    fn kernels(s: &Sim) -> Kern {
        Kern::new(4.0 / 3.0, CFL, 1.5, &s.geom.allocated)
            .with_solver(Solver::Hlle)
            .expect("solver")
            .ct_method(CtMethod::Uct)
    }

    fn fraction(k: &Kern) -> f64 {
        f64::from_bits(k.min_source_fraction.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn single() -> (Run, f64) {
        let whole = [(0usize, N); 3];
        let sim = build(whole, Boundaries::uniform(BoundaryType::Outflow));
        let k = kernels(&sim);
        let mut hier = Hierarchy::single(sim, k);
        let mut times = Vec::new();
        let d = hier
            .evolve_with_callback(100.0, 1, |h| {
                times.push(h.levels[0].state.time.to_bits());
                if times.len() >= STEPS { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
            })
            .expect("single-grid run");
        let (mut state, mut copies) = (State::new(), Copies::new());
        record(&mut state, &mut copies, &hier.levels[0].state, [0; 3], [N; 3]);
        let run = Run { times, troubled: d.guards.troubled_cells.total, frozen: d.guards.frozen_cells.total, state, copies };
        (run, fraction(&hier.levels[0].kernels))
    }

    fn decomposed(cut: usize) -> (Run, f64) {
        let partition = Partition::explicit([N; 3], [vec![cut], Vec::new(), Vec::new()]).expect("cut inside the grid");
        let counts = partition.counts();
        let extents: Vec<[(usize, usize); 3]> =
            (0..partition.n_tiles()).map(|f| partition.tile_extents(unflatten(f, counts))).collect();
        let mut sims: Vec<Sim> = extents
            .iter()
            .enumerate()
            .map(|(f, ext)| {
                let tc = unflatten(f, counts);
                let bnd = Boundaries(std::array::from_fn(|a| {
                    if counts[a] == 1 {
                        [BoundaryType::Outflow; 2]
                    } else {
                        [
                            if tc[a] == 0 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                            if tc[a] == counts[a] - 1 { BoundaryType::Outflow } else { BoundaryType::CoarseFine },
                        ]
                    }
                }));
                build(*ext, bnd)
            })
            .collect();
        let ks: Vec<Kern> = sims.iter().map(kernels).collect();
        let (times, d) = {
            let mut stores: Vec<&mut FieldStore<3, 3, HostMemory, f64>> = sims.iter_mut().map(|s| &mut **s).collect();
            scheduled(&mut stores, &ks, counts, &Topology::wrapping([false; 3]), STEPS)
        };
        let (mut state, mut copies) = (State::new(), Copies::new());
        for (s, ext) in sims.iter().zip(&extents) {
            record(&mut state, &mut copies, s, std::array::from_fn(|a| ext[a].0), std::array::from_fn(|a| ext[a].1));
        }
        let run = Run { times, troubled: d.guards.troubled_cells.total, frozen: d.guards.frozen_cells.total, state, copies };
        (run, ks.iter().map(fraction).fold(1.0, f64::min))
    }

    fn assert_clipping_at(cut: usize) {
        let (one, clipped_one) = single();
        assert!(clipped_one < 1.0, "the single-grid run clipped no source; the gate is vacuous");
        let (many, clipped_many) = decomposed(cut);
        assert!(clipped_many < 1.0, "cut {cut}: the decomposed run clipped no source");
        assert_eq!(many.times, one.times, "cut {cut}: the decomposed run accepted different times");
        assert_eq!((many.troubled, many.frozen), (one.troubled, one.frozen), "cut {cut}: the guard census differs");
        let differing: Vec<_> = one.state.iter().filter(|(k, v)| many.state.get(k) != Some(v)).map(|(k, _)| k.clone()).collect();
        assert!(differing.is_empty(), "cut {cut}: {} values differ from the single grid, first {:?}", differing.len(), differing.first());
        let split: Vec<_> = many.copies.iter().filter(|(_, v)| v.iter().any(|b| *b != v[0])).map(|(k, _)| k.clone()).collect();
        assert!(split.is_empty(), "cut {cut}: {} shared faces disagree, first {:?}", split.len(), split.first());
    }

    #[test]
    fn a_clipping_replay_split_by_a_cut() {
        assert_clipping_at(8);
    }

    #[test]
    fn a_clipping_replay_against_a_cut_from_one_side() {
        assert_clipping_at(9);
    }
}
