// =============================================================================
// timestep_agreement.rs
//
// the root timestep every march agrees on. the gates:
// - a fine-level limit that binds sets the root step in root units: the fine
//   level subcycles ratio times, so its limit enters scaled by the ratio
//   (real adiabatic kernels, a stream seeded on the fine level alone);
// a mock kernel set estimating a fixed limit, NaN from a chosen root iteration on,
// drives the rest:
// - a march meeting an estimate it cannot step on halts on its last state and
//   reports it, on the single grid and on two tiles alike, a positive user cap
//   notwithstanding: the decomposed march returns the halt without a panic and
//   without a further observer call that could publish the halted state;
// - a march whose last step reaches its final time takes it shortened to the
//   time left and ends complete, with no halt;
// - the standalone single-grid driver steps at the user cap.
//
// run: cargo test -p symbi --test timestep_agreement
// =============================================================================

use std::ops::ControlFlow;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const HALT_AT: u64 = 3;
const DT: f64 = 0.01;
const ROOT_DX: f64 = 1.0 / 16.0;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

/// a kernel set that recovers primitives exactly and estimates DT, NaN from the root iteration
/// `halt_at` on
#[derive(Clone, Copy)]
struct FixedCfl {
    halt_at: Option<u64>,
}

impl FixedCfl {
    fn uniform(halt_at: Option<u64>) -> Self {
        Self { halt_at }
    }
}

impl KernelSet<1, 1, HostMemory, f64> for FixedCfl {
    fn reconstruction_reach(&self) -> u8 {
        1
    }
    fn flux(&self, _store: &FieldStore<1, 1, HostMemory>, _dir: usize) {}
    fn c2p(&self, store: &FieldStore<1, 1, HostMemory>) {
        for cell in store.geom.interior.iter() {
            let rho = *store.fields.cons.den.view().at(cell);
            let mom = *store.fields.cons.mom[0].view().at(cell);
            let nrg = *store.fields.cons.nrg_field().unwrap().view().at(cell);
            store.fields.prim.rho.view_mut().set(cell, rho);
            store.fields.prim.vel[0].view_mut().set(cell, mom / rho);
            store
                .fields
                .prim
                .pre_field()
                .unwrap()
                .view_mut()
                .set(cell, (GAMMA - 1.0) * (nrg - 0.5 * mom * mom / rho));
        }
        store.mark_primitives_recovered();
    }
    fn godunov_stage(&self, _store: &FieldStore<1, 1, HostMemory>, _dt: f64, _a0: f64, _ac: f64) {}
    fn cfl(&self, store: &FieldStore<1, 1, HostMemory>) -> f64 {
        if self.halt_at.is_some_and(|at| store.iteration >= at) {
            f64::NAN
        } else {
            DT
        }
    }
    fn ghost_fill(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn snapshot(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn horizon_accretion(
        &self,
        _store: &FieldStore<1, 1, HostMemory>,
        _diagnostic_radius: f64,
    ) -> (f64, f64) {
        (0.0, 0.0)
    }
}

fn store(cells: usize, offset: usize, lo: BoundaryType, hi: BoundaryType, max_dt: f64) -> Sim {
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
        .origin([0.0])
        .index_offset([offset as isize])
        .spacing([ROOT_DX])
        .boundaries(Boundaries::per_axis([[lo, hi]]))
        .timestepping(Timestepping::Euler)
        .cfl(0.4)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build();
    sim.max_dt = max_dt;
    sim
}

fn single(
    kernels: FixedCfl,
    max_dt: f64,
) -> Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, FixedCfl> {
    Hierarchy::single(
        store(16, 0, BoundaryType::Outflow, BoundaryType::Outflow, max_dt),
        kernels,
    )
}

// a stream at speed three seeded on the fine level alone gives that level the tighter limit:
// in root units (its own limit times the ratio of two substeps) it falls below the root's, and
// one root step advances the clock by exactly that fine candidate
#[test]
fn a_binding_fine_level_limit_sets_the_root_step_in_root_units() {
    type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
    let coarse = store(16, 0, BoundaryType::Outflow, BoundaryType::Outflow, 0.0);
    let kernels = Kset::new(GAMMA, 0.4, &coarse.geom.allocated);
    let region = RefinementRegion {
        x_lo: [0.25],
        x_hi: [0.75],
    };
    let mut hier = Hierarchy::with_refinement(coarse, kernels, &[region], ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, 0.4, &s.geom.allocated)
    })
    .unwrap();
    hier.seed_fine_from_coarse().unwrap();
    let fine = &hier.levels[1].state;
    for cell in fine.geom.interior.iter() {
        fine.fields.cons.mom[0].view_mut().set(cell, 3.0);
        fine.fields
            .cons
            .nrg_field()
            .unwrap()
            .view_mut()
            .set(cell, 1.0 / (GAMMA - 1.0) + 0.5 * 9.0);
    }
    hier.prime();
    let candidates: Vec<f64> = hier.level_cfl_candidates().into_iter().map(|(_, c)| c).collect();
    assert_eq!(candidates.len(), 2, "premise: both levels evolve cells of their own");
    assert!(
        candidates[1] < candidates[0],
        "premise: the fine limit in root units binds: {candidates:?}"
    );
    hier.evolve_with_callback(1.0, 1, |h| {
        if h.levels[0].state.iteration >= 1 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .expect("the refined run returns");
    assert!(hier.crash.is_none(), "{:?}", hier.crash);
    assert_eq!(
        hier.levels[0].state.time, candidates[1],
        "the root step is the fine candidate in root units"
    );
}

#[test]
fn a_single_grid_records_the_halt_under_a_positive_cap() {
    let mut hier = single(FixedCfl::uniform(Some(HALT_AT)), 0.005);
    hier.evolve_with_callback(1.0, 1, |_| ControlFlow::Continue(()))
        .expect("the single-grid run returns");
    let crash = hier
        .crash
        .as_ref()
        .expect("the single grid records the halt");
    assert_eq!(crash.iter, HALT_AT);
    assert!(crash.dt_cfl.is_nan() && crash.panic.is_none(), "{crash:?}");
    assert_eq!(
        crash.dt_prev, DT,
        "the previous estimate is the raw one, before the cap"
    );
    assert_eq!(
        hier.levels[0].state.iteration, HALT_AT,
        "the halt leaves the last state in place"
    );
    assert!(
        (hier.levels[0].state.time - HALT_AT as f64 * 0.005).abs() < 1e-15,
        "the cap shaped every step"
    );
}

#[test]
fn a_decomposed_march_returns_the_halt_under_a_positive_cap_without_a_panic() {
    let mut a = store(8, 0, BoundaryType::Outflow, BoundaryType::CoarseFine, 0.005);
    let mut b = store(8, 8, BoundaryType::CoarseFine, BoundaryType::Outflow, 0.005);
    let kernels = FixedCfl::uniform(Some(HALT_AT));
    let mut observed = Vec::new();
    let mut stores = vec![&mut *a, &mut *b];
    let diagnostics = evolve_decomposed(
        &mut stores,
        &[&kernels, &kernels],
        [2],
        &[0, 0],
        Timestepping::Euler,
        0.0,
        1.0,
        1,
        &LocalCopy,
        |iter, _, _| {
            observed.push(iter);
            ControlFlow::Continue(())
        },
    );
    let halt = diagnostics
        .halt
        .expect("the decomposed march reports the halt");
    assert_eq!(halt.iter, HALT_AT);
    assert!(halt.dt_cfl.is_nan(), "{halt:?}");
    assert_eq!(
        halt.dt_prev, DT,
        "the previous estimate is the raw one, before the cap"
    );
    assert!(
        (halt.time - HALT_AT as f64 * 0.005).abs() < 1e-15,
        "{halt:?}"
    );
    assert_eq!(
        observed,
        vec![1, 2, 3],
        "the observer sees each accepted step and nothing after the halt"
    );
    for s in &stores {
        assert_eq!(s.iteration, HALT_AT, "every tile holds its last state");
    }
}

// a final time of 0.025 under steps of 0.01 ends on a third step shortened to the time left
#[test]
fn the_last_step_is_shortened_to_the_final_time_and_the_run_completes() {
    let t_final = 0.025;
    let mut hier = single(FixedCfl::uniform(None), 0.0);
    hier.evolve_with_callback(t_final, 1, |_| ControlFlow::Continue(()))
        .expect("the single-grid run returns");
    assert!(
        hier.crash.is_none(),
        "a completed run recorded a crash: {:?}",
        hier.crash
    );
    let root = &hier.levels[0].state;
    assert_eq!(root.iteration, 3);
    assert!(
        (root.time - t_final).abs() <= f64::EPSILON * t_final,
        "the run ended at {}",
        root.time
    );

    let mut a = store(8, 0, BoundaryType::Outflow, BoundaryType::CoarseFine, 0.0);
    let mut b = store(8, 8, BoundaryType::CoarseFine, BoundaryType::Outflow, 0.0);
    let kernels = FixedCfl::uniform(None);
    let mut stores = vec![&mut *a, &mut *b];
    let diagnostics = evolve_decomposed(
        &mut stores,
        &[&kernels, &kernels],
        [2],
        &[0, 0],
        Timestepping::Euler,
        0.0,
        t_final,
        1,
        &LocalCopy,
        |_, _, _| ControlFlow::Continue(()),
    );
    assert!(
        diagnostics.halt.is_none(),
        "a completed march reported a halt: {:?}",
        diagnostics.halt
    );
    for s in &stores {
        assert_eq!(s.iteration, 3);
        assert!(
            (s.time - t_final).abs() <= f64::EPSILON * t_final,
            "a tile ended at {}",
            s.time
        );
    }
}

// the standalone single-grid driver honors the user cap: steps of 0.01 under a cap of 0.004 reach
// 0.02 in five steps
#[test]
fn the_standalone_driver_steps_at_the_user_cap() {
    let mut sim = store(16, 0, BoundaryType::Outflow, BoundaryType::Outflow, 0.004);
    symbi::sim::evolve::evolve(&mut sim, &FixedCfl::uniform(None), 0.02).expect("the capped run completes");
    assert_eq!(sim.iteration, 5, "the run stepped at {} per step", sim.time / sim.iteration as f64);
}
