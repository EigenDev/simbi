// =============================================================================
// timestep_halt.rs
//
// a march that meets a cfl estimate it cannot step on (non-finite or
// non-positive) halts on its last state and reports the halt: the single grid
// records it on the hierarchy, a decomposed march returns it in its run
// diagnostics. a mock kernel set steps cleanly until the third root iteration
// and estimates a NaN timestep from then on; both drivers stop at iteration
// three with the same numbers, the decomposed one without a panic and without
// a further observer call that could publish the halted state.
//
// run: cargo test -p symbi --test timestep_halt
// =============================================================================

use std::ops::ControlFlow;

use symbi::sim::decomp::{LocalCopy, evolve_decomposed};
use symbi::sim::refinement::Hierarchy;
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

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

/// a kernel set that recovers primitives exactly and estimates DT until the root iteration
/// reaches HALT_AT, NaN from then on
struct PoisonedCfl;

impl KernelSet<1, 1, HostMemory, f64> for PoisonedCfl {
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
            store.fields.prim.pre_field().unwrap().view_mut().set(cell, (GAMMA - 1.0) * (nrg - 0.5 * mom * mom / rho));
        }
        store.mark_primitives_recovered();
    }
    fn godunov_stage(&self, _store: &FieldStore<1, 1, HostMemory>, _dt: f64, _a0: f64, _ac: f64) {}
    fn cfl(&self, store: &FieldStore<1, 1, HostMemory>) -> f64 {
        if store.iteration >= HALT_AT { f64::NAN } else { DT }
    }
    fn ghost_fill(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn snapshot(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn horizon_accretion(&self, _store: &FieldStore<1, 1, HostMemory>, _diagnostic_radius: f64) -> (f64, f64) {
        (0.0, 0.0)
    }
}

fn store(cells: usize, offset: usize, lo: BoundaryType, hi: BoundaryType) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
        .origin([0.0])
        .index_offset([offset as isize])
        .spacing([1.0 / 16.0])
        .boundaries(Boundaries::per_axis([[lo, hi]]))
        .timestepping(Timestepping::Euler)
        .cfl(0.4)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build()
}

#[test]
fn a_single_grid_records_the_halt_on_its_hierarchy() {
    let mut hier = Hierarchy::single(store(16, 0, BoundaryType::Outflow, BoundaryType::Outflow), PoisonedCfl);
    hier.evolve_with_callback(1.0, 1, |_| ControlFlow::Continue(())).expect("the single-grid run returns");
    let crash = hier.crash.as_ref().expect("the single grid records the halt");
    assert_eq!(crash.iter, HALT_AT);
    assert!(crash.dt_cfl.is_nan() && crash.panic.is_none(), "{crash:?}");
    assert_eq!(hier.levels[0].state.iteration, HALT_AT, "the halt leaves the last state in place");
}

#[test]
fn a_decomposed_march_returns_the_halt_without_a_panic() {
    let mut a = store(8, 0, BoundaryType::Outflow, BoundaryType::CoarseFine);
    let mut b = store(8, 8, BoundaryType::CoarseFine, BoundaryType::Outflow);
    let mut observed = Vec::new();
    let mut stores = vec![&mut *a, &mut *b];
    let diagnostics = evolve_decomposed(&mut stores, &[&PoisonedCfl, &PoisonedCfl], [2], &[0, 0], Timestepping::Euler, 0.0, 1.0, 1, &LocalCopy, |iter, _, _| {
        observed.push(iter);
        ControlFlow::Continue(())
    });
    let halt = diagnostics.halt.expect("the decomposed march reports the halt");
    assert_eq!(halt.iter, HALT_AT);
    assert!(halt.dt_cfl.is_nan(), "{halt:?}");
    assert_eq!(halt.dt_prev, DT);
    assert!((halt.time - HALT_AT as f64 * DT).abs() < 1e-12, "{halt:?}");
    assert_eq!(observed, vec![1, 2, 3], "the observer sees each accepted step and nothing after the halt");
    for s in &stores {
        assert_eq!(s.iteration, HALT_AT, "every tile holds its last state");
    }
}
