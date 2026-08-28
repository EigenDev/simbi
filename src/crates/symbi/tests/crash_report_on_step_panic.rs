// =============================================================================
// crash_report_on_step_panic.rs
//
// a panic inside the evolve step ends the run as a reported crash: the evolve
// loop catches it, records a crash report carrying the panic message, and
// fires the observer once with that report visible — the seam the driver uses
// to snapshot the `.crashed` checkpoint. the injector is a delegating kernel
// set that panics inside `godunov_stage` on its third substage, the same call
// path the FOFC freeze-streak halt panics on in production, so the march has
// completed a full healthy step before the failing one. the watchdog's own
// crash class (a bad cfl dt, reported with an empty panic field) is covered
// by its detection tests; this gate pins the panic class.
//
// run: cargo test -p symbi --test crash_report_on_step_panic
// =============================================================================

use std::sync::atomic::{AtomicU32, Ordering};

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi::sim::state::FieldStore;
use symbi::sim::substrate_seam::KernelSet;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 16;
const PANIC_MSG: &str = "injected step panic: the evolve loop converts this into a crash report";

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Inner = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

/// delegates every stage to the production adiabatic kernel set and panics on
/// the third godunov substage: with rk2 that is the second stage of the second
/// root step, so the panic fires from inside a step of a marching run.
struct PanicOnThirdStage {
    inner: Inner,
    stages: AtomicU32,
}

impl KernelSet<3, 3, HostMemory, f64> for PanicOnThirdStage {
    fn flux(&self, store: &FieldStore<3, 3, HostMemory, f64>, dir: usize) {
        self.inner.flux(store, dir);
    }
    fn c2p(&self, store: &FieldStore<3, 3, HostMemory, f64>) {
        self.inner.c2p(store);
    }
    fn godunov_stage(&self, store: &FieldStore<3, 3, HostMemory, f64>, dt: f64, a0: f64, ac: f64) {
        if self.stages.fetch_add(1, Ordering::Relaxed) + 1 >= 3 {
            panic!("{PANIC_MSG}");
        }
        self.inner.godunov_stage(store, dt, a0, ac);
    }
    fn cfl(&self, store: &FieldStore<3, 3, HostMemory, f64>) -> f64 {
        self.inner.cfl(store)
    }
    fn ghost_fill(&self, store: &FieldStore<3, 3, HostMemory, f64>) {
        self.inner.ghost_fill(store);
    }
    fn snapshot(&self, store: &FieldStore<3, 3, HostMemory, f64>) {
        self.inner.snapshot(store);
    }
}

fn build() -> Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, PanicOnThirdStage>
{
    let uniform = |_x: [f64; 3]| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0; 3]),
        pre: 1.0,
    };
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .origin([-0.5, -0.5, -0.5])
        .spacing([dx, dx, dx])
        .ghosts(2)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(uniform)
        .build();
    let inner = Inner::new(GAMMA, CFL, &coarse.geom.allocated)
        .with_solver(Solver::HllcPlus)
        .expect("solver/regime mismatch")
        .reconstruction(Recon::Plm);
    Hierarchy::single(
        coarse,
        PanicOnThirdStage {
            inner,
            stages: AtomicU32::new(0),
        },
    )
}

/// the panic inside the step surfaces as a crash report with the message
/// attached, the observer fires with the report visible, and the march has
/// advanced past its first healthy step before the crash.
#[test]
fn a_step_panic_becomes_a_crash_report_the_observer_sees() {
    let mut hier = build();

    let mut observer_saw_crash = false;
    let result = hier.evolve_with_callback(0.5, 1, |h| {
        if h.crash.is_some() {
            observer_saw_crash = true;
        }
        std::ops::ControlFlow::Continue(())
    });

    assert!(result.is_ok(), "the caught panic ends the march by report");
    let crash = hier
        .crash
        .as_ref()
        .expect("the step panic must record a crash report");
    assert!(
        crash.iter >= 1,
        "the injector fires on the second step, after one healthy accepted step"
    );
    let msg = crash
        .panic
        .as_ref()
        .expect("a panic-origin crash carries its message");
    assert!(
        msg.contains(PANIC_MSG),
        "the report carries the panic payload verbatim; got: {msg}"
    );
    assert!(
        observer_saw_crash,
        "the observer fires once with the crash visible — the driver's snapshot seam"
    );
}
