// =============================================================================
// dispatch_profile_reports.rs
//
// `SYMBI_DISPATCH_PROF` must produce numbers. the profiler counts calls and splits the
// per-call cost of the AMR-transfer dispatch into the registry NAME LOOKUP and the kernel
// execution, so the prolong/restrict overhead can be attributed to scheduling rather than
// to arithmetic.
//
// the failure this closes is silence, not a crash: counters that are written and never read
// make a set variable look like evidence that dispatch cost was measured, when nothing was
// reported at all. a run with refinement drives the transfer dispatch, so a zero call count
// after evolving a two-level hierarchy means the instrument is inert.
//
// the profiler is env-gated through a process-global OnceLock, so this binary holds ONLY
// this law (the flag must be set before the first dispatch).
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi::symbi_exec::policy::report_dispatch_profile;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 64;
const CFL: f64 = 0.4;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

fn kset(s: &Sim) -> Kset {
    Kset::new(GAMMA, CFL, &s.geom.allocated)
}

#[test]
fn the_dispatch_profiler_reports_numbers_when_enabled() {
    // the profiler latches SYMBI_DISPATCH_PROF once per process; set it before any
    // dispatch. sound here because this binary runs only this law.
    unsafe { std::env::set_var("SYMBI_DISPATCH_PROF", "1") };

    let dx = 1.0 / N as f64;
    // a sod jump so the refined patch carries real gradients: a uniform state would still
    // exercise the transfer dispatch, but a stationary one invites a future short-circuit
    // that skips it.
    let ic = |x: [f64; 1]| {
        let (rho, pre) = if x[0] < 0.5 { (1.0, 1.0) } else { (0.125, 0.1) };
        Prim {
            rho,
            vel: symbi_algebra::Tensor::new([0.0]),
            pre,
        }
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(ic)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [0.4],
        x_hi: [0.6],
    }];
    let mut hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| kset(s)).unwrap();
    hier.levels[1].state.seed_cells(ic);

    hier.evolve(0.05).unwrap();

    // NON-VACUITY: the run has to have actually refined and stepped, or a zero call count
    // below would mean the setup never reached a transfer rather than that the instrument
    // is inert.
    assert!(
        hier.levels.len() > 1,
        "the hierarchy is unrefined, so no transfer dispatch runs and this law is vacuous"
    );
    assert!(
        hier.levels[0].state.iteration > 1,
        "the run took {} steps; a transfer needs at least one",
        hier.levels[0].state.iteration
    );

    let (calls, lookup_ns, exec_ns) = report_dispatch_profile();
    assert!(
        calls > 0,
        "SYMBI_DISPATCH_PROF is set and a refined run stepped {} times, yet the profiler \
         counted no dispatches -- setting the variable produces no report",
        hier.levels[0].state.iteration
    );
    // both halves must be attributable: a zero on either side means the split the profiler
    // exists to make is not being taken, so its percentages would be meaningless.
    assert!(
        lookup_ns > 0 && exec_ns > 0,
        "the lookup/execution split collapsed (lookup {lookup_ns} ns, exec {exec_ns} ns) \
         over {calls} calls, so the attribution the profiler reports is not measured"
    );
}
