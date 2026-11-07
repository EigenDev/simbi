// =============================================================================
// recovery_failure_without_correction.rs
//
// a failed recovery on a kernel set with no first-order flux correction, through the production
// single-hierarchy driver: the isothermal gas on the lifted (r, theta) and (R, z) planes. the
// premise is asserted: the kernel set reports no correction and the root step no retry source,
// so nothing can restore a failed attempt. stage 0 of root step 2 has one cell's density poisoned
// before its recovery; a second slot poisons another cell at stage 1, so a stage run after the
// failure would mark it fired. the gates:
// - the run stops on the failed attempt as a crash a restart refuses: a panic report, no halt, not
//   restartable
// - the report names step 1, stage 0, the poisoned cell and its values
// - no later stage runs: the stage-1 slot stays unfired, and the attempted receipts hold no stage
//   of root step 2
// - the clock stays at the entry of root step 2: the iteration and the time of a clean run stopped
//   after step 1
// - the accepted receipts are the clean run's after step 1, bit for bit, and carry wall transport
// the first-order correction's own failure path is gated in lifted_recovery_acceptance.rs.
//
// run: cargo test -p symbi --test recovery_failure_without_correction
// =============================================================================

use std::f64::consts::PI;
use std::ops::ControlFlow;

use symbi::prelude::SimSubstrate;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_sim::conservation_ledger::{self, ConservationLedger};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const NI: usize = 16;
const NJ: usize = 8;
const POISONED: [isize; 2] = [7, 3];
const LATER: [isize; 2] = [9, 5];

macro_rules! lifted {
    ($chart:expr, $origin:expr, $extent:expr, $armed:expr) => {{
        type S<C> = SimStateGeneric<IsoNewtonian, 2, 3, C, Isothermal<f64>, CpuSpace, HostMemory>;
        let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
        let sim = S::build(IsoNewtonian, Isothermal { cs: 0.5 }, $chart)
            .cells([NI, NJ])
            .origin($origin)
            .spacing([$extent[0] / NI as f64, $extent[1] / NJ as f64])
            .boundaries(Boundaries::per_axis(walls))
            .cfl(0.2)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("the plane")
            .set_initial(|x: [f64; 2]| PrimG::<f64, 3, IsoModel>::isothermal(Density(1.0 + 0.2 * x[1].sin()), Tensor::new([-0.1, 0.02, 0.2])))
            .build();
        if $armed {
            // root step 2 enters at iteration 1: its stage 0 fails; the stage-1 slot reports a later stage
            *sim.workspace.unrecoverable_once.lock().unwrap() = vec![(1, 0, POISONED, false), (1, 1, LATER, false)];
        }
        let kernels = sim.substrate();
        let mut hier = Hierarchy::single(sim, kernels);
        hier.prime();
        hier
    }};
}

fn bits(l: &ConservationLedger) -> Vec<u64> {
    conservation_ledger::to_values(l).iter().map(|v| v.to_bits()).collect()
}

macro_rules! gate {
    ($label:expr, $armed:expr, $clean:expr) => {{
        let label = $label;
        let mut armed = $armed;
        // the premise: no first-order correction and no retry source, so no restore exists
        assert!(!KernelSet::<2, 3, HostMemory, f64>::fofc_active(&armed.levels[0].kernels), "{label}: the kernel set carries the correction");
        assert!(!armed.root_step_rejectable(), "{label}: the root step has a retry source");
        armed.evolve_with_callback(10.0, 1, |_| ControlFlow::Continue(())).expect("the armed run");
        let (attempted, accepted) = conservation_ledger::ledger_report();

        let crash = armed.crash.clone().expect("the armed run never stopped; the failed recovery was accepted");
        let report = crash.panic.clone().expect("the stop is not the unrestored crash a failure without a retry entry ends in");
        assert!(crash.halt.is_none() && !crash.restartable(), "{label}: an unrestored stop is published as restartable");
        assert!(report.contains("recovery failed at step 1 stage 0"), "{label}: the report names another stage: {report}");
        assert!(report.contains("first cell [7, 3]") && report.contains("rho "), "{label}: the report omits the cell or its values: {report}");

        let slots = armed.levels[0].state.workspace.unrecoverable_once.lock().unwrap().clone();
        assert!(slots[0].3, "{label}: the stage-0 poison never fired; the gate is vacuous");
        assert!(!slots[1].3, "{label}: stage 1 of the failed step ran after its stage 0 failed");
        assert_eq!(attempted.stages, 2, "{label}: the attempted receipts hold {} stages, root step 1's two and none after", attempted.stages);

        let mut clean = $clean;
        clean
            .evolve_with_callback(10.0, 1, |h| if h.levels[0].state.iteration >= 1 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
            .expect("the clean run");
        let (_, clean_accepted) = conservation_ledger::ledger_report();
        let (a, c) = (&armed.levels[0].state, &clean.levels[0].state);
        assert_eq!(crash.iter, 1, "{label}: the stop is reported past the entry of root step 2");
        assert_eq!((a.iteration, a.time.to_bits()), (c.iteration, c.time.to_bits()), "{label}: the clock left the entry of root step 2");
        assert!(accepted.stages == 2 && accepted.total_outflow().mass != 0.0, "{label}: the accepted receipts carry no wall transport: {accepted:?}");
        assert_eq!(bits(&accepted), bits(&clean_accepted), "{label}: the accepted receipts differ from the clean run's after step 1");
    }};
}

#[test]
fn a_failed_recovery_without_the_correction_stops_on_the_spherical_plane() {
    gate!(
        "sphere",
        lifted!(Spherical, [1.0, 0.0], [1.0, PI], true),
        lifted!(Spherical, [1.0, 0.0], [1.0, PI], false)
    );
}

#[test]
fn a_failed_recovery_without_the_correction_stops_on_the_cylindrical_plane() {
    gate!(
        "cylinder",
        lifted!(Cylindrical, [0.5, -1.0], [1.0, 2.0], true),
        lifted!(Cylindrical, [0.5, -1.0], [1.0, 2.0], false)
    );
}
