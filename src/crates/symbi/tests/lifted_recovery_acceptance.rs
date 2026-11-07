// =============================================================================
// lifted_recovery_acceptance.rs
//
// a failed recovery never becomes an accepted step on the lifted (r, theta) and (R, z) planes,
// through the production single-hierarchy driver. every attempt of root step 2 has one cell's
// stage output poisoned to an unrecoverable total energy before its recovery, more attempts than
// the retry guard allows. whichever way the step ends, by a retry the guard finally refuses or by
// a stop on an attempt no retry entry restores (the lifted planes carry the first-order correction,
// so the step retries until the correction's freeze limit halts it at the restored entry):
// - the clock stays at the entry of root step 2
// - the report names the failing stage and cell with its values
// - the accepted conservation receipts hold root step 1 alone
// - a stop reported restartable holds exactly the state after step 1; any other stop is marked
//   not restartable
//
// run: cargo test -p symbi --test lifted_recovery_acceptance
// =============================================================================

use std::f64::consts::PI;
use std::ops::ControlFlow;

use symbi::prelude::SimSubstrate;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const NI: usize = 16;
const NJ: usize = 8;
const POISONED: [isize; 2] = [7, 3];
const ARMED_ATTEMPTS: usize = 40;

macro_rules! lifted {
    ($chart:expr, $origin:expr, $extent:expr, $armed:expr) => {{
        type S<C> = SimStateGeneric<Newtonian, 2, 3, C, IdealGas<f64>, CpuSpace, HostMemory>;
        let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
        let sim = S::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, $chart)
            .cells([NI, NJ])
            .origin($origin)
            .spacing([$extent[0] / NI as f64, $extent[1] / NJ as f64])
            .boundaries(Boundaries::per_axis(walls))
            .cfl(0.2)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("the plane")
            .set_initial(|x: [f64; 2]| Prim::adiabatic(Density(1.0 + 0.2 * x[1].sin()), Tensor::new([0.05, 0.0, 0.3]), Pressure(0.5)))
            .build();
        if $armed {
            // root step 2 enters at iteration 1; each attempt's stage 0 consumes one slot
            *sim.workspace.unrecoverable_once.lock().unwrap() = vec![(1, 0, POISONED, false); ARMED_ATTEMPTS];
        }
        let kernels = sim.substrate();
        let mut hier = Hierarchy::single(sim, kernels);
        hier.prime();
        hier
    }};
}

fn bits(store: &FieldStore<2, 3, HostMemory>) -> Vec<u64> {
    let (c, d) = (&store.fields.cons, &store.geom.interior);
    let mut out = Vec::new();
    for f in [&c.den, c.nrg_field().expect("nrg"), &c.mom[0], &c.mom[1], &c.mom[2]] {
        let v = f.view();
        out.extend(d.iter().map(|x| v.at(x).to_bits()));
    }
    out
}

macro_rules! gate {
    ($label:expr, $armed:expr, $clean:expr) => {{
        let mut armed = $armed;
        armed.evolve_with_callback(10.0, 1, |_| ControlFlow::Continue(())).expect("the armed run");
        let (_, accepted) = symbi_sim::conservation_ledger::ledger_report();
        let crash = armed.crash.clone().expect("the armed run never stopped; a failed recovery was accepted");
        let report = crash.panic.clone().or_else(|| crash.halt.as_ref().map(|h| h.detail().to_string())).unwrap_or_default();
        // without the first-order correction the stop names the stage's failed recovery; with it the
        // retries end at the correction's freeze limit, which names the cell's state
        let named_stage = report.contains("recovery failed at step 1 stage 0") || report.contains("FOFC last-resort freeze");
        assert!(named_stage, "{}: the stop does not name the failing stage: {report}", $label);
        let named_cell = report.contains("first cell [7, 3]") || report.contains("coord=[7, 3]");
        let named_values = report.contains("specific internal energy") || report.contains("cons=(den=");
        assert!(named_cell && named_values, "{}: the stop does not name the cell and its values: {report}", $label);
        assert_eq!(crash.iter, 1, "{}: the stop is reported past the entry of root step 2", $label);
        assert_eq!(armed.levels[0].state.iteration, 1, "{}: the clock left the entry of root step 2", $label);
        assert_eq!(accepted.stages, 2, "{}: the accepted receipts hold {} stages, not root step 1's two", $label, accepted.stages);
        if crash.restartable() {
            let mut clean = $clean;
            clean
                .evolve_with_callback(10.0, 1, |h| if h.levels[0].state.iteration >= 1 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
                .expect("the clean run");
            assert!(bits(&armed.levels[0].state) == bits(&clean.levels[0].state), "{}: a restartable stop left the state after step 1", $label);
        } else {
            assert!(crash.panic.is_some() && crash.halt.is_none(), "{}: an unrestored stop was reported as a restored halt", $label);
        }
    }};
}

#[test]
fn a_failed_recovery_on_the_spherical_plane_is_never_accepted() {
    gate!(
        "sphere",
        lifted!(Spherical, [1.0, 0.0], [1.0, PI], true),
        lifted!(Spherical, [1.0, 0.0], [1.0, PI], false)
    );
}

#[test]
fn a_failed_recovery_on_the_cylindrical_plane_is_never_accepted() {
    gate!(
        "cylinder",
        lifted!(Cylindrical, [0.5, -1.0], [1.0, 2.0], true),
        lifted!(Cylindrical, [0.5, -1.0], [1.0, 2.0], false)
    );
}
