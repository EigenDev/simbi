// =============================================================================
// run_diagnostics.rs
//
// the run-owned diagnostics one run produced, returned by value across the
// driver boundary. holds settled, accepted evidence only — no transaction
// buckets, no process-global read. the run's driver extracts it from that run's
// own ledger scope and hands it back, so two sequential runs and two concurrent
// runs each report their own evidence with no cross-contamination.
//
// usage:
//   let diagnostics = hier.evolve_with_callback(t_final, interval, cb)?;
//   let projected = diagnostics.projection.projected_cells;
// =============================================================================

use crate::guard_ledger::GuardTotals;
use crate::projection_ledger::ProjectionLedger;

/// a march that stopped on a timestep it could not take: the root iteration and time its last
/// state holds, the cfl estimate that failed (non-finite or non-positive, or the reduced retry that
/// could no longer advance time) and the estimate before it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestepHalt {
    pub iter: u64,
    pub time: f64,
    pub dt_cfl: f64,
    pub dt_prev: f64,
}

/// the accepted evidence of one run.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunDiagnostics {
    /// the admissible-boundary projection's accepted intervention totals, booked
    /// over the states that survived into the solution.
    pub projection: ProjectionLedger,
    /// the FOFC guard acts — troubled and frozen cell counts — accepted over the
    /// states that survived into the solution.
    pub guards: GuardTotals,
    /// the timestep the march halted on, when it stopped short of its final time on a state it
    /// could not advance; the caller reports it and publishes the state as crashed.
    pub halt: Option<TimestepHalt>,
}
