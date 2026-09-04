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

/// the accepted evidence of one run.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunDiagnostics {
    /// the admissible-boundary projection's accepted intervention totals, booked
    /// over the states that survived into the solution.
    pub projection: ProjectionLedger,
    /// the FOFC guard acts — troubled and frozen cell counts — accepted over the
    /// states that survived into the solution.
    pub guards: GuardTotals,
}
