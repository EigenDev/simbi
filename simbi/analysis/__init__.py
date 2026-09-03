# =============================================================================
# analysis
#
# offline science diagnostics over run outputs (docs/ideas/accretor.md §5):
# the body-exchange time series (Mdot(t), F_acc(t)) written by the run into
# each checkpoint's `body_diagnostics` group, the steady-state detector and
# windowed averaging, and the sonic-surface / stagnation-point extractors.
# usage:
#   from simbi.analysis import load_body_diagnostics, steady_state_time
#   diag = load_body_diagnostics("run.chkpt.final.h5")
#   t0 = steady_state_time(diag.time, diag.mdot[:, 0])
# =============================================================================

from .accretion import (
    BodyDiagnostics,
    DatDiagnostics,
    averaged_rate,
    load_body_diagnostics,
    load_diagnostics_dat,
    mdot_from_cumulative,
    sonic_radius_vs_angle,
    sphere_flux,
    stagnation_distance,
    steady_state_time,
)
from .anchor_ab import (
    AnchorComparisonError,
    RunRecord,
    SuiteComparison,
    compare_pair,
    compare_suite,
    to_dict,
    to_json,
    to_markdown,
)

__all__ = [
    "AnchorComparisonError",
    "BodyDiagnostics",
    "DatDiagnostics",
    "RunRecord",
    "SuiteComparison",
    "averaged_rate",
    "compare_pair",
    "compare_suite",
    "load_body_diagnostics",
    "load_diagnostics_dat",
    "mdot_from_cumulative",
    "sonic_radius_vs_angle",
    "sphere_flux",
    "stagnation_distance",
    "steady_state_time",
    "to_dict",
    "to_json",
    "to_markdown",
]
