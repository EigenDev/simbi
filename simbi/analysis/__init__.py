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
    averaged_rate,
    load_body_diagnostics,
    sonic_radius_vs_angle,
    stagnation_distance,
    steady_state_time,
)

__all__ = [
    "BodyDiagnostics",
    "averaged_rate",
    "load_body_diagnostics",
    "sonic_radius_vs_angle",
    "stagnation_distance",
    "steady_state_time",
]
