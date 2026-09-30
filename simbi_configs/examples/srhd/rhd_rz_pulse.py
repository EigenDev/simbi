# =============================================================================
# rhd_rz_pulse.py
#
# the off-axis pressure pulse on the cylindrical (r, z) plane in special-
# relativistic hydro: the same initial state as the newtonian problem, every
# speed well below c, under an ideal gas with adiabatic index 4/3.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi_configs.examples.newtonian.rz_pulse import RzPulse


class RhdRzPulse(RzPulse):
    """the off-axis pressure pulse on the cylindrical (r, z) plane in special-relativistic hydro."""

    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
