# =============================================================================
# rhd_spherical_pulse.py
#
# the latitude-shaded radial pulse on a log-radial spherical shell in special-
# relativistic hydro: the same initial state as the newtonian problem, every
# speed well below c, under an ideal gas with adiabatic index 4/3.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi_configs.examples.newtonian.spherical_pulse import SphericalPulse


class RhdSphericalPulse(SphericalPulse):
    """the latitude-shaded radial pulse on a log-radial spherical shell in special-relativistic hydro."""

    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
