# =============================================================================
# rhd_spherical_pulse_3d.py
#
# the latitude- and longitude-shaded radial pulse on a 3d spherical shell in
# special-relativistic hydro: the same initial state as the newtonian problem,
# every speed well below c, under an ideal gas with adiabatic index 4/3.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi_configs.examples.newtonian.spherical_pulse_3d import SphericalPulse3D


class RhdSphericalPulse3D(SphericalPulse3D):
    """the latitude- and longitude-shaded radial pulse on a 3d spherical shell in special-relativistic hydro."""

    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
