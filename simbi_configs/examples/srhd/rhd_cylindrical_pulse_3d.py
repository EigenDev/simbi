# =============================================================================
# rhd_cylindrical_pulse_3d.py
#
# the shaded pressure pulse on a rotating 3d cylindrical annulus in special-
# relativistic hydro: the same initial state as the newtonian problem, every
# speed well below c, under an ideal gas with adiabatic index 4/3.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi_configs.examples.newtonian.cylindrical_pulse_3d import CylindricalPulse3D


class RhdCylindricalPulse3D(CylindricalPulse3D):
    """the shaded pressure pulse on a rotating 3d cylindrical annulus in special-relativistic hydro."""

    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
