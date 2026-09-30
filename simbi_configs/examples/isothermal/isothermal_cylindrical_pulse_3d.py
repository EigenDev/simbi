# =============================================================================
# isothermal_cylindrical_pulse_3d.py
#
# the shaded pulse on a rotating 3d cylindrical annulus in isothermal hydro: the
# newtonian problem's density and velocity with the closure p = cs^2 rho in place
# of its pressure.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi.types.typing import GasStateGenerator, InitialStateType
from simbi_configs.examples.newtonian.cylindrical_pulse_3d import CylindricalPulse3D


class IsothermalCylindricalPulse3D(CylindricalPulse3D):
    """the shaded pulse on a rotating 3d cylindrical annulus in isothermal hydro."""

    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]

    def initial_primitive_state(self) -> InitialStateType:
        """the base problem's density and velocity; the closure p = cs^2 rho replaces its pressure"""
        gas_state = super().initial_primitive_state()

        def isothermal_state() -> GasStateGenerator:
            for state in gas_state():
                yield state[:-1]

        return isothermal_state
