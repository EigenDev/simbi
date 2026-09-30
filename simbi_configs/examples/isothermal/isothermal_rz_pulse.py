# =============================================================================
# isothermal_rz_pulse.py
#
# the off-axis pulse on the cylindrical (r, z) plane in isothermal hydro: the
# newtonian problem's density and velocity with the closure p = cs^2 rho in place
# of its pressure. the axis at r = 0, outflow elsewhere.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi.types.typing import GasStateGenerator, InitialStateType
from simbi_configs.examples.newtonian.rz_pulse import RzPulse


class IsothermalRzPulse(RzPulse):
    """the off-axis pulse on the cylindrical (r, z) plane in isothermal hydro."""

    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]

    def initial_primitive_state(self) -> InitialStateType:
        """the base problem's density and velocity; the closure p = cs^2 rho replaces its pressure"""
        gas_state = super().initial_primitive_state()

        def isothermal_state() -> GasStateGenerator:
            for state in gas_state():
                yield state[:-1]

        return isothermal_state
