# =============================================================================
# imhd_jet_rz.py
#
# the swirling magnetized jet core on the cylindrical (r, z) plane in isothermal
# MHD: the newtonian problem's density, velocity and in-plane field with the
# closure p = cs^2 rho in place of its pressure.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi.types.typing import GasStateGenerator, InitialStateType
from simbi_configs.examples.newtonian.nmhd_jet_rz import NewtonianJetRz


class IsothermalJetRz(NewtonianJetRz):
    """the swirling magnetized jet core on the (r, z) plane in isothermal MHD."""

    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="physics regime")]
    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]

    def initial_primitive_state(self) -> InitialStateType:
        """the base problem's density and velocity; the closure p = cs^2 rho replaces its pressure"""
        gas_state, *faces = super().initial_primitive_state()

        def isothermal_state() -> GasStateGenerator:
            for state in gas_state():
                yield state[:-1]

        return (isothermal_state, *faces)
