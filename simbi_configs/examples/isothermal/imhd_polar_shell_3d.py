# =============================================================================
# imhd_polar_shell_3d.py
#
# the swirling, azimuthally shaded pulse on a 3D spherical shell spanning both
# poles, in a uniform vertical field, in isothermal MHD: the newtonian
# problem's density, velocity and divergence-free face field with the closure
# p = cs^2 rho in place of its pressure.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi.types.typing import GasStateGenerator, InitialStateType
from simbi_configs.examples.newtonian.nmhd_polar_shell_3d import NewtonianPolarShell3D


class IsothermalPolarShell3D(NewtonianPolarShell3D):
    """the swirling pulse on a 3D spherical shell through both poles, isothermal MHD."""

    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="physics regime")]
    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]

    def initial_primitive_state(self) -> InitialStateType:
        """the base problem's density and velocity; the closure p = cs^2 rho replaces its pressure"""
        gas_state, *faces = super().initial_primitive_state()

        def isothermal_state() -> GasStateGenerator:
            for state in gas_state():
                yield state[:-1]

        return (isothermal_state, *faces)
