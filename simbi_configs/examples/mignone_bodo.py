from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *


class MignoneBodo(BaseConfig):
    """
    Mignone & Bodo (2005), Relativistic Test Problems in 1D Fluid
    """

    nzones = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    adiabatic_index = DynamicArg(
        "ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type=float
    )
    problem = DynamicArg(
        "problem", 1, help="test problem to compute", var_type=int, choices=[1, 2]
    )

    @simbi_property
    def initial_primitive_state(self) -> Sequence[Sequence[float]]:
        if self.problem == 2:
            return ((1.0, -0.6, 10.0), (10.0, 0.5, 20.0))
        return ((1.0, 0.9, 1.0), (1.0, 0.0, 10.0))

    @simbi_property
    def bounds(self) -> Sequence[float]:
        return (0.0, 1.0, 0.5)

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> DynamicArg:
        return self.nzones

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srhd"
