# =============================================================================
# viscous_imhd_orszag_tang_3d.py
#
# the three-dimensional isothermal orszag-tang vortex under a constant kinematic
# viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.imhd_orszag_tang_3d import IsothermalOrszagTang3D


class ViscousIsothermalOrszagTang3D(IsothermalOrszagTang3D):
    """the 3d isothermal orszag-tang vortex under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
