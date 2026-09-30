# =============================================================================
# viscous_nmhd_orszag_tang_3d.py
#
# the three-dimensional newtonian orszag-tang vortex under a constant kinematic
# viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.newtonian.nmhd_orszag_tang_3d import NewtonianOrszagTang3D


class ViscousNewtonianOrszagTang3D(NewtonianOrszagTang3D):
    """the 3d newtonian orszag-tang vortex under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
