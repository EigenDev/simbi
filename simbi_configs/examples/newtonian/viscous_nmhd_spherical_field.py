# =============================================================================
# viscous_nmhd_spherical_field.py
#
# the swirling pulse on a spherical shell in a uniform vertical field under a
# constant kinematic viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.newtonian.nmhd_spherical_field import NewtonianSphericalField


class ViscousNewtonianSphericalField(NewtonianSphericalField):
    """the magnetized spherical-shell pulse under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
