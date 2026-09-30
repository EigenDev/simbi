# =============================================================================
# viscous_imhd_spherical_field.py
#
# the isothermal swirling pulse on a spherical shell in a vertical field under a
# constant kinematic viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.imhd_spherical_field import IsothermalSphericalField


class ViscousIsothermalSphericalField(IsothermalSphericalField):
    """the isothermal magnetized spherical-shell pulse under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
