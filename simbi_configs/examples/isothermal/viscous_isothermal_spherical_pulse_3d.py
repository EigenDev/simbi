# =============================================================================
# viscous_isothermal_spherical_pulse_3d.py
#
# the isothermal shaded pulse on a 3d spherical shell under a constant kinematic
# viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.isothermal_spherical_pulse_3d import IsothermalSphericalPulse3D


class ViscousIsothermalSphericalPulse3D(IsothermalSphericalPulse3D):
    """the isothermal 3d spherical shell pulse under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
