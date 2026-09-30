# =============================================================================
# viscous_spherical_pulse_3d.py
#
# the shaded radial pulse on a 3d spherical shell under a constant kinematic
# viscosity nu, the axis at both poles and a periodic azimuth.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.newtonian.spherical_pulse_3d import SphericalPulse3D


class ViscousSphericalPulse3D(SphericalPulse3D):
    """the 3d spherical shell pulse under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
