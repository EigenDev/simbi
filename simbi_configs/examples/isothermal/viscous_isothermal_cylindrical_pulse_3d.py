# =============================================================================
# viscous_isothermal_cylindrical_pulse_3d.py
#
# the isothermal shaded pulse on a rotating 3d cylindrical annulus under a constant
# kinematic viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.isothermal_cylindrical_pulse_3d import IsothermalCylindricalPulse3D


class ViscousIsothermalCylindricalPulse3D(IsothermalCylindricalPulse3D):
    """the isothermal 3d annulus pulse under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
