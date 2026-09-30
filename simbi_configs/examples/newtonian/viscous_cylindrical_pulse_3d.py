# =============================================================================
# viscous_cylindrical_pulse_3d.py
#
# the shaded pulse on a rotating 3d cylindrical annulus under a constant kinematic
# viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.newtonian.cylindrical_pulse_3d import CylindricalPulse3D


class ViscousCylindricalPulse3D(CylindricalPulse3D):
    """the 3d cylindrical annulus pulse under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
