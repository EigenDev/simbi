# =============================================================================
# viscous_isothermal_spherical_pulse.py
#
# the isothermal spherical pulse under a constant kinematic viscosity nu on a
# uniform radial grid, with the coordinate axis at both poles.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi.types import CellSpacing
from simbi_configs.examples.isothermal.isothermal_spherical_pulse import IsothermalSphericalPulse


class ViscousIsothermalSphericalPulse(IsothermalSphericalPulse):
    """the isothermal spherical pulse under a constant viscosity on a uniform radial grid."""

    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LINEAR, description="radial spacing")]
    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
