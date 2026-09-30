# =============================================================================
# viscous_spherical_pulse.py
#
# the spherical pulse under a constant kinematic viscosity nu on a uniform radial
# grid: the meridional flow v_theta = 0.05 sin(2 theta) and the pulse's radial
# structure diffuse through the curvilinear stress on the (r, theta) chart.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi.types import CellSpacing
from simbi_configs.examples.newtonian.spherical_pulse import SphericalPulse


class ViscousSphericalPulse(SphericalPulse):
    """the spherical pulse under a constant viscosity on a uniform radial grid."""

    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LINEAR, description="radial spacing")]
    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
