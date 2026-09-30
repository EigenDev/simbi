# =============================================================================
# viscous_isothermal_shear_3d.py
#
# the three-axis isothermal shear on a periodic box under a constant kinematic
# viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.isothermal_shear_3d import IsothermalShear3D


class ViscousIsothermalShear3D(IsothermalShear3D):
    """the three-axis isothermal shear under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
