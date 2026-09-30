# =============================================================================
# viscous_isothermal_disk_cyl.py
#
# the rotating isothermal core on the (r, phi) disk plane under a constant
# kinematic viscosity nu: the stress transports angular momentum out of the
# solid-body core across its edge at r = 0.3.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.isothermal_disk_cyl import IsothermalDiskCyl


class ViscousIsothermalDiskCyl(IsothermalDiskCyl):
    """the rotating isothermal core under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
