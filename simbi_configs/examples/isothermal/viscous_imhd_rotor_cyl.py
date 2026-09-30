# =============================================================================
# viscous_imhd_rotor_cyl.py
#
# the rotating isothermal magnetized core on the (r, phi) disk plane under a
# constant kinematic viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.imhd_rotor_cyl import IsothermalRotorCyl


class ViscousIsothermalRotorCyl(IsothermalRotorCyl):
    """the rotating isothermal magnetized core under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
