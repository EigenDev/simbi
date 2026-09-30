# =============================================================================
# viscous_isothermal_rz_pulse.py
#
# the isothermal off-axis pulse on the cylindrical (r, z) plane under a constant
# kinematic viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.isothermal_rz_pulse import IsothermalRzPulse


class ViscousIsothermalRzPulse(IsothermalRzPulse):
    """the isothermal (r, z) pulse under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
