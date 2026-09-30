# =============================================================================
# viscous_nmhd_rotor_cyl.py
#
# the magnetized rotor on the (r, phi) disk plane under a constant kinematic
# viscosity nu: the stress transports angular momentum out of the solid-body
# core and heats the gas.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.newtonian.nmhd_rotor_cyl import NewtonianRotorCyl


class ViscousNewtonianRotorCyl(NewtonianRotorCyl):
    """the magnetized rotor on the disk plane under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
