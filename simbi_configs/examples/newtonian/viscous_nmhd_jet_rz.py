# =============================================================================
# viscous_nmhd_jet_rz.py
#
# the swirling magnetized jet core on the cylindrical (r, z) plane under a constant
# kinematic viscosity nu.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.newtonian.nmhd_jet_rz import NewtonianJetRz


class ViscousNewtonianJetRz(NewtonianJetRz):
    """the magnetized (r, z) jet core under a constant viscosity."""

    nu: Annotated[float, ProblemParam(1.0e-2, cli=True, description="constant kinematic viscosity")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return self.nu
