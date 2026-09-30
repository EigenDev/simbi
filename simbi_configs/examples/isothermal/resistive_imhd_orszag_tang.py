# =============================================================================
# resistive_imhd_orszag_tang.py
#
# the isothermal orszag-tang vortex under a uniform ohmic resistivity eta: the
# induction equation carries the diffusive emf eta*J, which reconnects the
# current sheets the ideal vortex thins to the grid scale.
# =============================================================================
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang


class ResistiveIsothermalOrszagTang(IsothermalOrszagTang):
    """the isothermal orszag-tang vortex with a uniform ohmic resistivity."""

    eta: Annotated[float, ProblemParam(1.0e-2, cli=True, description="ohmic resistivity; 0 = ideal")]

    @computed_field
    @property
    def resistivity(self) -> float:
        return self.eta
