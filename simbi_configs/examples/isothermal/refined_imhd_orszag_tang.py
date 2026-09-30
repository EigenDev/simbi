# =============================================================================
# refined_imhd_orszag_tang.py
#
# the isothermal orszag-tang vortex with outflow walls, with one fine box of ratio
# two over the central half of the unit square, [0.25, 0.75]^2: the coarse-fine
# boundary refluxes the root and the fine level restricts into it, in the
# globally isothermal closure p = cs^2 rho.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.isothermal.imhd_orszag_tang_walls import IsothermalOrszagTangWalls


class RefinedIsothermalOrszagTang(IsothermalOrszagTangWalls):
    """the isothermal orszag-tang vortex with outflow walls, refined over the central half."""

    refinement_enabled: Annotated[bool, ProblemParam(True, description="enable mesh refinement")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="coarse + 1 fine")]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam([[0.25, 0.75, 0.25, 0.75]], description="one fine box over the central half"),
    ]
    refinement_ratios: Annotated[
        list[int], ProblemParam([2], description="coarse->fine refinement ratio per jump")
    ]
