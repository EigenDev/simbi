# =============================================================================
# imhd_orszag_tang_walls.py
#
# the isothermal orszag-tang vortex with outflow walls in place of the periodic
# box: the vortex and its field leave through the open faces.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import BoundaryCondition
from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang


class IsothermalOrszagTangWalls(IsothermalOrszagTang):
    """the isothermal orszag-tang vortex with outflow walls."""

    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="outflow walls")
    ]
