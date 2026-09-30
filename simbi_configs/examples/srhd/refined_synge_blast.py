# =============================================================================
# refined_synge_blast.py
#
# the relativistic blast under the synge perfect gas with one fine box of ratio
# two over the hot disk and its first expansion: [0.25, 0.75]^2 of the unit
# square. the blast wave crosses the coarse-fine boundary, where the flux
# register refluxes the root and the fine level restricts into it.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.srhd.synge_blast import SyngeBlast


class RefinedSyngeBlast(SyngeBlast):
    """the synge blast with a fine box over the hot disk."""

    refinement_enabled: Annotated[bool, ProblemParam(True, description="enable mesh refinement")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="coarse + 1 fine")]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam([[0.25, 0.75, 0.25, 0.75]], description="one fine box over the hot disk"),
    ]
    refinement_ratios: Annotated[
        list[int], ProblemParam([2], description="coarse->fine refinement ratio per jump")
    ]
