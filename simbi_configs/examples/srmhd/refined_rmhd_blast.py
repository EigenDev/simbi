# =============================================================================
# refined_rmhd_blast.py
#
# the magnetized blast on a statically refined 2.5D mesh in special-relativistic
# MHD: the central overpressure in a uniform oblique field, the fine box of
# ratio two around it and the outflow walls of the newtonian problem, evolved
# under the relativistic equations. the staggered faces start divergence-free
# and the fine faces are seeded by divergence-free prolongation of the coarse
# faces.
#
# usage:
#  simbi run refined_rmhd_blast.py
#  simbi launch refined_rmhd_blast.py --layout layout.toml
# =============================================================================
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi_configs.examples.newtonian.refined_mhd_blast import RefinedMhdBlast


class RefinedRmhdBlast(RefinedMhdBlast):
    """the refined magnetized blast in special-relativistic MHD."""

    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="physics regime")]
