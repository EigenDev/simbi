# =============================================================================
# simbi/simulation/__init__.py
#
# clean exports for simulation module.
# this is the public api for defining and running simbi problems.
#
# usage:
#   from simbi.simulation import SimbiProblem, ProblemParam, run
# =============================================================================
# re-export commonly needed types for convenience
from simbi.types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Reconstruction,
    Regime,
    Solver,
    TimeStepping,
)
from simbi.types.typing import (
    GasStateFunction,
    GasStateGenerator,
    InitialStateType,
    MHDStateGenerators,
)

from .checkpoint import merge_with_checkpoint
from .param import ProblemParam, get_param_metadata
from .problem import SimbiProblem
from .runner import run, to_execution_dict

__all__ = [
    # core api
    "SimbiProblem",
    "ProblemParam",
    "run",
    # utilities
    "to_execution_dict",
    "merge_with_checkpoint",
    "get_param_metadata",
    # types - input
    "BoundaryCondition",
    "CellSpacing",
    "CoordSystem",
    "Reconstruction",
    "Regime",
    "Solver",
    "TimeStepping",
    # types - generators
    "GasStateGenerator",
    "GasStateFunction",
    "InitialStateType",
    "MHDStateGenerators",
]
