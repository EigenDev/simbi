# =============================================================================
# simbi - relativistic hydrodynamics simulation framework
#
# primary exports:
#   - SimbiProblem, ProblemParam, run: for defining and running simulations
#   - read_simulation: for loading simulation data
#   - viz: for visualization
# =============================================================================
# utilities
from .functional.helpers import (
    calc_any_mean,
    calc_cell_volume,
    calc_centroid,
    compute_num_polar_zones,
    find_nearest,
)

# data loading
from .reader import read_simulation

# core simulation api
from .simulation import (
    ProblemParam,
    SimbiProblem,
    run,
)

# types for problem definition
from .types.bodies import (
    BinaryComponentConfig,
    BinaryConfig,
    GravitationalSystemConfig,
    ImmersedBodyConfig,
)
from .types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Reconstruction,
    Regime,
    Solver,
    TimeStepping,
)
from .types.typing import (
    GasStateGenerator,
    InitialStateType,
    MHDStateGenerators,
)
from .version import __version_tuple__

# optional: backend extensions (may not be available)
try:
    from .libs.rad_hydro import py_calc_fnu, py_log_events
except ImportError:
    py_calc_fnu = None
    py_log_events = None

__version__ = ".".join(map(str, __version_tuple__))

__all__ = [
    # core api
    "SimbiProblem",
    "ProblemParam",
    "run",
    # data
    "read_simulation",
    # types
    "InitialStateType",
    "GasStateGenerator",
    "MHDStateGenerators",
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
    "BinaryComponentConfig",
    "BoundaryCondition",
    "CellSpacing",
    "CoordSystem",
    "Reconstruction",
    "Regime",
    "Solver",
    "TimeStepping",
    # utilities
    "calc_cell_volume",
    "find_nearest",
    "compute_num_polar_zones",
    "calc_centroid",
    "calc_any_mean",
    # optional
    "py_calc_fnu",
    "py_log_events",
    # version
    "__version__",
]
