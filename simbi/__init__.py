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
    calc_dlogt,
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

# optional: radiation post-processing (rust symbi-afterglow via rad_hydro). the
# rust module exposes the self-contained `lightcurve` / `skymap` pipeline (it
# reads the checkpoint itself), replacing the legacy C++ py_calc_fnu/py_log_events.
try:
    from .libs.rad_hydro import lightcurve as afterglow_lightcurve
    from .libs.rad_hydro import skymap as afterglow_skymap
except ImportError:
    afterglow_lightcurve = None
    afterglow_skymap = None

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
    "calc_dlogt",
    # optional
    "afterglow_lightcurve",
    "afterglow_skymap",
    # version
    "__version__",
]
