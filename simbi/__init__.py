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
# the afterglow light curve, STREAMED over checkpoints via the single cpu_ext catalog path
# (afterglow/lightcurve.py). this replaces the former symbi-rad-py `rad_hydro.lightcurve`
# (a parallel self-contained binding) so there is ONE afterglow code path with all the fixes.
# images use the `simbi afterglow skymap` cli (cpu_ext), not a top-level export.
try:
    from .afterglow.lightcurve import afterglow_lightcurve
except ImportError:
    afterglow_lightcurve = None

# the installed package version (set by maturin from pyproject `[project] version`).
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("simbi")
except PackageNotFoundError:
    __version__ = "0.0.0"

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
    # version
    "__version__",
]
