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
    Neumann,
    Robin,
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
# the afterglow light curve, streamed over checkpoints via the single cpu_ext catalog path
# (afterglow/lightcurve.py). this replaces the former symbi-rad-py `rad_hydro.lightcurve`
# (a parallel self-contained binding) so there is one afterglow code path with all the fixes.
# images are produced through the `simbi afterglow skymap` cli (cpu_ext).
try:
    from .afterglow.lightcurve import afterglow_lightcurve
except ImportError:
    afterglow_lightcurve = None

# the analytic transonic bondi solution (docs/ideas/accretor.md §8): the
# config-side initial condition and the validation target for the emergent
# drain rate. code units G*M = c_inf = rho_inf = 1; bondi_profile(r, gamma)
# returns (rho, u_inflow, pre) with the radial velocity -u * rhat.
try:
    from .libs.cpu_ext import bondi_profile, bondi_sonic_radius, mdot_bondi
except ImportError:
    # a wheel built before these bindings existed: bind stubs that name the fix;
    # a bound None would raise a bare TypeError far from the cause.
    def _stale_wheel_stub(name):
        def _stub(*_a, **_k):
            raise RuntimeError(
                f"simbi.{name} is not in the installed backend — the wheel predates "
                "it. rebuild with `./dev.py build`."
            )
        return _stub

    bondi_profile = _stale_wheel_stub("bondi_profile")
    bondi_sonic_radius = _stale_wheel_stub("bondi_sonic_radius")
    mdot_bondi = _stale_wheel_stub("mdot_bondi")

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
    "Neumann",
    "Robin",
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
    # accretor validation (docs/ideas/accretor.md)
    "bondi_profile",
    "bondi_sonic_radius",
    "mdot_bondi",
    # version
    "__version__",
]
