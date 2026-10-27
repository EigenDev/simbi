# =============================================================================
# test_parity_coverage_sources.py
#
# user sources and relativistic riemann solvers through the tiled and launched
# drivers, each held to the exact parity contract against the single device on
# the session's backend and placement (parity_table, parity_harness):
# - every checkpoint of two host tiles, two launched workers on an even cut and
#   three on an uneven cut equals the single device's
# - half the steps on two workers resumed onto three uneven workers equals the
#   resumed and the uninterrupted single device
# - a stage rejected on one worker replays as the clean launch capped at the
#   accepted timestep
# - each entry's control (the same fixture without the setting) runs
#   differently on the single device, so the setting reaches the run
# the table:
# - cooling, velocity relaxation and conserved-vector injection on a drifting
#   pressure bump on a 24-cell plane
# - an outer sponge on the relativistic ram44 shock, relaxing through the
#   relativistic state law
# - an inverse-square radial force on the log-radial spherical (r, theta) shell
# - a trilinear tabulated energy source in 3D under one fine box (outside the
#   gpu device scope: refined user sources)
# - hllc and hllc+ on the relativistic ram44 shock; hllc on the relativistic
#   orszag-tang vortex
# entries outside the gpu admission run under SIMBI_LAUNCH_EXPERIMENTAL.
#
# run: pytest -m simulation simbi/simulation/tests/test_parity_coverage_sources.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
import pytest

from simbi.simulation.tests.parity_table import REJECTED_STEP, Feature, install
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation


SOURCE_PLANE = '''
import math
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime


class {name}(SimbiProblem):
    """a gaussian pressure bump in gas drifting at (0.2, 0.1) across [-1, 1]^2 on a 24-cell plane"""

    resolution: Annotated[tuple[int, int], ProblemParam((24, 24), cli=True, description="grid")]
    bounds: list[tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 1.0

    @property
    def source_expressions(self) -> list:
        x, y = expr.coords(2)
        disc = expr.where(x * x + y * y < 0.16, 1.0, 0.0)
        return [{source}]

    def initial_primitive_state(self):
        def gas_state():
            nx, ny = self.resolution
            for jj in range(ny):
                y = -1.0 + (jj + 0.5) * 2.0 / ny
                for ii in range(nx):
                    x = -1.0 + (ii + 0.5) * 2.0 / nx
                    yield (1.0, 0.2, 0.1, 1.0 + 0.5 * math.exp(-(x * x + y * y) / 0.1))

        return gas_state
'''
RHD_SPONGE = '''
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.srhd.ram44 import Ram44


class RhdSponge(Ram44):
    """the ram44 shock with an upper-band sponge relaxing to gas at rest at unit density and
    pressure, converted to conserved variables by the relativistic state law"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), cli=True, description="grid")]

    @property
    def source_expressions(self) -> list:
        x, y = expr.coords(2)
        kappa = expr.where(y > 0.75, 5.0, 0.0)
        return [expr.sponge([kappa, 1.0, 0.0, 0.0, 1.0], dim=2)]
'''
RHD_SOLVER = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Solver
from simbi_configs.examples.srhd.ram44 import Ram44


class {name}(Ram44):
    """the ram44 shock on a 32 x 32 plane under the {label} flux"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), cli=True, description="grid")]
    solver: Annotated[Solver, ProblemParam(Solver.{solver}, description="{label}")]
'''
RHD_CONTROL = ('''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.srhd.ram44 import Ram44


class RhdControl(Ram44):
    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), cli=True, description="grid")]
''', "RhdControl")
RMHD_HLLC = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Solver
from simbi_configs.examples.srmhd.rmhd_orszag_tang import OrszagTang


class RmhdHllc(OrszagTang):
    """the relativistic orszag-tang vortex on a 32 x 32 plane under the hllc flux"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 32, 1), cli=True, description="grid")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="hllc")]
'''
RMHD_CONTROL = ('''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.srmhd.rmhd_orszag_tang import OrszagTang


class RmhdControl(OrszagTang):
    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 32, 1), cli=True, description="grid")]
''', "RmhdControl")
SPHERICAL_FORCE = '''
import simbi.expression as expr
from simbi_configs.examples.newtonian.spherical_pulse import SphericalPulse


class SphericalForce(SphericalPulse):
    """the latitude-shaded pulse on the log-radial (r, theta) shell under the inverse-square
    radial acceleration -1 / r^2"""

    @property
    def source_expressions(self) -> list:
        r, theta = expr.coords(2)
        return [expr.force([-1.0 / (r * r), 0.0 * theta], dim=2)]
'''
SPHERICAL_CONTROL = ('''
from simbi_configs.examples.newtonian.spherical_pulse import SphericalPulse


class SphericalControl(SphericalPulse):
    pass
''', "SphericalControl")
REFINED_TABULATED = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.tabulated_source_3d import TabulatedSource3D


class RefinedTabulated(TabulatedSource3D):
    """the trilinear tabulated heating of the unit cube at 16^3 under one fine box over the
    central half"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((16, 16, 16), cli=True, description="grid")]
    end_time: float = 1.0
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]] | None, ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]], description="box")]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2], description="ratios")]
'''
REFINED_TABULATED_CONTROL = ('''
from feature_tabulated_refined_3d import RefinedTabulated


class Control(RefinedTabulated):
    @property
    def source_expressions(self):
        return []
''', "Control")


def _plane(name: str, source: str) -> str:
    return SOURCE_PLANE.format(name=name, source=source)


PLANE_CONTROL = (_plane("PlaneControl", ""), "PlaneControl")
FLAT_2D = {"resolution": None, "even": [[12], []], "uneven": [[7, 15], []], "rejection": f"reject:1:{REJECTED_STEP}:1"}
RAM44 = {"resolution": (32, 32), "even": [[16], []], "uneven": [[10, 21], []], "rejection": f"reject:1:{REJECTED_STEP}:1"}


FEATURES = {
    "cooling-2d": Feature(_plane("Cooling", "expr.cooling([0.3 * disc], dim=2)"), "Cooling", **FLAT_2D, control=PLANE_CONTROL),
    "velocity-relaxation-2d": Feature(_plane("Relaxation", "expr.velocity_relaxation([2.0, 0.0, 0.0], dim=2)"), "Relaxation", **FLAT_2D, control=PLANE_CONTROL),
    "inject-2d": Feature(_plane("Inject", "expr.inject([0.05 * disc, 0.02 * disc, 0.0 * disc, 0.1 * disc], dim=2)"), "Inject",
                         **FLAT_2D, control=PLANE_CONTROL),
    "rhd-sponge-2d": Feature(RHD_SPONGE, "RhdSponge", **RAM44, control=RHD_CONTROL),
    "spherical-force-2d": Feature(SPHERICAL_FORCE, "SphericalForce", (32, 32), [[16], []], [[10, 21], []], f"reject:1:{REJECTED_STEP}:1",
                                  control=SPHERICAL_CONTROL),
    "tabulated-refined-3d": Feature(REFINED_TABULATED, "RefinedTabulated", (16, 16, 16), [[8], [], []], [[5, 11], [], []],
                                    f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True, control=REFINED_TABULATED_CONTROL),
    "rhd-hllc-2d": Feature(RHD_SOLVER.format(name="RhdHllc", label="hllc", solver="HLLC"), "RhdHllc", **RAM44, control=RHD_CONTROL),
    "rhd-hllc-plus-2d": Feature(RHD_SOLVER.format(name="RhdHllcPlus", label="hllc+", solver="HLLC_PLUS"), "RhdHllcPlus", **RAM44,
                                control=RHD_CONTROL),
    # the 2.5D vortex partitions its two grid axes
    "rmhd-hllc-2d": Feature(RMHD_HLLC, "RmhdHllc", (32, 32, 1), [[16], []], [[10, 21], []], f"reject:1:{REJECTED_STEP}:1",
                            control=RMHD_CONTROL),
}


install(globals(), FEATURES)
