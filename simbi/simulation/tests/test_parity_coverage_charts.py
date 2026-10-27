# =============================================================================
# test_parity_coverage_charts.py
#
# one table of single-device configurations on charts, regimes and boundaries
# that carry no other table-driven parity gate, each run through the tiled
# and launched drivers on the session's backend and placement (parity_harness)
# and held to the exact contract against the single device on that backend
# (parity_table): every checkpoint of two host tiles, two launched workers on
# an even cut and three on an uneven cut; a launch resumed onto three uneven
# workers; a stage rejected on one worker replayed as the capped clean launch.
# the table:
# - 1D cartesian MHD: the relativistic mignone-bodo tube and the newtonian
#   brio-wu tube, each against the same tube with every field component zero
# - the refined 3D kerr-schild GRMHD box: a fine box beside the horizon, the
#   root cut through it; the fine level publishes a developed transverse field
#   and the restriction changes the root against the unrefined box
# - ohmic resistivity on the 3D orszag-tang box, against the ideal box
# - a driven x1-inner face prescribing an inflow varying along the face and in
#   time on relativistic gas, relativistic MHD and isothermal MHD, against the
#   same grid with an outflow face
# - spherical and cylindrical radial lines of relativistic and isothermal gas
#   carrying a pulse, against the same pulse on the cartesian line
# - mesh motion: the homologously expanding uniform sphere on the spherical
#   line, recording a grown scale factor, and a cartesian mesh translating at
#   a constant frame velocity on a relativistic and an isothermal plane, each
#   against the static mesh
# - a declared stationary target on relativistic gas: the target run holds
#   the gas at rest to roundoff, the same profile without the target moves
# the refined kerr-schild entry lies outside the gpu device admission and runs
# there under SIMBI_LAUNCH_EXPERIMENTAL, the explicit route for a combination
# whose gates are being established.
# each table entry is a module of one problem class, written beside the test.
#
# run: pytest -m simulation simbi/simulation/tests/test_parity_coverage_charts.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.parity_table import REJECTED_STEP, Feature, install
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation


def _control(key: str, base: str, body: str, imports: str = "") -> tuple[str, str]:
    """the entry's fixture with its setting reverted: a subclass of the feature's own class, from the
    feature module written beside the test"""
    module = f"feature_{key.replace('-', '_')}"
    return (f"{imports}from {module} import {base}\n\n\nclass Control({base}):\n{body}\n", "Control")


# ---------------------------------------------------------------------------
# 1D cartesian MHD
# ---------------------------------------------------------------------------
RMHD_TUBE = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.srmhd.magnetic_shock_tube import MagneticShockTube


class RmhdTube(MagneticShockTube):
    """the mignone-bodo relativistic MHD tube 1 on a 32-cell cartesian line, a timestep long enough
    that every step's checkpoint carries its own time in its name"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 1, 1), cli=True, description="grid")]
'''
NMHD_TUBE = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.simulation.tests.fixtures.mhd_brio_wu_1p5d import MhdBrioWu1p5d


class NmhdTube(MhdBrioWu1p5d):
    """the brio-wu newtonian MHD tube on a 128-cell cartesian line over [0, 100], a timestep long
    enough that every step's checkpoint carries its own time in its name"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((128, 1, 1), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 100.0)], description="x")]
    end_time: Annotated[float, ProblemParam(100.0, cli=True, checkpoint_safe=True, description="t")]
'''
# the tube's gas state with every field component seeded zero, face and cell alike
ZERO_FIELD = (
    "    def initial_primitive_state(self):\n"
    "        gas, *fields = super().initial_primitive_state()\n"
    "        return (gas, *[partial(_zeros, field) for field in fields])\n"
)
ZERO_FIELD_IMPORTS = "from functools import partial\n\n\ndef _zeros(field):\n    return (0.0 for _ in field())\n\n\n"

# ---------------------------------------------------------------------------
# the refined 3D kerr-schild GRMHD box
# ---------------------------------------------------------------------------
REFINED_KS_MHD = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.simulation.tests.test_cartesian_ks_mhd_3d import _RefinedCartesianKsMhd3D


class RefinedKsMhd(_RefinedCartesianKsMhd3D):
    """the vertical densitized field on a 16^3 schwarzschild kerr-schild root over [-5, 5]^3 with one
    fine box at x in [2.5, 3.75], |y|, |z| <= 1.25, clear of the horizon; the end time lies past
    every step the gates take"""

    end_time: Annotated[float, ProblemParam(10.0, checkpoint_safe=True)]
'''
UNREFINED = (
    "    refinement_enabled: Annotated[bool, ProblemParam(False, description='flat')]\n"
)
UNREFINED_IMPORTS = "from typing import Annotated\n\nfrom simbi import ProblemParam\n"


def _refined_ks_premise(mine: Path, control: Path | None) -> None:
    """the fine level carries a transverse field the infall developed, and its restriction changes the
    root against the unrefined box"""
    assert control is not None
    with h5py.File(mine, "r") as h:
        assert "level_1" in h, "the run published no refined level; the refined arm is vacuous"
        fine_bx = float(np.abs(h["level_1/partition_0/hydro/magnetic/B1/data"][()]).max())
    assert fine_bx > 1e-10, f"the fine level's transverse field never developed (max |B1| = {fine_bx:e})"
    xs, ys = _all_levels(mine), _all_levels(control)
    root = [name for name in ys if name.startswith("level_0/") and "@" not in name]
    assert root, "the unrefined control published no root dataset"
    assert any(not np.array_equal(xs[name], ys[name]) for name in root), (
        "the root equals the unrefined box's; the fine level never feeds the root"
    )


# ---------------------------------------------------------------------------
# resistivity in 3D
# ---------------------------------------------------------------------------
OT_3D = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.nmhd_orszag_tang_3d import NewtonianOrszagTang3D


class {name}(NewtonianOrszagTang3D):
    """the newtonian orszag-tang vortex on a periodic 16^3 box"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((16, 16, 16), cli=True, description="grid")]
{body}'''
RESISTIVE_BODY = '''
    @computed_field
    @property
    def resistivity(self) -> float:
        return 1.0e-2
'''
RESISTIVE_OT_3D = "from pydantic import computed_field\n" + OT_3D.format(name="ResistiveOt3D", body=RESISTIVE_BODY)
IDEAL_OT_3D = (OT_3D.format(name="IdealOt3D", body=""), "IdealOt3D")

# ---------------------------------------------------------------------------
# driven faces
# ---------------------------------------------------------------------------
DRIVEN = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi.types import BoundaryCondition
from {module} import {base}

DRIVEN = [BoundaryCondition.DYNAMIC, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW]


class {name}({base}):
    """{doc}, its x1-inner face driven with a density and transverse velocity varying along the face
    and in time, outflow on the other faces"""

    resolution: Annotated[tuple, ProblemParam({resolution!r}, cli=True, description="grid")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(DRIVEN, description="driven")]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        x1, x2 = expr.coords(2)
        t = expr.variable("t", x1)
        rho = {rho0} * (1.0 + 0.2 * expr.sin(2.0 * math.pi * x2 + 20.0 * t))
        vy = 0.05 * expr.sin(2.0 * math.pi * x2)
        return expr.boundary({outputs}, dim={dim})
'''
OUTFLOW_FACE = (
    "    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW] * 4, description='outflow')]\n\n"
    "    @computed_field\n    @property\n    def bx1_inner_expressions(self) -> dict:\n        return {}\n"
)
OUTFLOW_FACE_IMPORTS = (
    "from typing import Annotated\n\nfrom pydantic import computed_field\n\nfrom simbi import ProblemParam\n"
    "from simbi.types import BoundaryCondition\n"
)
# (key, class, module, base, description, resolution, rho0, outputs, dim)
DRIVEN_CASES = [
    ("driven-rhd-2d", "DrivenRhd", "simbi_configs.examples.srhd.ram44", "Ram44", "the relativistic split shock on a 32 x 32 plane",
     (32, 32), 1.0, "[rho, 0.3, vy, 1.0e3]", 2),
    ("driven-rmhd-2d", "DrivenRmhd", "simbi_configs.examples.srmhd.rmhd_orszag_tang", "OrszagTang",
     "the relativistic orszag-tang vortex on a 32 x 32 plane", (32, 32, 1), 25.0 / 9.0,
     "[rho, 0.3, vy, 0.0, 5.0 / 3.0, 0.1, 0.02 * expr.cos(2.0 * math.pi * x2), 0.0]", 3),
    ("driven-imhd-2d", "DrivenImhd", "simbi_configs.examples.isothermal.imhd_orszag_tang", "IsothermalOrszagTang",
     "the isothermal orszag-tang vortex on a 32 x 32 plane", (32, 32, 1), 1.0,
     "[rho, 0.3, vy, 0.0, 0.1, 0.02 * expr.cos(2.0 * math.pi * x2), 0.0]", 3),
]

# ---------------------------------------------------------------------------
# 1D curvilinear lines
# ---------------------------------------------------------------------------
LINE = '''
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime


class {name}(SimbiProblem):
    """a gaussian {bump} bump at x = 0.6 in gas at rest on a 64-cell {chart} line over [0.1, 1.1],
    a reflecting inner face and an outflow outer face"""

    {closure}
    resolution: Annotated[tuple[int], ProblemParam((64,), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.1, 1.1)], description="x1")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.{chart_enum}, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.{regime}, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(
        [BoundaryCondition.REFLECTING, BoundaryCondition.OUTFLOW], description="walls")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]

    def initial_primitive_state(self):
        def gas():
            (n,) = self.resolution
            for ii in range(n):
                x = 0.1 + (ii + 0.5) / n
                bump = 1.0 + 2.0 * math.exp(-(((x - 0.6) / 0.1) ** 2))
                yield {row}

        return gas
'''
LINE_REGIMES = {
    "rhd": dict(regime="RHD", bump="pressure",
                closure='adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="gamma")]',
                row="(1.0, 0.0, bump)"),
    "isothermal": dict(regime="ISOTHERMAL", bump="density",
                       closure='sound_speed: Annotated[float, ProblemParam(1.0, description="sound speed")]',
                       row="(bump, 0.0)"),
}
LINE_CHARTS = {"spherical": "SPHERICAL", "cylindrical": "CYLINDRICAL"}


def _line(name: str, regime: str, chart: str, chart_enum: str) -> str:
    return LINE.format(name=name, chart=chart, chart_enum=chart_enum, **LINE_REGIMES[regime])


# ---------------------------------------------------------------------------
# mesh motion
# ---------------------------------------------------------------------------
SPHERE = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.uniform_sphere import UniformSphere


class ExpandingSphere(UniformSphere):
    """uniform gas at rest on a 64-cell spherical line over [1, 10], the mesh expanding homologously
    as a(t) = 1 + t; the cells are wide enough that every step's checkpoint carries its own time in
    its name"""

    nr: Annotated[int, ProblemParam(64, cli=True, description="radial zones")]
    r_inner: Annotated[float, ProblemParam(1.0, cli=True, description="inner radius")]
    r_outer: Annotated[float, ProblemParam(10.0, cli=True, description="outer radius")]
    expansion_rate: Annotated[float, ProblemParam(1.0, cli=True, description="a_dot")]
'''
STATIC_MESH = (
    "    @property\n    def scale_factor(self):\n        return None\n\n"
    "    @property\n    def scale_factor_derivative(self):\n        return None\n"
)
TRANSLATING = '''
from typing import Annotated

from simbi import ProblemParam
from {module} import {base}


class {name}({base}):
    """{doc}, the cartesian mesh carried at a constant rate: a(t) = 1 + 0.5 t"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), cli=True, description="grid")]

    @property
    def scale_factor(self):
        return lambda t: 1.0 + 0.5 * t

    @property
    def scale_factor_derivative(self):
        return lambda _t: 0.5
'''


def _motion(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as h:
        return float(h["metadata"].attrs["scale_factor"]), float(h["metadata"].attrs["scale_factor_dot"])


def _expanding_premise(mine: Path, control: Path | None) -> None:
    """the expanding run records a scale factor grown past one at a positive rate; the static control
    records a unit scale factor at rest"""
    a, adot = _motion(mine)
    assert a > 1.0 and adot > 0.0, f"the expanding mesh records a = {a}, a_dot = {adot}; the motion never reached the run"
    assert control is not None
    assert _motion(control) == (1.0, 0.0), f"the static control records (a, a_dot) = {_motion(control)}"


def _translating_premise(mine: Path, control: Path | None) -> None:
    """a cartesian mesh under a(t) = 1 + 0.5 t translates: its scale factor stays one and a_dot is the
    frame velocity along x1; the static control records a frame at rest"""
    assert _motion(mine) == (1.0, 0.5), f"the translating mesh records (a, a_dot) = {_motion(mine)}; the motion never reached the run"
    assert control is not None
    assert _motion(control) == (1.0, 0.0), f"the static control records (a, a_dot) = {_motion(control)}"


# ---------------------------------------------------------------------------
# a declared stationary target on relativistic gas
# ---------------------------------------------------------------------------
RHD_TARGET = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime


class RhdTarget(SimbiProblem):
    """relativistic gas at rest and uniform density under the pressure p = 1 + 0.5 sin(2 pi x) on a
    64-cell line between reflecting walls, the profile declared as the run's stationary target and
    seeded from it"""

    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int], ProblemParam((64,), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0)], description="x")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    seed_from_equilibrium: Annotated[bool, ProblemParam(True, description="seeded")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        (x,) = expr.coords(1)
        pressure = 1.0 + 0.5 * expr.sin(2.0 * math.pi * x)
        return x._graph.compile([1.0 + 0.0 * x, 0.0 * x, pressure]).serialize_equilibrium(dim=1)

    def initial_primitive_state(self):
        def gas():
            (n,) = self.resolution
            for ii in range(n):
                x = (ii + 0.5) / n
                yield (1.0, 0.0, 1.0 + 0.5 * math.sin(2.0 * math.pi * x))

        return gas
'''
NO_TARGET = (
    "    seed_from_equilibrium: Annotated[bool, ProblemParam(False, description='unseeded')]\n\n"
    "    @computed_field\n    @property\n    def equilibrium_expressions(self) -> dict:\n        return {}\n"
)
NO_TARGET_IMPORTS = "from typing import Annotated\n\nfrom pydantic import computed_field\n\nfrom simbi import ProblemParam\n"


def _target_premise(mine: Path, control: Path | None) -> None:
    """the declared target is a fixed point of the run: the gas stays at rest to roundoff, while the
    same pressure profile without the target accelerates"""
    assert control is not None
    with h5py.File(mine, "r") as h:
        held = float(np.abs(h["level_0/partition_0/hydro/primitives/v1"][()]).max())
    with h5py.File(control, "r") as h:
        moved = float(np.abs(h["level_0/partition_0/hydro/primitives/v1"][()]).max())
    # the imbalance added back cancels the scheme's residual to roundoff of O(1) pressures
    assert held < 1e-12, f"the target run moves the gas (max |v| = {held:e}); the target never reached the run"
    assert moved > 1e-3, f"the gas without the target stays at rest (max |v| = {moved:e}); the profile is no test of the target"


REJECT = f"reject:1:{REJECTED_STEP}:1"

FEATURES = {
    # a 1D run partitions its one grid axis
    "rmhd-tube-1d": Feature(RMHD_TUBE, "RmhdTube", (32, 1, 1), [[16]], [[10, 21]], REJECT,
                            control=_control("rmhd-tube-1d", "RmhdTube", ZERO_FIELD, ZERO_FIELD_IMPORTS)),
    "nmhd-tube-1d": Feature(NMHD_TUBE, "NmhdTube", (128, 1, 1), [[64]], [[40, 90]], REJECT,
                            control=_control("nmhd-tube-1d", "NmhdTube", ZERO_FIELD, ZERO_FIELD_IMPORTS)),
    # the root cut at y = 8 runs through the fine box (root cells 12..14 x 6..9 x 6..9)
    "refined-ks-grmhd-3d": Feature(REFINED_KS_MHD, "RefinedKsMhd", None, [[], [8], []], [[], [3, 8], []],
                                   f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True,
                                   control=_control("refined-ks-grmhd-3d", "RefinedKsMhd", UNREFINED, UNREFINED_IMPORTS),
                                   premise=_refined_ks_premise),
    "resistive-ot-3d": Feature(RESISTIVE_OT_3D, "ResistiveOt3D", (16, 16, 16), [[8], [], []], [[5, 11], [], []], REJECT,
                               control=IDEAL_OT_3D),
    # the even cut gives one worker the whole driven face, the uneven cut shares it among three
    **{key: Feature(DRIVEN.format(name=cls, module=module, base=base, doc=doc, resolution=resolution, rho0=rho0,
                                  outputs=outputs, dim=dim),
                    cls, resolution, [[16], []], [[], [10, 21]], REJECT,
                    control=_control(key, cls, OUTFLOW_FACE, OUTFLOW_FACE_IMPORTS))
       for key, cls, module, base, doc, resolution, rho0, outputs, dim in DRIVEN_CASES},
    **{f"{chart}-line-{regime}": Feature(_line(f"Line{chart.title()}{regime.title()}", regime, chart, enum),
                                         f"Line{chart.title()}{regime.title()}", (64,), [[32]], [[20, 41]], REJECT,
                                         control=(_line("CartesianLine", regime, "cartesian", "CARTESIAN"), "CartesianLine"))
       for chart, enum in LINE_CHARTS.items() for regime in LINE_REGIMES},
    # the sphere's radial zone count is its resolution
    "homologous-sphere-1d": Feature(SPHERE, "ExpandingSphere", None, [[32]], [[20, 41]], REJECT,
                                    control=_control("homologous-sphere-1d", "ExpandingSphere", STATIC_MESH),
                                    premise=_expanding_premise),
    "moving-mesh-rhd-2d": Feature(TRANSLATING.format(name="MovingRhd", module="simbi_configs.examples.srhd.ram44", base="Ram44",
                                                     doc="the relativistic split shock on a 32 x 32 plane"),
                                  "MovingRhd", (32, 32), [[16], []], [[10, 21], []], REJECT,
                                  control=_control("moving-mesh-rhd-2d", "MovingRhd", STATIC_MESH), premise=_translating_premise),
    "moving-mesh-isothermal-2d": Feature(TRANSLATING.format(name="MovingIsothermal",
                                                            module="simbi_configs.examples.isothermal.isothermal_kh",
                                                            base="IsothermalKelvinHelmholtz",
                                                            doc="the isothermal kelvin-helmholtz plane at 32 x 32"),
                                         "MovingIsothermal", (32, 32), [[16], []], [[10, 21], []], REJECT,
                                         control=_control("moving-mesh-isothermal-2d", "MovingIsothermal", STATIC_MESH),
                                         premise=_translating_premise),
    "rhd-target-1d": Feature(RHD_TARGET, "RhdTarget", (64,), [[32]], [[20, 41]], REJECT,
                             control=_control("rhd-target-1d", "RhdTarget", NO_TARGET, NO_TARGET_IMPORTS),
                             premise=_target_premise),
}


install(globals(), FEATURES)
