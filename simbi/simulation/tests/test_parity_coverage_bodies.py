# =============================================================================
# test_parity_coverage_bodies.py
#
# one table of single-device configurations with immersed bodies, each run
# through the tiled and launched drivers on the session's backend and placement
# (parity_harness) and held to the exact contract of parity_table against the
# single device on that backend: every checkpoint of two host tiles, two
# launched workers on an even cut and three on an uneven cut; a launch resumed
# onto three uneven workers; and a stage rejected on one worker against the
# clean launch capped at the accepted timestep.
# the table:
# - an accreting bondi sink on isothermal gas, flat and under one fine box
# - a rigid sphere in a driven 3D wind, cut through by the workers
# - a prescribed spinning rotor in still gas on a plane, and a free card
#   tumbling and drifting in a 3D wind under two-way coupling
# - a resistive magnetized drain under one fine box
# - a prescribed circular accreting binary on a 3D grid and on isothermal gas
# every entry carries a control, the same fixture without its body (or, for
# the spin and the two-way coupling, the same body without that setting), whose
# single-device run must differ; the accreting entries assert every sink
# drained mass, and the two-way entry that the gas reaction moved the body. refined entries outside the gpu admission run under
# SIMBI_LAUNCH_EXPERIMENTAL. each table entry is a module of one problem class,
# written beside the test.
#
# run: pytest -m simulation simbi/simulation/tests/test_parity_coverage_bodies.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.parity_table import REJECTED_STEP, Feature, install
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation


ISOTHERMAL_SINK = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime, Solver
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


class {name}(SphericalBondiTest):
    """the bondi sink at rest at the origin of a 24-cell root on [-5, 5]^3, on isothermal gas of unit
    sound speed starting from the adiabatic hydrostatic density"""

    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam(False, description="no sponge source")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
    # isothermal hydro evaluates the hlle flux
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="hlle")]
    sound_speed: Annotated[float, ProblemParam(1.0, description="sound speed")]
{refinement}
    def initial_primitive_state(self):
        adiabatic = super().initial_primitive_state()

        def rows():
            for row in adiabatic():
                yield tuple(row[:-1])

        return rows
'''
ONE_BOX = '''    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]] | None, ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5]], description="box")]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2], description="ratios")]
'''
SPHERE_WIND = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Shape, Solver
from simbi.types.bodies import BodyCapability, ImmersedBodyConfig, RigidProperties


class SphereWind(SimbiProblem):
    """a fixed no-slip rigid sphere of radius 0.4 at the origin of [-2, 2] x [-1, 1]^2 in a mach-2
    wind driven through the x-inner face"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 16, 16), cli=True, description="grid")]
    bounds: list[tuple[float, float]] = [(-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 1.4
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="hllc")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(
        [BoundaryCondition.DYNAMIC] + [BoundaryCondition.OUTFLOW] * 5, description="driven x-inner face, outflow elsewhere")]
    end_time: float = 10.0

    def freestream_velocity(self) -> float:
        return 2.0 * math.sqrt(self.adiabatic_index)

    @computed_field
    @property
    def immersed_bodies(self) -> list:
        return [ImmersedBodyConfig(
            capability=BodyCapability.RIGID, mass=0.0, radius=0.8, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
            rigid=RigidProperties(inertia=1.0, apply_no_slip=True, k_eta_n=50.0, k_eta_t=50.0,
                                  shape=Shape.sphere((0.0, 0.0, 0.0), 0.4)),
        )]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        return expr.boundary([1.0, self.freestream_velocity(), 0.0, 0.0, 1.0], dim=3)

    def initial_primitive_state(self):
        def gas_state():
            for _ in range(math.prod(self.resolution)):
                yield (1.0, self.freestream_velocity(), 0.0, 0.0, 1.0)

        return gas_state
'''
SPINNING_ROTOR = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.ibm.spinning_rotor import SpinningRotor


class SpinningRotorFeature(SpinningRotor):
    """the three-blade rotor spinning at omega = 4 in still gas on a 48 x 48 plane over [-2, 2]^2"""

    resolution: Annotated[tuple[int, int], ProblemParam((48, 48), cli=True, description="grid")]
'''
TUMBLING_BODY = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.ibm.tumbling_body import TumblingBody


class TumblingBodyFeature(TumblingBody):
    """the free card spun about a tilted axis in the mach-0.9 wind, two-way coupled, on 48 x 24 x 24
    cells over [-1, 3] x [-1, 1]^2: the card's thinnest half-extent spans 1.8 cells"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((48, 24, 24), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(-1.0, 3.0), (-1.0, 1.0), (-1.0, 1.0)], description="box")]
'''
MAGNETIZED_DRAIN = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.ibm.magnetized_sink import MagnetizedBondiSink


class RefinedMagnetizedDrain(MagnetizedBondiSink):
    """the resistive magnetized drain at the origin of a 32-cell root on [-4, 4]^3 under one fine box
    over [-3.5, 3.5]^3. the drain mask radius 0.5 (four fine cells) plus the mask's f64 support and
    the stencil reach, 22.7 fine cells, lies inside the box"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 32, 32), description="grid")]
    r_acc_scale: Annotated[float, ProblemParam(2.0, description="mask radius in root cells")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-3.5, 3.5, -3.5, 3.5, -3.5, 3.5]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]
'''
BINARY = '''
import math
from typing import Annotated, Optional

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime, Solver
from simbi.types.bodies import BinaryComponentConfig, BinaryConfig, BodySystemConfig, GravitationalSystemConfig


def _component() -> BinaryComponentConfig:
    return BinaryComponentConfig(mass=0.5, radius=0.0, is_an_accretor=True, softening_length={softening},
                                 two_way_coupling=False, accretion_radius={accretion_radius})


class {name}(SimbiProblem):
    """a prescribed circular binary of two accreting point masses orbiting the center of a uniform
    gas at rest on [-2, 2]^d"""

    resolution: Annotated[{rtype}, ProblemParam({resolution!r}, cli=True, description="grid")]
    bounds: list[tuple[float, float]] = [(-2.0, 2.0)] * {dims}
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="hlle")]
    end_time: float = 10.0
{gas}
    @computed_field
    @property
    def body_system(self) -> Optional[BodySystemConfig]:
        return GravitationalSystemConfig(
            prescribed_motion=True, reference_frame="com", system_type="binary",
            binary_config=BinaryConfig(semi_major=0.8, eccentricity=0.0, mass_ratio=1.0, total_mass=1.0,
                                       components=[_component(), _component()]),
        )

    def initial_primitive_state(self):
        def gas_state():
            for _ in range(math.prod(self.resolution)):
                yield {row!r}

        return gas_state
'''
ADIABATIC_GAS = '''    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
'''
ISOTHERMAL_GAS = '''    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
    sound_speed: Annotated[float, ProblemParam(1.0, description="sound speed")]
'''


def _control(key: str, base: str, body: str, imports: str = "") -> tuple[str, str]:
    """the entry's fixture with its setting reverted: a subclass of the feature's own class, from the
    feature module written beside the test"""
    name = "Control"
    module = f"feature_{key.replace('-', '_')}"
    return (f"{imports}from {module} import {base}\n\n\nclass {name}({base}):\n{body}\n", name)


COMPUTED_FIELD_IMPORTS = "from pydantic import computed_field\n\n"
NO_BODIES = "    @computed_field\n    @property\n    def immersed_bodies(self) -> list:\n        return []\n"
# the magnetized sink declares its bodies as a plain property
NO_PLAIN_BODIES = "    @property\n    def immersed_bodies(self) -> list:\n        return []\n"
NO_BODY_SYSTEM = (
    "    @computed_field\n    @property\n    def body_system(self) -> Optional[BodySystemConfig]:\n        return None\n"
)
BODY_SYSTEM_IMPORTS = "from typing import Optional\n\nfrom pydantic import computed_field\n\nfrom simbi.types.bodies import BodySystemConfig\n"
STILL_ROTOR = "    omega: Annotated[float, ProblemParam(0.0, description='still')]\n"
STILL_ROTOR_IMPORTS = "from typing import Annotated\n\nfrom simbi import ProblemParam\n"
ONE_WAY_CARD = (
    "    @computed_field\n    @property\n    def immersed_bodies(self) -> list:\n"
    "        return [replace(body, two_way_coupling=False) for body in super().immersed_bodies]\n"
)
ONE_WAY_IMPORTS = "from dataclasses import replace\n\nfrom pydantic import computed_field\n\n"


def _drained(final: Path, control: Path | None) -> None:
    """every accreting body drained mass in the run under test"""
    with h5py.File(final, "r") as h:
        accreted = np.asarray(h["bodies/total_accreted_mass"][()])
    assert np.all(accreted > 0.0), f"accreted mass {accreted}; the drain never acted"


def _moved_under_feedback(final: Path, control: Path | None) -> None:
    """the gas reaction moved the two-way body from rest, and the one-way control stayed at rest"""
    with h5py.File(final, "r") as h:
        velocity = np.asarray(h["bodies/velocity"][()])[0]
    assert np.any(velocity != 0.0), "the two-way body stayed at rest; the feedback never reached it"
    with h5py.File(control, "r") as h:
        still = np.asarray(h["bodies/velocity"][()])[0]
    assert np.all(still == 0.0), f"the one-way control moved ({still}); the control carries the feedback"


FLAT_3D = ([[12], [], []], [[7, 15], [], []])
FEATURES = {
    "isothermal-sink-3d": Feature(ISOTHERMAL_SINK.format(name="IsothermalSink", refinement=""), "IsothermalSink", None,
                                  *FLAT_3D, f"reject:1:{REJECTED_STEP}:1",
                                  control=_control("isothermal-sink-3d", "IsothermalSink", NO_BODIES, COMPUTED_FIELD_IMPORTS),
                                  premise=_drained),
    # the gpu admission gates the refined sink on the newtonian regime alone
    "isothermal-sink-refined-3d": Feature(ISOTHERMAL_SINK.format(name="IsothermalSinkRefined", refinement=ONE_BOX), "IsothermalSinkRefined",
                                          None, *FLAT_3D, f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True,
                                          control=_control("isothermal-sink-refined-3d", "IsothermalSinkRefined", NO_BODIES,
                                                           COMPUTED_FIELD_IMPORTS),
                                          premise=_drained),
    # the sphere spans root cells 12.8..19.2 along x: the even cut at 16 and the uneven cut at 14
    # run through it
    "rigid-sphere-3d": Feature(SPHERE_WIND, "SphereWind", (32, 16, 16), [[16], [], []], [[14, 21], [], []], f"reject:1:{REJECTED_STEP}:1",
                               control=_control("rigid-sphere-3d", "SphereWind", NO_BODIES, COMPUTED_FIELD_IMPORTS)),
    # the rotor spans root cells 17..31 of 48 along x: both cuts run through it
    "spinning-rotor-2d": Feature(SPINNING_ROTOR, "SpinningRotorFeature", (48, 48), [[24], []], [[20, 30], []], f"reject:1:{REJECTED_STEP}:1",
                                 control=_control("spinning-rotor-2d", "SpinningRotorFeature", STILL_ROTOR, STILL_ROTOR_IMPORTS)),
    # the card spans root cells 6.6..17.4 of 48 along x: both cuts run through it
    "tumbling-body-3d": Feature(TUMBLING_BODY, "TumblingBodyFeature", (48, 24, 24), [[12], [], []], [[9, 15], [], []],
                                f"reject:1:{REJECTED_STEP}:1",
                                control=_control("tumbling-body-3d", "TumblingBodyFeature", ONE_WAY_CARD, ONE_WAY_IMPORTS),
                                premise=_moved_under_feedback),
    # the gpu admission gates the refined sink on the newtonian regime without a magnetic coupling
    "magnetized-drain-refined-3d": Feature(MAGNETIZED_DRAIN, "RefinedMagnetizedDrain", None, [[16], [], []], [[10, 21], [], []],
                                           f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True,
                                           control=_control("magnetized-drain-refined-3d", "RefinedMagnetizedDrain", NO_PLAIN_BODIES),
                                           premise=_drained),
    # the components orbit at x = +/-0.4, root cells 12.8 and 19.2 of 32
    "binary-3d": Feature(BINARY.format(name="Binary3D", softening=0.1, accretion_radius=0.3, resolution=(32, 32, 32), rtype="tuple[int, int, int]", dims=3,
                                       gas=ADIABATIC_GAS, row=(1.0, 0.0, 0.0, 0.0, 1.0)),
                         "Binary3D", (32, 32, 32), [[16], [], []], [[10, 21], [], []], f"reject:1:{REJECTED_STEP}:1",
                         control=_control("binary-3d", "Binary3D", NO_BODY_SYSTEM, BODY_SYSTEM_IMPORTS), premise=_drained),
    "binary-isothermal-2d": Feature(BINARY.format(name="BinaryIsothermal", softening=0.05, accretion_radius=0.15, resolution=(48, 48), rtype="tuple[int, int]",
                                                  dims=2, gas=ISOTHERMAL_GAS, row=(1.0, 0.0, 0.0)),
                                    "BinaryIsothermal", (48, 48), [[24], []], [[14, 30], []], f"reject:1:{REJECTED_STEP}:1",
                                    control=_control("binary-isothermal-2d", "BinaryIsothermal", NO_BODY_SYSTEM, BODY_SYSTEM_IMPORTS),
                                    premise=_drained),
}


install(globals(), FEATURES)
