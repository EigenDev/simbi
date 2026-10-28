# =============================================================================
# test_parity_features.py
#
# one table of single-device configurations, each run through the tiled and
# launched drivers on the session's backend and placement (parity_harness) and
# held to the exact contract against the single device on that backend:
# - every checkpoint (a checkpoint every step, the opening one included) of
#   two in-process host tiles, of two launched workers on an even cut, and of
#   three launched workers on an uneven cut equals the single device's: every
#   level's cells and faces, the bodies group, the body history and the body
#   diagnostics file
# - restart: half the steps on two workers, the rest resumed on three uneven
#   workers, against the single device resumed from the same checkpoint (every
#   dataset: a run's body history holds the steps it took) and against the
#   uninterrupted single device (every level and the bodies state)
# - rollback: a stage rejected on one worker (on a refined grid, in the last
#   stage of the last substep of the deepest level), stopped after that step,
#   equals the clean launch capped at the accepted timestep
# the table:
# - sources and sponges: the rotating frame and full-state sponge on a plane;
#   the bondi sink with its outer sponge, flat, under one fine box and under
#   two nested boxes
# - hllc+: the kelvin-helmholtz plane and the refined 3D blast
# - moving binaries: a prescribed circular accreting binary on a plane, flat
#   and under a fine box covering the orbit
# - a moving body under its own feedback: the bondi sink launched across the
#   grid with two-way coupling, flat and under one fine box; the gas reaction
#   and the accretion receipts set its trajectory, so every step's body state
#   carries the feedback sums into the evolution. the transverse components,
#   zero by symmetry, hold roundoff-level values whose rounding depends on the
#   order of summation, which makes these gates sensitive to it; the exactness
#   of the sums themselves rests on the direct reduction tests (the arithmetic
#   reference and the partition tests of exact_device_sum and
#   reduction_partitions)
# - single-device settings without a parity gate elsewhere: the refined 3D
#   blast under ppm, rk3, hllc and the van leer limiter; the refined bondi
#   sink with a porous surface, compact softening, two sinks, a torque-free
#   surface, launched without two-way coupling, under hllc+ with its sponge,
#   with well-balanced reconstruction and with reflecting outer walls; the rotating sponge under
#   a fine box; first-order kelvin-helmholtz; the orszag-tang vortex under hllc
# entries outside the gpu admission run under SIMBI_LAUNCH_EXPERIMENTAL, the
# explicit route for a combination whose gates are being established.
# each table entry is a module of one problem class, written beside the test.
#
# run: pytest -m simulation simbi/simulation/tests/test_parity_features.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.parity_table import REJECTED_STEP, Feature, final_of, install, single
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation


BONDI = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


class {name}(SphericalBondiTest):
    """the bondi point mass and its drain at rest at the origin of a 24-cell root on [-5, 5]^3 with
    the outer sponge relaxing to the far-field state"""

    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam(True, description="the outer sponge")]
{refinement}
'''
ONE_BOX = '''    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]] | None, ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5]], description="box")]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2], description="ratios")]
'''
TWO_BOXES = '''    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[
        list[list[float]] | None,
        ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], [-1.25, 1.25, -1.25, 1.25, -1.25, 1.25]], description="boxes"),
    ]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2, 2], description="ratios")]
'''
ROTATING_SPONGE = '''
from simbi_configs.examples.newtonian.rotating_sponge import RotatingSponge


class RotatingSpongeFeature(RotatingSponge):
    """the rotating frame and the outer sponge on the 24-cell plane"""
'''
HLLC_PLUS_KH = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Solver
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz


class HllcPlusKh(KelvinHelmholtz):
    """the kelvin-helmholtz plane under the hllc+ flux"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), description="grid")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC_PLUS, description="hllc+")]
'''
HLLC_PLUS_BLAST = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Solver
from simbi_configs.examples.newtonian.refined_blast_3d import RefinedBlast3D


class HllcPlusBlast(RefinedBlast3D):
    """the 3D blast under one fine box with the hllc+ flux"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((24, 24, 24), description="grid")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC_PLUS, description="hllc+")]
'''
BINARY = '''
from typing import Annotated, Optional

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime
from simbi.types.bodies import BinaryComponentConfig, BinaryConfig, BodySystemConfig, GravitationalSystemConfig


def _component() -> BinaryComponentConfig:
    return BinaryComponentConfig(mass=0.5, radius=0.0, is_an_accretor=True, softening_length=0.05,
                                 two_way_coupling=False, accretion_radius=0.15)


class {name}(SimbiProblem):
    """a prescribed circular binary of two accreting point masses orbiting the center of a uniform
    gas at rest on [-2, 2]^2"""

    resolution: Annotated[tuple[int, int], ProblemParam((48, 48), cli=True, description="grid")]
    bounds: list[tuple[float, float]] = [(-2.0, 2.0), (-2.0, 2.0)]
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 10.0
{refinement}
    @computed_field
    @property
    def body_system(self) -> Optional[BodySystemConfig]:
        return GravitationalSystemConfig(
            prescribed_motion=True, reference_frame="inertial", system_type="binary",
            binary_config=BinaryConfig(semi_major=0.8, eccentricity=0.0, mass_ratio=1.0, total_mass=1.0,
                                       components=[_component(), _component()]),
        )

    def initial_primitive_state(self):
        def gas_state():
            for _jj in range(self.resolution[1]):
                for _ii in range(self.resolution[0]):
                    yield (1.0, 0.0, 0.0, 1.0)

        return gas_state
'''
TWO_WAY_SINK = '''
from dataclasses import replace
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


class {name}(SphericalBondiTest):
    """the bondi sink launched along x at 0.3 with two-way coupling on a 24-cell root on [-5, 5]^3"""

    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam(False, description="no sponge source")]
{refinement}
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return [replace(body, velocity=(0.3, 0.0, 0.0), two_way_coupling=True)]
'''
BLAST_VARIANT = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import BoundaryCondition, Reconstruction, Solver, TimeStepping
from simbi.types.input import Limiter
from simbi_configs.examples.newtonian.refined_blast_3d import RefinedBlast3D


class {name}(RefinedBlast3D):
    """the 3D blast under one fine box at 24^3 with one setting changed"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((24, 24, 24), cli=True, description="grid")]
    {line}
'''
SINK_VARIANT = '''
from dataclasses import replace
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi.types import BoundaryCondition, Solver
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


class {name}(SphericalBondiTest):
    """the bondi sink at the origin of a 24-cell root on [-5, 5]^3 under one fine box, one setting changed"""

    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam({sponge}, description="the outer sponge")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]] | None, ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5]], description="box")]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2], description="ratios")]
{fields}
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return {bodies}
'''
REFINED_ROTATING_SPONGE = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.rotating_sponge import RotatingSponge


class RefinedRotatingSponge(RotatingSponge):
    """the rotating frame and the outer sponge on the 24-cell plane under a fine box over the center"""

    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.5, 0.5, -0.5, 0.5]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]
'''
FIRST_ORDER_KH = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import TimeStepping
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz


class FirstOrderKh(KelvinHelmholtz):
    """the kelvin-helmholtz plane under first-order time stepping"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), cli=True, description="grid")]
    timestepping: Annotated[TimeStepping, ProblemParam(TimeStepping.RK1, description="rk1")]
'''
HLLC_ORSZAG_TANG = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Solver
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang


class HllcOrszagTang(NewtonianOrszagTang):
    """the orszag-tang vortex on a 32 x 32 plane under the hllc flux"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 32, 1), cli=True, description="grid")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="hllc")]
'''
BINARY_BOX = '''    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-1.0, 1.0, -1.0, 1.0]], description="box over the orbit")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]
'''


def _control(key: str, base: str, body: str, imports: str = "") -> tuple[str, str]:
    """the entry's fixture with its setting reverted: a subclass of the feature's own class, from the
    feature module written beside the test"""
    name = "Control"
    module = f"feature_{key.replace('-', '_')}"
    return (f"{imports}from {module} import {base}\n\n\nclass {name}({base}):\n{body}\n", name)


NO_SOURCES = (
    "    @property\n    def source_expressions(self):\n        return []\n"
)
NO_BODY_SYSTEM = (
    "    @computed_field\n    @property\n    def body_system(self) -> Optional[BodySystemConfig]:\n        return None\n"
)
BODY_SYSTEM_IMPORTS = "from typing import Optional\n\nfrom pydantic import computed_field\n\nfrom simbi.types.bodies import BodySystemConfig\n"
BLAST_CONTROL = (BLAST_VARIANT.format(name="BlastControl", line=""), "BlastControl")
SINK_CONTROL = (SINK_VARIANT.format(name="SinkControl", sponge=False, fields="", bodies="[body]"), "SinkControl")
SINK_SPONGE_CONTROL = (SINK_VARIANT.format(name="SinkSpongeControl", sponge=True, fields="", bodies="[body]"), "SinkSpongeControl")
KH_CONTROL = ('''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz


class KhControl(KelvinHelmholtz):
    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), cli=True, description="grid")]
''', "KhControl")
OT_CONTROL = ('''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang


class OtControl(NewtonianOrszagTang):
    resolution: Annotated[tuple[int, int, int], ProblemParam((32, 32, 1), cli=True, description="grid")]
''', "OtControl")


FEATURES = {
    "rotating-sponge-2d": Feature(ROTATING_SPONGE, "RotatingSpongeFeature", (24, 24), [[12], []], [[7, 15], []], f"reject:1:{REJECTED_STEP}:1",
                                  control=_control("rotating-sponge-2d", "RotatingSpongeFeature", NO_SOURCES)),
    "sponge-bondi-3d": Feature(BONDI.format(name="SpongeBondi", refinement=""), "SpongeBondi", None,
                               [[12], [], []], [[7, 15], [], []], f"reject:1:{REJECTED_STEP}:1",
                               control=_control("sponge-bondi-3d", "SpongeBondi", NO_SOURCES)),
    "sponge-bondi-refined-3d": Feature(BONDI.format(name="SpongeBondiRefined", refinement=ONE_BOX), "SpongeBondiRefined", None,
                                       [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1",
                                       control=_control("sponge-bondi-refined-3d", "SpongeBondiRefined", NO_SOURCES)),
    "sponge-bondi-nested-3d": Feature(BONDI.format(name="SpongeBondiNested", refinement=TWO_BOXES), "SpongeBondiNested", None,
                                      [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:2:3:1",
                                      control=_control("sponge-bondi-nested-3d", "SpongeBondiNested", NO_SOURCES)),
    "hllc-plus-kh-2d": Feature(HLLC_PLUS_KH, "HllcPlusKh", (32, 32), [[16], []], [[10, 21], []], f"reject:1:{REJECTED_STEP}:1",
                               control=KH_CONTROL),
    "hllc-plus-refined-3d": Feature(HLLC_PLUS_BLAST, "HllcPlusBlast", (24, 24, 24), [[12], [], []], [[7, 15], [], []],
                                    f"reject_level:1:{REJECTED_STEP}:1:1:1", control=BLAST_CONTROL),
    "moving-binary-2d": Feature(BINARY.format(name="MovingBinary", refinement=""), "MovingBinary", (48, 48), [[24], []], [[14, 30], []],
                                f"reject:1:{REJECTED_STEP}:1",
                                control=_control("moving-binary-2d", "MovingBinary", NO_BODY_SYSTEM, BODY_SYSTEM_IMPORTS)),
    "moving-binary-refined-2d": Feature(BINARY.format(name="MovingBinaryRefined", refinement=BINARY_BOX), "MovingBinaryRefined", (48, 48),
                                        [[24], []], [[14, 30], []], f"reject_level:1:{REJECTED_STEP}:1:1:1",
                                        control=_control("moving-binary-refined-2d", "MovingBinaryRefined", NO_BODY_SYSTEM,
                                                         BODY_SYSTEM_IMPORTS)),
    **{f"refined-blast-{key}": Feature(BLAST_VARIANT.format(name=f"Blast{cls}", line=line), f"Blast{cls}", (24, 24, 24),
                                       [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1",
                                       control=BLAST_CONTROL)
       for key, cls, line in [
           ("ppm", "Ppm", "reconstruction: Annotated[Reconstruction, ProblemParam(Reconstruction.PPM, description='ppm')]"),
           ("rk3", "Rk3", "timestepping: Annotated[TimeStepping, ProblemParam(TimeStepping.RK3, description='rk3')]"),
           ("hllc", "Hllc", "solver: Annotated[Solver, ProblemParam(Solver.HLLC, description='hllc')]"),
           ("van-leer", "VanLeer", "limiter: Annotated[Limiter, ProblemParam(Limiter.VAN_LEER, description='van leer')]"),
       ]},
    **{f"refined-sink-{key}": Feature(SINK_VARIANT.format(name=f"Sink{cls}", sponge=sponge, fields=fields, bodies=bodies), f"Sink{cls}", None,
                                      [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True,
                                      control=SINK_SPONGE_CONTROL if sponge else SINK_CONTROL)
       for key, cls, sponge, fields, bodies in [
           ("porous", "Porous", False, "", "[replace(body, accretion=replace(body.accretion, porosity=0.5))]"),
           ("compact-softening", "CompactSoftening", False, "",
            "[replace(body, gravitational=replace(body.gravitational, softening_kind='compact'))]"),
           ("two-sinks", "TwoSinks", False, "", "[replace(body, mass=0.5, position=(-1.2, 0.0, 0.0)), replace(body, mass=0.5, position=(1.2, 0.0, 0.0))]"),
           ("torque-free", "TorqueFree", False, "", "[replace(body, accretion=replace(body.accretion, torque_free_xi=0.5))]"),
           ("moving-one-way", "MovingOneWay", False, "", "[replace(body, velocity=(0.3, 0.0, 0.0))]"),
           ("hllc-plus-sponge", "HllcPlusSponge", True,
            "    solver: Annotated[Solver, ProblemParam(Solver.HLLC_PLUS, description='hllc+')]\n", "[body]"),
           ("well-balanced", "WellBalanced", False,
            "    wb_reconstruction: Annotated[bool, ProblemParam(True, description='balanced')]\n", "[body]"),
           # the hydrostatic atmosphere meets the outer walls from the first step: a reflecting
           # wall's ghost cells mirror it, an outflow wall's copy it
           ("reflecting", "Reflecting", False,
            "    boundary_conditions: Annotated[list, ProblemParam([BoundaryCondition.REFLECTING], description='walls')]\n", "[body]"),
       ]},
    "refined-rotating-sponge-2d": Feature(REFINED_ROTATING_SPONGE, "RefinedRotatingSponge", (24, 24), [[12], []], [[7, 15], []],
                                          f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True,
                                          control=_control("refined-rotating-sponge-2d", "RefinedRotatingSponge", NO_SOURCES)),
    # first-order time stepping has one stage, stage 0
    "first-order-kh-2d": Feature(FIRST_ORDER_KH, "FirstOrderKh", (32, 32), [[16], []], [[10, 21], []], f"reject:1:{REJECTED_STEP}:0",
                                 control=KH_CONTROL),
    # the 2.5D vortex partitions its two grid axes
    "hllc-orszag-tang-2d": Feature(HLLC_ORSZAG_TANG, "HllcOrszagTang", (32, 32, 1), [[16], []], [[10, 21], []],
                                   f"reject:1:{REJECTED_STEP}:1", control=OT_CONTROL),
    "two-way-sink-3d": Feature(TWO_WAY_SINK.format(name="TwoWaySink", refinement=""), "TwoWaySink", None,
                               [[12], [], []], [[7, 15], [], []], f"reject:1:{REJECTED_STEP}:1", opt_in=True),
    "two-way-sink-refined-3d": Feature(TWO_WAY_SINK.format(name="TwoWaySinkRefined", refinement=ONE_BOX), "TwoWaySinkRefined", None,
                                       [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True),
}


install(globals(), FEATURES)


@pytest.mark.parametrize("feature", ["two-way-sink-3d", "two-way-sink-refined-3d"], indirect=True)
def test_the_two_way_sink_moves_under_its_feedback(tmp_path: Path, feature) -> None:
    """the trajectory entries' premise: the sink moves, the feedback changes its velocity, and its
    transverse components, zero by symmetry, hold nonzero roundoff-level values, so the trajectory
    gates compare sums whose rounding depends on the order of summation"""
    f, script = feature
    single_final = final_of(single(f, script, tmp_path / "single"))
    with h5py.File(single_final, "r") as h:
        position = np.asarray(h["bodies/position"][()])[0]
        velocity = np.asarray(h["bodies/velocity"][()])[0]
        force = np.asarray(h["body_diagnostics/force"][()])
    assert position[0] > 0.0, f"the sink stayed at x = {position[0]}"
    assert velocity[0] != 0.3, "the feedback left the launch velocity unchanged"
    assert np.any(np.abs(velocity[1:]) > 0.0), "the transverse velocity is exactly zero; the sums it tests cancel trivially"
    assert np.abs(force).max() > 0.0, "the body history books no force"
