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
#   blast under ppm, rk3, hllc, the van leer limiter and reflecting walls; the
#   refined bondi sink with a porous surface, compact softening, two sinks, a
#   torque-free surface, launched without two-way coupling, under hllc+ with
#   its sponge, and with well-balanced reconstruction; the rotating sponge under
#   a fine box; first-order kelvin-helmholtz; the orszag-tang vortex under hllc
# entries outside the gpu admission run under SIMBI_LAUNCH_EXPERIMENTAL, the
# explicit route for a combination whose gates are being established.
# each table entry is a module of one problem class, written beside the test.
#
# run: pytest -m simulation simbi/simulation/tests/test_parity_features.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_bodies import _assert_same, _body_groups
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 6
REJECTED_STEP = 2

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
            prescribed_motion=True, reference_frame="com", system_type="binary",
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
from simbi.types import Solver
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


@dataclass(frozen=True)
class Feature:
    source: str
    problem: str
    resolution: tuple | None
    even: list
    uneven: list
    rejection: str
    opt_in: bool = False


FEATURES = {
    "rotating-sponge-2d": Feature(ROTATING_SPONGE, "RotatingSpongeFeature", (24, 24), [[12], []], [[7, 15], []], f"reject:1:{REJECTED_STEP}:1"),
    "sponge-bondi-3d": Feature(BONDI.format(name="SpongeBondi", refinement=""), "SpongeBondi", None,
                               [[12], [], []], [[7, 15], [], []], f"reject:1:{REJECTED_STEP}:1"),
    "sponge-bondi-refined-3d": Feature(BONDI.format(name="SpongeBondiRefined", refinement=ONE_BOX), "SpongeBondiRefined", None,
                                       [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1"),
    "sponge-bondi-nested-3d": Feature(BONDI.format(name="SpongeBondiNested", refinement=TWO_BOXES), "SpongeBondiNested", None,
                                      [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:2:3:1"),
    "hllc-plus-kh-2d": Feature(HLLC_PLUS_KH, "HllcPlusKh", (32, 32), [[16], []], [[10, 21], []], f"reject:1:{REJECTED_STEP}:1"),
    "hllc-plus-refined-3d": Feature(HLLC_PLUS_BLAST, "HllcPlusBlast", (24, 24, 24), [[12], [], []], [[7, 15], [], []],
                                    f"reject_level:1:{REJECTED_STEP}:1:1:1"),
    "moving-binary-2d": Feature(BINARY.format(name="MovingBinary", refinement=""), "MovingBinary", (48, 48), [[24], []], [[14, 30], []],
                                f"reject:1:{REJECTED_STEP}:1"),
    "moving-binary-refined-2d": Feature(BINARY.format(name="MovingBinaryRefined", refinement=BINARY_BOX), "MovingBinaryRefined", (48, 48),
                                        [[24], []], [[14, 30], []], f"reject_level:1:{REJECTED_STEP}:1:1:1"),
    **{f"refined-blast-{key}": Feature(BLAST_VARIANT.format(name=f"Blast{cls}", line=line), f"Blast{cls}", (24, 24, 24),
                                       [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1")
       for key, cls, line in [
           ("ppm", "Ppm", "reconstruction: Annotated[Reconstruction, ProblemParam(Reconstruction.PPM, description='ppm')]"),
           ("rk3", "Rk3", "timestepping: Annotated[TimeStepping, ProblemParam(TimeStepping.RK3, description='rk3')]"),
           ("hllc", "Hllc", "solver: Annotated[Solver, ProblemParam(Solver.HLLC, description='hllc')]"),
           ("van-leer", "VanLeer", "limiter: Annotated[Limiter, ProblemParam(Limiter.VAN_LEER, description='van leer')]"),
           ("reflecting", "Reflecting", "boundary_conditions: Annotated[list, ProblemParam([BoundaryCondition.REFLECTING], description='walls')]"),
       ]},
    **{f"refined-sink-{key}": Feature(SINK_VARIANT.format(name=f"Sink{cls}", sponge=sponge, fields=fields, bodies=bodies), f"Sink{cls}", None,
                                      [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True)
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
       ]},
    "refined-rotating-sponge-2d": Feature(REFINED_ROTATING_SPONGE, "RefinedRotatingSponge", (24, 24), [[12], []], [[7, 15], []],
                                          f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True),
    # first-order time stepping has one stage, stage 0
    "first-order-kh-2d": Feature(FIRST_ORDER_KH, "FirstOrderKh", (32, 32), [[16], []], [[10, 21], []], f"reject:1:{REJECTED_STEP}:0"),
    # the 2.5D vortex partitions its two grid axes
    "hllc-orszag-tang-2d": Feature(HLLC_ORSZAG_TANG, "HllcOrszagTang", (32, 32, 1), [[16], []], [[10, 21], []],
                                   f"reject:1:{REJECTED_STEP}:1"),
    "two-way-sink-3d": Feature(TWO_WAY_SINK.format(name="TwoWaySink", refinement=""), "TwoWaySink", None,
                               [[12], [], []], [[7, 15], [], []], f"reject:1:{REJECTED_STEP}:1", opt_in=True),
    "two-way-sink-refined-3d": Feature(TWO_WAY_SINK.format(name="TwoWaySinkRefined", refinement=ONE_BOX), "TwoWaySinkRefined", None,
                                       [[12], [], []], [[7, 15], [], []], f"reject_level:1:{REJECTED_STEP}:1:1:1", opt_in=True),
}


@pytest.fixture
def feature(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Feature, Path]:
    f = FEATURES[request.param]
    module = f"feature_{request.param.replace('-', '_')}"
    script = tmp_path / f"{module}.py"
    script.write_text(f.source)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return f, script


def _run_snippet(f: Feature, module: str, out: Path, fields: str, steps: int, checkpoint: Path | None) -> str:
    if f.resolution is not None:
        fields += f", resolution={f.resolution!r}"
    if checkpoint is not None:
        fields += f", checkpoint_file={str(checkpoint)!r}"
    return (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {f.problem}\n"
        f"p = {f.problem}(data_directory=sys.argv[1]{fields})\n"
        "p.checkpoint_interval = 1e-9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )


def _single(f: Feature, script: Path, out: Path, steps: int = STEPS, checkpoint: Path | None = None) -> dict[str, Path]:
    """the single device on the session's backend, a checkpoint at every step"""
    ph.python(ph.snippet_args(_run_snippet(f, script.stem, out, ", gpus=1", steps, checkpoint), str(out)), "single device")
    return _files(out)


def _tiles(f: Feature, script: Path, out: Path, cuts: list) -> dict[str, Path]:
    """two in-process host tiles cut at `cuts`, a checkpoint at every step"""
    tiles = int(np.prod([len(axis) + 1 for axis in cuts]))
    snippet = _run_snippet(f, script.stem, out, f", gpus={tiles}, decompose={cuts!r}", STEPS, None)
    ph.python(ph.snippet_args(snippet, str(out)), "host tiles", devices=tiles, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    return _files(out)


def _launch(f: Feature, script: Path, out: Path, cuts: list, steps: int = STEPS, checkpoint: Path | None = None,
            inject: str | None = None) -> ph.Launch:
    workers = int(np.prod([len(axis) + 1 for axis in cuts]))
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e-9"]
    if f.resolution is not None:
        args += ["--resolution", ",".join(str(r) for r in f.resolution)]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return ph.launch(script, out, workers=workers, cuts=cuts, args=args, env=ph.environment(inject, opt_in=f.opt_in))


def _files(out: Path) -> dict[str, Path]:
    files = {p.name: p for p in out.glob("*.h5")}
    if not any("final" in name for name in files):
        raise ph.HarnessFailure(f"no final checkpoint in {out}")
    return files


def _final(files: dict[str, Path]) -> Path:
    return [p for name, p in files.items() if "final" in name][0]


def _same_run(got: dict[str, Path], want: dict[str, Path], got_dir: Path, want_dir: Path, label: str) -> None:
    """every checkpoint's levels and bodies, and the body diagnostics file, exactly"""
    if sorted(got) != sorted(want):
        raise ph.NumericalDisagreement(f"{label}: checkpoints {sorted(got)} against {sorted(want)}")
    for name in sorted(got):
        ph.disagreement(_assert_same_levels_and_bodies, got[name], want[name], f"{label}: {name}")
    a, b = got_dir / "diagnostics.dat", want_dir / "diagnostics.dat"
    if b.exists() != a.exists() or (b.exists() and a.read_text() != b.read_text()):
        raise ph.NumericalDisagreement(f"{label}: the body diagnostics files differ")


def _assert_same_levels_and_bodies(got: Path, want: Path, label: str) -> None:
    if _body_groups(want):
        _assert_same(got, want, label)
        return
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _assert_same_levels_and_state(got: Path, want: Path, label: str) -> None:
    """every level, and the bodies group when the reference carries one"""
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    bx = {k: v for k, v in _body_groups(got).items() if k.startswith("bodies/")}
    by = {k: v for k, v in _body_groups(want).items() if k.startswith("bodies/")}
    assert bx.keys() == by.keys(), f"{label}: body datasets {sorted(set(bx) ^ set(by))} differ"
    for name, x in bx.items():
        np.testing.assert_array_equal(x, by[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("feature", list(FEATURES), indirect=True)
def test_tiles_and_launches_equal_one_device_at_every_checkpoint(tmp_path: Path, feature) -> None:
    f, script = feature
    single = _single(f, script, tmp_path / "single")
    assert len(single) >= 4, f"{len(single)} checkpoints; the cadence is not exercised"
    tiles = _tiles(f, script, tmp_path / "tiles", f.even)
    _same_run(tiles, single, tmp_path / "tiles", tmp_path / "single", "two host tiles")
    for label, cuts in (("two workers", f.even), ("three uneven workers", f.uneven)):
        out = tmp_path / label.replace(" ", "-")
        run = _launch(f, script, out, cuts)
        workers = int(np.prod([len(axis) + 1 for axis in cuts]))
        assert run.reports() == {w: (STEPS, 0) for w in range(workers)}, f"{label}: (steps, rejections) per worker {run.reports()}"
        _same_run(run.files, single, out, tmp_path / "single", label)


@pytest.mark.parametrize("feature", list(FEATURES), indirect=True)
def test_a_launch_resumed_onto_uneven_workers_equals_one_device(tmp_path: Path, feature) -> None:
    f, script = feature
    single = _single(f, script, tmp_path / "single")
    first = _launch(f, script, tmp_path / "first", f.even, steps=STEPS // 2)
    with h5py.File(first.final, "r") as h:
        assert int(h["level_0"].attrs["iteration"]) == STEPS // 2
    resumed = _launch(f, script, tmp_path / "resumed", f.uneven, checkpoint=first.final)
    workers = int(np.prod([len(axis) + 1 for axis in f.uneven]))
    assert resumed.reports() == {w: (STEPS - STEPS // 2, 0) for w in range(workers)}, f"(steps, rejections) per worker {resumed.reports()}"
    resumed_single = _single(f, script, tmp_path / "single-resumed", checkpoint=first.final)
    _same_run({"final": resumed.final}, {"final": _final(resumed_single)}, tmp_path / "resumed", tmp_path / "single-resumed",
              "resumed onto three uneven workers against the resumed single device")
    ph.disagreement(_assert_same_levels_and_state, resumed.final, _final(single), "resumed onto three uneven workers against the uninterrupted single device")


@pytest.mark.parametrize("feature", list(FEATURES), indirect=True)
def test_a_rejected_stage_replays_as_the_capped_clean_launch(tmp_path: Path, feature) -> None:
    f, script = feature
    steps = REJECTED_STEP + 1
    rejected = _launch(f, script, tmp_path / "rejected", f.even, steps=steps, inject=f.rejection)
    assert rejected.reports() == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {rejected.reports()}: a worker kept out of the rollback"
    clean = _launch(f, script, tmp_path / "clean", f.even, steps=steps)
    assert clean.reports() == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(rejected.final, "r") as fr, h5py.File(clean.final, "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    capped = _launch(f, script, tmp_path / "capped", f.even, steps=steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert capped.reports() == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(capped.final, "r") as h:
        assert float(h["level_0"].attrs["dt"]) == accepted, "the capped control's step took another timestep"
    _same_run({"final": rejected.final}, {"final": capped.final}, tmp_path / "rejected", tmp_path / "capped",
              "rejected stage against the capped clean launch")


@pytest.mark.parametrize("feature", ["two-way-sink-3d", "two-way-sink-refined-3d"], indirect=True)
def test_the_two_way_sink_moves_under_its_feedback(tmp_path: Path, feature) -> None:
    """the trajectory entries' premise: the sink moves, the feedback changes its velocity, and its
    transverse components, zero by symmetry, hold nonzero roundoff-level values, so the trajectory
    gates compare sums whose rounding depends on the order of summation"""
    f, script = feature
    single = _final(_single(f, script, tmp_path / "single"))
    with h5py.File(single, "r") as h:
        position = np.asarray(h["bodies/position"][()])[0]
        velocity = np.asarray(h["bodies/velocity"][()])[0]
        force = np.asarray(h["body_diagnostics/force"][()])
    assert position[0] > 0.0, f"the sink stayed at x = {position[0]}"
    assert velocity[0] != 0.3, "the feedback left the launch velocity unchanged"
    assert np.any(np.abs(velocity[1:]) > 0.0), "the transverse velocity is exactly zero; the sums it tests cancel trivially"
    assert np.abs(force).max() > 0.0, "the body history books no force"
