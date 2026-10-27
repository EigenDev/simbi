# =============================================================================
# test_launch_refined_device_bodies.py
#
# a stationary gravitating, draining sink under one fine box on two launched
# refined workers, on the session's backend and placement (parity_harness).
# launched workers sum every tile's body feedback in flat tile order, the order
# host tiles sum it in, so the reference is the in-process host tiles cut at
# the same cells on the same backend: on the gpu build SYMBI_GPU_OVERSUBSCRIBE
# folds their logical devices onto the visible one. every comparison is exact:
# every level's grid, the bodies group, the per-step body history and the body
# diagnostics file.
# - evolution: a cut on the box's low x face (the fine level and the sink on
#   one worker, the reflux contributions crossing) and a cut through the box
#   and the sink (both workers hold a partial of the sink's feedback)
# - restart: half the steps cut through the sink, the rest resumed on the cut
#   on the box face, against the host tiles resumed from the same checkpoint
#   on the same cut
# - rollback: a rejection in the last stage of the last fine substep of one
#   step, on one worker, with the sink cut by the workers; stopped after the
#   run it equals the clean launch capped at the accepted timestep, so no
#   accretion, force, torque or heat of the rejected attempt survives
# - admission: on the gpu the sink runs without SIMBI_LAUNCH_EXPERIMENTAL at
#   any mass, radius, softening length and position; each untested body
#   property (two bodies, a gravity-only mass, a moving sink, a porous surface,
#   compact softening) is refused by name on the gpu without it and runs on the
#   cpu launch
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_refined_device_bodies.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_bodies import _assert_same, _assert_same_diagnostics, _final, _iteration

pytestmark = pytest.mark.simulation

STEPS = 6
SINK = '''
from dataclasses import replace
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi.types.bodies import BodyCapability, GravitationalProperties
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


class RefinedSink(SphericalBondiTest):
    """the bondi point mass and its drain at rest at the origin of a 24-cell root on [-5, 5]^3,
    under one fine box over root cells 6..18; the outer sponge source is off"""

    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam(False, description="no sponge source")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]] | None, ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5]], description="box")]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2], description="ratios")]


class ScaledSink(RefinedSink):
    """the gated sink at another mass, drain radius, softening length and position"""

    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return [replace(body, mass=0.7, position=(0.3, -0.2, 0.1), gravitational=GravitationalProperties(softening_length=0.05),
                        accretion=replace(body.accretion, accretion_radius=0.8))]


class TwoSinks(RefinedSink):
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return [replace(body, mass=0.5, position=(-1.2, 0.0, 0.0)), replace(body, mass=0.5, position=(1.2, 0.0, 0.0))]


class GravityOnly(RefinedSink):
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return [replace(body, capability=BodyCapability.GRAVITATIONAL, accretion=None)]


class MovingSink(RefinedSink):
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return [replace(body, velocity=(0.05, 0.0, 0.0))]


class PorousSink(RefinedSink):
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return [replace(body, accretion=replace(body.accretion, porosity=0.5))]


class CompactSoftening(RefinedSink):
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        (body,) = super().immersed_bodies
        return [replace(body, gravitational=replace(body.gravitational, softening_kind="compact"))]
'''
LAUNCHED = "from refined_sink import RefinedSink as Base\n\n\nclass LaunchedRefinedSink(Base):\n    pass\n"
# the untested body variants of the sink and the property each is refused by on the gpu
UNTESTED_BODIES = {
    "TwoSinks": "2 immersed bodies",
    "GravityOnly": "a body other than a gravitating sink",
    "MovingSink": "a moving body",
    "PorousSink": "a porous sink surface",
    "CompactSoftening": "compact softening",
}
# root cells of the 24-cell grid on [-5, 5]: the box spans 6..18, the sink of radius ~1.04 sits at
# the face of cell 12 (x = 0)
ROOT_CELL = 10.0 / 24
GEOMETRIES = {
    "cut-on-box-face": ([[6], [], []], ("FluxContribution", "PrimStencil")),
    "cut-through-sink": ([[12], [], []], ("Fine",)),
}
REJECTED_STEP = 2
# worker 1 holds the sink's upper half under the cut through it; the fine level takes two substeps
# per root step under rk2, so substep 1 stage 1 is the last stage of the last substep of the step
LEVEL_REJECTION = f"reject_level:1:{REJECTED_STEP}:1:1:1"


@pytest.fixture
def sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "refined_sink.py").write_text(SINK)
    (tmp_path / "launched_refined_sink.py").write_text(LAUNCHED)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "launched_refined_sink.py"


def _tiles(out: Path, cuts: list, steps: int, checkpoint: Path | None = None) -> Path:
    """the in-process host tiles cut at `cuts`, on the session's backend"""
    fields = f"gpus=2, decompose={cuts!r}, data_directory=sys.argv[1]"
    if checkpoint is not None:
        fields += f", checkpoint_file={str(checkpoint)!r}"
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from refined_sink import RefinedSink\n"
        f"p = RefinedSink({fields})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), "host tiles", env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"), devices=2)
    return _final(out)


def _launch(script: Path, out: Path, cuts: list, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> ph.Launch:
    """the two-worker launch with no experimental opt-in: the gated sink runs on either backend without it"""
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    run = ph.launch(script, out, workers=2, cuts=cuts, args=args, env=ph.environment(inject))
    assert not run.experimental, "the gated sink ran as experimental"
    return run


def _accreted(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        assert "body_diagnostics/mass_delta" in f, f"{path.name} publishes no body history"
        return np.asarray(f["body_diagnostics/mass_delta"][()])


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a_refined_sink_launch_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path, geometry: str) -> None:
    cuts, roles = GEOMETRIES[geometry]
    want = _tiles(tmp_path / "tiles", cuts, STEPS)
    with h5py.File(want, "r") as f:
        assert "level_1" in f, "the host tiles published no fine level"
        sink_x = float(f["bodies/position"][()][0, 0])
    # the drain books accretion at every step: the receipts are the quantity under comparison
    assert np.all(np.abs(_accreted(want)) > 0.0), f"{geometry}: the sink accreted nothing at some step"
    # a draining ball centered on the cut holds drained cells on both workers, so each sends a
    # partial of every receipt
    if geometry == "cut-through-sink":
        assert sink_x == -5.0 + cuts[0][0] * ROOT_CELL, f"the sink at x = {sink_x} sits off the cut; the cross-worker sum is untested"
    run = _launch(sink, tmp_path / "launched", cuts, STEPS)
    remote = run.remote_sends()
    assert sorted(remote) == [0, 1], f"plan reports from workers {sorted(remote)}"
    for role in roles:
        assert any(remote[w].get(role, 0) > 0 for w in remote), f"{geometry}: the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    assert run.reports() == {0: (STEPS, 0), 1: (STEPS, 0)}, f"(steps, rejections) per worker {run.reports()}"
    ph.disagreement(_assert_same, run.final, want, f"{ph.BACKEND} {geometry}: two launched refined workers")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "launched", tmp_path / "tiles", f"{ph.BACKEND} {geometry}")


def test_a_refined_sink_resumed_onto_another_partition_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path) -> None:
    face_cuts = GEOMETRIES["cut-on-box-face"][0]
    first = _launch(sink, tmp_path / "first", GEOMETRIES["cut-through-sink"][0], STEPS // 2).final
    assert _iteration(first) == STEPS // 2
    # `max_steps` caps the absolute root iteration: both resumed runs take STEPS - STEPS // 2 steps
    want = _tiles(tmp_path / "tiles", face_cuts, STEPS, checkpoint=first)
    resumed = _launch(sink, tmp_path / "resumed", face_cuts, STEPS, checkpoint=first)
    assert resumed.reports() == {0: (STEPS - STEPS // 2, 0), 1: (STEPS - STEPS // 2, 0)}, f"(steps, rejections) per worker {resumed.reports()}"
    # a run's history holds the steps it took, past the checkpoint's clock; the accreted total
    # carries the receipts booked before the restart
    with h5py.File(first, "r") as f0, h5py.File(resumed.final, "r") as f1:
        resumed_times = np.asarray(f1["body_diagnostics/time"][()]).ravel()
        assert resumed_times.shape[0] == STEPS - STEPS // 2, f"{resumed_times.shape[0]} history rows for {STEPS - STEPS // 2} resumed steps"
        assert resumed_times[0] > float(f0["level_0"].attrs["time"]), "the resumed history repeats a step before the checkpoint"
        before, after = float(f0["bodies/total_accreted_mass"][0]), float(f1["bodies/total_accreted_mass"][0])
        assert 0.0 < before < after, f"the accreted total {after} after the restart drops the {before} booked before it"
    ph.disagreement(_assert_same, resumed.final, want, f"{ph.BACKEND}: resumed onto the cut on the box face")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "resumed", tmp_path / "tiles", f"{ph.BACKEND}: resumed")


def test_a_fine_substep_rejection_with_a_cut_sink_replays_as_the_capped_launch(tmp_path: Path, sink: Path) -> None:
    through = GEOMETRIES["cut-through-sink"][0]
    steps = REJECTED_STEP + 1
    rejected = _launch(sink, tmp_path / "rejected", through, steps, inject=LEVEL_REJECTION)
    assert rejected.reports() == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {rejected.reports()}: a worker kept out of the rollback"
    clean = _launch(sink, tmp_path / "clean", through, steps)
    assert clean.reports() == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(rejected.final, "r") as fr, h5py.File(clean.final, "r") as fc:
        accepted = float(fr["body_diagnostics/dt"][REJECTED_STEP])
        assert accepted < float(fc["body_diagnostics/dt"][REJECTED_STEP]), "the rejection left the timestep unreduced; the replay is never exercised"
    # the drain booked accretion in the rejected attempt's first fine substep: discarding it is the
    # transaction under test
    assert abs(_accreted(clean.final)[REJECTED_STEP]).max() > 0.0, "the sink accreted nothing at the rejected step"
    capped = _launch(sink, tmp_path / "capped", through, steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert capped.reports() == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(capped.final, "r") as f:
        assert float(f["body_diagnostics/dt"][REJECTED_STEP]) == accepted, "the capped control's step took another timestep"
    ph.disagreement(_assert_same, rejected.final, capped.final, f"{ph.BACKEND}: fine-substep rejection against the capped clean launch")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "rejected", tmp_path / "capped", f"{ph.BACKEND}: fine-substep rejection")


def _variant_launch(sink: Path, problem: str, tmp_path: Path, expect_ok: bool = True) -> ph.Launch:
    """one step of a sink variant on two workers cut through the sink, with no experimental opt-in"""
    script = sink.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from refined_sink import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    return ph.launch(script, tmp_path / "out", workers=2, cuts=GEOMETRIES["cut-through-sink"][0], args=["--max-steps", "1", "--checkpoint-interval", "1e9"],
                     env=ph.environment(), expect_ok=expect_ok, timeout=600)


@ph.gpu_only
def test_the_gated_sink_at_other_scalars_launches_on_gpus_without_the_opt_in(tmp_path: Path, sink: Path) -> None:
    run = _variant_launch(sink, "ScaledSink", tmp_path)
    assert not run.experimental, "the gated sink ran as experimental"
    assert run.reports() == {0: (1, 0), 1: (1, 0)}, f"(steps, rejections) per worker {run.reports()}"


@ph.gpu_only
@pytest.mark.parametrize("problem", list(UNTESTED_BODIES))
def test_an_untested_body_is_refused_by_name_on_gpus(tmp_path: Path, sink: Path, problem: str) -> None:
    run = _variant_launch(sink, problem, tmp_path, expect_ok=False)
    assert run.returncode != 0, f"the launch ran:\n{run.stdout}"
    want = f"mesh refinement on gpu device memory with {UNTESTED_BODIES[problem]} is installed on the launched path without a distributed gate"
    assert want in " ".join(run.stderr.split()), run.stderr[-3000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


@ph.cpu_only
@pytest.mark.parametrize("problem", ["ScaledSink", *UNTESTED_BODIES])
def test_each_sink_variant_runs_on_the_cpu_launch(tmp_path: Path, sink: Path, problem: str) -> None:
    """the gpu refusals are the device scope's: each variant is a valid refined cpu launch"""
    run = _variant_launch(sink, problem, tmp_path)
    assert run.reports() == {0: (1, 0), 1: (1, 0)}, f"(steps, rejections) per worker {run.reports()}"
