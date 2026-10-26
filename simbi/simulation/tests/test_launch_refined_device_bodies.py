# =============================================================================
# test_launch_refined_device_bodies.py
#
# a stationary gravitating, draining sink under one fine box on two launched
# refined workers, on the cpu and (with SIMBI_LAUNCH_GPU_TESTS) on one gpu per
# worker. launched workers sum every tile's body feedback in flat tile order,
# the order host tiles sum it in, so the reference is the in-process host
# tiles cut at the same cells on the same device kind: on the gpu build
# SYMBI_GPU_OVERSUBSCRIBE folds their logical devices onto the visible one.
# every comparison is exact: every level's grid, the bodies group, the per-step
# body history and the body diagnostics file.
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
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import MODES, _bound_devices, _layout, bounded_processes, needs_gpu  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_bodies import _assert_same, _assert_same_diagnostics, _final, _iteration
from simbi.simulation.tests.test_launch_refined_devices import REPORT, _remote_sends

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


def _env(inject: str | None = None) -> dict[str, str]:
    """no experimental opt-in: the gated sink runs on either device without it"""
    env = dict(os.environ)
    env.pop("SIMBI_FABRIC_INJECT", None)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    return env


def _tiles(out: Path, mode: str, cuts: list, steps: int, checkpoint: Path | None = None) -> Path:
    """the in-process host tiles cut at `cuts`, on the mode's device kind"""
    fields = f"gpus=2, decompose={cuts!r}, data_directory=sys.argv[1]"
    if checkpoint is not None:
        fields += f", checkpoint_file={str(checkpoint)!r}"
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from refined_sink import RefinedSink\n"
        f"p = RefinedSink({fields})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={mode!r}, max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"the refined sink on two {mode} host tiles failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(script: Path, out: Path, layout: Path, mode: str, steps: int, checkpoint: Path | None = None,
            inject: str | None = None) -> tuple[Path, str]:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    if mode == "gpu":
        cmd.append("--gpu")
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=_env(inject))
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-3000:]}"
    assert "experimental combination" not in done.stderr, "the gated sink ran as experimental"
    if mode == "gpu":
        bound = _bound_devices(done.stderr)
        assert sorted(bound) == [0, 1] and bound[0] != bound[1], f"device bindings {bound}"
    return _final(out), done.stderr


def _reports(stderr: str) -> dict[int, tuple[int, int]]:
    return {int(w): (int(n), int(r)) for w, n, r in REPORT.findall(stderr)}


def _accreted(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        assert "body_diagnostics/mass_delta" in f, f"{path.name} publishes no body history"
        return np.asarray(f["body_diagnostics/mass_delta"][()])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a_refined_sink_launch_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path, geometry: str, mode: str) -> None:
    cuts, roles = GEOMETRIES[geometry]
    want = _tiles(tmp_path / "tiles", mode, cuts, STEPS)
    with h5py.File(want, "r") as f:
        assert "level_1" in f, "the host tiles published no fine level"
        sink_x = float(f["bodies/position"][()][0, 0])
    # the drain books accretion at every step: the receipts are the quantity under comparison
    assert np.all(np.abs(_accreted(want)) > 0.0), f"{geometry}: the sink accreted nothing at some step"
    # a draining ball centered on the cut holds drained cells on both workers, so each sends a
    # partial of every receipt
    if geometry == "cut-through-sink":
        assert sink_x == -5.0 + cuts[0][0] * ROOT_CELL, f"the sink at x = {sink_x} sits off the cut; the cross-worker sum is untested"
    two = _layout(tmp_path / "two.toml", workers=2, cuts=cuts)
    got, stderr = _launch(sink, tmp_path / "launched", two, mode, STEPS)
    remote = _remote_sends(stderr)
    for role in roles:
        assert any(remote[w].get(role, 0) > 0 for w in remote), f"{geometry}: the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    assert _reports(stderr) == {0: (STEPS, 0), 1: (STEPS, 0)}, f"(steps, rejections) per worker {_reports(stderr)}"
    _assert_same(got, want, f"{mode} {geometry}: two launched refined workers")
    _assert_same_diagnostics(tmp_path / "launched", tmp_path / "tiles", f"{mode} {geometry}")


@pytest.mark.parametrize("mode", MODES)
def test_a_refined_sink_resumed_onto_another_partition_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path, mode: str) -> None:
    through = _layout(tmp_path / "through.toml", workers=2, cuts=GEOMETRIES["cut-through-sink"][0])
    face_cuts = GEOMETRIES["cut-on-box-face"][0]
    face = _layout(tmp_path / "face.toml", workers=2, cuts=face_cuts)
    first, _ = _launch(sink, tmp_path / "first", through, mode, STEPS // 2)
    assert _iteration(first) == STEPS // 2
    # `max_steps` caps the absolute root iteration: both resumed runs take STEPS - STEPS // 2 steps
    want = _tiles(tmp_path / "tiles", mode, face_cuts, STEPS, checkpoint=first)
    got, stderr = _launch(sink, tmp_path / "resumed", face, mode, STEPS, checkpoint=first)
    assert _reports(stderr) == {0: (STEPS - STEPS // 2, 0), 1: (STEPS - STEPS // 2, 0)}, f"(steps, rejections) per worker {_reports(stderr)}"
    # a run's history holds the steps it took, past the checkpoint's clock; the accreted total
    # carries the receipts booked before the restart
    with h5py.File(first, "r") as f0, h5py.File(got, "r") as f1:
        resumed_times = np.asarray(f1["body_diagnostics/time"][()]).ravel()
        assert resumed_times.shape[0] == STEPS - STEPS // 2, f"{resumed_times.shape[0]} history rows for {STEPS - STEPS // 2} resumed steps"
        assert resumed_times[0] > float(f0["level_0"].attrs["time"]), "the resumed history repeats a step before the checkpoint"
        before, after = float(f0["bodies/total_accreted_mass"][0]), float(f1["bodies/total_accreted_mass"][0])
        assert 0.0 < before < after, f"the accreted total {after} after the restart drops the {before} booked before it"
    _assert_same(got, want, f"{mode}: resumed onto the cut on the box face")
    _assert_same_diagnostics(tmp_path / "resumed", tmp_path / "tiles", f"{mode}: resumed")


@pytest.mark.parametrize("mode", MODES)
def test_a_fine_substep_rejection_with_a_cut_sink_replays_as_the_capped_launch(tmp_path: Path, sink: Path, mode: str) -> None:
    through = _layout(tmp_path / "through.toml", workers=2, cuts=GEOMETRIES["cut-through-sink"][0])
    steps = REJECTED_STEP + 1
    rejected, stderr = _launch(sink, tmp_path / "rejected", through, mode, steps, inject=LEVEL_REJECTION)
    assert _reports(stderr) == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {_reports(stderr)}: a worker kept out of the rollback"
    clean, clean_err = _launch(sink, tmp_path / "clean", through, mode, steps)
    assert _reports(clean_err) == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(rejected, "r") as fr, h5py.File(clean, "r") as fc:
        accepted = float(fr["body_diagnostics/dt"][REJECTED_STEP])
        assert accepted < float(fc["body_diagnostics/dt"][REJECTED_STEP]), "the rejection left the timestep unreduced; the replay is never exercised"
    # the drain booked accretion in the rejected attempt's first fine substep: discarding it is the
    # transaction under test
    assert abs(_accreted(clean)[REJECTED_STEP]).max() > 0.0, "the sink accreted nothing at the rejected step"
    capped, capped_err = _launch(sink, tmp_path / "capped", through, mode, steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert _reports(capped_err) == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(capped, "r") as f:
        assert float(f["body_diagnostics/dt"][REJECTED_STEP]) == accepted, "the capped control's step took another timestep"
    _assert_same(rejected, capped, f"{mode}: fine-substep rejection against the capped clean launch")
    _assert_same_diagnostics(tmp_path / "rejected", tmp_path / "capped", f"{mode}: fine-substep rejection")


def _variant_launch(sink: Path, problem: str, tmp_path: Path, mode: str) -> subprocess.CompletedProcess:
    """one step of a sink variant on two workers cut through the sink"""
    script = sink.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from refined_sink import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    two = _layout(tmp_path / "two.toml", workers=2, cuts=GEOMETRIES["cut-through-sink"][0])
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(two), "--max-steps", "1",
           "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    if mode == "gpu":
        cmd.append("--gpu")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env())


@needs_gpu
def test_the_gated_sink_at_other_scalars_launches_on_gpus_without_the_opt_in(tmp_path: Path, sink: Path) -> None:
    done = _variant_launch(sink, "ScaledSink", tmp_path, "gpu")
    assert done.returncode == 0, f"the gated sink was refused:\n{done.stderr[-3000:]}"
    assert "experimental combination" not in done.stderr, "the gated sink ran as experimental"
    assert _reports(done.stderr) == {0: (1, 0), 1: (1, 0)}, f"(steps, rejections) per worker {_reports(done.stderr)}"


@needs_gpu
@pytest.mark.parametrize("problem", list(UNTESTED_BODIES))
def test_an_untested_body_is_refused_by_name_on_gpus(tmp_path: Path, sink: Path, problem: str) -> None:
    done = _variant_launch(sink, problem, tmp_path, "gpu")
    assert done.returncode != 0, f"the launch ran:\n{done.stdout}"
    want = f"mesh refinement on gpu device memory with {UNTESTED_BODIES[problem]} is installed on the launched path without a distributed gate"
    assert want in " ".join(done.stderr.split()), done.stderr[-3000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


@pytest.mark.parametrize("problem", ["ScaledSink", *UNTESTED_BODIES])
def test_each_sink_variant_runs_on_the_cpu_launch(tmp_path: Path, sink: Path, problem: str) -> None:
    """the gpu refusals are the device scope's: each variant is a valid refined cpu launch"""
    done = _variant_launch(sink, problem, tmp_path, "cpu")
    assert done.returncode == 0, f"{problem} is no valid refined cpu launch:\n{done.stderr[-3000:]}"
    assert _reports(done.stderr) == {0: (1, 0), 1: (1, 0)}, f"(steps, rejections) per worker {_reports(done.stderr)}"
