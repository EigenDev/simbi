# =============================================================================
# test_launch_nested_device_bodies.py
#
# a stationary gravitating, draining sink under two nested refined levels on
# two launched workers, on the cpu and (with SIMBI_LAUNCH_GPU_TESTS) on one gpu
# per worker. the 24-cell root spans [-5, 5]^3; level 1 covers root cells 6..18,
# level 2 root cells 9..15, and the sink of radius ~0.52 sits at the origin
# inside level 2. launched workers sum every tile's body feedback in flat tile
# order, the order host tiles sum it in, so the reference is the in-process
# host tiles cut at the same cells on the same device kind (on the gpu build
# SYMBI_GPU_OVERSUBSCRIBE folds their logical devices onto the visible ones).
# every comparison is exact: every level's grid, the bodies group, the per-step
# body history and the body diagnostics file.
# - evolution: a cut on the level-2 box's low x face (root cell 9: worker 0
#   holds level 1 and no level-2 tile, receives the level-2 face's
#   contributions and books no body feedback) and a cut through level 2 and
#   the sink (both workers hold a partial of the sink's feedback)
# - restart: half the steps cut through the sink, the rest resumed on the cut
#   on the level-2 face, against the host tiles resumed from the same
#   checkpoint on the same cut
# - rollback: a rejection in the last stage of the last level-2 substep of one
#   step, with the sink cut by the workers, against the clean launch capped at
#   the accepted timestep: no accretion, force, torque or heat of the rejected
#   attempt survives
# nested refinement on gpu device memory runs under SIMBI_LAUNCH_EXPERIMENTAL.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_nested_device_bodies.py
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import MODES, _bound_devices, _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_bodies import _assert_same, _assert_same_diagnostics, _final, _iteration
from simbi.simulation.tests.test_launch_nested_devices import EXPERIMENTAL, REPORT, _remote_sends

pytestmark = pytest.mark.simulation

STEPS = 6
NESTED_SINK = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


class NestedSink(SphericalBondiTest):
    """the bondi point mass and its drain at rest at the origin of a 24-cell root on [-5, 5]^3,
    under two nested fine boxes over root cells 6..18 and 9..15; the outer sponge source is off"""

    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam(False, description="no sponge source")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[
        list[list[float]] | None,
        ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], [-1.25, 1.25, -1.25, 1.25, -1.25, 1.25]], description="boxes"),
    ]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2, 2], description="ratios")]
'''
LAUNCHED = "from nested_sink import NestedSink as Base\n\n\nclass LaunchedNestedSink(Base):\n    pass\n"
ROOT_CELL = 10.0 / 24
FACE_CELL = 9
GEOMETRIES = {
    "cut-on-level-2-face": ([[FACE_CELL], [], []], ("NestedFluxContribution{level:2}", "NestedPrimStencil{level:2}")),
    "cut-through-sink": ([[12], [], []], ("Nested{level:2}",)),
}
REJECTED_STEP = 2
# both workers hold level 2 and the sink under the cut through it; level 2 takes four substeps per
# root step under rk2, so substep 3 stage 1 is the last stage of the last level-2 substep
LEVEL_REJECTION = f"reject_level:1:{REJECTED_STEP}:2:3:1"


@pytest.fixture
def sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "nested_sink.py").write_text(NESTED_SINK)
    (tmp_path / "launched_nested_sink.py").write_text(LAUNCHED)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "launched_nested_sink.py"


def _env(mode: str, inject: str | None = None) -> dict[str, str]:
    """nested refinement on gpu device memory runs under the experimental opt-in"""
    env = dict(os.environ)
    env.pop("SIMBI_FABRIC_INJECT", None)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if mode == "gpu":
        env["SIMBI_LAUNCH_EXPERIMENTAL"] = "1"
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
        "from nested_sink import NestedSink\n"
        f"p = NestedSink({fields})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={mode!r}, max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"the nested sink on two {mode} host tiles failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(script: Path, out: Path, layout: Path, mode: str, steps: int, checkpoint: Path | None = None,
            inject: str | None = None) -> tuple[Path, str]:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    if mode == "gpu":
        cmd.append("--gpu")
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=_env(mode, inject))
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-3000:]}"
    if mode == "gpu":
        assert EXPERIMENTAL in " ".join(done.stderr.split()), "the nested gpu launch ran outside the experimental opt-in"
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
def test_a_nested_sink_launch_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path, geometry: str, mode: str) -> None:
    cuts, roles = GEOMETRIES[geometry]
    want = _tiles(tmp_path / "tiles", mode, cuts, STEPS)
    with h5py.File(want, "r") as f:
        assert "level_2" in f, "the host tiles published no second refined level"
        sink_x = float(f["bodies/position"][()][0, 0])
        dim = f["level_2"]["mesh/geometry/dim_0"].attrs
        level_2_lo = round((float(dim["start"]) + 5.0) / ROOT_CELL)
    assert level_2_lo == FACE_CELL, f"level 2 starts at root cell {level_2_lo}, off the face cut"
    # the drain books accretion at every step: the receipts are the quantity under comparison
    assert np.all(np.abs(_accreted(want)) > 0.0), f"{geometry}: the sink accreted nothing at some step"
    if geometry == "cut-through-sink":
        assert sink_x == -5.0 + cuts[0][0] * ROOT_CELL, f"the sink at x = {sink_x} sits off the cut; the cross-worker sum is untested"
    two = _layout(tmp_path / "two.toml", workers=2, cuts=cuts)
    got, stderr = _launch(sink, tmp_path / "launched", two, mode, STEPS)
    remote = _remote_sends(stderr)
    for role in roles:
        assert any(remote[w].get(role, 0) > 0 for w in remote), f"{geometry}: the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    if geometry == "cut-on-level-2-face":
        assert remote[0].get("Nested{level:2}", 0) == 0, f"worker 0 holds a level-2 tile: {remote[0]}"
        assert remote[1].get("NestedFluxContribution{level:2}", 0) > 0, f"the level-2 face sends no contribution to worker 0: {remote[1]}"
    assert _reports(stderr) == {0: (STEPS, 0), 1: (STEPS, 0)}, f"(steps, rejections) per worker {_reports(stderr)}"
    _assert_same(got, want, f"{mode} {geometry}: two launched nested workers")
    _assert_same_diagnostics(tmp_path / "launched", tmp_path / "tiles", f"{mode} {geometry}")


@pytest.mark.parametrize("mode", MODES)
def test_a_nested_sink_resumed_onto_another_partition_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path, mode: str) -> None:
    through = _layout(tmp_path / "through.toml", workers=2, cuts=GEOMETRIES["cut-through-sink"][0])
    face_cuts = GEOMETRIES["cut-on-level-2-face"][0]
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
    _assert_same(got, want, f"{mode}: resumed onto the cut on the level-2 face")
    _assert_same_diagnostics(tmp_path / "resumed", tmp_path / "tiles", f"{mode}: resumed")


@pytest.mark.parametrize("mode", MODES)
def test_a_deepest_substep_rejection_with_a_cut_sink_replays_as_the_capped_launch(tmp_path: Path, sink: Path, mode: str) -> None:
    through = _layout(tmp_path / "through.toml", workers=2, cuts=GEOMETRIES["cut-through-sink"][0])
    steps = REJECTED_STEP + 1
    rejected, stderr = _launch(sink, tmp_path / "rejected", through, mode, steps, inject=LEVEL_REJECTION)
    assert _reports(stderr) == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {_reports(stderr)}: a worker kept out of the rollback"
    clean, clean_err = _launch(sink, tmp_path / "clean", through, mode, steps)
    assert _reports(clean_err) == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(rejected, "r") as fr, h5py.File(clean, "r") as fc:
        accepted = float(fr["body_diagnostics/dt"][REJECTED_STEP])
        assert accepted < float(fc["body_diagnostics/dt"][REJECTED_STEP]), "the rejection left the timestep unreduced; the replay is never exercised"
    # the drain booked accretion in the rejected attempt's earlier level-2 substeps: discarding it is
    # the transaction under test
    assert abs(_accreted(clean)[REJECTED_STEP]).max() > 0.0, "the sink accreted nothing at the rejected step"
    capped, capped_err = _launch(sink, tmp_path / "capped", through, mode, steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert _reports(capped_err) == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(capped, "r") as f:
        assert float(f["body_diagnostics/dt"][REJECTED_STEP]) == accepted, "the capped control's step took another timestep"
    _assert_same(rejected, capped, f"{mode}: level-2 substep rejection against the capped clean launch")
    _assert_same_diagnostics(tmp_path / "rejected", tmp_path / "capped", f"{mode}: level-2 substep rejection")
