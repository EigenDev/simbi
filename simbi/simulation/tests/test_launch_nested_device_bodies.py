# =============================================================================
# test_launch_nested_device_bodies.py
#
# a stationary gravitating, draining sink under two nested refined levels on
# two launched workers, on the session's backend and placement (parity_harness).
# the 24-cell root spans [-5, 5]^3; level 1 covers root cells 6..18, level 2
# root cells 9..15, and the sink of radius ~0.52 sits at the origin inside
# level 2. launched workers sum every tile's body feedback in flat tile order,
# the order host tiles sum it in, so the reference is the in-process host tiles
# cut at the same cells on the same backend (on the gpu build
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
# the sink under two nested regions runs on either device without
# SIMBI_LAUNCH_EXPERIMENTAL.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_nested_device_bodies.py
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


def _tiles(out: Path, cuts: list, steps: int, checkpoint: Path | None = None) -> Path:
    """the in-process host tiles cut at `cuts`, on the session's backend"""
    fields = f"gpus=2, decompose={cuts!r}, data_directory=sys.argv[1]"
    if checkpoint is not None:
        fields += f", checkpoint_file={str(checkpoint)!r}"
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from nested_sink import NestedSink\n"
        f"p = NestedSink({fields})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), "host tiles", env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"), devices=2)
    return _final(out)


def _launch(script: Path, out: Path, cuts: list, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> ph.Launch:
    """the two-worker launch with no experimental opt-in: the sink under two nested regions runs on
    either backend without it"""
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    run = ph.launch(script, out, workers=2, cuts=cuts, args=args, env=ph.environment(inject))
    assert not run.experimental, "the gated nested sink ran as experimental"
    return run


def _accreted(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        assert "body_diagnostics/mass_delta" in f, f"{path.name} publishes no body history"
        return np.asarray(f["body_diagnostics/mass_delta"][()])


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a_nested_sink_launch_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path, geometry: str) -> None:
    cuts, roles = GEOMETRIES[geometry]
    want = _tiles(tmp_path / "tiles", cuts, STEPS)
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
    run = _launch(sink, tmp_path / "launched", cuts, STEPS)
    remote = run.remote_sends()
    assert sorted(remote) == [0, 1], f"plan reports from workers {sorted(remote)}"
    for role in roles:
        assert any(remote[w].get(role, 0) > 0 for w in remote), f"{geometry}: the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    if geometry == "cut-on-level-2-face":
        assert remote[0].get("Nested{level:2}", 0) == 0, f"worker 0 holds a level-2 tile: {remote[0]}"
        assert remote[1].get("NestedFluxContribution{level:2}", 0) > 0, f"the level-2 face sends no contribution to worker 0: {remote[1]}"
    assert run.reports() == {0: (STEPS, 0), 1: (STEPS, 0)}, f"(steps, rejections) per worker {run.reports()}"
    ph.disagreement(_assert_same, run.final, want, f"{ph.BACKEND} {geometry}: two launched nested workers")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "launched", tmp_path / "tiles", f"{ph.BACKEND} {geometry}")


def test_a_nested_sink_resumed_onto_another_partition_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path) -> None:
    face_cuts = GEOMETRIES["cut-on-level-2-face"][0]
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
    ph.disagreement(_assert_same, resumed.final, want, f"{ph.BACKEND}: resumed onto the cut on the level-2 face")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "resumed", tmp_path / "tiles", f"{ph.BACKEND}: resumed")


def test_a_deepest_substep_rejection_with_a_cut_sink_replays_as_the_capped_launch(tmp_path: Path, sink: Path) -> None:
    through = GEOMETRIES["cut-through-sink"][0]
    steps = REJECTED_STEP + 1
    rejected = _launch(sink, tmp_path / "rejected", through, steps, inject=LEVEL_REJECTION)
    assert rejected.reports() == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {rejected.reports()}: a worker kept out of the rollback"
    clean = _launch(sink, tmp_path / "clean", through, steps)
    assert clean.reports() == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(rejected.final, "r") as fr, h5py.File(clean.final, "r") as fc:
        accepted = float(fr["body_diagnostics/dt"][REJECTED_STEP])
        assert accepted < float(fc["body_diagnostics/dt"][REJECTED_STEP]), "the rejection left the timestep unreduced; the replay is never exercised"
    # the drain booked accretion in the rejected attempt's earlier level-2 substeps: discarding it is
    # the transaction under test
    assert abs(_accreted(clean.final)[REJECTED_STEP]).max() > 0.0, "the sink accreted nothing at the rejected step"
    capped = _launch(sink, tmp_path / "capped", through, steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert capped.reports() == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(capped.final, "r") as f:
        assert float(f["body_diagnostics/dt"][REJECTED_STEP]) == accepted, "the capped control's step took another timestep"
    ph.disagreement(_assert_same, rejected.final, capped.final, f"{ph.BACKEND}: level-2 substep rejection against the capped clean launch")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "rejected", tmp_path / "capped", f"{ph.BACKEND}: level-2 substep rejection")


# root cells 6 and 9: the cuts lie on level 1's and on level 2's low x faces, so the session
# gathers a parent prim stencil on both levels
BOTH_FACES = [[6, FACE_CELL], [], []]


def test_a_nested_sink_cut_on_both_level_faces_publishes_the_host_tiles_bodies(tmp_path: Path, sink: Path) -> None:
    """three workers, one cut on each refined level's low x face: the launch gathers prim stencils on
    level 1 (from the root) and level 2 (from level 1) and publishes the three host tiles' grid,
    bodies, body history and diagnostics"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from nested_sink import NestedSink\n"
        f"p = NestedSink(gpus=3, decompose={BOTH_FACES!r}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={STEPS})\n"
    )
    ph.python(ph.snippet_args(snippet, str(tmp_path / "tiles")), "host tiles", devices=3)
    want = _final(tmp_path / "tiles")
    assert np.all(np.abs(_accreted(want)) > 0.0), "the sink accreted nothing at some step"
    run = ph.launch(sink, tmp_path / "launched", workers=3, cuts=BOTH_FACES, args=["--max-steps", str(STEPS), "--checkpoint-interval", "1e9"],
                    env=ph.environment())
    remote = run.remote_sends()
    for role in ("PrimStencil", "NestedPrimStencil{level:2}"):
        assert any(r.get(role, 0) > 0 for r in remote.values()), f"the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    assert run.reports() == {w: (STEPS, 0) for w in range(3)}, f"(steps, rejections) per worker {run.reports()}"
    ph.disagreement(_assert_same, run.final, want, f"{ph.BACKEND}: three launched workers cut on both level faces")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "launched", tmp_path / "tiles", f"{ph.BACKEND}: cut on both level faces")
