# =============================================================================
# test_launch_nested_devices.py
#
# a 3D newtonian blast under two nested refined levels (ratio two each) on two
# launched workers against the single-device hierarchy, on the session's
# backend and placement (parity_harness). the 24-cell root spans [-0.5, 0.5]^3;
# level 1 covers root cells 6..18, level 2 root cells 9..15, and the blast of
# radius 0.15 straddles the level-2 faces from the first step. two cuts:
# - on the level-2 box's low x face (root cell 9): worker 0 holds part of level
#   1 and no level-2 tile; the level-2 face's flux-register contributions
#   travel to it, and its level-1 cells feed the level-2 parent stencil
# - through the level-2 box (root cell 12): the level-2 halos travel between
#   the workers
# each launch reports, by plan role and level, the transfers its compiled plans
# schedule to the other worker; the gate asserts the nested roles its cut is
# built to exercise, and the bitwise equality with the single device is the
# runtime evidence.
# - evolution: priming, subcycling, parent stencils and reflux on both levels;
#   every published checkpoint, the opening one included, equals the single
#   device's, whole file, all three levels
# - restart: half the steps cut through the level-2 box, the rest resumed on
#   the cut on its face, against the uninterrupted single device
# - rollback: a rejection in the last stage of the last level-2 substep of one
#   step, on the worker holding level 2, rolls both workers back; it equals the
#   clean launch capped at the accepted timestep
# - admission: two nested regions run on the gpu without SIMBI_LAUNCH_EXPERIMENTAL;
#   a third nested region is refused by name on the gpu without it and runs on
#   the cpu launch
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_nested_devices.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

RESOLUTION = (24, 24, 24)
STEPS = 8
NESTED = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.refined_blast_3d import RefinedBlast3D


class NestedBlast3D(RefinedBlast3D):
    """the blast under a second fine box inside the first, straddling its faces"""

    r_blast: Annotated[float, ProblemParam(0.15, description="blast radius")]
    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam([[-0.25, 0.25, -0.25, 0.25, -0.25, 0.25], [-0.125, 0.125, -0.125, 0.125, -0.125, 0.125]], description="boxes"),
    ]
    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description="ratios")]
'''
THREE_LEVELS = '''
from typing import Annotated

from simbi import ProblemParam
from nested_blast import NestedBlast3D


class ThreeLevels(NestedBlast3D):
    """a third fine box inside the second, over root cells 10.5..13.5"""

    refinement_max_levels: Annotated[int, ProblemParam(4, description="levels")]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam(
            [[-0.25, 0.25, -0.25, 0.25, -0.25, 0.25], [-0.125, 0.125, -0.125, 0.125, -0.125, 0.125],
             [-0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625]],
            description="boxes",
        ),
    ]
    refinement_ratios: Annotated[list[int], ProblemParam([2, 2, 2], description="ratios")]
'''
# root cells of the level-2 box's low x face and of its middle
FACE_CELL = 9
GEOMETRIES = {
    "cut-on-level-2-face": ([[FACE_CELL], [], []], ("NestedFluxContribution{level:2}", "NestedPrimStencil{level:2}")),
    "cut-through-level-2": ([[12], [], []], ("Nested{level:2}",)),
}
UNTESTED_DEPTH = "mesh refinement on gpu device memory with 3 refined regions"
REJECTED_STEP = 2
# worker 1 holds the level-2 box under the cut on its face; level 2 takes four substeps per root
# step under rk2, so substep 3 stage 1 is the last stage of the last level-2 substep of the step
LEVEL_REJECTION = f"reject_level:1:{REJECTED_STEP}:2:3:1"
LAUNCH_ARGS = ["--resolution", ",".join(str(r) for r in RESOLUTION)]


@pytest.fixture
def nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "nested_blast.py").write_text(NESTED)
    (tmp_path / "three_levels.py").write_text(THREE_LEVELS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "nested_blast.py"


def _reference(out: Path, steps: int = STEPS) -> dict[str, Path]:
    """the single-device hierarchy on the session's backend, a checkpoint at every step, by file name"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from nested_blast import NestedBlast3D\n"
        f"runner.run(NestedBlast3D(resolution={RESOLUTION!r}, data_directory=sys.argv[1], checkpoint_interval=1e-9), "
        f"compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), "reference")
    files = {p.name: p for p in out.glob("*.h5")}
    assert any("final" in name for name in files), f"no final checkpoint in {out}"
    return files


def _final(files: dict[str, Path]) -> Path:
    finals = [p for name, p in files.items() if "final" in name]
    assert len(finals) == 1, f"final checkpoints {finals}"
    return finals[0]


def _launch(script: Path, out: Path, cuts, steps: int = STEPS, checkpoint: Path | None = None, inject: str | None = None) -> ph.Launch:
    """the two-worker launch, a checkpoint at every step, with no experimental opt-in: two nested
    regions run on either backend without it"""
    args = ["--max-steps", str(steps), *LAUNCH_ARGS, "--checkpoint-interval", "1e-9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    run = ph.launch(script, out, workers=2, cuts=cuts, args=args, env=ph.environment(inject))
    assert not run.experimental, "the gated nested launch ran as experimental"
    assert any("final" in name for name in run.files), f"no final checkpoint in {out}"
    return run


def _root_cells(path: Path, level: str) -> tuple[int, int]:
    """the root cells a level spans"""
    with h5py.File(path, "r") as f:
        lo, hi = (float(f[level]["mesh/geometry/dim_0"].attrs[k]) for k in ("start", "end"))
        root_lo, root_hi = (float(f["level_0"]["mesh/geometry/dim_0"].attrs[k]) for k in ("start", "end"))
    width = (root_hi - root_lo) / RESOLUTION[0]
    return round((lo - root_lo) / width), round((hi - root_lo) / width)


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a_nested_launch_equals_one_device_at_every_checkpoint(tmp_path: Path, nested: Path, geometry: str) -> None:
    cuts, roles = GEOMETRIES[geometry]
    reference = _reference(tmp_path / "single")
    final = _final(reference)
    with h5py.File(final, "r") as f:
        assert "level_2" in f, "the single device published no second refined level"
    assert _root_cells(final, "level_1") == (6, 18) and _root_cells(final, "level_2") == (FACE_CELL, 15), "the boxes moved off the cuts"
    run = _launch(nested, tmp_path / "two", cuts)
    remote = run.remote_sends()
    assert sorted(remote) == [0, 1], f"plan reports from workers {sorted(remote)}"
    for role in roles:
        assert any(remote[w].get(role, 0) > 0 for w in remote), f"{geometry}: the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    if geometry == "cut-on-level-2-face":
        # worker 0 holds root cells below the level-2 face: level 1 there and no level-2 tile, so it
        # sends no level-2 state and receives the face's contributions
        assert remote[0].get("Nested{level:2}", 0) == 0, f"worker 0 holds a level-2 tile: {remote[0]}"
        assert remote[1].get("NestedFluxContribution{level:2}", 0) > 0, f"the level-2 face sends no contribution to worker 0: {remote[1]}"
    assert run.reports() == {0: (STEPS, 0), 1: (STEPS, 0)}, f"(steps, rejections) per worker {run.reports()}"
    assert sorted(run.files) == sorted(reference), f"checkpoints {sorted(run.files)} against {sorted(reference)}"
    assert len(run.files) >= 4, f"{len(run.files)} checkpoints; the cadence is not exercised"
    for name in sorted(run.files):
        ph.same_checkpoint(run.files[name], reference[name], f"{ph.BACKEND} {geometry}: {name}", expect=("level_2",))


def test_a_nested_launch_resumed_onto_another_partition_equals_one_device(tmp_path: Path, nested: Path) -> None:
    reference = _reference(tmp_path / "single")
    first = _launch(nested, tmp_path / "first", GEOMETRIES["cut-through-level-2"][0], steps=STEPS // 2)
    with h5py.File(first.final, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == STEPS // 2
    resumed = _launch(nested, tmp_path / "resumed", GEOMETRIES["cut-on-level-2-face"][0], checkpoint=first.final)
    assert resumed.reports() == {0: (STEPS - STEPS // 2, 0), 1: (STEPS - STEPS // 2, 0)}, f"(steps, rejections) per worker {resumed.reports()}"
    ph.same_checkpoint(resumed.final, _final(reference), f"{ph.BACKEND}: resumed onto the cut on the level-2 face", expect=("level_2",))


def test_a_deepest_substep_rejection_rolls_both_workers_back_as_the_capped_clean_launch(tmp_path: Path, nested: Path) -> None:
    face = GEOMETRIES["cut-on-level-2-face"][0]
    steps = REJECTED_STEP + 1
    rejected = _launch(nested, tmp_path / "rejected", face, steps=steps, inject=LEVEL_REJECTION)
    assert rejected.reports() == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {rejected.reports()}: a worker kept out of the rollback"
    clean = _launch(nested, tmp_path / "clean", face, steps=steps)
    assert clean.reports() == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(rejected.final, "r") as fr, h5py.File(clean.final, "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    capped = _launch(nested, tmp_path / "capped", face, steps=steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert capped.reports() == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(capped.final, "r") as f:
        assert float(f["level_0"].attrs["dt"]) == accepted, "the capped control's step took another timestep"
    ph.same_checkpoint(rejected.final, capped.final, f"{ph.BACKEND}: level-2 substep rejection against the capped clean launch", expect=("level_2",))


def _three_levels(nested: Path, tmp_path: Path, expect_ok: bool = True) -> ph.Launch:
    """one step of the three-region variant on two workers cut through the boxes, with no experimental opt-in"""
    return ph.launch(nested.parent / "three_levels.py", tmp_path / "out", workers=2, cuts=GEOMETRIES["cut-through-level-2"][0],
                     args=["--max-steps", "1", *LAUNCH_ARGS, "--checkpoint-interval", "1e9"], env=ph.environment(), expect_ok=expect_ok, timeout=600)


@ph.gpu_only
def test_a_third_nested_region_on_gpus_is_refused_by_name_without_the_opt_in(tmp_path: Path, nested: Path) -> None:
    run = _three_levels(nested, tmp_path, expect_ok=False)
    assert run.returncode != 0, f"the launch ran:\n{run.stdout}"
    assert f"{UNTESTED_DEPTH} is installed on the launched path without a distributed gate" in " ".join(run.stderr.split()), run.stderr[-3000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


@ph.cpu_only
def test_a_third_nested_region_runs_on_the_cpu_launch(tmp_path: Path, nested: Path) -> None:
    """the gpu refusal is the device scope's: three nested regions are a valid refined cpu launch"""
    run = _three_levels(nested, tmp_path)
    assert run.reports() == {0: (1, 0), 1: (1, 0)}, f"(steps, rejections) per worker {run.reports()}"
    with h5py.File(run.final, "r") as f:
        assert "level_3" in f, "the cpu launch published no third refined level"
