# =============================================================================
# test_launch_nested_devices.py
#
# a 3D newtonian blast under two nested refined levels (ratio two each) on two
# launched workers against the single-device hierarchy, on the cpu and (with
# SIMBI_LAUNCH_GPU_TESTS) on one gpu per worker. the 24-cell root spans [-0.5,
# 0.5]^3; level 1 covers root cells 6..18, level 2 root cells 9..15, and the
# blast of radius 0.15 straddles the level-2 faces from the first step. two cuts:
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
# =============================================================================
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.checkpoint_compare import assert_same_checkpoint
from simbi.simulation.tests.test_launch import EXIT, MODES, _bound_devices, _layout, bounded_processes, needs_gpu  # noqa: F401  the autouse process guard

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
REMOTE = re.compile(r"SIMBI launch refined plans: worker=(\d+)/\d+ remote_sends=(\S+)")
REPORT = re.compile(r"SIMBI launch: worker=(\d+)/\d+ steps=(\d+) rejections=(\d+) refined=1")
UNTESTED_DEPTH = "mesh refinement on gpu device memory with 3 refined regions"
REJECTED_STEP = 2
# worker 1 holds the level-2 box under the cut on its face; level 2 takes four substeps per root
# step under rk2, so substep 3 stage 1 is the last stage of the last level-2 substep of the step
LEVEL_REJECTION = f"reject_level:1:{REJECTED_STEP}:2:3:1"


@pytest.fixture
def nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "nested_blast.py").write_text(NESTED)
    (tmp_path / "three_levels.py").write_text(THREE_LEVELS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "nested_blast.py"


def _env(inject: str | None = None) -> dict[str, str]:
    """no experimental opt-in: two nested regions run on either device without it"""
    env = dict(os.environ)
    env.pop("SIMBI_FABRIC_INJECT", None)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    return env


def _reference(out: Path, mode: str, steps: int = STEPS) -> dict[str, Path]:
    """the single-device hierarchy, a checkpoint at every step, by file name"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from nested_blast import NestedBlast3D\n"
        f"runner.run(NestedBlast3D(resolution={RESOLUTION!r}, data_directory=sys.argv[1], checkpoint_interval=1e-9), "
        f"compute_mode={mode!r}, max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, f"reference failed:\n{done.stderr[-3000:]}"
    files = {p.name: p for p in out.glob("*.h5")}
    assert any("final" in name for name in files), f"no final checkpoint in {out}"
    return files


def _launch(script: Path, out: Path, layout: Path, mode: str, steps: int = STEPS, checkpoint: Path | None = None,
            inject: str | None = None) -> tuple[dict[str, Path], str]:
    """the two-worker launch, a checkpoint at every step: its files by name and its stderr"""
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out), "--checkpoint-interval", "1e-9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    if mode == "gpu":
        cmd.append("--gpu")
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=_env(inject))
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-3000:]}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == [0, 1] and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    assert "experimental combination" not in done.stderr, "the gated nested launch ran as experimental"
    if mode == "gpu":
        bound = _bound_devices(done.stderr)
        assert sorted(bound) == [0, 1] and bound[0] != bound[1], f"device bindings {bound}"
    files = {p.name: p for p in out.glob("*.h5")}
    assert any("final" in name for name in files), f"no final checkpoint in {out}"
    return files, done.stderr


def _remote_sends(stderr: str) -> dict[int, dict[str, int]]:
    """each worker's remote transfers by plan role; a nested role names its level"""
    out = {}
    for worker, roles in REMOTE.findall(stderr):
        out[int(worker)] = {role: int(n) for role, n in (item.rsplit(":", 1) for item in roles.split(","))}
    assert sorted(out) == [0, 1], f"plan reports from workers {sorted(out)}"
    return out


def _reports(stderr: str) -> dict[int, tuple[int, int]]:
    return {int(w): (int(n), int(r)) for w, n, r in REPORT.findall(stderr)}


def _final(files: dict[str, Path]) -> Path:
    finals = [p for name, p in files.items() if "final" in name]
    assert len(finals) == 1, f"final checkpoints {finals}"
    return finals[0]


def _root_cells(path: Path, level: str) -> tuple[int, int]:
    """the root cells a level spans"""
    with h5py.File(path, "r") as f:
        lo, hi = (float(f[level]["mesh/geometry/dim_0"].attrs[k]) for k in ("start", "end"))
        root_lo, root_hi = (float(f["level_0"]["mesh/geometry/dim_0"].attrs[k]) for k in ("start", "end"))
    width = (root_hi - root_lo) / RESOLUTION[0]
    return round((lo - root_lo) / width), round((hi - root_lo) / width)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a_nested_launch_equals_one_device_at_every_checkpoint(tmp_path: Path, nested: Path, geometry: str, mode: str) -> None:
    cuts, roles = GEOMETRIES[geometry]
    reference = _reference(tmp_path / "single", mode)
    final = _final(reference)
    with h5py.File(final, "r") as f:
        assert "level_2" in f, "the single device published no second refined level"
    assert _root_cells(final, "level_1") == (6, 18) and _root_cells(final, "level_2") == (FACE_CELL, 15), "the boxes moved off the cuts"
    layout = _layout(tmp_path / "two.toml", workers=2, cuts=cuts)
    launched, stderr = _launch(nested, tmp_path / "two", layout, mode)
    remote = _remote_sends(stderr)
    for role in roles:
        assert any(remote[w].get(role, 0) > 0 for w in remote), f"{geometry}: the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    if geometry == "cut-on-level-2-face":
        # worker 0 holds root cells below the level-2 face: level 1 there and no level-2 tile, so it
        # sends no level-2 state and receives the face's contributions
        assert remote[0].get("Nested{level:2}", 0) == 0, f"worker 0 holds a level-2 tile: {remote[0]}"
        assert remote[1].get("NestedFluxContribution{level:2}", 0) > 0, f"the level-2 face sends no contribution to worker 0: {remote[1]}"
    assert _reports(stderr) == {0: (STEPS, 0), 1: (STEPS, 0)}, f"(steps, rejections) per worker {_reports(stderr)}"
    assert sorted(launched) == sorted(reference), f"checkpoints {sorted(launched)} against {sorted(reference)}"
    assert len(launched) >= 4, f"{len(launched)} checkpoints; the cadence is not exercised"
    for name in sorted(launched):
        assert_same_checkpoint(launched[name], reference[name], f"{mode} {geometry}: {name}", expect=("level_2",))


@pytest.mark.parametrize("mode", MODES)
def test_a_nested_launch_resumed_onto_another_partition_equals_one_device(tmp_path: Path, nested: Path, mode: str) -> None:
    reference = _reference(tmp_path / "single", mode)
    through = _layout(tmp_path / "through.toml", workers=2, cuts=GEOMETRIES["cut-through-level-2"][0])
    face = _layout(tmp_path / "face.toml", workers=2, cuts=GEOMETRIES["cut-on-level-2-face"][0])
    first, _ = _launch(nested, tmp_path / "first", through, mode, steps=STEPS // 2)
    partial = _final(first)
    with h5py.File(partial, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == STEPS // 2
    resumed, stderr = _launch(nested, tmp_path / "resumed", face, mode, checkpoint=partial)
    assert _reports(stderr) == {0: (STEPS - STEPS // 2, 0), 1: (STEPS - STEPS // 2, 0)}, f"(steps, rejections) per worker {_reports(stderr)}"
    assert_same_checkpoint(_final(resumed), _final(reference), f"{mode}: resumed onto the cut on the level-2 face", expect=("level_2",))


@pytest.mark.parametrize("mode", MODES)
def test_a_deepest_substep_rejection_rolls_both_workers_back_as_the_capped_clean_launch(tmp_path: Path, nested: Path, mode: str) -> None:
    face = _layout(tmp_path / "face.toml", workers=2, cuts=GEOMETRIES["cut-on-level-2-face"][0])
    steps = REJECTED_STEP + 1
    rejected, stderr = _launch(nested, tmp_path / "rejected", face, mode, steps=steps, inject=LEVEL_REJECTION)
    assert _reports(stderr) == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {_reports(stderr)}: a worker kept out of the rollback"
    clean, clean_err = _launch(nested, tmp_path / "clean", face, mode, steps=steps)
    assert _reports(clean_err) == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(_final(rejected), "r") as fr, h5py.File(_final(clean), "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    capped, capped_err = _launch(nested, tmp_path / "capped", face, mode, steps=steps,
                                 inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert _reports(capped_err) == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(_final(capped), "r") as f:
        assert float(f["level_0"].attrs["dt"]) == accepted, "the capped control's step took another timestep"
    assert_same_checkpoint(_final(rejected), _final(capped), f"{mode}: level-2 substep rejection against the capped clean launch", expect=("level_2",))


def _three_levels(nested: Path, tmp_path: Path, mode: str) -> subprocess.CompletedProcess:
    """one step of the three-region variant on two workers cut through the boxes"""
    script = nested.parent / "three_levels.py"
    layout = _layout(tmp_path / "two.toml", workers=2, cuts=GEOMETRIES["cut-through-level-2"][0])
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    if mode == "gpu":
        cmd.append("--gpu")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env())


@needs_gpu
def test_a_third_nested_region_on_gpus_is_refused_by_name_without_the_opt_in(tmp_path: Path, nested: Path) -> None:
    done = _three_levels(nested, tmp_path, "gpu")
    assert done.returncode != 0, f"the launch ran:\n{done.stdout}"
    assert f"{UNTESTED_DEPTH} is installed on the launched path without a distributed gate" in " ".join(done.stderr.split()), done.stderr[-3000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


def test_a_third_nested_region_runs_on_the_cpu_launch(tmp_path: Path, nested: Path) -> None:
    """the gpu refusal is the device scope's: three nested regions are a valid refined cpu launch"""
    done = _three_levels(nested, tmp_path, "cpu")
    assert done.returncode == 0, f"three nested regions are no valid refined cpu launch:\n{done.stderr[-3000:]}"
    assert _reports(done.stderr) == {0: (1, 0), 1: (1, 0)}, f"(steps, rejections) per worker {_reports(done.stderr)}"
    with h5py.File(_final({p.name: p for p in (tmp_path / "out").glob("*.h5")}), "r") as f:
        assert "level_3" in f, "the cpu launch published no third refined level"
