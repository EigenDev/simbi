# =============================================================================
# test_tiled_nested_face_on_cut.py
#
# a level-2 patch face on a cut of level 1, on host tiles under the production
# halo. the refined blast carries level 1 over [-0.25, 0.25]^2, spanning the
# root cuts at x = 0 and y = 0, and a level-2 patch whose faces lie on those
# cuts: the tile across the cut holds level 1 and none of level 2, receives
# the face's register units and refluxes the level-1 cells they correct, and
# the face's fine ghosts prolong from a level-1 stencil gathered across the
# cut. host tiles equal the single grid bit for bit on every level:
# - with the low x face, the high x face, and a corner (low x and high y faces
#   on the cuts of both axes);
# - resumed on a different partition from a checkpoint the two-tile run wrote
#   halfway;
# - after a level-2 stage rejected in the second substep of a root step,
#   once the first substep's units crossed the cut: the run cut on the face
#   equals the run cut clear of it and the clean run whose rejected step takes
#   the timestep the replay accepted.
# a launch publishes the same state: the faces on two and four workers, a
# two-worker checkpoint resumed on four, and the rejected stage replayed alike
# on and clear of the face, counted once on every worker, equal to the capped
# clean launch. a nested unit listed twice ends the session before any output,
# and a launch carrying parent prim stencils on two levels is refused by name.
# every arm runs as a fresh process, the tiled ones as host tiles on a cpu
# build.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_nested_face_on_cut.py
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
from simbi.simulation.tests.test_launch import EXIT, _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 6
# level 1 over [-0.25, 0.25]^2 on the 32^2 root of [-0.5, 0.5]^2: root cells 8..24 on each axis
LEVEL_1 = [-0.25, 0.25, -0.25, 0.25]
FACES = {
    "low": [0.0, 0.125, -0.125, 0.125],
    "high": [-0.125, 0.0, -0.125, 0.125],
    "corner": [0.0, 0.125, -0.125, 0.0],
}
# the nested prolongation reach in level-1 cells past a level-2 face: half the level-2 halo
# (one deeper than level 1's, itself one deeper than the root's) rounded up, plus the stencil
# half-width of 2
NESTED_REACH = 4


def _script(path: Path, level_2: list[float], level_1: list[float] = LEVEL_1) -> str:
    path.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n\n"
        "class NestedFaceOnCut(RefinedBlast):\n"
        "    refinement_max_levels: Annotated[int, ProblemParam(3, description='coarse + 2 fine')]\n"
        "    refinement_regions: Annotated[\n"
        "        list[list[float]],\n"
        f"        ProblemParam([{level_1!r}, {level_2!r}], description='nested patches'),\n"
        "    ]\n"
        "    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratio per jump')]\n"
    )
    return str(path)


def _capped(script: str, out: Path, gpus: int, steps: int, cuts: list[list[int]] | None = None,
            checkpoint: Path | None = None, inject: str | None = None) -> Path:
    """the nested problem of `script` on `gpus` devices cut at `cuts`, capped at the absolute root
    iteration `steps` and resumed from `checkpoint` when given; its final checkpoint"""
    settings = f"p.gpus = {gpus}\n" + (f"p.decompose = {cuts!r}\n" if cuts is not None else "")
    resume = f"p.checkpoint_file = {str(checkpoint)!r}\n" if checkpoint is not None else ""
    snippet = (
        "import importlib, pathlib, sys\n"
        "from simbi.simulation import runner\n"
        "script = pathlib.Path(sys.argv[1])\n"
        "sys.path.insert(0, str(script.parent))\n"
        "module = importlib.import_module(script.stem)\n"
        "p = module.NestedFaceOnCut.from_cli(['--resolution', '32,32,1'])\n"
        + settings
        + "p.checkpoint_interval = 1e9\n"
        "p.data_directory = sys.argv[2]\n"
        + resume
        + f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    env.pop("SIMBI_FABRIC_INJECT", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    done = subprocess.run([sys.executable, "-c", snippet, script, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{script} on {gpus} device(s) cut at {cuts} to step {steps} failed:\n{done.stderr[-3000:]}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final


def _assert_premises(one: Path) -> None:
    """the single grid holds three levels under a halo short of the nested reach, and level 2
    carries structure the face's flux moves"""
    with h5py.File(one, "r") as f:
        assert "level_2" in f, "the single grid wrote no level 2; the gate is vacuous"
        halo = int(f["level_1/mesh"].attrs["halo_width"])
        assert halo < NESTED_REACH, f"level 1's halo ({halo}) holds the nested reach; the cut face reads no gathered cell"
        den = f["level_2/conserved/den"][()]
        assert np.ptp(den) > 1e-2, f"level 2 carries a density spread of {np.ptp(den):.3e}; the gate is blind"


@pytest.mark.parametrize(("face", "cuts"), [("low", [[16], []]), ("high", [[16], []]), ("corner", [[16], [16]])])
def test_a_level_2_face_on_a_level_1_cut_matches_the_single_grid(tmp_path: Path, face: str, cuts: list[list[int]]) -> None:
    """the level-2 face lies on the root cut at x = 0 (and y = 0 for the corner), inside level 1:
    the tile across the cut holds level 1 alone and receives the face's units"""
    script = _script(tmp_path / f"{face}.py", FACES[face])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    gpus = int(np.prod([len(c) + 1 for c in cuts]))
    tiles = _capped(script, tmp_path / "tiles", gpus, STEPS, cuts)
    assert_same_checkpoint(tiles, one, f"a level-2 {face} face on a level-1 cut", expect=["level_1", "level_2"])


def test_a_level_2_face_on_a_level_1_cut_resumes_on_another_partition(tmp_path: Path) -> None:
    """two tiles cut on the low face write a checkpoint at step 3; four tiles cut on both axes
    resume it to step 6 and equal the uninterrupted single grid"""
    script = _script(tmp_path / "low.py", FACES["low"])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    half = _capped(script, tmp_path / "half", 2, STEPS // 2, [[16], []])
    resumed = _capped(script, tmp_path / "resumed", 4, STEPS, [[16], [16]], checkpoint=half)
    assert_same_checkpoint(resumed, one, "a level-2 face on a level-1 cut resumed on four tiles", expect=["level_1", "level_2"])


# the root step whose level-2 stage a rejecting run rejects; the rejecting runs stop after it
REJECTED_STEP = 2


def _accepted_dt(rejecting: Path, clean: Path) -> float:
    """the root timestep the replay of the rejected step accepted, which the clean run's exceeds"""
    with h5py.File(rejecting, "r") as fr, h5py.File(clean, "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    return accepted


def _capped_clean_launch(script: str, tmp_path: Path, accepted: float) -> Path:
    """the clean two-worker launch cut on the face whose rejected step takes the timestep the replay
    accepted: the same accepted timestep sequence, entered from a state no rollback touched"""
    layout = _layout(tmp_path / "capped.toml", workers=2, cuts=[[16], []])
    out = tmp_path / "capped"
    done = _launch(script, out, layout, REJECTED_STEP + 1, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    capped = _launched(done, out, 2)
    assert {int(r) for r in REJECTIONS.findall(done.stderr)} == {0}, "the capped control rejected a step"
    return capped


def test_a_rejected_level_2_stage_after_booked_units_replays_to_the_capped_clean_run(tmp_path: Path) -> None:
    """a level-2 stage of the second substep of root step 2 rejected after the first substep's
    units crossed the level-1 cut: the host tiles cut on the face equal those cut clear of it (level
    2 on one tile, at x = -0.125), and equal the clean launch whose step 2 takes the timestep the
    replay accepted"""
    script = _script(tmp_path / "low.py", FACES["low"])
    reject = f"reject_level:0:{REJECTED_STEP}:2:1:1"
    on_face = _capped(script, tmp_path / "on_face", 2, REJECTED_STEP + 1, [[16], []], inject=reject)
    clear = _capped(script, tmp_path / "clear", 2, REJECTED_STEP + 1, [[12], []], inject=reject)
    assert_same_checkpoint(on_face, clear, "a rejected level-2 stage cut on and clear of the face", expect=["level_1", "level_2"])
    clean = _capped(script, tmp_path / "clean", 2, REJECTED_STEP + 1, [[16], []])
    capped = _capped_clean_launch(script, tmp_path, _accepted_dt(on_face, clean))
    assert_same_checkpoint(on_face, capped, "a rejected level-2 stage on host tiles against the capped clean launch", expect=["level_1", "level_2"])


REJECTIONS = re.compile(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)")


def _launch(script: str, out: Path, layout: Path, steps: int, checkpoint: Path | None = None,
            inject: str | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", "32,32,1", "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(script).parent), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_FABRIC_INJECT", None)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)


def _launched(done: subprocess.CompletedProcess, out: Path, workers: int) -> Path:
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout[-3000:]}\nstderr:\n{done.stderr[-3000:]}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final


@pytest.mark.parametrize(("face", "cuts"), [("low", [[16], []]), ("high", [[16], []]), ("corner", [[16], [16]])])
def test_a_level_2_face_on_a_level_1_cut_matches_the_single_grid_on_launch(tmp_path: Path, face: str, cuts: list[list[int]]) -> None:
    """the launched workers carry the face's units on level 1's partition and gather its level-1
    stencil there, the worker across the cut holding level 1 alone"""
    script = _script(tmp_path / f"{face}.py", FACES[face])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    workers = int(np.prod([len(c) + 1 for c in cuts]))
    layout = _layout(tmp_path / "layout.toml", workers=workers, cuts=cuts)
    launched = _launched(_launch(script, tmp_path / "launched", layout, STEPS), tmp_path / "launched", workers)
    assert_same_checkpoint(launched, one, f"a level-2 {face} face on a level-1 cut on {workers} workers", expect=["level_1", "level_2"])


def test_a_level_2_face_on_a_level_1_cut_resumes_on_more_workers(tmp_path: Path) -> None:
    script = _script(tmp_path / "low.py", FACES["low"])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    two = _layout(tmp_path / "two.toml", workers=2, cuts=[[16], []])
    half = _launched(_launch(script, tmp_path / "half", two, STEPS // 2), tmp_path / "half", 2)
    four = _layout(tmp_path / "four.toml", workers=4, cuts=[[16], [16]])
    resumed = _launched(_launch(script, tmp_path / "resumed", four, STEPS, checkpoint=half), tmp_path / "resumed", 4)
    assert_same_checkpoint(resumed, one, "a level-2 face on a level-1 cut resumed on four workers", expect=["level_1", "level_2"])


def test_a_rejected_level_2_stage_after_booked_units_replays_to_the_capped_clean_run_on_launch(tmp_path: Path) -> None:
    """worker 1, holding level 2, rejects a stage of the second level-2 substep of root step 2 after
    the first substep's units crossed the level-1 cut: the launch cut on the face equals the launch
    cut clear of it, every worker counts the rejection once, and both equal the clean launch whose
    step 2 takes the timestep the replay accepted"""
    script = _script(tmp_path / "low.py", FACES["low"])
    reject = f"reject_level:1:{REJECTED_STEP}:2:1:1"
    finals = {}
    for cut in (16, 12):
        layout = _layout(tmp_path / f"cut{cut}.toml", workers=2, cuts=[[cut], []])
        out = tmp_path / f"rejecting_{cut}"
        done = _launch(script, out, layout, REJECTED_STEP + 1, inject=reject)
        finals[cut] = _launched(done, out, 2)
        counts = {int(r) for r in REJECTIONS.findall(done.stderr)}
        assert counts == {1}, f"cut at {cut}: the rejection was not counted once on every worker: {counts}"
    assert_same_checkpoint(finals[16], finals[12], "a rejected level-2 stage launched cut on and clear of the face", expect=["level_1", "level_2"])
    layout = _layout(tmp_path / "clean.toml", workers=2, cuts=[[16], []])
    clean = _launched(_launch(script, tmp_path / "clean", layout, REJECTED_STEP + 1), tmp_path / "clean", 2)
    capped = _capped_clean_launch(script, tmp_path, _accepted_dt(finals[16], clean))
    assert_same_checkpoint(finals[16], capped, "a rejected level-2 stage launched against the capped clean launch", expect=["level_1", "level_2"])


def test_a_duplicated_nested_unit_ends_the_launch(tmp_path: Path) -> None:
    """worker 1 lists its sends of the level-2 face's units twice at the first stage of root step 2:
    the receiving worker's grant counts the plan's sends, and the session ends before any final
    output"""
    script = _script(tmp_path / "low.py", FACES["low"])
    layout = _layout(tmp_path / "two.toml", workers=2, cuts=[[16], []])
    out = tmp_path / "dup"
    done = _launch(script, out, layout, STEPS, inject="nested_contribution:1:2:2:0:0:dup")
    assert done.returncode != 0, "the launch with a duplicated nested unit succeeded"
    assert not list(out.glob("*final*.h5")), "the launch published a final checkpoint"
    assert "this phase sends it" in done.stderr + done.stdout, (done.stderr + done.stdout)[-3000:]


def test_parent_prim_stencils_on_two_levels_are_refused_on_launch(tmp_path: Path) -> None:
    """level 1's low x face lies on the root cut at x = 0 and level 2's low y face on level 1's cut
    at y = 0, each past its parent's halo: the launch gathers every stencil at the primed point in
    one phase, so it refuses the second level by name before any output"""
    script = _script(tmp_path / "two_levels.py", [0.0625, 0.1875, 0.0, 0.125], [0.0, 0.25, -0.25, 0.25])
    layout = _layout(tmp_path / "four.toml", workers=4, cuts=[[16], [16]])
    out = tmp_path / "out"
    done = _launch(script, out, layout, STEPS)
    assert done.returncode != 0, "the launch with stencils on two levels ran"
    assert not list(out.glob("*.h5")), "output was written before the refusal"
    assert "parent prim stencils on 2 levels of one launched session" in " ".join((done.stderr + done.stdout).split()), done.stderr[-3000:]
