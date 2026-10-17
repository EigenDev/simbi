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
#   equals the run cut clear of it, and both differ from the run rejecting
#   nothing.
# every arm runs as a fresh process, the tiled ones as host tiles on a cpu
# build.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_nested_face_on_cut.py
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.checkpoint_compare import assert_same_checkpoint

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


def _script(path: Path, level_2: list[float]) -> str:
    path.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n\n"
        "class NestedFaceOnCut(RefinedBlast):\n"
        "    refinement_max_levels: Annotated[int, ProblemParam(3, description='coarse + 2 fine')]\n"
        "    refinement_regions: Annotated[\n"
        "        list[list[float]],\n"
        f"        ProblemParam([{LEVEL_1!r}, {level_2!r}], description='nested patches'),\n"
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


def test_a_rejected_level_2_stage_after_booked_units_replays_alike(tmp_path: Path) -> None:
    """a level-2 stage of the second substep of root step 2 rejected after the first substep's
    units crossed the level-1 cut: the run cut on the face equals the run cut clear of it (level 2
    on one tile, at x = -0.125), and both differ from the run rejecting nothing"""
    script = _script(tmp_path / "low.py", FACES["low"])
    reject = "reject_level:0:2:2:1:1"
    on_face = _capped(script, tmp_path / "on_face", 2, STEPS, [[16], []], inject=reject)
    clear = _capped(script, tmp_path / "clear", 2, STEPS, [[12], []], inject=reject)
    assert_same_checkpoint(on_face, clear, "a rejected level-2 stage cut on and clear of the face", expect=["level_1", "level_2"])
    clean = _capped(script, tmp_path / "clean", 2, STEPS, [[16], []])
    with h5py.File(on_face, "r") as a, h5py.File(clean, "r") as b:
        assert not np.array_equal(a["level_2/conserved/den"][()], b["level_2/conserved/den"][()]), (
            "the rejection left level 2 unchanged; the replay is never exercised"
        )
