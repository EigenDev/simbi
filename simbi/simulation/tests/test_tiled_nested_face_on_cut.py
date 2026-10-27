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
# clean launch. a nested unit listed twice ends the session before any output.
# four workers cut on both axes gather prim stencils on two levels (level 1's
# face on the root cut, level 2's on level 1's cut, and level 2 close enough to
# level 1's face that its stencil reads cells level 1's stencil fills): the
# launch and the host tiles equal the single grid, a two-worker checkpoint
# resumes on the four, and a rejection in the last level-2 substep replays as
# the capped clean launch.
# a ppm level-2 face and a plm level-3 face on cuts prime from their parents'
# crossed cut halos: their step-zero and evolved checkpoints equal the single
# grid's on host tiles and launch. every arm runs as a fresh process on the
# session's backend and placement (parity_harness), the tiled ones as host
# tiles; the arms on host tiles alone involve no launch and run under the local
# placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_nested_face_on_cut.py
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
    return _nested_script(path, [level_1, level_2])


def _nested_script(path: Path, regions: list[list[float]], order: int | None = None) -> str:
    """the refined blast over the nested `regions`, each [x_lo, x_hi, y_lo, y_hi] refining the one
    before it by two; `order` 3 runs ppm with rk3"""
    path.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n\n"
        "class NestedFaceOnCut(RefinedBlast):\n"
        f"    refinement_max_levels: Annotated[int, ProblemParam({len(regions) + 1}, description='levels')]\n"
        "    refinement_regions: Annotated[\n"
        "        list[list[float]],\n"
        f"        ProblemParam({regions!r}, description='nested patches'),\n"
        "    ]\n"
        f"    refinement_ratios: Annotated[list[int], ProblemParam({[2] * len(regions)!r}, description='ratio per jump')]\n"
        + (f"    order: Annotated[int, ProblemParam({order}, description='order of accuracy')]\n" if order is not None else "")
    )
    return str(path)


def _capped(script: str, out: Path, gpus: int, steps: int, cuts: list[list[int]] | None = None,
            checkpoint: Path | None = None, inject: str | None = None) -> Path:
    """the nested problem of `script` on the session's backend, one device or `gpus` host tiles cut
    at `cuts`, capped at the absolute root iteration `steps` and resumed from `checkpoint` when
    given; its final checkpoint"""
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
        + f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    env.pop("SIMBI_FABRIC_INJECT", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    ph.python(ph.snippet_args(snippet, script, str(out)), f"{gpus} device(s) cut at {cuts} to step {steps}", env=env, devices=gpus)
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


@ph.local_only
@pytest.mark.parametrize(("face", "cuts"), [("low", [[16], []]), ("high", [[16], []]), ("corner", [[16], [16]])])
def test_a_level_2_face_on_a_level_1_cut_matches_the_single_grid(tmp_path: Path, face: str, cuts: list[list[int]]) -> None:
    """the level-2 face lies on the root cut at x = 0 (and y = 0 for the corner), inside level 1:
    the tile across the cut holds level 1 alone and receives the face's units"""
    script = _script(tmp_path / f"{face}.py", FACES[face])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    gpus = int(np.prod([len(c) + 1 for c in cuts]))
    tiles = _capped(script, tmp_path / "tiles", gpus, STEPS, cuts)
    ph.same_checkpoint(tiles, one, f"a level-2 {face} face on a level-1 cut", expect=["level_1", "level_2"])


@ph.local_only
def test_a_level_2_face_on_a_level_1_cut_resumes_on_another_partition(tmp_path: Path) -> None:
    """two tiles cut on the low face write a checkpoint at step 3; four tiles cut on both axes
    resume it to step 6 and equal the uninterrupted single grid"""
    script = _script(tmp_path / "low.py", FACES["low"])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    half = _capped(script, tmp_path / "half", 2, STEPS // 2, [[16], []])
    resumed = _capped(script, tmp_path / "resumed", 4, STEPS, [[16], [16]], checkpoint=half)
    ph.same_checkpoint(resumed, one, "a level-2 face on a level-1 cut resumed on four tiles", expect=["level_1", "level_2"])


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
    run = _launch(script, tmp_path / "capped", 2, [[16], []], REJECTED_STEP + 1,
                  inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert {r for _, r in run.reports().values()} == {0}, "the capped control rejected a step"
    return run.final


def test_a_rejected_level_2_stage_after_booked_units_replays_to_the_capped_clean_run(tmp_path: Path) -> None:
    """a level-2 stage of the second substep of root step 2 rejected after the first substep's
    units crossed the level-1 cut: the host tiles cut on the face equal those cut clear of it (level
    2 on one tile, at x = -0.125), and equal the clean launch whose step 2 takes the timestep the
    replay accepted"""
    script = _script(tmp_path / "low.py", FACES["low"])
    reject = f"reject_level:0:{REJECTED_STEP}:2:1:1"
    on_face = _capped(script, tmp_path / "on_face", 2, REJECTED_STEP + 1, [[16], []], inject=reject)
    clear = _capped(script, tmp_path / "clear", 2, REJECTED_STEP + 1, [[12], []], inject=reject)
    ph.same_checkpoint(on_face, clear, "a rejected level-2 stage cut on and clear of the face", expect=["level_1", "level_2"])
    clean = _capped(script, tmp_path / "clean", 2, REJECTED_STEP + 1, [[16], []])
    capped = _capped_clean_launch(script, tmp_path, _accepted_dt(on_face, clean))
    ph.same_checkpoint(on_face, capped, "a rejected level-2 stage on host tiles against the capped clean launch", expect=["level_1", "level_2"])


def _launch(script: str, out: Path, workers: int, cuts: list[list[int]], steps: int, checkpoint: Path | None = None,
            inject: str | None = None, expect_ok: bool = True) -> ph.Launch:
    """`simbi launch` of `script` on the session's backend and placement without the experimental
    opt-in; with `expect_ok` every worker exits zero"""
    args = ["--max-steps", str(steps), "--resolution", "32,32,1", "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    env = ph.environment(inject)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(script).parent), os.getcwd(), os.environ.get("PYTHONPATH", "")])
    return ph.launch(script, out, workers, cuts=cuts, args=args, env=env, timeout=600, expect_ok=expect_ok)


@pytest.mark.parametrize(("face", "cuts"), [("low", [[16], []]), ("high", [[16], []]), ("corner", [[16], [16]])])
def test_a_level_2_face_on_a_level_1_cut_matches_the_single_grid_on_launch(tmp_path: Path, face: str, cuts: list[list[int]]) -> None:
    """the launched workers carry the face's units on level 1's partition and gather its level-1
    stencil there, the worker across the cut holding level 1 alone"""
    script = _script(tmp_path / f"{face}.py", FACES[face])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    workers = int(np.prod([len(c) + 1 for c in cuts]))
    launched = _launch(script, tmp_path / "launched", workers, cuts, STEPS).final
    ph.same_checkpoint(launched, one, f"a level-2 {face} face on a level-1 cut on {workers} workers", expect=["level_1", "level_2"])


def test_a_level_2_face_on_a_level_1_cut_resumes_on_more_workers(tmp_path: Path) -> None:
    script = _script(tmp_path / "low.py", FACES["low"])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    _assert_premises(one)
    half = _launch(script, tmp_path / "half", 2, [[16], []], STEPS // 2).final
    resumed = _launch(script, tmp_path / "resumed", 4, [[16], [16]], STEPS, checkpoint=half).final
    ph.same_checkpoint(resumed, one, "a level-2 face on a level-1 cut resumed on four workers", expect=["level_1", "level_2"])


def test_a_rejected_level_2_stage_after_booked_units_replays_to_the_capped_clean_run_on_launch(tmp_path: Path) -> None:
    """worker 1, holding level 2, rejects a stage of the second level-2 substep of root step 2 after
    the first substep's units crossed the level-1 cut: the launch cut on the face equals the launch
    cut clear of it, every worker counts the rejection once, and both equal the clean launch whose
    step 2 takes the timestep the replay accepted"""
    script = _script(tmp_path / "low.py", FACES["low"])
    reject = f"reject_level:1:{REJECTED_STEP}:2:1:1"
    finals = {}
    for cut in (16, 12):
        run = _launch(script, tmp_path / f"rejecting_{cut}", 2, [[cut], []], REJECTED_STEP + 1, inject=reject)
        finals[cut] = run.final
        counts = {r for _, r in run.reports().values()}
        assert counts == {1}, f"cut at {cut}: the rejection was not counted once on every worker: {counts}"
    ph.same_checkpoint(finals[16], finals[12], "a rejected level-2 stage launched cut on and clear of the face", expect=["level_1", "level_2"])
    clean = _launch(script, tmp_path / "clean", 2, [[16], []], REJECTED_STEP + 1).final
    capped = _capped_clean_launch(script, tmp_path, _accepted_dt(finals[16], clean))
    ph.same_checkpoint(finals[16], capped, "a rejected level-2 stage launched against the capped clean launch", expect=["level_1", "level_2"])


def test_a_duplicated_nested_unit_ends_the_launch(tmp_path: Path) -> None:
    """worker 1 lists its sends of the level-2 face's units twice at the first stage of root step 2:
    the receiving worker's grant counts the plan's sends, and the session ends before any final
    output"""
    script = _script(tmp_path / "low.py", FACES["low"])
    out = tmp_path / "dup"
    done = _launch(script, out, 2, [[16], []], STEPS, inject="nested_contribution:1:2:2:0:0:dup", expect_ok=False)
    assert done.returncode != 0, "the launch with a duplicated nested unit succeeded"
    assert not list(out.glob("*final*.h5")), "the launch published a final checkpoint"
    assert "this phase sends it" in done.stderr + done.stdout, (done.stderr + done.stdout)[-3000:]


# level 1's low x face on the root cut at x = 0 and level 2's low y face on level 1's cut at y = 0,
# each past its parent's halo: the launch gathers level 1's prim stencil from the root and level 2's
# from level 1, each at its level's primed point and at its parent level's last stage. "coupled"
# places level 2 two level-1 cells inside level 1's face on the root cut, so level 2's stencil
# prolongs from level-1 cells whose coarse-fine ghosts level 1's own stencil fills
STENCIL_LEVELS = {
    "two-levels": ([0.0625, 0.1875, 0.0, 0.125], [0.0, 0.25, -0.25, 0.25]),
    "coupled": ([0.03125, 0.15625, 0.0, 0.125], [0.0, 0.25, -0.25, 0.25]),
}
STENCIL_CUTS = [[16], [16]]


def _stencil_roles_cross(run: ph.Launch) -> None:
    """both prim stencil roles schedule transfers to another worker"""
    remote = run.remote_sends()
    for role in ("PrimStencil", "NestedPrimStencil{level:2}"):
        assert any(r.get(role, 0) > 0 for r in remote.values()), f"the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"


@pytest.mark.parametrize("layout", list(STENCIL_LEVELS))
def test_parent_prim_stencils_on_two_levels_launch_as_the_single_grid(tmp_path: Path, layout: str) -> None:
    """four workers cut on both axes gather a prim stencil on level 1 and on level 2: the launch and
    four host tiles equal the single grid bit for bit at the final step, every level included"""
    script = _script(tmp_path / f"{layout}.py", *STENCIL_LEVELS[layout])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    with h5py.File(one, "r") as f:
        assert "level_2" in f, "the single grid wrote no level 2; the gate is vacuous"
    tiles = _capped(script, tmp_path / "tiles", 4, STEPS, STENCIL_CUTS)
    ph.same_checkpoint(tiles, one, f"{layout}: four host tiles", expect=["level_1", "level_2"])
    run = _launch(script, tmp_path / "launched", 4, STENCIL_CUTS, STEPS)
    _stencil_roles_cross(run)
    ph.same_checkpoint(run.final, one, f"{layout}: four launched workers", expect=["level_1", "level_2"])


def test_parent_prim_stencils_on_two_levels_resume_onto_more_workers(tmp_path: Path) -> None:
    """two workers cut on level 1's face (one stencil level) write step 3; four workers cut on both
    axes (stencils on both levels) resume it and equal the uninterrupted single grid"""
    script = _script(tmp_path / "coupled.py", *STENCIL_LEVELS["coupled"])
    one = _capped(script, tmp_path / "one", 1, STEPS)
    half = _launch(script, tmp_path / "half", 2, [[16], []], STEPS // 2)
    resumed = _launch(script, tmp_path / "resumed", 4, STENCIL_CUTS, STEPS, checkpoint=half.final)
    _stencil_roles_cross(resumed)
    ph.same_checkpoint(resumed.final, one, "stencils on two levels resumed on four workers", expect=["level_1", "level_2"])


# worker 3 holds the tile of level 2 (x > 0, y > 0); level 2 takes four substeps per root step under
# rk2, so substep 3 stage 1 is the last stage of the last level-2 substep, after both stencils gathered
STENCIL_REJECTION = f"reject_level:3:{REJECTED_STEP}:2:3:1"


def test_a_rejection_with_stencils_on_two_levels_replays_as_the_capped_clean_launch(tmp_path: Path) -> None:
    """a rejection in the last level-2 substep of root step 2 rolls the four workers back past both
    levels' gathered stencils: every worker counts it once and the run equals the clean launch capped
    at the accepted timestep"""
    script = _script(tmp_path / "coupled.py", *STENCIL_LEVELS["coupled"])
    rejected = _launch(script, tmp_path / "rejected", 4, STENCIL_CUTS, REJECTED_STEP + 1, inject=STENCIL_REJECTION)
    assert {r for _, r in rejected.reports().values()} == {1}, f"the rejection was not counted once on every worker: {rejected.reports()}"
    clean = _launch(script, tmp_path / "clean", 4, STENCIL_CUTS, REJECTED_STEP + 1)
    accepted = _accepted_dt(rejected.final, clean.final)
    capped = _launch(script, tmp_path / "capped", 4, STENCIL_CUTS, REJECTED_STEP + 1,
                     inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert {r for _, r in capped.reports().values()} == {0}, "the capped control rejected a step"
    ph.same_checkpoint(rejected.final, capped.final, "a rejection with stencils on two levels against the capped clean launch",
                       expect=["level_1", "level_2"])


@pytest.mark.parametrize(
    ("regions", "order"),
    [
        # ppm: a level-2 face on the level-1 cut at x = 0 reads level 1's exchanged halo
        ([LEVEL_1, FACES["low"]], 3),
        # plm: a level-3 face on the level-2 cut at x = 0 reads a level-2 prim stencil
        ([LEVEL_1, [-0.125, 0.125, -0.125, 0.125], [0.0, 0.0625, -0.0625, 0.0625]], None),
        # ppm: levels 1 and 2 span the cut, no face on it; level 2's coarse-fine ghosts beside the
        # cut prolong from level 1's cells across it
        ([LEVEL_1, [-0.125, 0.125, -0.125, 0.125]], 3),
    ],
    ids=["ppm-level-2", "plm-level-3", "ppm-spanning"],
)
def test_the_primed_state_of_a_nested_face_on_a_cut_matches_the_single_grid(tmp_path: Path, regions: list[list[float]], order: int | None) -> None:
    """every level primes from its parent's crossed cut halos, coarsest first: the step-zero
    checkpoint of host tiles and of a launch, the deepest patch's coarse-fine ghosts beside the cut
    included, equals the single grid's bit for bit, and so does the state after several steps"""
    script = _nested_script(tmp_path / "nested.py", regions, order)
    deepest = f"level_{len(regions)}"
    one = _capped(script, tmp_path / "one", 1, STEPS)
    tiles = _capped(script, tmp_path / "tiles", 2, STEPS, [[16], []])
    launched = _launch(script, tmp_path / "launched", 2, [[16], []], STEPS).final

    def initial(final: Path) -> Path:
        (first,) = sorted(final.parent.glob("*.chkpt.000_000.h5"))
        return first

    with h5py.File(initial(one), "r") as f:
        assert deepest in f, f"the single grid wrote no {deepest}; the gate is vacuous"
        assert int(f["level_0"].attrs["iteration"]) == 0, "the first checkpoint is not the primed state"
    for label, run in (("host tiles", tiles), ("a launch", launched)):
        ph.same_checkpoint(initial(run), initial(one), f"{label} at step zero", expect=["level_1", deepest])
        ph.same_checkpoint(run, one, f"{label} after {STEPS} steps", expect=["level_1", deepest])
