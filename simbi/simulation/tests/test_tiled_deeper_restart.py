# =============================================================================
# test_tiled_deeper_restart.py
#
# a refined run resumed from a checkpoint holding fewer levels than it builds:
# the levels the file carries are loaded, and each level it lacks is injected
# from its parent (every fine cell takes its covering coarse cell's conserved
# state) with its parent's clock, as the single hierarchy's deeper restart
# does. two rungs of the ladder, an unrefined blast resumed with one fine box
# and a refined one resumed with a nested box, each written on two host tiles:
# - resumed at the checkpoint's own step cap, every driver publishes the
#   stored levels exactly as the file holds them and each added level as its
#   parent's cells, so the load reinitializes nothing;
# - resumed onward, one device, three uneven host tiles, two launched workers
#   and three workers of which two hold no tile of an added level publish the
#   single hierarchy's levels and censuses bit for bit, and every census
#   continues the first segment's schedule (the two segments' rows concatenate
#   with no duplicate and no early interval sample), a per-level census on the
#   added level included.
# a checkpoint deeper than the run is refused by name. every arm runs on the
# session's backend and placement (parity_harness), its single-device runs and
# host tiles as fresh processes; the refusal on host tiles involves no launch
# and runs under the local placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_deeper_restart.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

RESOLUTION = (24, 24, 1)
FIRST_STEPS = 3
STEPS = 6
VARIANTS = '''
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast as Base


def _censuses():
    g = expr.ExprGraph()
    x, y = expr.variable("x", g), expr.variable("y", g)
    mass = expr.density(g) * expr.cell_volume(g)
    r = expr.sqrt(x * x + y * y)
    shells = expr.Census(name="shells", axes=[expr.BinAxis("r", r, [0.0, 0.1, 0.2, 0.45, 0.8])], values={"mass": mass}).serialize()
    peak = expr.Census(name="peak", axes=[expr.BinAxis("x", x, [-0.5, 0.0, 0.5])], values={"pressure": expr.pressure(g)},
                       op=expr.ReductionOp.MAX, sample_interval=2.5e-3, cadence=expr.Cadence.PER_LEVEL_STEP).serialize()
    return [shells, peak]


class RefinedBlast(Base):
    @property
    def census_expressions(self):
        return _censuses()


class FlatBlast(RefinedBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class NestedBlast(RefinedBlast):
    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.25, 0.25, -0.25, 0.25], [-0.125, 0.125, -0.125, 0.125]], description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description="ratios")]
'''
# the rungs: (the checkpoint's problem and level count, the resumed problem and its level count)
RUNGS = {"one-to-two": ("FlatBlast", 1, "RefinedBlast", 2), "two-to-three": ("RefinedBlast", 2, "NestedBlast", 3)}


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _levels(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return sum(1 for name in f if name.startswith("level_"))


def _in_process(problem: str, out: Path, steps: int, gpus: int = 1, cuts: list | None = None, checkpoint: Path | None = None,
                expect_failure: bool = False) -> subprocess.CompletedProcess:
    """`problem` on the session's backend in a fresh process, one device or `gpus` host tiles cut at
    `cuts`, capped at the absolute root iteration `steps`; a failure raises its category unless
    `expect_failure`"""
    fields = [f"resolution={RESOLUTION!r}", f"gpus={gpus}", "checkpoint_interval=1e9", "data_directory=sys.argv[1]"]
    if cuts is not None:
        fields.append(f"decompose={cuts!r}")
    if checkpoint is not None:
        fields.append(f"checkpoint_file={str(checkpoint)!r}")
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}({', '.join(fields)}), compute_mode=sys.argv[2], max_steps={steps})\n"
    )
    return ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{problem} on {gpus} device(s) to step {steps}", devices=gpus,
                     check=not expect_failure)


def _launch(variants: Path, problem: str, out: Path, workers: int, cuts: list, checkpoint: Path, steps: int = STEPS) -> Path:
    """`problem` launched on `workers` workers cut at `cuts` on the session's backend and placement,
    resumed from `checkpoint` with no experimental opt-in; its final checkpoint"""
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    args = ["--max-steps", str(steps), "--resolution", ",".join(str(r) for r in RESOLUTION), "--checkpoint-interval", "1e9",
            "--checkpoint", str(checkpoint)]
    ph.launch(script, out, workers=workers, cuts=cuts, args=args, env=ph.environment())
    return _final(out)


def _census_groups(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    def keep(prefix: str, name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            out[f"{prefix}/{name}"] = obj[()]

    with h5py.File(path, "r") as f:
        for group in ("census", "census_schedule"):
            if group in f:
                f[group].visititems(lambda name, obj, g=group: keep(g, name, obj))
    return out


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = {**_all_levels(got), **_census_groups(got)}, {**_all_levels(want), **_census_groups(want)}
    assert any(k.startswith("census/") for k in ys), f"{label}: the reference wrote no census"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(xs.keys() ^ ys.keys())} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _assert_load_reinitializes_nothing(got: Path, checkpoint: Path, stored: int, depth: int, label: str) -> None:
    """the stored levels as the file holds them; each added level its parent's cells, every fine
    cell its covering coarse cell's conserved state"""
    have, file = _all_levels(got), _all_levels(checkpoint)
    for name, x in file.items():
        if int(name.split("/")[0].split("@")[0].split("_")[1]) < stored and "@" not in name:
            np.testing.assert_array_equal(have[name], x, err_msg=f"{label}: stored {name} changed at load")
    for level in range(stored, depth):
        # each added box spans parent cells 6..18 on both axes of its parent's interior
        for field in ("den", "m1", "m2", "nrg"):
            parent = have[f"level_{level - 1}/conserved/{field}"][6:18, 6:18]
            child = have[f"level_{level}/conserved/{field}"]
            np.testing.assert_array_equal(child, np.repeat(np.repeat(parent, 2, axis=0), 2, axis=1),
                                          err_msg=f"{label}: added level {level} {field} is no injection of its parent")


def _assert_continues(got: Path, first: Path, label: str) -> None:
    """each segment's file holds its own rows: the first segment's and the resumed one's
    concatenate in time with no duplicate, and the interval census takes no extra first sample"""
    from simbi.reader.census import read_census

    for name in ("shells", "peak"):
        head, tail = read_census(str(first), name), read_census(str(got), name)
        assert len(head.time) > 0 and len(tail.time) > 0, f"{label} {name}: a segment recorded nothing"
        for level in tail.levels:
            before = head.for_level(level).time if level in head.levels else np.array([])
            times = np.concatenate([before, tail.for_level(level).time])
            assert np.all(np.diff(times) > 0.0), f"{label} {name} level {level}: a duplicate or reordered sample"
    peak_head, peak_tail = read_census(str(first), "peak").for_level(0), read_census(str(got), "peak").for_level(0)
    assert peak_tail.time[0] >= peak_head.time[-1] + 2.5e-3, f"{label}: the interval census sampled early after the restart"
    assert 1 in read_census(str(got), "peak").levels, f"{label}: the per-level census never sampled level 1"


@pytest.mark.parametrize("rung", list(RUNGS))
def test_a_deeper_restart_on_tiles_and_workers_publishes_the_single_hierarchys_levels(tmp_path: Path, variants: Path, rung: str) -> None:
    shallow, stored, deep, depth = RUNGS[rung]
    _in_process(shallow, tmp_path / "first", FIRST_STEPS, gpus=2, cuts=[[12], []])
    first = _final(tmp_path / "first")
    assert _levels(first) == stored, f"the checkpoint holds {_levels(first)} levels, not {stored}; the rung is vacuous"
    # resumed at the checkpoint's own cap: the load and the seeding alone
    _in_process(deep, tmp_path / "loaded", FIRST_STEPS, checkpoint=first)
    ph.disagreement(_assert_load_reinitializes_nothing, _final(tmp_path / "loaded"), first, stored, depth, f"{rung}: one device")
    _in_process(deep, tmp_path / "loaded-tiles", FIRST_STEPS, gpus=3, cuts=[[8, 13], []], checkpoint=first)
    ph.disagreement(_assert_load_reinitializes_nothing, _final(tmp_path / "loaded-tiles"), first, stored, depth, f"{rung}: three host tiles")
    loaded = _launch(variants, deep, tmp_path / "loaded-three", 3, [[4, 20], []], first, steps=FIRST_STEPS)
    ph.disagreement(_assert_load_reinitializes_nothing, loaded, first, stored, depth, f"{rung}: three workers")
    # resumed onward
    _in_process(deep, tmp_path / "single", STEPS, checkpoint=first)
    want = _final(tmp_path / "single")
    assert _levels(want) == depth, f"the single hierarchy resumed with {_levels(want)} levels"
    _assert_continues(want, first, f"{rung}: one device")
    _in_process(deep, tmp_path / "tiles", STEPS, gpus=3, cuts=[[8, 13], []], checkpoint=first)
    ph.disagreement(_assert_same, _final(tmp_path / "tiles"), want, f"{rung}: three uneven host tiles")
    two = _launch(variants, deep, tmp_path / "two", 2, [[12], []], first)
    ph.disagreement(_assert_same, two, want, f"{rung}: two workers")
    # the patches span root cells 6..18: the cuts at 4 and 20 leave the outer workers no added tile
    three = _launch(variants, deep, tmp_path / "three", 3, [[4, 20], []], first)
    ph.disagreement(_assert_same, three, want, f"{rung}: three workers, two holding no tile of an added level")


@ph.local_only
def test_a_checkpoint_deeper_than_the_run_is_refused_by_name_on_tiles(tmp_path: Path, variants: Path) -> None:
    _in_process("NestedBlast", tmp_path / "first", FIRST_STEPS)
    done = _in_process("RefinedBlast", tmp_path / "tiles", STEPS, gpus=2, cuts=[[12], []], checkpoint=_final(tmp_path / "first"),
                       expect_failure=True)
    assert done.returncode != 0, "a restart dropping a level ran on host tiles"
    assert "may add levels to a checkpoint, never drop them" in " ".join(done.stderr.split()), done.stderr[-2000:]
