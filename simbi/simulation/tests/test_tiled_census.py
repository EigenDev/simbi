# =============================================================================
# test_tiled_census.py
#
# a census on host tiles and on launched workers records the single grid's (or
# hierarchy's) samples bit for bit: every tile reduces its own cells, the exact
# partial sums merge across tiles and workers into the whole grid's and round
# once, extrema merge exactly, and every tile records the merged sample. so,
# against one device:
# - an additive census binned in radius and an extremal one binned in x, on six
#   uneven host tiles of an unrefined blast and on two and three launched
#   workers, write the same census groups and sampling schedules bit for bit;
# - the same censuses over a refined blast on two host tiles and two workers cut
#   through the patch, every level's leaf cells counted once;
# - a run resumed onto another partition, on host tiles and on workers,
#   continues the sampling schedule: the records of the two segments
#   concatenate to the uninterrupted record, with no duplicate and no extra
#   first sample;
# - a census sampling every refined level's own steps on tiles is refused by
#   name before any output.
# the censuses vary over the run and the sampling interval skips steps, so the
# comparisons exercise the merge and the schedule.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_census.py
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest

import simbi.expression as expr
from simbi import ProblemParam
from simbi.reader.census import read_census
from simbi.simulation import runner
from simbi.simulation.tests.test_launch import _layout
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

pytestmark = pytest.mark.simulation

RESOLUTION = (24, 24, 1)
STEPS = 8
# the blast's unrefined timestep is near 1e-3 at this resolution: the interval samples every third
# step, so either half of a resumed run holds a sample
INTERVAL = 2.5e-3


def _shells(cadence=expr.Cadence.ROOT_STEP):
    g = expr.ExprGraph()
    x, y = expr.variable("x", g), expr.variable("y", g)
    mass = expr.density(g) * expr.cell_volume(g)
    r = expr.sqrt(x * x + y * y)
    return expr.Census(
        name="shells",
        axes=[expr.BinAxis("r", r, [0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.8])],
        values={"mass": mass, "energy": expr.pressure(g) * expr.cell_volume(g), "mass_x": mass * x},
        cadence=cadence,
    ).serialize()


def _peak():
    g = expr.ExprGraph()
    x = expr.variable("x", g)
    return expr.Census(
        name="peak",
        axes=[expr.BinAxis("x", x, [-0.5, -0.2, 0.0, 0.2, 0.5])],
        values={"pressure": expr.pressure(g)},
        op=expr.ReductionOp.MAX,
        sample_interval=INTERVAL,
    ).serialize()


class CensusedBlast(RefinedBlast):
    @property
    def census_expressions(self):
        return [_shells(), _peak()]


class FlatCensusedBlast(CensusedBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class PerLevelCensusedBlast(RefinedBlast):
    @property
    def census_expressions(self):
        return [_shells(expr.Cadence.PER_LEVEL_STEP)]


ARMS = {False: "FlatCensusedBlast", True: "CensusedBlast"}


@pytest.fixture
def host_tiles(monkeypatch):
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")


def _final(out: Path) -> Path:
    (final,) = sorted(out.glob("*final*.h5"))
    return final


def _run(out: Path, gpus: int, refined: bool, steps: int = STEPS, cls=None, **kwargs) -> Path:
    cls = cls or globals()[ARMS[refined]]
    runner.run(cls(resolution=RESOLUTION, gpus=gpus, data_directory=out, checkpoint_interval=1.0e9, **kwargs),
               compute_mode="cpu", max_steps=steps)
    return _final(out)


def _launch(tmp_path: Path, refined: bool, out: Path, layout: Path, steps: int = STEPS, checkpoint: Path | None = None) -> Path:
    problem = ARMS[refined]
    script = tmp_path / f"launched_{problem.lower()}.py"
    script.write_text(f"from simbi.simulation.tests.test_tiled_census import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"the launch failed:\nstdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


def _census_groups(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    def keep(prefix: str, name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            out[f"{prefix}/{name}"] = obj[()]
        for key, value in obj.attrs.items():
            out[f"{prefix}/{name}@{key}"] = np.asarray(value)

    with h5py.File(path, "r") as f:
        for group in ("census", "census_schedule"):
            if group in f:
                f[group].visititems(lambda name, obj, g=group: keep(g, name, obj))
    return out


def _assert_same_census(got: Path, want: Path, label: str) -> None:
    xs, ys = _census_groups(got), _census_groups(want)
    assert ys, f"{label}: the reference wrote no census"
    assert xs.keys() == ys.keys(), f"{label}: census entries differ: {sorted(xs.keys() ^ ys.keys())}"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _assert_exercised(path: Path) -> None:
    shells, peak = read_census(str(path), "shells"), read_census(str(path), "peak")
    assert len(shells.time) == STEPS, f"the per-step census holds {len(shells.time)} rows"
    assert 1 < len(peak.time) < STEPS, f"the interval census holds {len(peak.time)} rows; the interval skips nothing"
    assert not np.array_equal(shells.values[0], shells.values[-1]), "the census never changed; the comparison is vacuous"


def _assert_continues(whole: Path, first: Path, resumed: Path, label: str) -> None:
    for name in ("shells", "peak"):
        want, head, tail = (read_census(str(p), name) for p in (whole, first, resumed))
        assert len(head.time) > 0 and len(tail.time) > 0, f"{label} {name}: a segment recorded nothing"
        np.testing.assert_array_equal(np.concatenate([head.time, tail.time]), want.time, err_msg=f"{label} {name}: sample times")
        np.testing.assert_array_equal(np.concatenate([head.values, tail.values]), want.values, err_msg=f"{label} {name}: samples")


@pytest.mark.parametrize("refined", [False, True])
def test_a_census_on_host_tiles_records_the_single_grids_samples(tmp_path: Path, host_tiles, refined: bool) -> None:
    single = _run(tmp_path / "single", 1, refined)
    _assert_exercised(single)
    # x = 9 and 17 and y = 13 cut six uneven tiles, the blast's center beside the cut at y = 13;
    # x = 12 cuts the refined patch in two
    gpus, cuts = (2, [[12], []]) if refined else (6, [[9, 17], [13]])
    tiles = _run(tmp_path / "tiles", gpus, refined, decompose=cuts)
    _assert_same_census(tiles, single, f"{gpus} {'refined ' if refined else ''}host tiles")


@pytest.mark.parametrize("refined", [False, True])
def test_a_census_on_launched_workers_records_the_single_grids_samples(tmp_path: Path, host_tiles, refined: bool) -> None:
    single = _run(tmp_path / "single", 1, refined)
    _assert_exercised(single)
    layouts = [("two", 2, [[12], []])] + ([] if refined else [("three", 3, [[6, 15], []])])
    for tag, workers, cuts in layouts:
        got = _launch(tmp_path, refined, tmp_path / tag, _layout(tmp_path / f"{tag}.toml", workers=workers, cuts=cuts))
        _assert_same_census(got, single, f"{'refined ' if refined else ''}census on {workers} workers")


@pytest.mark.parametrize("refined", [False, True])
def test_a_tiled_census_resumed_onto_another_partition_continues_the_schedule(tmp_path: Path, host_tiles, refined: bool) -> None:
    whole = _run(tmp_path / "single", 1, refined)
    first = _run(tmp_path / "first", 2, refined, steps=STEPS // 2 + 1, decompose=[[12], []])
    resumed = _run(tmp_path / "resumed", 3, refined, decompose=[[8, 15], []], checkpoint_file=str(first))
    _assert_continues(whole, first, resumed, "host tiles")
    two = _layout(tmp_path / "two.toml", workers=2, cuts=[[12], []])
    launched_first = _launch(tmp_path, refined, tmp_path / "launched_first", two, steps=STEPS // 2 + 1)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[8, 15], []])
    launched_resumed = _launch(tmp_path, refined, tmp_path / "launched_resumed", three, checkpoint=launched_first)
    _assert_continues(whole, launched_first, launched_resumed, "launched workers")


def test_a_per_level_census_on_refined_tiles_is_refused_by_name(tmp_path: Path, host_tiles) -> None:
    with pytest.raises(Exception, match="samples every refined level's own steps"):
        _run(tmp_path / "tiles", 2, True, cls=PerLevelCensusedBlast)
    assert not list((tmp_path / "tiles").glob("*.h5")), "a refused run published a checkpoint"
