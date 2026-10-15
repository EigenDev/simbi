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
# - a census sampling every refined level's own steps records the single
#   hierarchy's rows, level by level, on two host tiles and on two and three
#   launched workers, the three cutting the root so two workers hold no tile
#   of the refined level, and a run resumed onto another partition continues
#   every level's schedule;
# - a rigid wall cut by the tiles leaves its interior out of every tile's and
#   every worker's census: the counted cells equal an independent count of the
#   leaf cells whose centroid lies outside the wall, unrefined and refined;
# - the excised core of a kerr-schild box cut by two launched workers stays out
#   of the census: the counted cells equal an independent count of the cells
#   whose centroid lies outside the excision sphere, by the excision kernels'
#   own predicate.
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
from simbi.types.bodies import BodyCapability, ImmersedBodyConfig, RigidProperties
from simbi.types.shape import Shape
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


def _peak(cadence=expr.Cadence.ROOT_STEP):
    g = expr.ExprGraph()
    x = expr.variable("x", g)
    return expr.Census(
        name="peak",
        axes=[expr.BinAxis("x", x, [-0.5, -0.2, 0.0, 0.2, 0.5])],
        values={"pressure": expr.pressure(g)},
        op=expr.ReductionOp.MAX,
        sample_interval=INTERVAL,
        cadence=cadence,
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
        return [_shells(expr.Cadence.PER_LEVEL_STEP), _peak(expr.Cadence.PER_LEVEL_STEP)]


# a wall wholly inside the refined patch [-0.25, 0.25]^2, crossed by the cuts x = 9, 12, 17 and y = 13
WALL_CENTER, WALL_RADIUS = (0.07, -0.04), 0.17


def _gas_cells():
    g = expr.ExprGraph()
    return expr.Census(name="gas", axes=[], values={"cells": expr.constant(1.0, g)}).serialize()


class WalledBlast(RefinedBlast):
    @property
    def census_expressions(self):
        return [_gas_cells()]

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.RIGID, mass=0.0, radius=WALL_RADIUS, position=WALL_CENTER,
                                   velocity=(0.0, 0.0), rigid=RigidProperties(inertia=1.0, apply_no_slip=True))]


class FlatWalledBlast(WalledBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


# a slab spinning about its center inside the refined patch, crossed by the same cuts; its mask
# turns as Rodrigues(z, omega t) between samples
SPIN_CENTER, SPIN_HALF, SPIN_OMEGA = (0.005, -0.01), (0.21, 0.11), 120.0


class SpinningWalledBlast(RefinedBlast):
    @property
    def census_expressions(self):
        return [_gas_cells()]

    @property
    def immersed_bodies(self):
        slab = Shape.box((0.0, 0.0, 0.0), (*SPIN_HALF, 1.0))
        return [ImmersedBodyConfig(capability=BodyCapability.RIGID, mass=0.0, radius=0.24, position=SPIN_CENTER,
                                   velocity=(0.0, 0.0), rigid=RigidProperties(inertia=1.0, apply_no_slip=True,
                                                                              shape=slab, omega=SPIN_OMEGA))]


class FlatSpinningWalledBlast(SpinningWalledBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


ARMS = {False: "FlatCensusedBlast", True: "CensusedBlast"}
PER_LEVEL = {True: "PerLevelCensusedBlast"}
WALLED = {False: "FlatWalledBlast", True: "WalledBlast"}
SPINNING = {False: "FlatSpinningWalledBlast", True: "SpinningWalledBlast"}


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


def _launch(tmp_path: Path, refined: bool, out: Path, layout: Path, steps: int = STEPS, checkpoint: Path | None = None,
            arms: dict = ARMS) -> Path:
    problem = arms[refined]
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


def _assert_per_level_exercised(path: Path) -> None:
    shells, peak = read_census(str(path), "shells"), read_census(str(path), "peak")
    assert shells.levels == (0, 1), f"the per-level census holds levels {shells.levels}"
    root, fine = shells.for_level(0), shells.for_level(1)
    assert len(root.time) == STEPS and len(fine.time) == 2 * STEPS, f"rows per level: {len(root.time)}, {len(fine.time)}"
    assert 1 < len(peak.for_level(1).time) < 2 * STEPS, "the per-level interval skips no fine substep"
    assert not np.array_equal(fine.values[0], fine.values[-1]), "the fine level's census never changed; the comparison is vacuous"


def test_a_per_level_census_on_refined_tiles_and_workers_records_the_single_hierarchys_rows(tmp_path: Path, host_tiles) -> None:
    single = _run(tmp_path / "single", 1, True, cls=PerLevelCensusedBlast)
    _assert_per_level_exercised(single)
    tiles = _run(tmp_path / "tiles", 2, True, cls=PerLevelCensusedBlast, decompose=[[12], []])
    _assert_same_census(tiles, single, "two refined host tiles")
    two = _launch(tmp_path, True, tmp_path / "two", _layout(tmp_path / "two.toml", workers=2, cuts=[[12], []]), arms=PER_LEVEL)
    _assert_same_census(two, single, "two workers")
    # the patch spans root cells 6..18: the cuts at 4 and 20 leave the outer workers no refined tile
    three = _launch(tmp_path, True, tmp_path / "three", _layout(tmp_path / "three.toml", workers=3, cuts=[[4, 20], []]), arms=PER_LEVEL)
    _assert_same_census(three, single, "three workers, two holding no refined tile")


def test_a_per_level_census_resumed_onto_another_partition_continues_every_level(tmp_path: Path, host_tiles) -> None:
    whole = _run(tmp_path / "whole", 1, True, cls=PerLevelCensusedBlast)
    first = _run(tmp_path / "first", 2, True, steps=STEPS // 2 + 1, cls=PerLevelCensusedBlast, decompose=[[12], []])
    resumed = _launch(tmp_path, True, tmp_path / "resumed", _layout(tmp_path / "three.toml", workers=3, cuts=[[4, 20], []]),
                      checkpoint=first, arms=PER_LEVEL)
    for name in ("shells", "peak"):
        want, head, tail = (read_census(str(p), name) for p in (whole, first, resumed))
        for level in (0, 1):
            w, h, t = want.for_level(level), head.for_level(level), tail.for_level(level)
            assert len(h.time) > 0 and len(t.time) > 0, f"{name} level {level}: a segment recorded nothing"
            np.testing.assert_array_equal(np.concatenate([h.time, t.time]), w.time, err_msg=f"{name} level {level}: sample times")
            np.testing.assert_array_equal(np.concatenate([h.values, t.values]), w.values, err_msg=f"{name} level {level}: samples")


def _leaf_cells_outside(refined: bool, inside) -> int:
    """the leaf cells whose centroid lies outside the body `inside(xx, yy)`, counted from the grid
    alone"""
    def outside(n: int, lo: float, hi: float, keep) -> int:
        h = (hi - lo) / n
        c = lo + (np.arange(n) + 0.5) * h
        xx, yy = np.meshgrid(c, c, indexing="ij")
        return int(np.count_nonzero(keep(xx, yy) & ~inside(xx, yy)))

    everywhere = lambda xx, yy: np.ones_like(xx, dtype=bool)  # noqa: E731
    if not refined:
        return outside(RESOLUTION[0], -0.5, 0.5, everywhere)
    uncovered = lambda xx, yy: ~((np.abs(xx) < 0.25) & (np.abs(yy) < 0.25))  # noqa: E731
    return outside(RESOLUTION[0], -0.5, 0.5, uncovered) + outside(RESOLUTION[0], -0.25, 0.25, everywhere)


def _gas_cell_count(refined: bool) -> int:
    """the leaf cells whose centroid lies outside the round wall"""
    return _leaf_cells_outside(refined, lambda xx, yy: np.hypot(xx - WALL_CENTER[0], yy - WALL_CENTER[1]) < WALL_RADIUS)


def _outside_slab(refined: bool, t: float) -> int:
    """the leaf cells whose centroid lies outside the slab turned by omega t: the body-frame point
    of a centroid is R(omega t)^T (x - center)"""
    c, s = np.cos(SPIN_OMEGA * t), np.sin(SPIN_OMEGA * t)

    def inside(xx, yy):
        dx, dy = xx - SPIN_CENTER[0], yy - SPIN_CENTER[1]
        return (np.abs(c * dx + s * dy) < SPIN_HALF[0]) & (np.abs(-s * dx + c * dy) < SPIN_HALF[1])

    return _leaf_cells_outside(refined, inside)


@pytest.mark.parametrize("refined", [False, True])
def test_a_wall_cut_by_tiles_leaves_its_interior_out_of_the_census(tmp_path: Path, host_tiles, refined: bool) -> None:
    want = _gas_cell_count(refined)
    leaves = RESOLUTION[0] ** 2 + (3 * RESOLUTION[0] ** 2 // 4 if refined else 0)
    assert 0 < want < leaves - 40, f"the wall masks {leaves - want} cells; the gate is vacuous"
    cls = globals()[WALLED[refined]]
    single = _run(tmp_path / "single", 1, refined, cls=cls)
    gpus, cuts = (2, [[12], []]) if refined else (6, [[9, 17], [13]])
    tiles = _run(tmp_path / "tiles", gpus, refined, cls=cls, decompose=cuts)
    launched = _launch(tmp_path, refined, tmp_path / "launched", _layout(tmp_path / "two.toml", workers=2, cuts=[[12], []]), arms=WALLED)
    for label, path in (("single", single), (f"{gpus} host tiles", tiles), ("two workers", launched)):
        gas = read_census(str(path), "gas")
        assert len(gas.time) == STEPS, f"{label}: {len(gas.time)} rows"
        counts = gas.values.reshape(len(gas.time), -1)[:, 0]
        np.testing.assert_array_equal(counts, np.full(STEPS, float(want)), err_msg=f"{label}: counted gas cells")
        np.testing.assert_array_equal(gas.dropped, 0, err_msg=f"{label}: a masked cell was reported out of the bins")


@pytest.mark.parametrize("refined", [False, True])
def test_a_spinning_slab_cut_by_tiles_masks_its_turned_geometry_at_each_sample(tmp_path: Path, host_tiles, refined: bool) -> None:
    cls = globals()[SPINNING[refined]]
    single = _run(tmp_path / "single", 1, refined, cls=cls)
    gpus, cuts = (2, [[12], []]) if refined else (6, [[9, 17], [13]])
    tiles = _run(tmp_path / "tiles", gpus, refined, cls=cls, decompose=cuts)
    launched = _launch(tmp_path, refined, tmp_path / "launched", _layout(tmp_path / "two.toml", workers=2, cuts=[[12], []]), arms=SPINNING)
    want = np.array([_outside_slab(refined, t) for t in read_census(str(single), "gas").time], dtype=float)
    assert len(np.unique(want)) > 2, f"the slab's count changes {len(np.unique(want)) - 1} times; the turn is vacuous"
    assert want.min() < _leaf_cells_outside(refined, lambda xx, yy: np.zeros_like(xx, dtype=bool)) - 40, "the slab masks too few cells"
    for label, path in (("single", single), (f"{gpus} host tiles", tiles), ("two workers", launched)):
        gas = read_census(str(path), "gas")
        assert len(gas.time) == STEPS, f"{label}: {len(gas.time)} rows"
        counts = gas.values.reshape(len(gas.time), -1)[:, 0]
        np.testing.assert_array_equal(counts, np.array([_outside_slab(refined, t) for t in gas.time], dtype=float),
                                      err_msg=f"{label}: counted gas cells against the slab turned to each sample time")
        np.testing.assert_array_equal(gas.dropped, 0, err_msg=f"{label}: a masked cell was reported out of the bins")


EXCISED = '''
import simbi.expression as expr
from simbi_configs.examples.grhd.gr_bondi_cartesian import GrBondiCartesian


class CensusedBondi(GrBondiCartesian):
    """the excised kerr-schild bondi box counting its gas cells"""

    @property
    def census_expressions(self):
        g = expr.ExprGraph()
        return [expr.Census(name="gas", axes=[], values={"cells": expr.constant(1.0, g)}).serialize()]
'''
GR_RES = 24
GR_STEPS = 3
GR_FLAGS = ["--domain-radius", "0.4", "--excision-radius", "1.6"]


def _outside_excision(half_width: float, r_exc: float) -> int:
    """the cells whose centroid lies outside the excision sphere, by the kernels' predicate at zero
    spin: r_ks^2 = d + sqrt(d^2) with d = |x|^2 / 2, excised below r_exc^2"""
    c = -half_width + (np.arange(GR_RES) + 0.5) * (2.0 * half_width / GR_RES)
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    rr2 = x * x + y * y + z * z
    d = 0.5 * rr2
    return int(np.count_nonzero(~(d + np.sqrt(d * d) < r_exc * r_exc)))


def test_an_excised_core_cut_by_workers_stays_out_of_the_census(tmp_path: Path) -> None:
    (tmp_path / "censused_bondi.py").write_text(EXCISED)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    single = subprocess.run(
        [sys.executable, "-c",
         "import sys\nfrom simbi.simulation import runner\nfrom censused_bondi import CensusedBondi\n"
         f"p = CensusedBondi(resolution=({GR_RES},) * 3, domain_radius=0.4, excision_radius=1.6, data_directory=sys.argv[1], checkpoint_interval=1e9)\n"
         f"runner.run(p, compute_mode='cpu', max_steps={GR_STEPS})\n"
         "p.setup()\nprint('HALF', p.bounds[0][1])\n", str(tmp_path / "single")],
        capture_output=True, text=True, timeout=900, env=env)
    assert single.returncode == 0, single.stderr[-3000:]
    half = float(next(line.split()[1] for line in single.stdout.splitlines() if line.startswith("HALF")))
    want = _outside_excision(half, 1.6)
    assert 0 < want < GR_RES**3 - 20, f"the core excises {GR_RES**3 - want} cells; the gate is vacuous"
    script = tmp_path / "launched_bondi.py"
    script.write_text("from censused_bondi import CensusedBondi as Base\n\n\nclass LaunchedBondi(Base):\n    pass\n")
    layout = _layout(tmp_path / "two.toml", workers=2, cuts=[[GR_RES // 2], [], []])
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(GR_STEPS),
           "--resolution", ",".join([str(GR_RES)] * 3), *GR_FLAGS, "--data-directory", str(tmp_path / "launched"), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, done.stderr[-3000:]
    for label, out in (("one device", tmp_path / "single"), ("two workers", tmp_path / "launched")):
        gas = read_census(str(_final(out)), "gas")
        assert len(gas.time) == GR_STEPS, f"{label}: {len(gas.time)} rows"
        counts = gas.values.reshape(len(gas.time), -1)[:, 0]
        np.testing.assert_array_equal(counts, np.full(GR_STEPS, float(want)), err_msg=f"{label}: counted gas cells")
        np.testing.assert_array_equal(gas.dropped, 0, err_msg=f"{label}: an excised cell was reported out of the bins")

