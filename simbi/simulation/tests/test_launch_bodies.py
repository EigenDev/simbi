# =============================================================================
# test_launch_bodies.py
#
# immersed bodies on launched workers. launched workers sum every tile's body
# feedback in flat tile order, the order host tiles sum it in, so two launched
# workers publish what two host tiles publish, bit for bit: every grid field,
# the bodies group, the per-step body history and the body diagnostics file.
# host tiles reach the single grid within the reassociation bound of their
# cross-tile sums (test_tiled_body_ledger, test_isothermal_tile_settings). the
# arms:
# - a shaped rigid wall in a driven wind tunnel, adiabatic and isothermal;
# - a bonded fragment cluster in the same wind;
# - an accreting bondi sink, fresh, and resumed onto three uneven workers
#   against three host tiles cut at the same cells;
# - the adiabatic alpha viscosity about a central body;
# - a shaped wall and an accreting sink on the magnetized grid;
# - with mesh refinement: a gravitating primary, the same with the alpha
#   viscosity, and an accreting bondi sink under a fine central box, launched
#   refined workers against refined host tiles; the gravitating arms' host
#   tiles against the single hierarchy (the sink's link is
#   test_tiled_body_ledger[refined]); three workers, one holding no fine tile;
#   and a late fine-substep rejection against its capped clean control, which
#   must publish no contribution of the rejected attempt.
# - an accreting bondi sink under two nested fine levels on three workers, two
#   of them holding level 1 without level 2, against the single hierarchy.
# a magnetized wall takes its shape (a box and a sphere leave different flows),
# and a bonded fragment assembly with mesh refinement is refused on launch by
# name.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_bodies.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels
from simbi.simulation.tests.test_tiled_body_ledger import U, _diagnostics, _pull

pytestmark = pytest.mark.simulation

STEPS = 6
VARIANTS = '''
import importlib
from typing import Annotated, Optional

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi.types import Regime, Solver
from simbi.types.bodies import BondedAssembly
from simbi_configs.examples.ibm.roche_passage import RochePassage
from simbi_configs.examples.ibm.rubble_wind import RubbleWind
from simbi_configs.examples.ibm.wind_tunnel import WindTunnel
from simbi.types import Shape
from simbi.types.bodies import BodyCapability, ImmersedBodyConfig, RigidProperties
from simbi_configs.examples.ibm.magnetized_sink import MagnetizedBondiSink
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang


def _isothermal(base):
    class Isothermal(base):
        regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
        # isothermal hydro evaluates the hlle flux
        solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="hlle")]
        sound_speed: Annotated[float, ProblemParam(3.0, description="sound speed")]

        @computed_field
        @property
        def bx1_inner_expressions(self) -> dict:
            freestream = importlib.import_module(base.__module__).RHO_INF
            return expr.boundary([freestream, self.freestream_velocity(), 0.0], dim=2)

        def initial_primitive_state(self):
            adiabatic = super().initial_primitive_state()

            def rows():
                for row in adiabatic():
                    yield tuple(row[:-1])

            return rows

    return Isothermal


IsoWindTunnel = _isothermal(WindTunnel)


class AlphaPrimary(RochePassage):
    viscosity_alpha: Annotated[float, ProblemParam(0.1, description="alpha")]

    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        return None


class Bondi(SphericalBondiTest):
    base_resolution: Annotated[int, ProblemParam(24, description="base")]


class RefinedPrimary(AlphaPrimary):
    viscosity_alpha: Annotated[float, ProblemParam(0.0, description="inviscid")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-2.0, 2.0, -2.0, 2.0]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]


class RefinedAlphaPrimary(RefinedPrimary):
    viscosity_alpha: Annotated[float, ProblemParam(0.1, description="alpha")]


class RefinedIsothermalAlphaPrimary(RefinedAlphaPrimary):
    """the refined alpha primary on globally isothermal gas"""

    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
    # isothermal hydro evaluates the hlle flux
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="hlle")]
    sound_speed: Annotated[float, ProblemParam(0.3, description="sound speed")]

    def initial_primitive_state(self):
        adiabatic = super().initial_primitive_state()

        def rows():
            for row in adiabatic():
                yield tuple(row[:-1])

        return rows


class RefinedBondi(Bondi):
    """the bondi sink under one fine level over the central box"""

    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]] | None, ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5]], description="box")]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2], description="ratios")]


class NestedBondi(Bondi):
    """the bondi sink under two nested fine levels, the sink inside the finest"""

    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]

    def setup(self) -> None:
        super().setup()
        self.refinement_max_levels = 3
        self.refinement_regions = self.refinement_regions[:2]
        self.refinement_ratios = self.refinement_ratios[:2]


class RefinedRubble(RubbleWind):
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]


def _magnetized_wall(shape):
    return [ImmersedBodyConfig(capability=BodyCapability.RIGID, mass=0.0, radius=0.3, position=(0.5, 0.5), velocity=(0.0, 0.0),
                               rigid=RigidProperties(inertia=1.0, apply_no_slip=True, k_eta_n=50.0, k_eta_t=50.0, shape=shape))]


class MagnetizedBoxWall(NewtonianOrszagTang):
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        return _magnetized_wall(Shape.box((0.5, 0.5, 0.0), (0.1, 0.1, 1.0)))


class MagnetizedBallWall(NewtonianOrszagTang):
    @computed_field
    @property
    def immersed_bodies(self) -> list:
        return _magnetized_wall(Shape.sphere((0.5, 0.5, 0.0), 0.05))


class MagnetizedSink(MagnetizedBondiSink):
    resolution: Annotated[tuple[int, int, int], ProblemParam((24, 24, 24), description="grid")]
'''
# (problem, resolution passed to the run, layout cuts or shape)
ARMS = {
    "wall": ("WindTunnel", (64, 32)),
    "wall-isothermal": ("IsoWindTunnel", (64, 32)),
    "fragments": ("RubbleWind", (64, 32)),
    "alpha": ("AlphaPrimary", (64, 64)),
    "sink": ("Bondi", None),
    "magnetized-wall": ("MagnetizedBoxWall", (32, 32, 1)),
    "magnetized-sink": ("MagnetizedSink", None),
}


# the wall and the fragment disk sit at x = 0 on [-2, 6] with 64 cells: the cut at cell 16 runs
# through each body, so the cross-tile sum of its feedback carries both tiles' partials
THROUGH_THE_BODY = {"wall": [[16], []], "wall-isothermal": [[16], []], "fragments": [[16], []]}


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _iteration(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"])


def _tiles(problem: str, resolution: tuple | None, out: Path, steps: int, checkpoint: Path | None = None, cuts: list | None = None) -> Path:
    fields = [f"resolution={resolution!r}"] if resolution is not None else []
    gpus = 2
    if cuts is not None:
        fields.append(f"decompose={cuts!r}")
        gpus = 1
        for axis in cuts:
            gpus *= len(axis) + 1
    if checkpoint is not None:
        fields.append(f"checkpoint_file={str(checkpoint)!r}")
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"p = {problem}(gpus={gpus}, data_directory=sys.argv[1]{''.join(', ' + f for f in fields)})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} on {gpus} host tiles failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(variants: Path, problem: str, resolution: tuple | None, out: Path, layout: Path, steps: int, checkpoint: Path | None = None) -> Path:
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if resolution is not None:
        cmd += ["--resolution", ",".join(str(r) for r in resolution)]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


def _body_groups(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as f:
        for group in ("bodies", "body_diagnostics"):
            if group in f:
                # every dataset of the group, a bonded assembly's fragment state included
                f[group].visititems(lambda name, obj, group=group: out.__setitem__(f"{group}/{name}", np.asarray(obj[()])) if isinstance(obj, h5py.Dataset) else None)
    return out


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    bx, by = _body_groups(got), _body_groups(want)
    assert by, f"{label}: the reference publishes no body state; the gate is vacuous"
    assert bx.keys() == by.keys(), f"{label}: body datasets {sorted(set(bx) ^ set(by))} differ"
    for name, x in bx.items():
        np.testing.assert_array_equal(x, by[name], err_msg=f"{label}: {name}")


def _assert_same_diagnostics(got: Path, want: Path, label: str) -> None:
    a, b = got / "diagnostics.dat", want / "diagnostics.dat"
    assert b.exists(), f"{label}: the host tiles wrote no body diagnostics"
    assert a.exists(), f"{label}: the launch wrote no body diagnostics"
    assert a.read_text() == b.read_text(), f"{label}: the body diagnostics differ"


@pytest.mark.parametrize("arm", list(ARMS))
def test_launched_workers_publish_the_host_tiles_bodies(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution = ARMS[arm]
    cuts = THROUGH_THE_BODY.get(arm)
    want = _tiles(problem, resolution, tmp_path / "tiles", STEPS, cuts=cuts)
    if cuts is not None:
        # premise: the body straddles the cut at x = 0, so both tiles hold a partial of its feedback
        with h5py.File(want, "r") as f:
            x = f["bodies/position"][()][:, 0]
        assert (x.min() < 0.0 < x.max()) or abs(float(x[0])) < 0.4, f"{arm}: the body lies on one side of the cut (x = {x})"
    # a planar grid cut once along x (through the body where named); the 3d sinks likewise
    planar = resolution is not None and sum(1 for r in resolution if r > 1) == 2
    two = _layout(tmp_path / "two.toml", workers=2, cuts=cuts) if cuts is not None else _layout(tmp_path / "two.toml", workers=2, shape=(2, 1) if planar else (2, 1, 1))
    got = _launch(variants, problem, resolution, tmp_path / "launched", two, STEPS)
    _assert_same(got, want, f"{arm}: two launched workers")
    if arm == "sink":
        _assert_same_diagnostics(tmp_path / "launched", tmp_path / "tiles", arm)


def test_a_sink_resumed_onto_three_uneven_workers_publishes_the_host_tiles_ledger(tmp_path: Path, variants: Path) -> None:
    # the reference sums over the same three tiles, cut at the same cells
    cuts = [[7, 15], [], []]
    first = _tiles("Bondi", None, tmp_path / "tiles-first", STEPS // 2)
    # `max_steps` caps the absolute root iteration: both resumed runs take STEPS - STEPS // 2 steps
    # past the checkpoint
    cap = _iteration(first) + STEPS - STEPS // 2
    want = _tiles("Bondi", None, tmp_path / "tiles", cap, checkpoint=first, cuts=cuts)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=cuts)
    got = _launch(variants, "Bondi", None, tmp_path / "launched", three, cap, checkpoint=first)
    _assert_same(got, want, "sink resumed onto three uneven workers")


def test_a_magnetized_wall_takes_its_shape(tmp_path: Path, variants: Path) -> None:
    """the shaped wall acts on the magnetized grid: a box and a sphere of the same mask scale
    leave different flows"""
    box = _all_levels(_tiles("MagnetizedBoxWall", (32, 32, 1), tmp_path / "box", STEPS, cuts=[[], []]))
    ball = _all_levels(_tiles("MagnetizedBallWall", (32, 32, 1), tmp_path / "ball", STEPS, cuts=[[], []]))
    assert not np.array_equal(box["level_0/conserved/den"], ball["level_0/conserved/den"]), "the wall's shape left the flow unchanged"


# refined arms: (problem, resolution); the host tiles cut once along x, as the launch
REFINED_ARMS = {
    "refined-primary": ("RefinedPrimary", (64, 64)),
    "refined-alpha": ("RefinedAlphaPrimary", (64, 64)),
    "refined-isothermal-alpha": ("RefinedIsothermalAlphaPrimary", (64, 64)),
    "refined-sink": ("RefinedBondi", None),
}


@pytest.mark.parametrize("arm", list(REFINED_ARMS))
def test_launched_refined_workers_publish_the_refined_host_tiles_bodies(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution = REFINED_ARMS[arm]
    want = _tiles(problem, resolution, tmp_path / "tiles", STEPS)
    assert any(name.startswith("level_1/") for name in _all_levels(want)), f"{arm}: the host tiles published no fine level"
    planar = resolution is not None
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1) if planar else (2, 1, 1))
    got = _launch(variants, problem, resolution, tmp_path / "launched", two, STEPS)
    _assert_same(got, want, f"{arm}: two launched refined workers")
    if arm == "refined-sink":
        _assert_same_diagnostics(tmp_path / "launched", tmp_path / "tiles", arm)


def test_a_refined_bonded_assembly_on_launch_is_refused(tmp_path: Path, variants: Path) -> None:
    script = variants.parent / "launched_refinedrubble.py"
    script.write_text("from variants import RefinedRubble as Base\n\n\nclass LaunchedRefinedRubble(Base):\n    pass\n")
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", "64,32", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    assert "does not support refinement yet (no fragment step on the hierarchy)" in " ".join(done.stderr.split()), done.stderr[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


def _single(problem: str, resolution: tuple | None, out: Path, steps: int) -> Path:
    fields = [f"resolution={resolution!r}"] if resolution is not None else []
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"p = {problem}(gpus=1, data_directory=sys.argv[1]{''.join(', ' + f for f in fields)})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, f"{problem} on one device failed:\n{done.stderr[-3000:]}"
    return _final(out)


@pytest.mark.parametrize("problem", ["RefinedPrimary", "RefinedAlphaPrimary", "RefinedIsothermalAlphaPrimary"])
def test_refined_host_tiles_publish_the_single_hierarchy_bodies(tmp_path: Path, variants: Path, problem: str) -> None:
    # a gravitating body takes no feedback, so no cross-tile sum enters its state: the refined host
    # tiles the launch is gated against publish the single hierarchy's grid, bodies and history bit
    # for bit. the accreting sink's link to the single hierarchy is test_tiled_body_ledger[refined],
    # the same configuration within the reassociation bound of its cross-tile sums.
    want = _single(problem, (64, 64), tmp_path / "single", STEPS)
    assert any(name.startswith("level_1/") for name in _all_levels(want)), f"{problem}: the single run published no fine level"
    _assert_same(_tiles(problem, (64, 64), tmp_path / "tiles", STEPS), want, f"{problem}: two refined host tiles")


# three workers, the first holding an unrefined tile alone: (problem, resolution, cuts). the planar
# patch spans root cells 24..40 of 64, the 3d one 6..18 of 24; every cut keeps clear of its faces
WITHOUT_FINE = {
    "primary": ("RefinedPrimary", (64, 64), [[12, 32], []]),
    "sink": ("RefinedBondi", None, [[3, 12], [], []]),
}


@pytest.mark.parametrize("arm", list(WITHOUT_FINE))
def test_a_worker_holding_no_fine_tile_joins_the_refined_body_step(tmp_path: Path, variants: Path, arm: str) -> None:
    # the first worker, the checkpoint coordinator, holds the unrefined first tile alone: it agrees
    # the finest clock without holding the finest level, contributes no body delta, and publishes
    # the bodies and their history from its own tile, whose finest level (its root) books the same
    # global delta every tile applies
    problem, resolution, cuts = WITHOUT_FINE[arm]
    want = _tiles(problem, resolution, tmp_path / "tiles", STEPS, cuts=cuts)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=cuts)
    got = _launch(variants, problem, resolution, tmp_path / "launched", three, STEPS)
    _assert_same(got, want, f"{arm}: three workers, the first without a fine tile")
    if arm == "sink":
        _assert_same_diagnostics(tmp_path / "launched", tmp_path / "tiles", arm)
    # resumed on the same layout from the coordinator's own checkpoint: the history the coordinator
    # publishes after the restart is the host tiles' after theirs
    first = _launch(variants, problem, resolution, tmp_path / "launched-first", three, STEPS // 2)
    tiles_first = _tiles(problem, resolution, tmp_path / "tiles-first", STEPS // 2, cuts=cuts)
    # `max_steps` caps the absolute root iteration: each resumed run takes STEPS - STEPS // 2 steps
    # past its checkpoint
    resumed = _launch(variants, problem, resolution, tmp_path / "launched-resumed", three, _iteration(first) + STEPS - STEPS // 2, checkpoint=first)
    tiles_resumed = _tiles(
        problem, resolution, tmp_path / "tiles-resumed", _iteration(tiles_first) + STEPS - STEPS // 2, checkpoint=tiles_first, cuts=cuts
    )
    with h5py.File(resumed, "r") as f:
        assert "body_diagnostics/time" in f and f["body_diagnostics/time"].shape[0] > 0, (
            f"{arm}: the coordinator published no body history after the restart"
        )
    _assert_same(resumed, tiles_resumed, f"{arm}: resumed on three workers, the coordinator without a fine tile")


# root cells of the 24-cell nested bondi grid: level 1 spans 6..18, level 2 spans 9..15; the cuts
# at 8 and 16 leave the first and third workers holding level 1 without level 2
NESTED_CUTS = [[8, 16], [], []]


def _level_root_span(path: Path, level: str) -> tuple[float, float]:
    with h5py.File(path, "r") as f:
        dim = f[level]["mesh/geometry/dim_0"].attrs
        return float(dim["start"]), float(dim["end"])


def test_tiles_and_workers_holding_fewer_levels_publish_the_single_hierarchy_nested_sink(tmp_path: Path, variants: Path) -> None:
    # every tile applies the global body delta on its own finest level, gravity-only proxies below;
    # a worker without the run's finest level publishes the bodies and history from its deepest
    # level. the grid matches the single hierarchy bit for bit: the drain acts on the finest level,
    # which one worker holds whole. the body state and history take the reassociation bound of the
    # cross-tile body sums (test_tiled_body_ledger)
    want = _single("NestedBondi", None, tmp_path / "single", STEPS)
    lo, hi = _level_root_span(want, "level_2")
    root_lo, root_hi = _level_root_span(want, "level_0")
    width = (root_hi - root_lo) / 24
    fine_cells = (round((lo - root_lo) / width), round((hi - root_lo) / width))
    # worker 0, the checkpoint coordinator, holds root cells 0..8 and worker 2 holds 16..24: both
    # hold level 1 without level 2
    assert NESTED_CUTS[0][0] <= fine_cells[0] and fine_cells[1] <= NESTED_CUTS[0][1], (
        f"level 2 spans root cells {fine_cells}; the coordinator holds the finest level and the gate is vacuous"
    )
    with h5py.File(want, "r") as f:
        assert float(f["bodies/total_accreted_mass"][0]) > 0.0, "the sink accreted nothing; the gate is vacuous"
    three = _layout(tmp_path / "three.toml", workers=3, cuts=NESTED_CUTS)
    got = _launch(variants, "NestedBondi", None, tmp_path / "launched", three, STEPS)
    _assert_nested_sink(got, tmp_path / "launched", want, tmp_path / "single", "nested sink, launched")
    # the host tiles on the same cuts: the tiles holding level 1 without level 2 add no reaction
    tiles = _tiles("NestedBondi", None, tmp_path / "tiles", STEPS, cuts=NESTED_CUTS)
    _assert_nested_sink(tiles, tmp_path / "tiles", want, tmp_path / "single", "nested sink, host tiles")


def _assert_nested_sink(got: Path, out: Path, want: Path, single_out: Path, label: str) -> None:
    """the grid bit for bit, the body state and history within the reassociation bound of the
    cross-tile body sums"""
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    bx, by = _body_groups(got), _body_groups(want)
    assert bx.keys() == by.keys(), f"body datasets {sorted(set(bx) ^ set(by))} differ"
    for name in ("body_diagnostics/time", "body_diagnostics/dt"):
        np.testing.assert_array_equal(bx[name], by[name], err_msg=name)
    with h5py.File(want, "r") as f:
        cells = sum(int(np.prod([s - 2 * int(f[k]["mesh"].attrs["halo_width"]) for s in f[k]["conserved/den"].shape]))
                    for k in f if k.startswith("level_"))
    pull, arm = _pull(want)
    for name, y in by.items():
        y = y.astype(float)
        scale = pull if "force" in name else pull * arm if "torque" in name else np.abs(y).max()
        worst = np.abs(bx[name].astype(float) - y).max()
        assert worst <= cells * U * scale, f"{label}: {name} differs by {worst:.3e} against {cells * U:.1e} of {scale:.3e}"
    # the published body state carries the published history: one row per step, and the accreted
    # mass is the sum of the per-step receipts (a sum of STEPS terms of one sign)
    assert bx["body_diagnostics/mass_delta"].shape[0] == STEPS, f"the history books {bx['body_diagnostics/mass_delta'].shape[0]} rows for {STEPS} steps"
    receipts = float(np.sum(bx["body_diagnostics/mass_delta"]))
    published = float(bx["bodies/total_accreted_mass"][0])
    assert abs(published - receipts) <= STEPS * U * receipts, f"the published accreted mass {published:.17e} departs from its history {receipts:.17e}"
    rows, single_rows = _diagnostics(out), _diagnostics(single_out)
    assert rows.shape == single_rows.shape, f"diagnostics rows {rows.shape} against {single_rows.shape}"
    np.testing.assert_array_equal(rows[:, 0], single_rows[:, 0], err_msg="diagnostics times")
    # nine printed significant digits (%.8e) on top of the reassociation bound; force and torque
    # columns take the bound on their unsummed scales
    unsummed = np.zeros(rows.shape[1])
    unsummed[8:11], unsummed[11:14] = pull, pull * arm
    tolerance = 1e-8 * np.maximum(np.abs(rows), np.abs(single_rows)) + cells * U * unsummed
    excess = np.abs(rows - single_rows) - tolerance
    assert np.all(excess <= 0.0), f"diagnostics differ beyond the printed precision: column {int(np.argmax(excess.max(axis=0)))}"


REJECTIONS = __import__("re").compile(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)")
# the sink's late rejection: worker 2 holds root cells 12..24, the patch's upper half; the last
# stage (rk2) of the second fine substep of root step 2, after the first substep drained the sink
LATE_STEP = 2
LATE_REJECTION = f"reject_level:2:{LATE_STEP}:1:1:1"


def _launch_counting(variants: Path, out: Path, layout: Path, inject: str | None) -> tuple[Path, set[int]]:
    script = variants.parent / "launched_refinedbondi.py"
    script.write_text("from variants import RefinedBondi as Base\n\n\nclass LaunchedRefinedBondi(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(STEPS),
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out), {int(r) for r in REJECTIONS.findall(done.stderr)}


def test_a_late_fine_rejection_publishes_no_body_contribution_of_the_rejected_attempt(tmp_path: Path, variants: Path) -> None:
    three = _layout(tmp_path / "three.toml", workers=3, cuts=WITHOUT_FINE["sink"][2])
    rejecting, counts = _launch_counting(variants, tmp_path / "rejecting", three, LATE_REJECTION)
    assert counts == {1}, f"the fine rejection was not counted once on every worker: {counts}"
    clean, clean_counts = _launch_counting(variants, tmp_path / "clean", three, None)
    assert clean_counts == {0}
    with h5py.File(rejecting, "r") as fr, h5py.File(clean, "r") as fc:
        accepted = float(fr["body_diagnostics/dt"][LATE_STEP])
        assert accepted < float(fc["body_diagnostics/dt"][LATE_STEP]), "the rejection never reduced the timestep"
        # the drain booked accretion in the rejected attempt's first fine substep: discarding it is
        # the transaction under test
        assert np.abs(fc["body_diagnostics/mass_delta"][LATE_STEP]).max() > 0.0, "the sink accreted nothing at the rejected step"
    cap = f"cap:{LATE_STEP}:{int(np.float64(accepted).view(np.uint64))}"
    capped, capped_counts = _launch_counting(variants, tmp_path / "capped", three, cap)
    assert capped_counts == {0}, "the capped control rejected a step"
    # the accepted run publishes the capped clean control's grid, bodies, history and diagnostics:
    # no accretion, force, torque or heat of the rejected attempt survives in any of them
    _assert_same(rejecting, capped, "late fine rejection against the capped control")
    _assert_same_diagnostics(tmp_path / "rejecting", tmp_path / "capped", "late fine rejection")
