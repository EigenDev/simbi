# =============================================================================
# test_launch_bodies.py
#
# immersed bodies on launched workers, on the session's backend and placement
# (parity_harness). each tile books its own cells' partial of every body
# feedback reduction exactly, and the partials merge in flat tile order and
# round once, so on the cpu the single device, host tiles and launched workers
# publish the same run bit for bit: every grid field, the bodies group, the
# per-step body history and the body diagnostics file. a gpu device rounds its
# feedback sum once per tile, so its tiles and workers agree with each other
# and its single-device comparisons carry that rounding. the arms:
# - a shaped rigid wall in a driven wind tunnel, adiabatic and isothermal;
# - a bonded fragment cluster in the same wind;
# - an accreting bondi sink, fresh, and resumed onto three uneven workers
#   against three host tiles cut at the same cells;
# - the adiabatic alpha viscosity about a central body;
# - a shaped wall and an accreting sink on the magnetized grid;
# - with mesh refinement: a gravitating primary, the same with the alpha
#   viscosity, and an accreting bondi sink under a fine central box, launched
#   refined workers against refined host tiles; the gravitating arms' host
#   tiles against the single hierarchy (the sink's host tiles against the
#   single hierarchy are test_tiled_body_ledger[refined]); three workers, one
#   holding no fine tile; and a late fine-substep rejection against its capped
#   clean control, which must publish no contribution of the rejected attempt.
# - an accreting bondi sink under two nested fine levels on three workers, two
#   of them holding level 1 without level 2, and on host tiles cut at the same
#   cells, each against the single hierarchy bit for bit;
# - on an unrefined grid, a stage rejected on one worker with an accreting
#   sink and with a wall cut by the workers replays as the run capped at the
#   accepted timestep: grid, bodies, history and diagnostics carry nothing of
#   the rejected attempt.
# a magnetized wall takes its shape (a box and a sphere leave different flows),
# and a bonded fragment assembly with mesh refinement is refused on launch by
# name.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_bodies.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

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


def _run(problem: str, resolution: tuple | None, out: Path, steps: int, gpus: int, fields: list[str]) -> Path:
    """`problem` on the session's backend over `gpus` devices, one device or in-process host tiles"""
    fields = ([f"resolution={resolution!r}"] if resolution is not None else []) + fields
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"p = {problem}(gpus={gpus}, data_directory=sys.argv[1]{''.join(', ' + f for f in fields)})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode=sys.argv[2], max_steps={steps})\n"
    )
    # host tiles beyond the visible devices share them
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{problem} on {gpus} device(s)", env=env, devices=gpus)
    return _final(out)


def _tiles(problem: str, resolution: tuple | None, out: Path, steps: int, checkpoint: Path | None = None, cuts: list | None = None) -> Path:
    fields = []
    gpus = 2
    if cuts is not None:
        fields.append(f"decompose={cuts!r}")
        gpus = 1
        for axis in cuts:
            gpus *= len(axis) + 1
    if checkpoint is not None:
        fields.append(f"checkpoint_file={str(checkpoint)!r}")
    return _run(problem, resolution, out, steps, gpus, fields)


def _single(problem: str, resolution: tuple | None, out: Path, steps: int) -> Path:
    return _run(problem, resolution, out, steps, 1, [])


def _launch(variants: Path, problem: str, resolution: tuple | None, out: Path, workers: int, steps: int, cuts: list | None = None,
            shape: tuple | None = None, checkpoint: Path | None = None, inject: str | None = None, expect_ok: bool = True) -> ph.Launch:
    """`problem` launched on `workers` workers partitioned by `cuts` (else `shape`), with no
    experimental opt-in"""
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e9"]
    if resolution is not None:
        args += ["--resolution", ",".join(str(r) for r in resolution)]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return ph.launch(script, out, workers=workers, cuts=cuts, shape=shape, args=args, env=ph.environment(inject), expect_ok=expect_ok)


def _rejections(run: ph.Launch) -> set[int]:
    """the rejection counts the workers report"""
    return {rejections for _, rejections in run.reports().values()}


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
    got = _launch(variants, problem, resolution, tmp_path / "launched", 2, STEPS, cuts=cuts, shape=(2, 1) if planar else (2, 1, 1))
    ph.disagreement(_assert_same, got.final, want, f"{arm}: two launched workers")
    if arm == "sink":
        ph.disagreement(_assert_same_diagnostics, tmp_path / "launched", tmp_path / "tiles", arm)


def test_a_sink_resumed_onto_three_uneven_workers_publishes_the_host_tiles_ledger(tmp_path: Path, variants: Path) -> None:
    # the reference sums over the same three tiles, cut at the same cells
    cuts = [[7, 15], [], []]
    first = _tiles("Bondi", None, tmp_path / "tiles-first", STEPS // 2)
    # `max_steps` caps the absolute root iteration: both resumed runs take STEPS - STEPS // 2 steps
    # past the checkpoint
    cap = _iteration(first) + STEPS - STEPS // 2
    want = _tiles("Bondi", None, tmp_path / "tiles", cap, checkpoint=first, cuts=cuts)
    got = _launch(variants, "Bondi", None, tmp_path / "launched", 3, cap, cuts=cuts, checkpoint=first)
    ph.disagreement(_assert_same, got.final, want, "sink resumed onto three uneven workers")


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
    got = _launch(variants, problem, resolution, tmp_path / "launched", 2, STEPS, shape=(2, 1) if planar else (2, 1, 1))
    ph.disagreement(_assert_same, got.final, want, f"{arm}: two launched refined workers")
    if arm == "refined-sink":
        ph.disagreement(_assert_same_diagnostics, tmp_path / "launched", tmp_path / "tiles", arm)


def test_a_refined_bonded_assembly_on_launch_is_refused(tmp_path: Path, variants: Path) -> None:
    run = _launch(variants, "RefinedRubble", (64, 32), tmp_path / "out", 2, 1, shape=(2, 1), expect_ok=False)
    assert run.returncode != 0, f"the launch succeeded:\n{run.stdout}"
    assert "does not support refinement yet (no fragment step on the hierarchy)" in " ".join(run.stderr.split()), run.stderr[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


@pytest.mark.parametrize("problem", ["RefinedPrimary", "RefinedAlphaPrimary", "RefinedIsothermalAlphaPrimary"])
def test_refined_host_tiles_publish_the_single_hierarchy_bodies(tmp_path: Path, variants: Path, problem: str) -> None:
    # the refined host tiles the launch is gated against publish the single hierarchy's grid, bodies
    # and history bit for bit. the accreting sink's host tiles against the single hierarchy are
    # test_tiled_body_ledger[refined].
    want = _single(problem, (64, 64), tmp_path / "single", STEPS)
    assert any(name.startswith("level_1/") for name in _all_levels(want)), f"{problem}: the single run published no fine level"
    ph.disagreement(_assert_same, _tiles(problem, (64, 64), tmp_path / "tiles", STEPS), want, f"{problem}: two refined host tiles")


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
    got = _launch(variants, problem, resolution, tmp_path / "launched", 3, STEPS, cuts=cuts)
    ph.disagreement(_assert_same, got.final, want, f"{arm}: three workers, the first without a fine tile")
    if arm == "sink":
        ph.disagreement(_assert_same_diagnostics, tmp_path / "launched", tmp_path / "tiles", arm)
    # resumed on the same layout from the coordinator's own checkpoint: the history the coordinator
    # publishes after the restart is the host tiles' after theirs
    first = _launch(variants, problem, resolution, tmp_path / "launched-first", 3, STEPS // 2, cuts=cuts).final
    tiles_first = _tiles(problem, resolution, tmp_path / "tiles-first", STEPS // 2, cuts=cuts)
    # `max_steps` caps the absolute root iteration: each resumed run takes STEPS - STEPS // 2 steps
    # past its checkpoint
    resumed = _launch(variants, problem, resolution, tmp_path / "launched-resumed", 3, _iteration(first) + STEPS - STEPS // 2, cuts=cuts,
                      checkpoint=first).final
    tiles_resumed = _tiles(
        problem, resolution, tmp_path / "tiles-resumed", _iteration(tiles_first) + STEPS - STEPS // 2, checkpoint=tiles_first, cuts=cuts
    )
    with h5py.File(resumed, "r") as f:
        assert "body_diagnostics/time" in f and f["body_diagnostics/time"].shape[0] > 0, (
            f"{arm}: the coordinator published no body history after the restart"
        )
    ph.disagreement(_assert_same, resumed, tiles_resumed, f"{arm}: resumed on three workers, the coordinator without a fine tile")


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
    # level. the tiles book the body sums exactly and round once, so the launch and the host tiles
    # publish the single hierarchy's grid, bodies, history and diagnostics bit for bit
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
    got = _launch(variants, "NestedBondi", None, tmp_path / "launched", 3, STEPS, cuts=NESTED_CUTS)
    _assert_nested_sink(got.final, tmp_path / "launched", want, tmp_path / "single", "nested sink, launched")
    # the host tiles on the same cuts: the tiles holding level 1 without level 2 add no reaction
    tiles = _tiles("NestedBondi", None, tmp_path / "tiles", STEPS, cuts=NESTED_CUTS)
    _assert_nested_sink(tiles, tmp_path / "tiles", want, tmp_path / "single", "nested sink, host tiles")


def _assert_nested_sink(got: Path, out: Path, want: Path, single_out: Path, label: str) -> None:
    """the grid, the body state and history, and the diagnostics file of the single hierarchy, bit
    for bit"""
    ph.disagreement(_assert_same, got, want, label)
    ph.disagreement(_assert_same_diagnostics, out, single_out, label)
    # the published body state carries the published history: one row per step, and the accreted
    # mass is the sum of the per-step receipts, a sum of STEPS terms of one sign within its
    # rounding of STEPS u, u = 2^-53
    bx = _body_groups(got)
    assert bx["body_diagnostics/mass_delta"].shape[0] == STEPS, f"the history books {bx['body_diagnostics/mass_delta'].shape[0]} rows for {STEPS} steps"
    receipts = float(np.sum(bx["body_diagnostics/mass_delta"]))
    published = float(bx["bodies/total_accreted_mass"][0])
    assert abs(published - receipts) <= STEPS * 2.0**-53 * receipts, f"the published accreted mass {published:.17e} departs from its history {receipts:.17e}"


# the sink's late rejection: worker 2 holds root cells 12..24, the patch's upper half; the last
# stage (rk2) of the second fine substep of root step 2, after the first substep drained the sink
LATE_STEP = 2
LATE_REJECTION = f"reject_level:2:{LATE_STEP}:1:1:1"


def test_a_late_fine_rejection_publishes_no_body_contribution_of_the_rejected_attempt(tmp_path: Path, variants: Path) -> None:
    cuts = WITHOUT_FINE["sink"][2]
    rejecting = _launch(variants, "RefinedBondi", None, tmp_path / "rejecting", 3, STEPS, cuts=cuts, inject=LATE_REJECTION)
    assert _rejections(rejecting) == {1}, f"the fine rejection was not counted once on every worker: {rejecting.reports()}"
    clean = _launch(variants, "RefinedBondi", None, tmp_path / "clean", 3, STEPS, cuts=cuts)
    assert _rejections(clean) == {0}
    with h5py.File(rejecting.final, "r") as fr, h5py.File(clean.final, "r") as fc:
        accepted = float(fr["body_diagnostics/dt"][LATE_STEP])
        assert accepted < float(fc["body_diagnostics/dt"][LATE_STEP]), "the rejection never reduced the timestep"
        # the drain booked accretion in the rejected attempt's first fine substep: discarding it is
        # the transaction under test
        assert np.abs(fc["body_diagnostics/mass_delta"][LATE_STEP]).max() > 0.0, "the sink accreted nothing at the rejected step"
    cap = f"cap:{LATE_STEP}:{int(np.float64(accepted).view(np.uint64))}"
    capped = _launch(variants, "RefinedBondi", None, tmp_path / "capped", 3, STEPS, cuts=cuts, inject=cap)
    assert _rejections(capped) == {0}, "the capped control rejected a step"
    # the accepted run publishes the capped clean control's grid, bodies, history and diagnostics:
    # no accretion, force, torque or heat of the rejected attempt survives in any of them
    ph.disagreement(_assert_same, rejecting.final, capped.final, "late fine rejection against the capped control")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "rejecting", tmp_path / "capped", "late fine rejection")


FLAT_REJECTION_STEP = 2
# the cut runs through the wall; the 3d sink splits along x
FLAT_REJECTING = {"sink": ("Bondi", None, None), "wall": ("WindTunnel", (64, 32), [[16], []])}


@pytest.mark.parametrize("arm", list(FLAT_REJECTING))
def test_a_flat_rejection_with_bodies_replays_as_the_capped_run(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution, cuts = FLAT_REJECTING[arm]
    step = FLAT_REJECTION_STEP

    def launch(name: str, inject: str | None) -> ph.Launch:
        return _launch(variants, problem, resolution, tmp_path / name, 2, STEPS, cuts=cuts, shape=(2, 1, 1), inject=inject)

    rejecting = launch("rejecting", f"reject:1:{step}:1")
    assert _rejections(rejecting) == {1}, f"{arm}: the rejection was not counted once on every worker: {rejecting.reports()}"
    clean = launch("clean", None)
    assert _rejections(clean) == {0}
    with h5py.File(rejecting.final, "r") as fr, h5py.File(clean.final, "r") as fc:
        accepted = float(fr["body_diagnostics/dt"][step])
        assert accepted < float(fc["body_diagnostics/dt"][step]), f"{arm}: the rejection never reduced the timestep"
        # the rejected attempt booked body feedback: discarding it is the transaction under test
        booked = [name for name, obj in fc["body_diagnostics"].items()
                  if name not in ("dt", "time") and np.asarray(obj[()]).ndim >= 1 and np.abs(np.asarray(obj[()])[step]).max() > 0.0]
        assert booked, f"{arm}: the body booked no feedback at the rejected step"
    capped = launch("capped", f"cap:{step}:{int(np.float64(accepted).view(np.uint64))}")
    assert _rejections(capped) == {0}, f"{arm}: the capped control rejected a step"
    ph.disagreement(_assert_same, rejecting.final, capped.final, f"{arm}: flat rejection against the capped control")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "rejecting", tmp_path / "capped", f"{arm}: flat rejection")
