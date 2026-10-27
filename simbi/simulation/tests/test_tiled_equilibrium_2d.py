# =============================================================================
# test_tiled_equilibrium_2d.py
#
# a declared stationary target on a two-dimensional grid. the isentropic
# atmosphere stratified along y in a point mass's gravity, uniform along x,
# with reflecting walls: its discrete imbalance is read off one probe stage of
# the decomposed pipeline and added back at every stage, so two host tiles cut
# across the stratification, four host tiles whose every tile holds a cut
# corner and a two-by-two launch publish the single grid's state bit for bit,
# and every decomposition probes at the single grid's own timestep. the target
# changes the run: the same atmosphere without it lands elsewhere.
#
# every arm runs as a fresh process on the session's backend and placement
# (parity_harness), the tiled ones as host tiles.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_equilibrium_2d.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
import re
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import _assert_same_state, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 8
RESOLUTION = (24, 32)
PROBE_DT = re.compile(r"SIMBI equilibrium probe: level=0 dt_bits=([0-9a-f]+)")
FIXTURE = '''
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver


class Atmosphere2d(SimbiProblem):
    """an isentropic atmosphere stratified along y, uniform along x, held as the run's target"""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int, int], ProblemParam((24, 32), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="solver")]
    seed_from_equilibrium: Annotated[bool, ProblemParam(True, description="seeded")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(1, cli=True, description="devices")]
    data_directory: Annotated[Path, ProblemParam(Path("data/atmosphere_2d"), cli=True, checkpoint_safe=True, description="out")]

    def _atmosphere(self) -> expr.Equilibrium:
        _, y = expr.coords(2)
        return expr.isentropic_atmosphere(
            -100.0 / (y + 1.0), gamma=self.adiabatic_index, k_entropy=1.0, dim=2, reference_density=1.0, reference_point=[1.0, 1.0]
        )

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return expr.equilibrium(self._atmosphere().primitives, dim=2)

    @computed_field
    @property
    def source_expressions(self) -> list[dict]:
        return [expr.force(self._atmosphere().acceleration, dim=2)]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            nx, ny = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    yield tuple(sample(x1=(ii + 0.5) / nx, x2=(jj + 0.5) / ny))

        return gas


class RefinedAtmosphere2d(Atmosphere2d):
    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), cli=True, description="grid")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]


class Atmosphere2dWithoutTarget(Atmosphere2d):
    seed_from_equilibrium: Annotated[bool, ProblemParam(False, description="unseeded")]

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return {}
'''


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "atmosphere_2d.py"
    path.write_text(FIXTURE)
    return path


def _single(fixture: Path, cls: str, out: Path, gpus: int = 1, extra: str = "") -> tuple[Path, str]:
    """`cls` on the session's backend, one device or `gpus` host tiles, in a fresh interpreter: its
    final checkpoint and output"""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixture.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from atmosphere_2d import {cls}\n"
        f"runner.run({cls}(gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9{extra}), compute_mode={ph.BACKEND!r}, max_steps={STEPS})\n"
    )
    done = ph.python(ph.snippet_args(snippet, str(out)), f"{cls} on {gpus} device(s){extra}", devices=gpus)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0], done.stdout + done.stderr


def _launch(fixture: Path, cls: str, resolution: tuple, out: Path, cuts: list[list[int]]) -> Path:
    """`simbi launch` of `cls` on the session's backend and placement without the experimental
    opt-in: its final checkpoint"""
    # the launch evolves the one problem its script defines
    script = fixture.parent / f"launched_{cls.lower()}.py"
    script.write_text(f"from atmosphere_2d import {cls} as Base\n\n\nclass Launched{cls}(Base):\n    pass\n")
    env = ph.environment()
    env["PYTHONPATH"] = os.pathsep.join([str(fixture.parent), os.getcwd(), os.environ.get("PYTHONPATH", "")])
    args = ["--max-steps", str(STEPS), "--resolution", ",".join(str(r) for r in resolution), "--checkpoint-interval", "1e9"]
    workers = int(np.prod([len(c) + 1 for c in cuts]))
    return ph.launch(script, out, workers, cuts=cuts, args=args, env=env, timeout=600).final


def _probe_dt(output: str, label: str) -> set[str]:
    bits = set(PROBE_DT.findall(output))
    assert bits, f"{label}: no equilibrium probe was reported; the target never reached the backend"
    return bits


def _differs(a: Path, b: Path) -> bool:
    try:
        _assert_same_state(a, b, "premise")
    except AssertionError:
        return True
    return False


def test_a_two_dimensional_target_holds_on_every_decomposition(tmp_path: Path, fixture: Path) -> None:
    reference, out = _single(fixture, "Atmosphere2d", tmp_path / "single")
    probe = _probe_dt(out, "the single grid")
    without, _ = _single(fixture, "Atmosphere2dWithoutTarget", tmp_path / "without")
    assert _differs(reference, without), "the target left the run unchanged; the gate is vacuous"
    for label, gpus, extra in [("two host tiles across the stratification", 2, ", decompose=[[], [13]]"), ("four host tiles", 4, ", decompose=[[11], [17]]")]:
        got, out = _single(fixture, "Atmosphere2d", tmp_path / f"tiles{gpus}", gpus=gpus, extra=extra)
        assert _probe_dt(out, label) == probe, f"{label} probed at another timestep"
        ph.same_checkpoint(got, reference, label)
    launched = _launch(fixture, "Atmosphere2d", RESOLUTION, tmp_path / "launched", cuts=[[11], [17]])
    ph.same_checkpoint(launched, reference, "a two-by-two launch")


def _same_levels(got: dict, want: dict, label: str) -> None:
    """every cell and face dataset of every level, halos trimmed, bit for bit"""
    assert got.keys() == want.keys(), f"{label}: datasets {sorted(got.keys() ^ want.keys())} differ"
    for name, x in want.items():
        np.testing.assert_array_equal(got[name], x, err_msg=f"{label}: {name}")


PROBE_LEVEL = re.compile(r"SIMBI equilibrium probe: level=(\d+) dt_bits=([0-9a-f]+)")


def test_a_refined_two_dimensional_target_holds_on_every_decomposition(tmp_path: Path, fixture: Path) -> None:
    from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

    reference, out = _single(fixture, "RefinedAtmosphere2d", tmp_path / "single")
    probes = set(PROBE_LEVEL.findall(out))
    assert {level for level, _ in probes} == {"0", "1"}, f"the single hierarchy probed levels {probes}"
    want = _all_levels(reference)
    assert "level_1/conserved/den" in want, "the single hierarchy published no fine level; the refined arm is vacuous"
    got, out = _single(fixture, "RefinedAtmosphere2d", tmp_path / "tiles", gpus=2, extra=", decompose=[[16], []]")
    assert set(PROBE_LEVEL.findall(out)) == probes, "the refined host tiles probed at other timesteps"
    ph.disagreement(_same_levels, _all_levels(got), want, "two refined host tiles")
    launched = _launch(fixture, "RefinedAtmosphere2d", (32, 32), tmp_path / "launched", cuts=[[16], []])
    ph.disagreement(_same_levels, _all_levels(launched), want, "two launched refined workers")
