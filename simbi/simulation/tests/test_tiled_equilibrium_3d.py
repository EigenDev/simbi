# =============================================================================
# test_tiled_equilibrium_3d.py
#
# a declared stationary target on a three-dimensional grid. the isentropic
# atmosphere stratified along z in a point mass's gravity, uniform across x and
# y, with reflecting walls: its discrete imbalance is read off one probe stage
# of the decomposed pipeline and added back at every stage, so two host tiles
# cut across the stratification, four host tiles cut on two axes and a
# two-by-two launch publish the single grid's state bit for bit, and every
# decomposition probes at the single grid's own timestep. the target changes
# the run: the same atmosphere without it lands elsewhere.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_equilibrium_3d.py
# =============================================================================

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _assert_same_state, _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_isothermal import _script_launch

pytestmark = pytest.mark.simulation

STEPS = 8
RESOLUTION = (12, 12, 16)
PROBE_DT = re.compile(r"SIMBI equilibrium probe: level=0 dt_bits=([0-9a-f]+)")
FIXTURE = '''
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver


class Atmosphere3d(SimbiProblem):
    """an isentropic atmosphere stratified along z, uniform across x and y, held as the run's target"""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int, int, int], ProblemParam((12, 12, 16), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="solver")]
    seed_from_equilibrium: Annotated[bool, ProblemParam(True, description="seeded")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(1, cli=True, description="devices")]
    data_directory: Annotated[Path, ProblemParam(Path("data/atmosphere_3d"), cli=True, checkpoint_safe=True, description="out")]

    def _atmosphere(self) -> expr.Equilibrium:
        _, _, z = expr.coords(3)
        return expr.isentropic_atmosphere(
            -100.0 / (z + 1.0), gamma=self.adiabatic_index, k_entropy=1.0, dim=3, reference_density=1.0, reference_point=[1.0, 1.0, 1.0]
        )

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return expr.equilibrium(self._atmosphere().primitives, dim=3)

    @computed_field
    @property
    def source_expressions(self) -> list[dict]:
        return [expr.force(self._atmosphere().acceleration, dim=3)]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            nx, ny, nz = self.resolution
            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        yield tuple(sample(x1=(ii + 0.5) / nx, x2=(jj + 0.5) / ny, x3=(kk + 0.5) / nz))

        return gas


class Atmosphere3dWithoutTarget(Atmosphere3d):
    seed_from_equilibrium: Annotated[bool, ProblemParam(False, description="unseeded")]

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return {}
'''


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "atmosphere_3d.py"
    path.write_text(FIXTURE)
    return path


def _single(fixture: Path, cls: str, out: Path, gpus: int = 1, extra: str = "") -> tuple[Path, str]:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixture.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from atmosphere_3d import {cls}\n"
        f"runner.run({cls}(gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9{extra}), compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    assert done.returncode == 0, f"{cls} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0], done.stdout + done.stderr


def _launch(fixture: Path, cls: str, resolution: tuple, out: Path, layout: Path, workers: int) -> Path:
    # the launch evolves the one problem its script defines
    script = fixture.parent / f"launched_{cls.lower()}.py"
    script.write_text(f"from atmosphere_3d import {cls} as Base\n\n\nclass Launched{cls}(Base):\n    pass\n")
    env = {"PYTHONPATH": os.pathsep.join([str(fixture.parent), os.getcwd(), os.environ.get("PYTHONPATH", "")])}
    return _script_launch(str(script), resolution, out, layout, STEPS, workers=workers, env=env)


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


def test_a_three_dimensional_target_holds_on_every_decomposition(tmp_path: Path, fixture: Path) -> None:
    reference, out = _single(fixture, "Atmosphere3d", tmp_path / "single")
    probe = _probe_dt(out, "the single grid")
    without, _ = _single(fixture, "Atmosphere3dWithoutTarget", tmp_path / "without")
    assert _differs(reference, without), "the target left the run unchanged; the gate is vacuous"
    for label, gpus, extra in [("two host tiles across the stratification", 2, ", decompose=[[], [], [7]]"),
                               ("four host tiles on two axes", 4, ", decompose=[[5], [], [9]]")]:
        got, out = _single(fixture, "Atmosphere3d", tmp_path / f"tiles{gpus}", gpus=gpus, extra=extra)
        assert _probe_dt(out, label) == probe, f"{label} probed at another timestep"
        _assert_same_state(got, reference, label)
    layout = _layout(tmp_path / "four.toml", workers=4, cuts=[[5], [], [9]])
    launched = _launch(fixture, "Atmosphere3d", RESOLUTION, tmp_path / "launched", layout, workers=4)
    _assert_same_state(launched, reference, "a two-by-two launch")
