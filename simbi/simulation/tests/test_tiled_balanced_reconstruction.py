# =============================================================================
# test_tiled_balanced_reconstruction.py
#
# the balanced reconstruction on tiles. an isentropic atmosphere at rest in
# the potential of a point mass below the box, with reflecting walls: the
# balanced face pair holds it near rest where the plain reconstruction drives
# a spurious flow, so the balance changes the run. two host tiles cut across
# the stratification and four host tiles cut on both axes publish the single
# grid's state bit for bit, and so does a two-by-two launch under the
# experimental opt-in. with mesh refinement, whose balanced coarse-fine
# transfer and seam band start from cells a cut separates, host tiles refuse
# the balanced reconstruction by name before any output.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_balanced_reconstruction.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _assert_same_state, _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_isothermal import _script_launch

pytestmark = pytest.mark.simulation

STEPS = 8
RESOLUTION = (24, 32)
FIXTURE = '''
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig

GM = 100.0
CENTER = (0.5, -1.0)


class BalancedAtmosphere(SimbiProblem):
    """an isentropic atmosphere in the potential of a point mass below the box, at rest"""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int, int], ProblemParam((24, 32), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="solver")]
    wb_reconstruction: Annotated[bool, ProblemParam(True, description="balanced")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(1, cli=True, description="devices")]
    data_directory: Annotated[Path, ProblemParam(Path("data/balanced_atmosphere"), cli=True, checkpoint_safe=True, description="out")]

    def _atmosphere(self) -> expr.Equilibrium:
        x, y = expr.coords(2)
        r = ((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) ** 0.5
        return expr.isentropic_atmosphere(-GM / r, gamma=self.adiabatic_index, k_entropy=1.0, dim=2, reference_density=1.0, reference_point=[0.5, 1.0])

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=GM, radius=0.0, position=(CENTER[0], CENTER[1], 0.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            nx, ny = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    yield tuple(sample(x1=(ii + 0.5) / nx, x2=(jj + 0.5) / ny))

        return gas


class RefinedBalancedAtmosphere(BalancedAtmosphere):
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]
'''


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "balanced_atmosphere.py"
    path.write_text(FIXTURE)
    return path


def _run(fixture: Path, cls: str, out: Path, gpus: int = 1, extra: str = "") -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixture.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from balanced_atmosphere import {cls}\n"
        f"runner.run({cls}(gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9{extra}), compute_mode='cpu', max_steps={STEPS})\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))


def _final(done: subprocess.CompletedProcess, out: Path, label: str) -> Path:
    assert done.returncode == 0, f"{label} failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _vertical_momentum(path: Path) -> float:
    with h5py.File(path, "r") as f:
        return float(np.abs(f["level_0/conserved/m2"][()]).max())


def test_the_balanced_reconstruction_on_host_tiles_matches_the_single_grid(tmp_path: Path, fixture: Path) -> None:
    want = _final(_run(fixture, "BalancedAtmosphere", tmp_path / "single"), tmp_path / "single", "the single grid")
    plain = _final(_run(fixture, "BalancedAtmosphere", tmp_path / "plain", extra=", wb_reconstruction=False"), tmp_path / "plain", "the plain run")
    # measured over eight steps: 11.7 plain against 0.053 balanced
    assert _vertical_momentum(want) < 0.1 * _vertical_momentum(plain), "the balance leaves the atmosphere's spurious flow in place; the gate is vacuous"
    for label, gpus, decompose in [("two host tiles across the stratification", 2, [[], [13]]), ("four host tiles", 4, [[11], [17]])]:
        out = tmp_path / f"tiles{gpus}"
        got = _final(_run(fixture, "BalancedAtmosphere", out, gpus, f", decompose={decompose!r}"), out, label)
        _assert_same_state(got, want, label)


def test_the_balanced_reconstruction_launched_under_the_opt_in_matches_the_single_grid(tmp_path: Path, fixture: Path) -> None:
    want = _final(_run(fixture, "BalancedAtmosphere", tmp_path / "single"), tmp_path / "single", "the single grid")
    script = fixture.parent / "launched_balanced.py"
    script.write_text("from balanced_atmosphere import BalancedAtmosphere as Base\n\n\nclass LaunchedBalancedAtmosphere(Base):\n    pass\n")
    env = {"PYTHONPATH": os.pathsep.join([str(fixture.parent), os.getcwd(), os.environ.get("PYTHONPATH", "")]), "SIMBI_LAUNCH_EXPERIMENTAL": "1"}
    layout = _layout(tmp_path / "four.toml", workers=4, cuts=[[11], [17]])
    got = _script_launch(str(script), RESOLUTION, tmp_path / "launched", layout, STEPS, workers=4, env=env)
    _assert_same_state(got, want, "a two-by-two launch under the opt-in")


def test_the_balanced_reconstruction_with_refinement_on_host_tiles_is_refused_by_name(tmp_path: Path, fixture: Path) -> None:
    done = _run(fixture, "RefinedBalancedAtmosphere", tmp_path / "out", 2)
    assert done.returncode != 0, "refined tiles ran the balanced reconstruction"
    assert "wb_reconstruction with mesh refinement and gpus > 1" in " ".join(done.stderr.split()), done.stderr[-1500:]
    assert not list((tmp_path / "out").glob("*.h5")), "output was written before the refusal"
