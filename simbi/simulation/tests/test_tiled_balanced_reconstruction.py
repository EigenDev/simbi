# =============================================================================
# test_tiled_balanced_reconstruction.py
#
# the balanced reconstruction on tiles. an isentropic atmosphere at rest in
# the potential of a point mass below the domain, with reflecting walls: the
# balanced face pair holds it near rest where the plain reconstruction drives
# a spurious flow, so the balance changes the run. on the 2D grid two host
# tiles cut across the stratification, four host tiles cut on both axes and a
# two-by-two launch publish the single grid's state bit for bit; on the 3D
# grid two and four host tiles and a four-worker launch do too. a grid outside
# that scope (the one-dimensional column) and a refined run, whose balanced
# coarse-fine transfer and seam band start from cells a cut separates, are
# refused by name on host tiles and launch before any output.
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


class BalancedAtmosphere3d(BalancedAtmosphere):
    """the same atmosphere in a cube, the point mass below its base"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((16, 16, 16), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="cube")]

    def _atmosphere(self) -> expr.Equilibrium:
        x, y, z = expr.coords(3)
        r = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z + 1.0) ** 2) ** 0.5
        return expr.isentropic_atmosphere(-GM / r, gamma=self.adiabatic_index, k_entropy=1.0, dim=3, reference_density=1.0, reference_point=[0.5, 0.5, 1.0])

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=GM, radius=0.0, position=(0.5, 0.5, -1.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            nx, ny, nz = self.resolution
            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        yield tuple(sample(x1=(ii + 0.5) / nx, x2=(jj + 0.5) / ny, x3=(kk + 0.5) / nz))

        return gas


class BalancedColumn(BalancedAtmosphere):
    """the atmosphere along one axis, the point mass below its base"""

    resolution: Annotated[tuple[int], ProblemParam((32,), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0)], description="column")]

    def _atmosphere(self) -> expr.Equilibrium:
        (x,) = expr.coords(1)
        return expr.isentropic_atmosphere(-GM / (x + 1.0), gamma=self.adiabatic_index, k_entropy=1.0, dim=1, reference_density=1.0, reference_point=[1.0])

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=GM, radius=0.0, position=(-1.0, 0.0, 0.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            (nx,) = self.resolution
            for ii in range(nx):
                yield tuple(sample(x1=(ii + 0.5) / nx))

        return gas
'''
# (class, resolution, momentum dataset along the stratification, host tile cuts, launch cuts)
GRIDS = {
    "2d": ("BalancedAtmosphere", (24, 32), "m2", [[[], [13]], [[11], [17]]], [[11], [17]]),
    "3d": ("BalancedAtmosphere3d", (16, 16, 16), "m3", [[[], [], [7]], [[9], [], [7]]], [[9], [], [7]]),
}


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


def _momentum(path: Path, name: str) -> float:
    with h5py.File(path, "r") as f:
        return float(np.abs(f[f"level_0/conserved/{name}"][()]).max())


def _launch_script(fixture: Path, cls: str) -> Path:
    # the launch evolves the one problem its script defines
    script = fixture.parent / f"launched_{cls.lower()}.py"
    script.write_text(f"from balanced_atmosphere import {cls} as Base\n\n\nclass Launched{cls}(Base):\n    pass\n")
    return script


def _launch_env(fixture: Path) -> dict[str, str]:
    return {"PYTHONPATH": os.pathsep.join([str(fixture.parent), os.getcwd(), os.environ.get("PYTHONPATH", "")])}


@pytest.mark.parametrize("grid", list(GRIDS))
def test_the_balanced_reconstruction_on_tiles_matches_the_single_grid(tmp_path: Path, fixture: Path, grid: str, monkeypatch) -> None:
    cls, resolution, momentum, host_cuts, launch_cuts = GRIDS[grid]
    want = _final(_run(fixture, cls, tmp_path / "single"), tmp_path / "single", "the single grid")
    plain = _final(_run(fixture, cls, tmp_path / "plain", extra=", wb_reconstruction=False"), tmp_path / "plain", "the plain run")
    # the 2D atmosphere measured over eight steps: 11.7 plain against 0.053 balanced
    assert _momentum(want, momentum) < 0.1 * _momentum(plain, momentum), f"{grid}: the balance leaves the spurious flow in place; the gate is vacuous"
    for cuts in host_cuts:
        gpus = int(np.prod([len(c) + 1 for c in cuts]))
        out = tmp_path / f"tiles{gpus}"
        got = _final(_run(fixture, cls, out, gpus, f", decompose={cuts!r}"), out, f"{grid} on {gpus} host tiles")
        _assert_same_state(got, want, f"{grid} on {gpus} host tiles cut at {cuts}")
    monkeypatch.delenv("SIMBI_LAUNCH_EXPERIMENTAL", raising=False)
    workers = int(np.prod([len(c) + 1 for c in launch_cuts]))
    layout = _layout(tmp_path / "launch.toml", workers=workers, cuts=launch_cuts)
    got = _script_launch(str(_launch_script(fixture, cls)), resolution, tmp_path / "launched", layout, STEPS, workers=workers, env=_launch_env(fixture))
    _assert_same_state(got, want, f"{grid} launched on {workers} workers")


@pytest.mark.parametrize(
    ("cls", "reason"),
    [
        ("BalancedColumn", "wb_reconstruction on the 1-dimensional cartesian grid with several tiles is outside the gated tiled runs"),
        ("RefinedBalancedAtmosphere", "wb_reconstruction with mesh refinement on several tiles"),
    ],
)
def test_the_balanced_reconstruction_outside_the_gated_scope_is_refused_on_host_tiles(tmp_path: Path, fixture: Path, cls: str, reason: str) -> None:
    done = _run(fixture, cls, tmp_path / "out", 2)
    assert done.returncode != 0, f"{cls} ran on host tiles"
    assert reason in " ".join(done.stderr.split()), done.stderr[-1500:]
    assert not list((tmp_path / "out").glob("*.h5")), "output was written before the refusal"


def test_the_balanced_column_is_refused_on_launch(tmp_path: Path, fixture: Path) -> None:
    env = dict(os.environ, **_launch_env(fixture))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2,))
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(_launch_script(fixture, "BalancedColumn")), "--layout", str(layout),
           "--max-steps", "1", "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    assert "wb_reconstruction on the 1-dimensional cartesian grid with several tiles is outside the gated tiled runs" in " ".join(done.stderr.split()), done.stderr
    assert not (out.exists() and list(out.glob("*.h5"))), "a checkpoint was published"
