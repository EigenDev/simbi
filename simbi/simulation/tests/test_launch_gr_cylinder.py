# =============================================================================
# test_launch_gr_cylinder.py
#
# a black hole on the horizon-penetrating cylindrical chart through `simbi
# launch`: the (R, z) meridional plane, which carries the azimuthal momentum
# on its two axes, the equatorial (R, phi) plane, periodic in phi with one
# momentum per axis, and the full three-dimensional (R, phi, z) chart around a
# schwarzschild hole and around a hole spinning at a = 0.5 M. each launches on
# two workers and on three uneven workers cut along the radius and publishes
# the single grid's state bit for bit; the gas moves in the steps compared and
# the spin changes the run.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_gr_cylinder.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import EXIT, _assert_same_state, _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 6
FIXTURES = '''
import math
from typing import Annotated

from simbi import ProblemParam
from simbi.types import BoundaryCondition, CoordSystem, Spacetime
from simbi.simulation.tests.fixtures.gr_cylindrical_3d_ks_bh import GrCylindrical3DKsBH
from simbi_configs.examples.grhd.gr_cylindrical_ks_bh import GrCylindricalKsBH


class Plane(GrCylindricalKsBH):
    pass


class Equatorial(GrCylindricalKsBH):
    """the equatorial (R, phi) plane of the same hole, periodic in phi, one momentum per axis"""

    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.PLANAR_CYLINDRICAL, description="(R, phi)")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(4.0, 12.0), (0.0, 2.0 * math.pi)], description="(R, phi)")]
    boundary_conditions: Annotated[list[str], ProblemParam([BoundaryCondition.OUTFLOW, BoundaryCondition.PERIODIC], description="R, phi")]

    def initial_primitive_state(self):
        rho, pre = self.rho_ambient, self.p_ambient
        n_r, n_phi = self.resolution

        def gas_state():
            # gas falling inward with a small azimuthal drift, so the plane evolves off its start
            for _ in range(n_phi):
                for ii in range(n_r):
                    yield (rho, -0.05, 0.01, pre)

        return gas_state


class Cylinder(GrCylindrical3DKsBH):
    pass


class SpinningCylinder(GrCylindrical3DKsBH):
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.KERR_KS, description="spinning hole")]
    kerr_spin: Annotated[float, ProblemParam(0.5, description="spin")]
'''
CASES = {
    "plane": ((32, 32), [[10, 21], []]),
    "equatorial": ((32, 32), [[10, 21], []]),
    "cylinder": ((20, 12, 16), [[6, 13], [], []]),
    "spinning-cylinder": ((20, 12, 16), [[6, 13], [], []]),
}
CLASSES = {"plane": "Plane", "equatorial": "Equatorial", "cylinder": "Cylinder", "spinning-cylinder": "SpinningCylinder"}


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    for case, cls in CLASSES.items():
        # the launch loads one problem from its script
        body = FIXTURES.split("\n\nclass ")[0] + "\n\nclass " + next(part for part in FIXTURES.split("\n\nclass ")[1:] if part.startswith(cls + "("))
        (tmp_path / f"{case.replace('-', '_')}.py").write_text(body)
    return tmp_path


def _single(fixtures: Path, case: str, out: Path) -> Path:
    module, cls = case.replace("-", "_"), CLASSES[case]
    resolution, _ = CASES[case]
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixtures)!r})\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {cls}\n"
        f"p = {cls}(resolution={resolution!r}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, f"{case} on one device failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(fixtures: Path, case: str, out: Path, layout: Path, workers: int) -> Path:
    resolution, _ = CASES[case]
    script = fixtures / f"{case.replace('-', '_')}.py"
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(STEPS),
           "--resolution", ",".join(str(r) for r in resolution), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_LAUNCH_EXPERIMENTAL"}
    env["PYTHONPATH"] = os.pathsep.join([str(fixtures), str(Path.cwd()), env.get("PYTHONPATH", "")])
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _moved(out: Path, final: Path) -> bool:
    initial = [p for p in out.glob("*.h5") if "final" not in p.name]
    assert initial, f"no initial checkpoint in {out} to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(final, "r") as f1:
        return not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()])


@pytest.mark.parametrize("case", list(CASES))
def test_the_cylindrical_kerr_schild_chart_launches_as_the_single_grid_runs(tmp_path: Path, fixtures: Path, case: str) -> None:
    want = _single(fixtures, case, tmp_path / "single")
    assert _moved(tmp_path / "single", want), f"{case}: the gas never moved; the gate is vacuous"
    if case == "spinning-cylinder":
        still = _single(fixtures, "cylinder", tmp_path / "still")
        with h5py.File(still, "r") as a, h5py.File(want, "r") as b:
            assert not np.array_equal(a["level_0/conserved/den"][()], b["level_0/conserved/den"][()]), "the spin left the run unchanged"
    _, cuts = CASES[case]
    ndim = len(cuts)
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, shape=(2,) + (1,) * (ndim - 1))),
        ("three uneven workers", 3, _layout(tmp_path / "three.toml", workers=3, cuts=cuts)),
    ]:
        _assert_same_state(_launch(fixtures, case, tmp_path / label.replace(" ", "_"), layout, workers), want, f"{case}: {label}")
