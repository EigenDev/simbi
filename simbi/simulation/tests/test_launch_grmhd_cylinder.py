# =============================================================================
# test_launch_grmhd_cylinder.py
#
# magnetized relativistic gas on the horizon-penetrating cylindrical chart
# through `simbi launch`: a poloidal field loop on the 2.5D (R, z) plane and an
# in-plane loop on the equatorial (R, phi) disk, each seeded div-free through
# the metric-weighted curl of a vector potential, under the contact and the UCT
# edge EMF. each launches on two workers and on three uneven workers cut along
# the radius and publishes the single grid's state bit for bit, faces and
# halos included; the field moves in the steps compared and the edge EMF
# changes the run.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_grmhd_cylinder.py
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
from typing import Annotated

from simbi import ProblemParam
from simbi.types import CtMethod
from simbi.simulation.tests.fixtures.gr_cylindrical_rz_field_loop import GrCylindricalRzFieldLoop
from simbi.simulation.tests.fixtures.gr_disk_field_loop import GrDiskFieldLoop


class Plane(GrCylindricalRzFieldLoop):
    nr: Annotated[int, ProblemParam(32, cli=True, description="R resolution")]
    nz: Annotated[int, ProblemParam(32, cli=True, description="z resolution")]
    ct_method: Annotated[CtMethod, ProblemParam(CtMethod.CONTACT, cli=True, description="edge EMF")]


class PlaneUct(Plane):
    ct_method: Annotated[CtMethod, ProblemParam(CtMethod.UCT, cli=True, description="edge EMF")]


class Disk(GrDiskFieldLoop):
    nr: Annotated[int, ProblemParam(32, cli=True, description="R resolution")]
    nphi: Annotated[int, ProblemParam(32, cli=True, description="phi resolution")]
    ct_method: Annotated[CtMethod, ProblemParam(CtMethod.CONTACT, cli=True, description="edge EMF")]


class DiskUct(Disk):
    ct_method: Annotated[CtMethod, ProblemParam(CtMethod.UCT, cli=True, description="edge EMF")]
'''
# case: (class, the case under the other edge EMF)
CASES = {
    "plane": ("Plane", "PlaneUct"),
    "plane-uct": ("PlaneUct", "Plane"),
    "disk": ("Disk", "DiskUct"),
    "disk-uct": ("DiskUct", "Disk"),
}
CUTS = [[10, 21], []]


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    where = tmp_path / "fixtures"
    where.mkdir()
    (where / "grmhd_cylinder.py").write_text(FIXTURES)
    for cls in {cls for pair in CASES.values() for cls in pair}:
        (where / f"{cls.lower()}.py").write_text(f"from grmhd_cylinder import {cls} as Base\n\n\nclass Launched{cls}(Base):\n    pass\n")
    return where


def _single(fixtures: Path, cls: str, out: Path) -> Path:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixtures)!r})\n"
        "from simbi.simulation import runner\n"
        f"from grmhd_cylinder import {cls}\n"
        f"p = {cls}()\n"
        "p.data_directory = sys.argv[1]\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, f"{cls} on one device failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(fixtures: Path, cls: str, out: Path, layout: Path, workers: int) -> Path:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(fixtures / f"{cls.lower()}.py"), "--layout", str(layout),
           "--max-steps", str(STEPS), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_LAUNCH_EXPERIMENTAL"}
    env["PYTHONPATH"] = os.pathsep.join([str(fixtures), str(Path.cwd()), env.get("PYTHONPATH", "")])
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _faces(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["level_0/partition_0/hydro/magnetic/B1/data"][()]


@pytest.mark.parametrize("case", list(CASES))
def test_the_magnetized_cylindrical_kerr_schild_chart_launches_as_the_single_grid_runs(tmp_path: Path, fixtures: Path, case: str) -> None:
    cls, other = CASES[case]
    want = _single(fixtures, cls, tmp_path / "single")
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, f"{case}: no initial checkpoint to measure the evolution against"
    assert not np.array_equal(_faces(initial[0]), _faces(want)), f"{case}: the field never moved; the gate is vacuous"
    assert not np.array_equal(_faces(_single(fixtures, other, tmp_path / "other")), _faces(want)), f"{case}: the edge EMF left the run unchanged"
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))),
        ("three uneven workers", 3, _layout(tmp_path / "three.toml", workers=3, cuts=CUTS)),
    ]:
        _assert_same_state(_launch(fixtures, cls, tmp_path / label.replace(" ", "_"), layout, workers), want, f"{case}: {label}")
