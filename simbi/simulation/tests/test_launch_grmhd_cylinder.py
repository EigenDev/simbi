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
# changes the run. every arm runs on the session's backend and placement
# (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_grmhd_cylinder.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

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
    """`cls` on one device of the session's backend"""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixtures)!r})\n"
        "from simbi.simulation import runner\n"
        f"from grmhd_cylinder import {cls}\n"
        f"p = {cls}()\n"
        "p.data_directory = sys.argv[1]\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={STEPS})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), f"{cls} on one device", timeout=1800)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(fixtures: Path, cls: str, out: Path, workers: int, cuts=None, shape=None) -> Path:
    """`cls` launched on the session's backend and placement without the experimental opt-in"""
    args = ["--max-steps", str(STEPS), "--checkpoint-interval", "1e9"]
    env = ph.environment()
    env["PYTHONPATH"] = os.pathsep.join([str(fixtures), str(Path.cwd()), env.get("PYTHONPATH", "")])
    return ph.launch(fixtures / f"{cls.lower()}.py", out, workers, cuts=cuts, shape=shape, args=args, env=env, timeout=1800).final


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
    ph.same_checkpoint(_launch(fixtures, cls, tmp_path / "two_workers", 2, shape=(2, 1)), want, f"{case}: two workers")
    ph.same_checkpoint(_launch(fixtures, cls, tmp_path / "three_uneven_workers", 3, cuts=CUTS), want, f"{case}: three uneven workers")
