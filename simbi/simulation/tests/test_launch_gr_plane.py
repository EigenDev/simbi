# =============================================================================
# test_launch_gr_plane.py
#
# the meridional plane of a black hole on the two-dimensional spherical
# kerr-schild chart through `simbi launch`, in each layout the single grid's
# dispatch picks: the spinning hole's fishbone-moncrief torus carries the
# azimuthal momentum the frame dragging needs (three momentum components on a
# two-axis grid); the same torus around a schwarzschild hole carries it too;
# and the schwarzschild plane without it (rows of one momentum per axis, the
# torus with its rotation dropped). each on two launched workers and on three
# uneven workers cut along the radius publishes the single grid's state bit
# for bit, and the gas moves off its initial state in the steps compared. the
# single grid and the launches run on the session's backend and placement
# (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_gr_plane.py
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
NR, NPOLAR = 48, 32
HEADER = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Spacetime
from simbi_configs.examples.grhd.gr_fishbone_moncrief import GrFishboneMoncrief
'''
# case: the problem class, as the launch script and the single grid both load it
CASES = {
    "kerr-torus": "class Plane(GrFishboneMoncrief):\n    pass\n",
    "schwarzschild-torus": (
        "class Plane(GrFishboneMoncrief):\n"
        "    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.SCHWARZSCHILD_KS, description='schwarzschild')]\n"
        "    kerr_spin: Annotated[float, ProblemParam(0.0, description='spin')]\n"
    ),
    "schwarzschild-plane": (
        "class Plane(GrFishboneMoncrief):\n"
        "    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.SCHWARZSCHILD_KS, description='schwarzschild')]\n"
        "    kerr_spin: Annotated[float, ProblemParam(0.0, description='spin')]\n"
        "\n"
        "    def initial_primitive_state(self):\n"
        "        swirl = super().initial_primitive_state()\n"
        "\n"
        "        def rows():\n"
        "            # one momentum per axis: the azimuthal velocity dropped\n"
        "            for row in swirl():\n"
        "                yield tuple(row[:3]) + tuple(row[4:])\n"
        "\n"
        "        return rows\n"
    ),
}


def _script(tmp_path: Path, case: str) -> Path:
    path = tmp_path / f"{case.replace('-', '_')}.py"
    path.write_text(HEADER + "\n\n" + CASES[case])
    return path


def _single(script: Path, out: Path) -> Path:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(script.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from {script.stem} import Plane\n"
        f"p = Plane(nr={NR}, npolar={NPOLAR}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={STEPS})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), "the single grid", timeout=1800)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(script: Path, out: Path, workers: int, cuts=None, shape=None) -> Path:
    """the launch on the session's backend and placement; its final checkpoint"""
    # the torus derives its resolution from nr and npolar
    args = ["--max-steps", str(STEPS), "--nr", str(NR), "--npolar", str(NPOLAR), "--checkpoint-interval", "1e9"]
    env = ph.environment()
    env["PYTHONPATH"] = os.pathsep.join([str(script.parent), str(Path.cwd()), env.get("PYTHONPATH", "")])
    return ph.launch(script, out, workers, cuts=cuts, shape=shape, args=args, env=env, timeout=1800).final


def _assert_same(got: Path, want: Path, label: str) -> None:
    """every level dataset bit for bit, a difference raised as a NumericalDisagreement"""
    ph.disagreement(_assert_same_levels, got, want, label)


def _assert_same_levels(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert "level_0/conserved/den" in ys, f"{label}: the reference publishes no density; the comparison is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("case", list(CASES))
def test_the_kerr_schild_plane_launches_as_the_single_grid_runs(tmp_path: Path, case: str) -> None:
    script = _script(tmp_path, case)
    want = _single(script, tmp_path / "single")
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, f"{case}: the reference wrote no initial checkpoint to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(want, "r") as f1:
        assert not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()]), f"{case}: the gas never moved; the gate is vacuous"
        momenta = sorted(k for k in f1["level_0/conserved"].keys() if k in ("m1", "m2", "m3"))
    # the layout the dispatch picked: the azimuthal momentum is carried unless the rows drop it
    assert len(momenta) == (2 if case == "schwarzschild-plane" else 3), f"{case}: momentum datasets {momenta}"
    for label, workers, layout in [
        ("two workers", 2, {"shape": (2, 1)}),
        ("three uneven workers", 3, {"cuts": [[15, 33], []]}),
    ]:
        _assert_same(_launch(script, tmp_path / label.replace(" ", "_"), workers, **layout), want, f"{case}: {label}")
