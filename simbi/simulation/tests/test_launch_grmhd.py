# =============================================================================
# test_launch_grmhd.py
#
# relativistic magnetized gas on curved spacetimes through `simbi launch`. the
# fixtures the host tiles are gated on: the magnetized bondi inflow and the
# magnetized michel inflow on the radial kerr-schild line, the frame dragging
# and the magnetized torus on the spinning (r, theta) plane, the magnetized
# torus in the three-dimensional kerr-schild box, and a field loop on the
# cylindrical (R, z) and (R, phi) planes. each launches on two workers
# and on three uneven workers and publishes the single grid's state bit for
# bit, every staggered face included; the gas moves in the steps compared.
# every arm runs on the session's backend and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_grmhd.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 6
# name: (module, class, constructor fields, launch flags, uneven cuts)
CASES = {
    "magnetized-ks-line": ("simbi_configs.examples.grmhd.gr_bondi_ks_magnetized", "GrBondiKsMagnetized", "resolution=128", ["--resolution", "128"], [[40, 90]]),
    "magnetized-michel-line": ("simbi_configs.examples.grmhd.gr_michel_magnetized", "GrMichelMagnetized", "resolution=128", ["--resolution", "128"], [[40, 90]]),
    "kerr-dragging": ("simbi_configs.examples.grmhd.gr_kerr_dragging", "GrKerrDragging", "nr=48, npolar=32", ["--nr", "48", "--npolar", "32"], [[15, 33], []]),
    "magnetized-kerr-torus": ("simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd", "GrFishboneMoncriefMhd", "nr=48, npolar=32", ["--nr", "48", "--npolar", "32"], [[15, 33], []]),
    "rz-field-loop": ("simbi.simulation.tests.fixtures.gr_cylindrical_rz_field_loop", "GrCylindricalRzFieldLoop", "nr=32, nz=32", ["--nr", "32", "--nz", "32"], [[10, 21], []]),
    "disk-field-loop": ("simbi.simulation.tests.fixtures.gr_disk_field_loop", "GrDiskFieldLoop", "nr=32, nphi=32", ["--nr", "32", "--nphi", "32"], [[10, 21], []]),
    "magnetized-cartesian-torus": (
        "simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd_cartesian",
        "GrFishboneMoncriefMhdCartesian",
        "resolution=(24, 24, 24)",
        ["--resolution", "24,24,24"],
        [[8, 17], [], []],
    ),
}


def _single(case: str, out: Path) -> Path:
    """the case on one device of the session's backend"""
    module, cls, fields, _, _ = CASES[case]
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {cls}\n"
        f"p = {cls}({fields}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={STEPS})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), f"{case} on one device", timeout=1800)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(case: str, out: Path, workers: int, cuts=None, shape=None) -> Path:
    """the case launched on the session's backend and placement without the experimental opt-in"""
    module, _, _, flags, _ = CASES[case]
    args = ["--max-steps", str(STEPS), *flags, "--checkpoint-interval", "1e9"]
    return ph.launch(module.replace(".", "/") + ".py", out, workers, cuts=cuts, shape=shape, args=args, env=ph.environment(), timeout=1800).final


@pytest.mark.parametrize("case", list(CASES))
def test_curved_magnetized_launches_match_the_single_grid(tmp_path: Path, case: str) -> None:
    want = _single(case, tmp_path / "single")
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, f"{case}: the reference wrote no initial checkpoint to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(want, "r") as f1:
        assert not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()]), f"{case}: the gas never moved; the gate is vacuous"
    cuts = CASES[case][4]
    ndim = len(cuts)
    ph.same_checkpoint(_launch(case, tmp_path / "two_workers", 2, shape=(2,) + (1,) * (ndim - 1)), want, f"{case}: two workers")
    ph.same_checkpoint(_launch(case, tmp_path / "three_uneven_workers", 3, cuts=cuts), want, f"{case}: three uneven workers")
