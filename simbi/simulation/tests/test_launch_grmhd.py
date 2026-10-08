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
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_grmhd.py
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
    module, cls, fields, _, _ = CASES[case]
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {cls}\n"
        f"p = {cls}({fields}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, f"{case} on one device failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _launch(case: str, out: Path, layout: Path, workers: int) -> Path:
    module, _, _, flags, _ = CASES[case]
    cmd = [sys.executable, "-m", "simbi.cli", "launch", module.replace(".", "/") + ".py", "--layout", str(layout), "--max-steps", str(STEPS),
           *flags, "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_LAUNCH_EXPERIMENTAL"}
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


@pytest.mark.parametrize("case", list(CASES))
def test_curved_magnetized_launches_match_the_single_grid(tmp_path: Path, case: str) -> None:
    want = _single(case, tmp_path / "single")
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, f"{case}: the reference wrote no initial checkpoint to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(want, "r") as f1:
        assert not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()]), f"{case}: the gas never moved; the gate is vacuous"
    cuts = CASES[case][4]
    ndim = len(cuts)
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, shape=(2,) + (1,) * (ndim - 1))),
        ("three uneven workers", 3, _layout(tmp_path / "three.toml", workers=3, cuts=cuts)),
    ]:
        _assert_same_state(_launch(case, tmp_path / label.replace(" ", "_"), layout, workers), want, f"{case}: {label}")
