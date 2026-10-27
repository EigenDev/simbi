# =============================================================================
# test_tiled_body_ledger.py
#
# the body ledger of an accreting sink on host tiles, on the session's backend
# (parity_harness). the spherical bondi sink, on a flat grid and under one
# refined box, on one device and on two tiles cutting through the sink. each
# tile books its own cells' partial of every sink reduction exactly, and the
# partials merge and round once, so on the cpu the tiles publish the single
# grid's run bit for bit:
# - every grid field of every level;
# - the bodies group (mass, accretion, kinematics) and the per-step body
#   history (time, dt, accreted mass and energy, force, normal force, torque);
# - the body diagnostics file, character for character.
# a gpu device rounds its feedback sum once per tile, so its single-device
# comparison carries that rounding.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_body_ledger.py
#      (SIMBI_PARITY_BACKEND selects the session's backend, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch_bodies import _assert_same, _assert_same_diagnostics

pytestmark = pytest.mark.simulation

BASE = 24
STEPS = 8
U = 2.0**-53


def _run(out: Path, gpus: int, refined: bool) -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from simbi_configs.examples.newtonian.bondi import SphericalBondiTest\n"
        f"p = SphericalBondiTest(base_resolution={BASE}, gpus={gpus}, refinement_enabled={refined}, data_directory=sys.argv[1])\n"
        "p.checkpoint_interval = 1e9\n"
        # one refined box around the sink: the tiles carry a body under a single region
        "if p.refinement_enabled:\n"
        "    p.refinement_max_levels = 2\n"
        "    p.refinement_regions = p.refinement_regions[:1]\n"
        "    p.refinement_ratios = p.refinement_ratios[:1]\n"
        f"runner.run(p, compute_mode=sys.argv[2], max_steps={STEPS})\n"
    )
    # host tiles beyond the visible devices share them
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"bondi on {gpus} device(s)", env=env, devices=gpus)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _groups(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """the bodies and body_diagnostics datasets, and the interior cell count over every level"""
    out = {}
    with h5py.File(path, "r") as f:
        for group in ("bodies", "body_diagnostics"):
            if group in f:
                for name in f[group]:
                    out[f"{group}/{name}"] = np.asarray(f[group][name][()], dtype=float)
        cells = 0
        for level in (k for k in f if k.startswith("level_")):
            ng = int(f[level]["mesh"].attrs["halo_width"])
            cells += int(np.prod([s - 2 * ng for s in f[level]["conserved/den"].shape]))
    return out, cells


def _pull(path: Path) -> tuple[float, float]:
    """the unsummed gravitational pull of the gas on the sink over every level's cells (covered
    cells counted on both levels, an upper bound), and the largest lever arm"""
    import importlib

    bondi = importlib.import_module("simbi_configs.examples.newtonian.bondi").SphericalBondiTest(base_resolution=BASE)
    gm, eps = bondi.central_mass, 0.01 * bondi.bondi_radius
    pull, arm = 0.0, 0.0
    with h5py.File(path, "r") as f:
        for level in (k for k in f if k.startswith("level_")):
            ng = int(f[level]["mesh"].attrs["halo_width"])
            rho = f[level]["conserved/den"][()][ng:-ng, ng:-ng, ng:-ng]
            axes = []
            for ax in range(3):
                dim = f[level][f"mesh/geometry/dim_{ax}"].attrs
                n = rho.shape[2 - ax]
                width = (dim["end"] - dim["start"]) / n
                axes.append((dim["start"] + (np.arange(n) + 0.5) * width, width))
            (x, dx), (y, dy), (z, dz) = axes
            r2 = z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2
            pull += gm * float(np.sum(rho * dx * dy * dz / (r2 + eps * eps)))
            arm = max(arm, float(np.sqrt(r2.max())))
    return pull, arm


def _diagnostics(out: Path) -> np.ndarray:
    path = out / "diagnostics.dat"
    assert path.exists(), f"no body diagnostics file in {out}: {sorted(p.name for p in out.iterdir())}"
    return np.atleast_2d(np.loadtxt(path))


@pytest.mark.parametrize("refined", [False, True], ids=["flat", "refined"])
def test_tiles_publish_the_single_grid_body_ledger(tmp_path: Path, refined: bool) -> None:
    want = _run(tmp_path / "single", 1, refined)
    got = _run(tmp_path / "tiles", 2, refined)
    history, _ = _groups(want)
    assert "body_diagnostics/energy_delta" in history, f"the single grid published no body history: {sorted(history)}"
    assert np.abs(history["body_diagnostics/energy_delta"]).max() > 0.0, "the sink accreted no energy; the ledger is vacuous"
    assert len(_diagnostics(tmp_path / "single")) > 1, "the run wrote one diagnostics row; the cadence is never exercised"
    if refined:
        with h5py.File(want, "r") as f:
            assert "level_1" in f, "the single grid published no refined level"
    ph.disagreement(_assert_same, got, want, "two host tiles against the single grid")
    ph.disagreement(_assert_same_diagnostics, tmp_path / "tiles", tmp_path / "single", "two host tiles against the single grid")
