# =============================================================================
# test_tiled_body_ledger.py
#
# the body ledger of an accreting sink on host tiles. the spherical bondi sink,
# on a flat grid and under one refined box, on one device and on two tiles
# cutting through the sink:
# - the per-step body history (time, dt, accreted mass and energy, force,
#   normal force, torque) publishes on the tiles as on the single grid;
# - the body diagnostics file carries the same rows, to its printed precision;
# - the bodies group (mass, accretion, kinematics) agrees.
# the clock columns agree bit for bit. a tile books its own cells' partial of
# each sink reduction and the tiles sum the partials, so the summed columns
# agree within the reassociation bound of a sum over the grid's n cells split
# in two: n u of the sum of the terms' magnitudes, u = 2^-53. the accreted mass
# and energy are sums of terms of one sign, so that sum is the column itself.
# the force and torque on a sink at the center of a spherical inflow cancel to
# roundoff; their terms are bounded by the unsummed pull of the gas,
# G M sum rho dV / (r^2 + eps^2), and that pull times the largest lever arm.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_body_ledger.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

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
        f"runner.run(p, compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"bondi on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
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
    want, cells = _groups(_run(tmp_path / "single", 1, refined))
    got, _ = _groups(_run(tmp_path / "tiles", 2, refined))
    assert "body_diagnostics/energy_delta" in want, f"the single grid published no body history: {sorted(want)}"
    assert np.abs(want["body_diagnostics/energy_delta"]).max() > 0.0, "the sink accreted no energy; the ledger is vacuous"
    assert got.keys() == want.keys(), f"datasets {sorted(set(got) ^ set(want))} differ"
    for name in ("body_diagnostics/time", "body_diagnostics/dt"):
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    bound = cells * U
    pull, arm = _pull(next((tmp_path / "single").glob("*final*.h5")))
    for name, x in want.items():
        if "force" in name:
            scale = pull
        elif "torque" in name:
            scale = pull * arm
        else:
            scale = np.abs(x).max()
        worst = np.abs(got[name] - x).max()
        assert worst <= bound * scale, f"{name}: tiles differ by {worst:.3e} against {bound:.1e} of {scale:.3e}"
    rows, single_rows = _diagnostics(tmp_path / "tiles"), _diagnostics(tmp_path / "single")
    assert len(single_rows) > 1, "the run wrote one diagnostics row; the cadence is never exercised"
    assert rows.shape == single_rows.shape, f"diagnostics rows {rows.shape} against {single_rows.shape}"
    np.testing.assert_array_equal(rows[:, 0], single_rows[:, 0], err_msg="diagnostics times")
    # the file prints nine significant digits (%.8e), so each printed value carries its own
    # rounding of 1e-8 relative on top of the reassociation bound; the force (fx, fy, fz) and
    # torque columns take the bound on their unsummed scales, as in the history
    unsummed = np.zeros(rows.shape[1])
    unsummed[8:11], unsummed[11:14] = pull, pull * arm
    tolerance = 1e-8 * np.maximum(np.abs(rows), np.abs(single_rows)) + bound * unsummed
    worst = np.abs(rows - single_rows) - tolerance
    assert np.all(worst <= 0.0), f"diagnostics rows differ beyond the printed precision: column {int(np.argmax(worst.max(axis=0)))}"
