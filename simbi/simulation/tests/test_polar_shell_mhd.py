# =============================================================================
# test_polar_shell_mhd.py
#
# newtonian MHD on a 3D spherical shell spanning both poles through the public
# runner. every azimuthal copy of a radial edge on the axis is one physical
# segment, and the stage reconciles the copies of each ring to one EMF, so:
# - under contact and under uct the unnormalized discrete magnetic flux out of
#   every cell, weighted by the constrained-transport face weights, is held to
#   roundoff over the run, the two pole rows included;
# - a run checkpointed halfway resumes to the uninterrupted state bit for bit;
# - host tiles cutting the azimuth into two halves reproduce the single grid
#   bit for bit.
#
# run: pytest -m simulation simbi/simulation/tests/test_polar_shell_mhd.py
# =============================================================================
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi_configs.examples.newtonian.nmhd_polar_shell_3d import NewtonianPolarShell3D

pytestmark = pytest.mark.simulation

RESOLUTION = (8, 8, 16)
STEPS = 10


def _run(out: Path, steps: int, ct: str = "contact", gpus: int = 1, decompose=None, checkpoint: Path | None = None) -> Path:
    kwargs = dict(resolution=RESOLUTION, data_directory=str(out), checkpoint_interval=1e9, ct_method=ct, gpus=gpus)
    if decompose is not None:
        kwargs["decompose"] = decompose
    if checkpoint is not None:
        kwargs["checkpoint_file"] = str(checkpoint)
    runner.run(NewtonianPolarShell3D(**kwargs), compute_mode="cpu", max_steps=steps)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _initial(out: Path) -> Path:
    for p in sorted(out.glob("*.h5")):
        with h5py.File(p, "r") as f:
            if int(f["level_0"].attrs["iteration"]) == 0:
                return p
    raise AssertionError(f"no initial checkpoint in {out}")


def _cell_flux(path: Path) -> tuple[np.ndarray, float]:
    """the unnormalized flux out of every cell with the CT face weights of the orthonormal field:
    r faces r_f^2 sin(theta_c) dtheta dphi, theta faces r_c sin(theta_f) dr dphi, phi faces
    r_c dr dtheta; and the largest weighted face flux, the scale a change is roundoff against"""
    with h5py.File(path, "r") as f:
        m = f["level_0/partition_0/hydro/magnetic"]
        b1, b2, b3 = (m[f"B{k}/data"][()] for k in (1, 2, 3))
    nr, nt, np_ = RESOLUTION
    r = np.linspace(1.0, 2.0, nr + 1)
    t = np.linspace(0.0, math.pi, nt + 1)
    dr, dt, dp = r[1] - r[0], t[1] - t[0], 2.0 * math.pi / np_
    rc, tc = 0.5 * (r[1:] + r[:-1]), 0.5 * (t[1:] + t[:-1])
    assert b1.shape == (np_, nt, nr + 1) and b2.shape == (np_, nt + 1, nr) and b3.shape[1:] == (nt, nr), (b1.shape, b2.shape, b3.shape)
    w1 = b1 * (r[None, None, :] ** 2) * (np.sin(tc) * dt * dp)[None, :, None]
    w2 = b2 * np.sin(t)[None, :, None] * (rc * dr * dp)[None, None, :]
    w3 = b3[:np_ + 1] * (rc * dr * dt)[None, None, :]
    flux = (w1[:, :, 1:] - w1[:, :, :-1]) + (w2[:, 1:, :] - w2[:, :-1, :]) + (w3[1:] - w3[:-1])
    scale = max(np.abs(w1).max(), np.abs(w2).max(), np.abs(w3).max())
    return flux, scale


def _assert_same_state(a: Path, b: Path, label: str) -> None:
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for key in ("time", "iteration"):
            assert fa["level_0"].attrs[key] == fb["level_0"].attrs[key], f"{label}: {key}"
        names: list[str] = []
        fb["level_0/partition_0"].visit(lambda n: names.append(n) if isinstance(fb["level_0/partition_0"][n], h5py.Dataset) else None)
        assert names, f"{label}: no datasets"
        for n in names:
            x, y = fa["level_0/partition_0"][n][()], fb["level_0/partition_0"][n][()]
            assert x.shape == y.shape and np.array_equal(x, y), f"{label}: {n} differs"


@pytest.fixture
def host_tiles(monkeypatch):
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")


@pytest.mark.parametrize("ct", ["contact", "uct"])
def test_every_cells_flux_is_held_at_both_poles(tmp_path: Path, ct: str) -> None:
    final = _run(tmp_path, STEPS, ct=ct)
    before, scale = _cell_flux(_initial(tmp_path))
    after, _ = _cell_flux(final)
    change = np.abs(after - before)
    # 1e-12 of the scale: the per-cell sums round once per stage (measured ~1e-16 of the scale)
    for label, rows in (("theta = 0 pole row", 0), ("theta = pi pole row", -1)):
        assert change[:, rows, :].max() <= 1e-12 * scale, f"{ct}: the {label} changed its flux by {change[:, rows, :].max():e}"
    assert change.max() <= 1e-12 * scale, f"{ct}: a cell changed its flux by {change.max():e}"


def test_a_resumed_run_continues_bit_for_bit(tmp_path: Path) -> None:
    whole = _run(tmp_path / "whole", STEPS)
    half = _run(tmp_path / "half", STEPS // 2)
    # the runner's step limit on a resumed run counts the checkpoint's iterations too
    resumed = _run(tmp_path / "resumed", STEPS, checkpoint=half)
    _assert_same_state(resumed, whole, "resumed against uninterrupted")


def test_azimuth_halves_match_the_single_grid(tmp_path: Path, host_tiles) -> None:
    single = _run(tmp_path / "single", STEPS)
    tiles = _run(tmp_path / "tiles", STEPS, gpus=2, decompose=[[], [], [8]])
    _assert_same_state(tiles, single, "two azimuth tiles")
