# =============================================================================
# test_accretion.py
#
# the accretor analysis laws (docs/ideas/accretor.md §5):
# - the steady detector fires after an exponential transient and never on a
#   secular drift
# - windowed averaging reproduces known dt-weighted means/amplitudes
# - the sonic extractor recovers an exact mach = r_s/r surface, isotropic and
#   shaped, and reports nan where no crossing exists
# - the stagnation crossing interpolates exactly on linear data
# - the h5 loader round-trips the checkpoint group layout
# =============================================================================

import numpy as np
import pytest

from simbi.analysis import (
    averaged_rate,
    load_body_diagnostics,
    sonic_radius_vs_angle,
    stagnation_distance,
    steady_state_time,
)


def test_steady_detector_fires_after_the_transient() -> None:
    # an exponential relaxation to 2.0 with time constant 1: settled well
    # before t = 20, and the detector must not fire inside the transient.
    t = np.linspace(0.0, 40.0, 4000)
    series = 2.0 + 3.0 * np.exp(-t)
    t0 = steady_state_time(t, series, window=5.0, tol=0.01)
    assert t0 is not None
    assert 10.0 <= t0 <= 20.0


def test_steady_detector_rejects_a_secular_drift() -> None:
    t = np.linspace(0.0, 40.0, 4000)
    series = 1.0 + 0.05 * t
    assert steady_state_time(t, series, window=5.0, tol=0.01) is None


def test_averaged_rate_weights_by_dt() -> None:
    # two constant segments with different step sizes: the mean is the
    # time-weighted value, not the sample average.
    time = np.array([1.0, 2.0, 2.5, 3.0])
    dt = np.array([1.0, 1.0, 0.5, 0.5])
    series = np.array([4.0, 4.0, 1.0, 1.0])
    mean, fluct, span = averaged_rate(time, dt, series, t_start=0.0)
    assert mean == pytest.approx((4.0 * 2.0 + 1.0 * 1.0) / 3.0)
    assert fluct > 0.0
    assert span == pytest.approx(3.0)


def test_sonic_surface_recovers_an_exact_spherical_surface() -> None:
    # mach = r_s / r crosses 1 exactly at r_s, every direction.
    rng = np.random.default_rng(7)
    pos = rng.uniform(-1.0, 1.0, size=(20000, 3))
    r = np.linalg.norm(pos, axis=1)
    keep = r > 0.05
    pos, r = pos[keep], r[keep]
    r_s = 0.4
    speed = r_s / r
    cs = np.ones_like(r)
    theta, r_sonic = sonic_radius_vs_angle(pos, speed, cs, nbins=16)
    assert theta.shape == (16,)
    valid = ~np.isnan(r_sonic)
    assert valid.sum() >= 14
    assert np.allclose(r_sonic[valid], r_s, rtol=0.05)


def test_sonic_surface_recovers_an_anisotropic_shape() -> None:
    # r_s(theta) = 0.5 + 0.15 cos(theta): the wind-flattened surface shape.
    rng = np.random.default_rng(11)
    pos = rng.uniform(-1.5, 1.5, size=(60000, 3))
    r = np.linalg.norm(pos, axis=1)
    keep = r > 0.05
    pos, r = pos[keep], r[keep]
    theta_true = np.arccos(np.clip(pos[:, 0] / r, -1, 1))
    r_s = 0.5 + 0.15 * np.cos(theta_true)
    speed = r_s / r
    cs = np.ones_like(r)
    theta, r_sonic = sonic_radius_vs_angle(pos, speed, cs, nbins=12)
    valid = ~np.isnan(r_sonic)
    expected = 0.5 + 0.15 * np.cos(theta[valid])
    assert np.allclose(r_sonic[valid], expected, rtol=0.08)


def test_sonic_surface_reports_nan_without_a_crossing() -> None:
    pos = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    speed = np.array([0.1, 0.1])  # subsonic everywhere
    cs = np.ones(2)
    _, r_sonic = sonic_radius_vs_angle(pos, speed, cs, nbins=4)
    assert np.all(np.isnan(r_sonic))


def test_stagnation_crossing_interpolates_linearly() -> None:
    x = np.array([-4.0, -3.0, -2.5, -1.0])
    u = x + 2.0  # zero at x = -2
    assert stagnation_distance(x, u) == pytest.approx(2.0)
    assert stagnation_distance(x, np.ones_like(x)) is None


def test_loader_round_trips_the_checkpoint_group(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "diag.h5"
    n, nb, d = 3, 1, 2
    with h5py.File(path, "w") as f:
        g = f.create_group("body_diagnostics")
        g.create_dataset("time", data=np.array([0.1, 0.2, 0.3]))
        g.create_dataset("dt", data=np.full(n, 0.1))
        g.create_dataset("mass_delta", data=np.array([[0.01], [0.02], [0.03]]))
        g.create_dataset("energy_delta", data=np.zeros((n, nb)))
        g.create_dataset("force", data=np.zeros((n, nb, d)))
    diag = load_body_diagnostics(str(path))
    assert diag.time.shape == (n,)
    assert diag.mdot.shape == (n, nb)
    assert diag.mdot[1, 0] == pytest.approx(0.2)
