# =============================================================================
# test_accretion.py
#
# the accretor analysis laws (docs/ideas/accretor.md §5):
# - the steady detector fires after an exponential transient and returns None on
#   a secular drift
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
    # before t = 20, and the detector reports a time past the transient.
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
    # time-weighted value.
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


def test_dat_loader_reads_the_legacy_2d_schema(tmp_path) -> None:
    from simbi.analysis import load_diagnostics_dat, mdot_from_cumulative

    # two bodies, three samples; body 0 accretes at a constant 0.5/unit time.
    path = tmp_path / "diagnostics.dat"
    header = "# time body x y vx vy fx fy torque_z mass accreted_mass accretion_rate\n"
    rows = []
    for t in [1.0, 2.0, 3.0]:
        rows.append(f"{t} 0 0 0 0 0 {0.1 * t} -0.2 0 1.0 {0.5 * t} 0.5\n")
        rows.append(f"{t} 1 1 1 0 0 0 0 0 2.0 0.0 0.0\n")
    path.write_text(header + "".join(rows))

    d = load_diagnostics_dat(str(path))
    assert d.time.shape == (3,)
    assert d.accreted_mass.shape == (3, 2)
    assert d.force[1, 0, 0] == pytest.approx(0.2)
    # the legacy schema stops at fy, so fz is absent here and comes back nan; a stored 0.0
    # would read as a real measurement.
    assert np.isnan(d.force[1, 0, 2])
    # the cumulative diff gives the exact mean rate regardless of cadence.
    assert mdot_from_cumulative(d.time, d.accreted_mass[:, 0], t_start=1.0) == pytest.approx(0.5)
    assert mdot_from_cumulative(d.time, d.accreted_mass[:, 1], t_start=1.0) == pytest.approx(0.0)


def test_dat_loader_reads_the_three_component_schema(tmp_path) -> None:
    from simbi.analysis import load_diagnostics_dat

    path = tmp_path / "diagnostics.dat"
    header = (
        "# time body x y z vx vy vz fx fy fz torque_x torque_y torque_z "
        "mass accreted_mass accretion_rate\n"
    )
    row = "1.0 0 1 2 3 0.1 0.2 0.3 4 5 6 7 8 9 1.0 0.25 0.5\n"
    path.write_text(header + row)
    d = load_diagnostics_dat(str(path))
    assert d.force[0, 0].tolist() == [4.0, 5.0, 6.0]
    assert d.accreted_mass[0, 0] == pytest.approx(0.25)


def test_dat_loader_rejects_a_headerless_file(tmp_path) -> None:
    from simbi.analysis import load_diagnostics_dat

    path = tmp_path / "diagnostics.dat"
    path.write_text("1.0 0 0 0 0 0 0 0 0 1.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="header"):
        load_diagnostics_dat(str(path))


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


def test_sphere_flux_recovers_the_analytic_rotating_inflow() -> None:
    # a uniform-density cloud in rigid rotation w_z falling radially at speed
    # u: Mdot(r) = 4 pi r^2 rho u exactly, and the z angular momentum carried
    # through the sphere is Ldot_z(r) = 4 pi r^2 rho u * w_z <x^2 + y^2>
    # with the shell average <x^2 + y^2> = 2 r^2 / 3.
    from simbi.analysis import sphere_flux

    rng = np.random.default_rng(7)
    n = 200_000
    pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    r = np.sqrt(np.sum(pos**2, axis=1))
    keep = (r > 0.05) & (r < 1.0)
    pos, r = pos[keep], r[keep]
    rho = np.full(len(pos), 1.3)
    u, w_z = 0.4, 0.9
    vel = -u * pos / r[:, None]
    vel[:, 0] += -w_z * pos[:, 1]
    vel[:, 1] += w_z * pos[:, 0]

    radii = np.array([0.3, 0.5, 0.8])
    mdot, ldot = sphere_flux(pos, rho, vel, radii, shell_width=0.02)
    for i, rs in enumerate(radii):
        assert mdot[i] == pytest.approx(4.0 * np.pi * rs**2 * 1.3 * u, rel=0.02)
        expect_l = 4.0 * np.pi * rs**2 * 1.3 * u * w_z * (2.0 * rs**2 / 3.0)
        assert ldot[i] == pytest.approx(expect_l, rel=0.05)


def test_sphere_flux_flags_empty_shells_with_nan() -> None:
    from simbi.analysis import sphere_flux

    pos = np.array([[0.5, 0.0, 0.0]])
    rho = np.array([1.0])
    vel = np.array([[-0.1, 0.0, 0.0]])
    mdot, ldot = sphere_flux(pos, rho, vel, np.array([0.5, 2.0]), shell_width=0.05)
    assert np.isfinite(mdot[0]) and np.isnan(mdot[1])
    assert np.isnan(ldot[1])


def test_lambda_c_handles_the_monoatomic_edge() -> None:
    # gamma = 5/3 makes the general formula 0/0; the limit is 1/4. also pin
    # continuity: a gamma just below the edge stays near 1/4.
    from simbi.analysis.__main__ import lambda_c

    assert lambda_c(5.0 / 3.0) == pytest.approx(0.25)
    assert lambda_c(5.0 / 3.0 - 1e-4) == pytest.approx(0.25, rel=1e-2)
    assert lambda_c(1.0) == pytest.approx(np.e**1.5 / 4.0)
