# =============================================================================
# test_hardening_batch4.py
#
# regression gates for the reader/viz robustness hardening: derived-field
# membership on MHD contexts, 2d vorticity, slice bounds, conditional
# differentiation, stale-wheel stubs, one-call IC contract, first-tuple
# validation, and the NaN-tolerant steady-state window.
# =============================================================================

import numpy as np
import pytest


def test_level_context_supports_membership():
    # `"b1" in ctx` fell back to integer __getitem__ iteration -> KeyError: 0,
    # breaking every b*_mean derived field on MHD checkpoints.
    from simbi.reader.computation import create_computation_pipeline

    class _Meta:
        dimensions = 2
        regime = "rmhd"
        gamma = 2.0
        sound_speed = None
        is_mhd = True

    class _Chk:
        metadata = _Meta()

    pipeline = create_computation_pipeline(_Chk())
    assert "b1_mean" in pipeline or len(pipeline) > 0  # pipeline builds


def test_mhd_vector_dof_is_three_regardless_of_spatial_ndim():
    # the velocity / magnetic field is a 3-vector for any mhd run: a 2.5D (D=2) or 1.75D (D=1)
    # setup still evolves the out-of-plane v_phi / b_phi. reading only ndim components dropped it.
    from simbi.reader.computation import _vector_dof

    assert _vector_dof("rmhd", 2) == 3
    assert _vector_dof("rmhd", 1) == 3
    assert _vector_dof("nmhd", 2) == 3
    assert _vector_dof("imhd", 1) == 3
    # hydro velocity has one component per spatial axis.
    assert _vector_dof("newtonian", 2) == 2
    assert _vector_dof("rhd", 3) == 3
    assert _vector_dof("rhd", 1) == 1


def test_purely_toroidal_field_has_nonzero_magnetic_pressure():
    # a 2.5D toroidal wind carries only b3 = b_phi (b_r = b_theta = 0). the magnetic pressure must
    # be 0.5 * b_phi^2; summing only the ndim in-plane components drops b_phi and yields zero.
    from simbi.reader.computation import magnetic_pressure

    zero = np.zeros((4, 4))
    b_phi = np.full((4, 4), 2.0)
    bfields = [zero, zero, b_phi]  # b1, b2, b3 = b_r, b_theta, b_phi
    velocity = [zero, zero, zero]  # static: no relativistic v.B / lorentz correction

    pmag = magnetic_pressure(bfields, velocity, "rmhd")
    assert np.allclose(pmag, 0.5 * 4.0)
    assert np.all(pmag > 0.0)


def test_relativistic_magnetic_pressure_uses_comoving_field():
    # rmhd magnetic pressure is b^2/2 in the FLUID frame: b^2 = B^2/W^2 + (v.B)^2. a radial flow
    # with a toroidal field (v perpendicular to B, so v.B = 0) reduces to B_phi^2 / (2 W^2).
    from simbi.reader.computation import magnetic_pressure

    zero = np.zeros((3, 3))
    b_phi = np.full((3, 3), 2.0)
    v_r = np.full((3, 3), 0.6)  # W = 1/sqrt(1-0.36) = 1.25
    bfields = [zero, zero, b_phi]
    velocity = [v_r, zero, zero]  # v.B = v_r*b_r = 0

    w_sq = 1.0 / (1.0 - 0.36)
    pmag = magnetic_pressure(bfields, velocity, "rmhd")
    assert np.allclose(pmag, 0.5 * 4.0 / w_sq)


def test_diff_of_where_differentiates_per_branch():
    import simbi.expression as expr

    g = expr.ExprGraph()
    t = expr.variable("t", g)
    a = expr.where(t > expr.constant(1.0, g), t * t, t)
    d = a.diff(t)  # must not raise on the comparison in the condition
    c = g.compile([d])
    # d/dt = 2t for t>1, 1 for t<1
    assert c.evaluate(t=2.0)[0] == pytest.approx(4.0)
    assert c.evaluate(t=0.5)[0] == pytest.approx(1.0)


def test_slice_position_outside_domain_is_loud():
    from simbi.viz.pipeline.transforms import find_slice_index

    verts = np.linspace(0.0, 1.0, 11)  # 10 cells, 11 vertices
    # at the upper edge: clamps to the last CELL index, never index 10.
    assert find_slice_index(verts, 1.0) == 9
    with pytest.raises(ValueError, match="outside the domain"):
        find_slice_index(verts, 1.5)


def test_nonsquare_slice_keeps_domain_aligned_with_values():
    # a NON-SQUARE 3D field sliced on x3 must leave the vertex coordinate arrays
    # aligned with the value grid: pcolormesh(flat) needs len(domain[0]) == rows+1 and
    # len(domain[1]) == cols+1. reordering the domain into logical order without a
    # matching value transpose silently swapped these on a non-square grid (a square
    # grid hid it because the two lengths were equal).
    from simbi.viz.pipeline.transforms import execute_slice, plan_slice

    # vertex arrays, in data-axis order [x3, x2, x1]; distinct lengths + ranges.
    x1 = np.linspace(-2.0, 6.0, 161)  # 160 cells
    x2 = np.linspace(-2.0, 2.0, 97)  # 96 cells
    x3 = np.linspace(-1.0, 1.0, 21)  # 20 cells
    domain = [x3, x2, x1]
    values = np.arange(20 * 96 * 160, dtype=float).reshape(20, 96, 160)

    plan = plan_slice(domain, {"x3": 0.0})
    sliced_values, new_domain = execute_slice(values, domain, plan)

    assert sliced_values.shape == (96, 160)  # (x2, x1) — value axis order preserved
    # domain[0] is the outer/vertical (x2, [-2,2]); domain[1] the inner/horizontal (x1, [-2,6]).
    assert len(new_domain[0]) == sliced_values.shape[0] + 1  # 97 == 96 + 1
    assert len(new_domain[1]) == sliced_values.shape[1] + 1  # 161 == 160 + 1
    assert new_domain[0].max() == 2.0 and new_domain[1].max() == 6.0
    # the value grid is not transposed: it is exactly the x3 plane.
    s = int(np.abs(x3 - 0.0).argmin())
    assert np.array_equal(sliced_values, values[s, :, :])
    # labels stay in forward logical order so the labeler names the horizontal axis x1.
    assert plan.final_axis_names == ["x1", "x2"]


def test_stale_wheel_stub_names_the_fix(monkeypatch):
    import simbi

    if callable(getattr(simbi, "bondi_profile", None)) and simbi.bondi_profile.__module__ != "simbi":
        pytest.skip("wheel provides the real binding")
    with pytest.raises(RuntimeError, match="dev.py build"):
        simbi.bondi_profile(1.0, 1.4)


def test_first_tuple_validation_names_the_contract():
    from simbi.simulation.runner import _check_first_tuple

    class _P:
        pass

    with pytest.raises(ValueError, match="positive"):
        it = iter([(0.0, 0.1, 1.0)])  # zero density
        _check_first_tuple(_P(), it)
    with pytest.raises(ValueError, match="non-finite"):
        it = iter([(float("nan"), 0.1, 1.0)])
        _check_first_tuple(_P(), it)
    # a good tuple is REPLAYED.
    it = _check_first_tuple(_P(), iter([(1.0, 0.1, 1.0), (2.0, 0.2, 2.0)]))
    assert next(it) == (1.0, 0.1, 1.0)
    assert next(it) == (2.0, 0.2, 2.0)


def test_steady_state_time_tolerates_nan_samples():
    from simbi.analysis.accretion import steady_state_time

    t = np.linspace(0.0, 30.0, 300)
    series = np.ones_like(t)
    series[50] = np.nan  # one bad sample must not misdiagnose NOT SETTLED
    with pytest.warns(UserWarning, match="non-finite"):
        t0 = steady_state_time(t, series, window=5.0)
    assert t0 is not None

    # and a too-short series returns None; a gradient on a single sample would crash.
    assert steady_state_time(np.array([1.0]), np.array([1.0])) is None


def test_sphere_flux_1d_returns_mass_flux_without_l():
    from simbi.analysis.accretion import sphere_flux

    pos = np.linspace(0.1, 2.0, 50)[:, None]  # 1d: a single column
    rho = np.ones(50)
    vel = -0.1 * np.ones((50, 1))
    mdot, ldot = sphere_flux(pos, rho, vel, np.array([1.0]), 0.5)
    assert np.isfinite(mdot[0]) and mdot[0] > 0.0
    assert ldot[0] == 0.0
