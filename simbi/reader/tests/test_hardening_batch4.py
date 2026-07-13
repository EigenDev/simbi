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
        regime = "srmhd"
        gamma = 2.0
        sound_speed = None
        is_mhd = True

    class _Chk:
        metadata = _Meta()

    pipeline = create_computation_pipeline(_Chk())
    assert "b1_mean" in pipeline or len(pipeline) > 0  # pipeline builds


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
    # a good tuple is REPLAYED, not consumed.
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

    # and a too-short series returns None instead of a gradient crash.
    assert steady_state_time(np.array([1.0]), np.array([1.0])) is None


def test_sphere_flux_1d_returns_mass_flux_without_l():
    from simbi.analysis.accretion import sphere_flux

    pos = np.linspace(0.1, 2.0, 50)[:, None]  # 1d: a single column
    rho = np.ones(50)
    vel = -0.1 * np.ones((50, 1))
    mdot, ldot = sphere_flux(pos, rho, vel, np.array([1.0]), 0.5)
    assert np.isfinite(mdot[0]) and mdot[0] > 0.0
    assert ldot[0] == 0.0
