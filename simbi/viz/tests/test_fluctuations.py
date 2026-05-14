# =============================================================================
# test_fluctuations.py
#
# unit tests for reynolds-decomposition helpers in reader.computation.
# validates that delta_<field> = field - <field> and that the relative
# form (field - <field>) / <field> is computed correctly.
# =============================================================================
import numpy as np
import pytest

from simbi.reader.computation import (
    make_fluctuation,
    make_relative_fluctuation,
)


class TestMakeFluctuation:
    """signed fluctuation: q' = q - <q>."""

    def test_constant_field_is_zero(self):
        ctx = {"rho": np.full((8, 8), 3.14)}
        compute = make_fluctuation(lambda c: c["rho"])
        result = compute(ctx)
        assert np.allclose(result, 0.0)

    def test_mean_of_output_is_zero(self):
        rng = np.random.default_rng(0)
        ctx = {"p": rng.normal(loc=5.0, scale=1.0, size=(32, 32))}
        compute = make_fluctuation(lambda c: c["p"])
        result = compute(ctx)
        # reynolds decomposition: <q'> = 0 by construction
        assert np.isclose(np.mean(result), 0.0, atol=1e-12)

    def test_preserves_shape(self):
        ctx = {"rho": np.arange(24).reshape(2, 3, 4).astype(float)}
        compute = make_fluctuation(lambda c: c["rho"])
        result = compute(ctx)
        assert result.shape == (2, 3, 4)

    def test_signed_fluctuation(self):
        # q = [1, 2, 3, 4] -> <q> = 2.5 -> q' = [-1.5, -0.5, 0.5, 1.5]
        ctx = {"rho": np.array([1.0, 2.0, 3.0, 4.0])}
        compute = make_fluctuation(lambda c: c["rho"])
        result = compute(ctx)
        expected = np.array([-1.5, -0.5, 0.5, 1.5])
        assert np.allclose(result, expected)

    def test_getter_can_derive(self):
        # getter may derive the field rather than look up a key
        ctx = {"v1": np.array([3.0, 4.0]), "v2": np.array([4.0, 3.0])}
        compute = make_fluctuation(
            lambda c: np.sqrt(c["v1"] ** 2 + c["v2"] ** 2)
        )
        # |v| = [5, 5] -> fluctuation is zero
        result = compute(ctx)
        assert np.allclose(result, 0.0)


class TestMakeRelativeFluctuation:
    """relative fluctuation: (q - <q>) / <q>."""

    def test_unit_mean_matches_signed(self):
        # when <q> == 1, relative form equals signed form
        ctx = {"rho": np.array([0.5, 1.0, 1.5])}
        rel = make_relative_fluctuation(lambda c: c["rho"])
        sgn = make_fluctuation(lambda c: c["rho"])
        assert np.allclose(rel(ctx), sgn(ctx))

    def test_known_values(self):
        # q = [2, 4] -> <q> = 3 -> (q - <q>)/<q> = [-1/3, 1/3]
        ctx = {"p": np.array([2.0, 4.0])}
        compute = make_relative_fluctuation(lambda c: c["p"])
        result = compute(ctx)
        expected = np.array([-1.0 / 3.0, 1.0 / 3.0])
        assert np.allclose(result, expected)

    def test_zero_mean_guarded(self):
        # mean zero -> tiny denominator prevents inf/nan
        ctx = {"rho": np.array([-1.0, 1.0])}
        compute = make_relative_fluctuation(lambda c: c["rho"])
        result = compute(ctx)
        assert np.all(np.isfinite(result))

    def test_preserves_shape(self):
        ctx = {"p": np.ones((5, 6, 7)) * 2.0}
        compute = make_relative_fluctuation(lambda c: c["p"])
        result = compute(ctx)
        assert result.shape == (5, 6, 7)


class TestFluctuationLabels:
    """delta_<field> names resolve to latex labels."""

    def test_delta_rho_label(self):
        from simbi.viz.utility import get_field_str

        assert get_field_str("delta_rho") == r"$\delta \rho$"

    def test_delta_p_label(self):
        from simbi.viz.utility import get_field_str

        assert get_field_str("delta_p") == r"$\delta p$"

    def test_delta_rel_rho_label(self):
        from simbi.viz.utility import get_field_str

        assert (
            get_field_str("delta_rel_rho")
            == r"$\delta \rho / \langle \rho \rangle$"
        )

    def test_unknown_delta_falls_through(self):
        # unknown names fall through to $<name>$ form (existing behavior)
        from simbi.viz.utility import get_field_str

        assert get_field_str("delta_foo") == "$delta_foo$"
