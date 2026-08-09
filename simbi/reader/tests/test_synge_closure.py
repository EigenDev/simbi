# =============================================================================
# test_synge_closure.py
#
# post-processing must compute derived fields with the equation of state the run was
# integrated with. the synge (taub-mathews) closure is PARAMETER-FREE, and the `gamma`
# attribute on its checkpoints is an inert placeholder written only so the plumbing has a
# float to carry; reading it as an adiabatic index computes a different gas than the one
# that was evolved — at p / rho = 20 the gamma-law enthalpy is 51 against the true 80.02,
# a 36% error in every energy, momentum and enthalpy field derived from it.
#
# the gates below pin the closure selection, the two thermodynamic limits of the
# taub-mathews gas, and the exact taub identity it saturates.
# =============================================================================

from types import SimpleNamespace

import numpy as np
import pytest

from simbi.reader.computation import (
    FieldComputationError,
    closure_of,
    enthalpy_density,
    gamma_law_t,
    labframe_energy_density,
    sound_speed,
    spec_enthalpy,
    taub_mathews_t,
)

# the placeholder a synge run records, and a state deep in the taub-mathews walk.
PLACEHOLDER_GAMMA = 5.0 / 3.0
RHO = np.array([1.0])
PRE = np.array([20.0])
THETA = 20.0


def _scalar(a) -> float:
    """the single value of a one-cell field."""
    return float(np.asarray(a).ravel()[0])


def _meta(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**{"gamma": PLACEHOLDER_GAMMA, "eos": "ideal", **kwargs})


def test_synge_metadata_selects_the_parameter_free_closure() -> None:
    assert closure_of(_meta(eos="synge")) == taub_mathews_t()
    assert closure_of(_meta(eos="ideal")) == gamma_law_t(PLACEHOLDER_GAMMA)


def test_a_checkpoint_predating_the_eos_attribute_reads_as_gamma_law() -> None:
    # an empty slug means "not recorded", and only gamma-law runs predate the attribute.
    assert closure_of(_meta(eos="")) == gamma_law_t(PLACEHOLDER_GAMMA)
    assert closure_of(SimpleNamespace(gamma=1.4)) == gamma_law_t(1.4)


def test_taub_mathews_enthalpy_saturates_the_taub_inequality() -> None:
    # (h - theta)(h - 4 theta) = 1 is an identity for this gas, not an approximation;
    # it holds at every temperature and is the sharpest available check that the closed
    # form was transcribed correctly.
    #
    # the residual is bounded by the CONDITIONING of the check rather than by a flat
    # number: h -> 4 theta hot, so `h - 4 theta` is a cancelling difference of two values
    # of size 4 theta. perturbing h by one ulp moves the product by
    # |2h - 5 theta| * eps * h -> 12 eps theta^2, and by 2 eps in the cold limit where
    # h -> 1. the bound below is 5 eps (1 + theta^2), which the exact closed form clears
    # with 3-5x of margin across twelve decades; an enthalpy that is merely CLOSE to
    # taub-mathews misses it by orders of magnitude at every theta.
    eos = taub_mathews_t()
    rho = np.ones(13)
    theta = np.logspace(-6.0, 6.0, 13)
    h = eos.specific_enthalpy(rho, theta * rho)
    residual = np.abs((h - theta) * (h - 4.0 * theta) - 1.0)
    allowed = 5.0 * np.finfo(float).eps * (1.0 + theta**2)
    worst = float((residual / allowed).max())
    assert worst < 1.0, (
        f"the taub identity is violated beyond roundoff (worst residual is {worst:.2f} "
        f"of the conditioning bound); the enthalpy is not the taub-mathews one"
    )


def test_taub_mathews_enthalpy_walks_between_its_two_gamma_law_limits() -> None:
    # the effective index runs from 5/3 cold to 4/3 hot, so h must approach the 5/3
    # gamma law at low theta and the 4/3 one at high theta — and sit between them in
    # neither place at theta ~ 1, which is what makes the closure worth having.
    tm = taub_mathews_t()
    cold, hot = gamma_law_t(5.0 / 3.0), gamma_law_t(4.0 / 3.0)
    rho = np.array([1.0])

    for theta, limit in ((1e-6, cold), (1e6, hot)):
        pre = np.array([theta])
        err = abs(_scalar(tm.specific_enthalpy(rho, pre) / limit.specific_enthalpy(rho, pre)) - 1.0)
        assert err < 1e-5, f"theta = {theta:g}: h departs its limit by {err:.3e}"

    mid = np.array([1.0])
    assert abs(_scalar(tm.specific_enthalpy(rho, mid) / cold.specific_enthalpy(rho, mid)) - 1.0) > 0.05
    assert abs(_scalar(tm.specific_enthalpy(rho, mid) / hot.specific_enthalpy(rho, mid)) - 1.0) > 0.05


def test_derived_fields_on_a_synge_checkpoint_are_not_the_placeholder_gamma_ones() -> None:
    synge = closure_of(_meta(eos="synge"))
    placeholder = gamma_law_t(PLACEHOLDER_GAMMA)
    h_exact = 2.5 * THETA + np.sqrt(2.25 * THETA**2 + 1.0)

    h = spec_enthalpy(synge, RHO, PRE, "rhd")
    assert np.allclose(h, h_exact, rtol=1e-12)
    # the gate is only meaningful where the two closures disagree.
    h_placeholder = spec_enthalpy(placeholder, RHO, PRE, "rhd")
    assert abs(_scalar(h / h_placeholder) - 1.0) > 0.1, (
        f"the closures agree at theta = {THETA} (h = {_scalar(h)} vs "
        f"{_scalar(h_placeholder)}); this gate cannot detect the placeholder being read"
    )

    vel = [np.array([0.0])]
    for field, args in (
        (labframe_energy_density, (RHO, PRE, vel, [], None, "rhd")),
        (enthalpy_density, (RHO, PRE, [], vel, None, "rhd")),
    ):
        eos_slot = args.index(None)
        with_synge = field(*args[:eos_slot], synge, *args[eos_slot + 1 :])
        with_placeholder = field(*args[:eos_slot], placeholder, *args[eos_slot + 1 :])
        assert abs(_scalar(with_synge / with_placeholder) - 1.0) > 0.1, (
            f"{field.__name__} is unchanged by the closure; it is reading the "
            f"placeholder adiabatic index"
        )


def test_the_relativistic_sound_speed_stays_subluminal() -> None:
    # a^2 = gamma p / rho alone passes 1 once theta exceeds 1 / gamma, which is the
    # ordinary state of a relativistic blast; the sound speed of the same gas is a^2 / h
    # and is bounded by 1 / sqrt(3).
    for eos in (taub_mathews_t(), gamma_law_t(4.0 / 3.0), gamma_law_t(5.0 / 3.0)):
        newtonian_form = np.sqrt(eos.sound_speed_sq(RHO, PRE))
        assert newtonian_form.max() > 1.0, (
            f"{type(eos).__name__}: theta = {THETA} no longer makes the newtonian form "
            f"superluminal, so this gate proves nothing"
        )
        cs = sound_speed(RHO, PRE, eos, "rhd")
        assert cs.max() < 1.0, f"{type(eos).__name__}: sound speed {cs.max()} >= c"

    # the ultrarelativistic limit of the taub-mathews gas.
    hot = sound_speed(np.array([1.0]), np.array([1e6]), taub_mathews_t(), "rhd")
    assert abs(_scalar(hot) - 1.0 / np.sqrt(3.0)) < 1e-6


def test_a_newtonian_checkpoint_refuses_the_relativistic_closure() -> None:
    # the parameter-free closure is rejected off the rhd regime at configuration time, so
    # a newtonian file carrying it means the regime and eos attributes disagree; the
    # alternative to failing here is inventing an adiabatic index it does not have.
    with pytest.raises(FieldComputationError, match="no adiabatic index"):
        spec_enthalpy(taub_mathews_t(), RHO, PRE, "newtonian")


def test_a_synge_pipeline_never_consults_an_adiabatic_index() -> None:
    # the wiring gate: a metadata object with NO gamma attribute at all. the pipeline
    # built for a parameter-free closure has nothing to read it for, so anything that
    # still reaches for one fails here rather than silently using a placeholder.
    from simbi.reader.computation import create_computation_pipeline

    class _Meta:
        dimensions = 1
        regime = "rhd"
        eos = "synge"
        sound_speed = None
        is_mhd = False

    class _Chk:
        metadata = _Meta()

    assert len(create_computation_pipeline(_Chk())) > 0
