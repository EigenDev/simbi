# =============================================================================
# test_softening_kind_wire.py
#
# the softening-FAMILY wire: a config declaring `softening_kind="compact"` must reach the backend
# as the `gravitational.softening_kind` key the rust body parser reads. the parser resolves the
# family with `sub_str(body, "gravitational", "softening_kind", "plummer")`, so a key that never
# arrives is indistinguishable from one that says "plummer" -- the run proceeds silently on the
# wrong gravitational field.
#
# the two families are not small perturbations of each other. plummer is an extended profile whose
# field sits below newtonian at EVERY radius (0.354 of it at r = h, reaching 0.99 only past r = 5h),
# so a length chosen to keep the field finite near a sink biases gravity across the whole domain.
# compact truncates the source at h: outside it the field is the bare point mass exactly. a
# measurement that fits a power law in radius therefore reads a different exponent depending on
# which family it silently ran, which is why this key is gated rather than trusted.
# =============================================================================

import pytest

from simbi.types.bodies import (
    AccretionProperties,
    BinaryComponentConfig,
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
)


def _accretor(**gravitational) -> dict:
    """the backend wire for a gravitating accretor, the shape a sink config emits."""
    body = ImmersedBodyConfig(
        capability=BodyCapability.GRAVITATIONAL | BodyCapability.ACCRETION,
        mass=1.0,
        radius=0.1,
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        gravitational=GravitationalProperties(**gravitational),
        accretion=AccretionProperties(accretion_radius=0.1),
    )
    return body.to_backend()


@pytest.mark.parametrize("kind", ["compact", "plummer"])
def test_the_declared_softening_family_reaches_the_backend_wire(kind: str) -> None:
    group = _accretor(softening_length=0.1, softening_kind=kind)["gravitational"]
    # the key name is the contract: the rust parser looks up exactly "softening_kind" inside the
    # "gravitational" group, and falls back to plummer when the lookup misses.
    assert "softening_kind" in group, (
        "the gravitational wire group dropped softening_kind; the backend would fall back to "
        "plummer and weaken gravity across the whole domain with no diagnostic"
    )
    assert group["softening_kind"] == kind
    assert group["softening_length"] == 0.1


def test_an_undeclared_family_is_plummer() -> None:
    # the default has to be the historical field, so an existing config's behavior is unchanged
    # by the family becoming selectable.
    group = _accretor(softening_length=0.1)["gravitational"]
    assert group["softening_kind"] == "plummer"


def _binary_component(**kwargs) -> dict:
    """the backend wire for one component of a gravitational binary."""
    return BinaryComponentConfig(
        mass=1.0,
        radius=0.1,
        is_an_accretor=True,
        softening_length=0.1,
        two_way_coupling=False,
        accretion_radius=0.1,
        **kwargs,
    ).to_body_config()


@pytest.mark.parametrize("kind", ["compact", "plummer"])
def test_a_binary_component_carries_the_softening_family(kind: str) -> None:
    # a binary component is serialized by its own emitter rather than by
    # GravitationalProperties, and the backend reads both through the same lookup.
    group = _binary_component(softening_kind=kind)["gravitational"]
    assert group["softening_kind"] == kind
    assert group["softening_length"] == 0.1


def test_both_body_kinds_emit_the_same_gravitational_key_set() -> None:
    # the two paths reach one backend lookup, which falls back to a default on a missing key
    # instead of failing. a key that exists on one emitter and not the other is therefore a
    # silent behavior difference between a standalone body and a binary component, and this is
    # the invariant that makes the two impossible to drift apart.
    standalone = set(_accretor(softening_length=0.1, softening_kind="compact")["gravitational"])
    component = set(_binary_component(softening_kind="compact")["gravitational"])
    assert standalone == component, (
        f"the gravitational wire groups disagree: standalone-only {standalone - component}, "
        f"binary-component-only {component - standalone}"
    )


def test_an_unknown_family_is_refused_on_a_binary_component() -> None:
    # the vocabulary is checked on every config that carries a softening length, not only on
    # the standalone property block.
    with pytest.raises(ValueError, match="softening_kind"):
        _binary_component(softening_kind="compct")


def test_an_unknown_family_is_refused_where_it_was_written() -> None:
    # the backend resolves anything that is not "compact" to plummer, so an unrecognized spelling
    # would otherwise run as plummer silently. the vocabulary is checked in the config layer, where
    # the error can name the line that wrote it.
    with pytest.raises(ValueError, match="softening_kind"):
        GravitationalProperties(softening_length=0.1, softening_kind="compct")
