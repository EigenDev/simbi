# =============================================================================
# test_velocity_relaxation_migration.py
#
# the RELAX -> VELOCITY_RELAXATION vocabulary migration. `VELOCITY_RELAXATION` is
# the canonical enum member and `velocity_relaxation(...)` the canonical
# constructor; `RELAX` is an identity-equal deprecated alias and `relax(...)` a
# deprecated constructor that warns and delegates exactly. the serialized wire
# value stays `"relax"`, so canonical and legacy inputs produce byte-identical
# source payloads.
# =============================================================================
import warnings

import pytest

import simbi.expression as expr
from simbi.expression.dag_expression import SourceKind


def _outputs():
    (x1,) = expr.coords(1)
    return [expr.constant(0.5, x1), expr.constant(1.0, x1)]  # [rate, v_0]


def test_relax_is_an_identity_equal_alias_of_velocity_relaxation():
    assert SourceKind.RELAX is SourceKind.VELOCITY_RELAXATION
    assert SourceKind.VELOCITY_RELAXATION.value == "relax"
    assert SourceKind.RELAX.value == "relax"


def test_iteration_exposes_the_canonical_member_without_duplicating_the_value():
    names = [m.name for m in SourceKind]
    assert "VELOCITY_RELAXATION" in names
    assert "RELAX" not in names  # aliases are excluded from iteration
    # exactly one member carries the "relax" value in iteration.
    assert sum(m.value == "relax" for m in SourceKind) == 1


def test_the_canonical_constructor_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        expr.velocity_relaxation(_outputs(), dim=1)


def test_the_deprecated_constructor_warns_once_and_delegates():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        payload = expr.relax(_outputs(), dim=1)
    deprecations = [w for w in rec if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "velocity_relaxation" in str(deprecations[0].message)
    assert payload["kind"] == "relax"


def test_canonical_and_legacy_constructors_serialize_identically():
    canonical = expr.velocity_relaxation(_outputs(), dim=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = expr.relax(_outputs(), dim=1)
    assert canonical == legacy
    assert canonical["kind"] == "relax"
