# =============================================================================
# test_spacetime_enum_matches_rust.py
#
# the python `Spacetime` enum and the rust `Spacetime` enum are two hand-maintained
# spellings of one vocabulary, and nothing else compares them.
#
# the failure this closes is a config that validates and then cannot run: a member the
# python side still offers but the backend no longer implements is selectable, passes
# every python-side check, and fails somewhere deep in dispatch -- or worse, is silently
# whitelisted by a validation guard that was never updated alongside the enum. that is
# exactly what happened to the areal `schwarzschild` chart, which outlived its rust
# variant by one refactor.
#
# the dispatch coverage gate does not reach this: it checks that every kernel the
# DISPATCH MATRIX accepts is baked, which says nothing about a name the python layer can
# produce and the rust layer cannot parse.
# =============================================================================

import re
from pathlib import Path

import pytest

from simbi.types import Spacetime

_METRIC_RS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "crates"
    / "symbi-geometry"
    / "src"
    / "metric.rs"
)


def _snake(variant: str) -> str:
    """rust variant name -> the wire string. `SchwarzschildKS` -> `schwarzschild_ks`,
    `KerrKS` -> `kerr_ks`, `Minkowski` -> `minkowski`."""
    return re.sub(r"(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])", "_", variant).lower()


def _rust_variants() -> set[str]:
    """the variants of `pub enum Spacetime` in metric.rs, as wire strings."""
    source = _METRIC_RS.read_text()
    start = source.index("pub enum Spacetime {")
    body = source[start : source.index("\n}", start)]
    # a variant is a bare CamelCase identifier at the head of a line, optionally with a
    # discriminant; attributes and doc comments are skipped by the leading-char filter.
    names = re.findall(r"^\s{4}([A-Z][A-Za-z0-9]*)\s*(?:=\s*\d+\s*)?,", body, re.M)
    return {_snake(n) for n in names}


def test_every_python_spacetime_has_a_rust_variant() -> None:
    if not _METRIC_RS.is_file():
        pytest.skip("the rust workspace is not present in this checkout")

    rust = _rust_variants()
    # PREMISE: the parse actually found the enum. an empty set would make every
    # comparison below vacuously interesting rather than meaningful.
    assert len(rust) >= 3, (
        f"parsed only {rust} from {_METRIC_RS.name} -- the enum shape changed and this "
        "gate is no longer reading it"
    )

    python = {member.value for member in Spacetime}
    orphaned = python - rust
    assert not orphaned, (
        f"these Spacetime members have no rust variant: {sorted(orphaned)}. a config can "
        f"select them and pass validation, then fail in dispatch. rust has {sorted(rust)}"
    )


def test_every_rust_spacetime_is_reachable_from_python() -> None:
    if not _METRIC_RS.is_file():
        pytest.skip("the rust workspace is not present in this checkout")

    rust = _rust_variants()
    python = {member.value for member in Spacetime}
    unreachable = rust - python
    assert not unreachable, (
        f"these rust spacetimes cannot be selected from python: {sorted(unreachable)}. "
        "the backend implements a metric no config can ask for"
    )


def test_the_areal_schwarzschild_chart_is_refused() -> None:
    # the specific regression: the areal chart's coordinate singularity at r = 2M puts the
    # horizon outside any evolvable domain, so it was removed from the backend. it must not
    # be constructible, or an old config resurrects a metric that cannot run.
    with pytest.raises(ValueError):
        Spacetime("schwarzschild")
