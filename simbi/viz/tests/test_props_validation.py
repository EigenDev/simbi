# =============================================================================
# test_props_validation.py
#
# component props reject unknown keys by name instead of ignoring them.
# a mistyped prop (cma for cmap) previously validated cleanly and the
# style silently never applied.
# =============================================================================

import pytest

from simbi.viz.config_loader import load_component_props, validate_props


def test_unknown_prop_is_rejected_with_a_close_match_suggestion() -> None:
    with pytest.raises(ValueError, match=r"did you mean 'cmap'"):
        validate_props("polygon", {"cma": "cmr.viola"})


def test_unknown_prop_without_a_close_match_lists_known_props() -> None:
    with pytest.raises(ValueError, match=r"known props"):
        validate_props("polygon", {"zzz_not_a_prop": 1})


def test_known_props_still_validate() -> None:
    props = validate_props("polygon", {"cmap": "viridis"})
    assert props.cmap == "viridis"


def test_cli_override_path_surfaces_the_typo() -> None:
    with pytest.raises(ValueError, match=r"did you mean 'cmap'"):
        load_component_props(overrides=["polygon.cma=cmr.viola"])


def test_direct_construction_forbids_extras() -> None:
    from simbi.viz.props_registry import get_props_class

    with pytest.raises(Exception, match=r"[Ee]xtra"):
        get_props_class("polygon")(cma="viridis")
