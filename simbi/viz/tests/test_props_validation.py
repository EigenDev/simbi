# =============================================================================
# test_props_validation.py
#
# component props reject unknown keys by name. absent that check a mistyped prop
# (cma for cmap) validates cleanly and the style silently never applies.
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

# a props key may name the data field it styles, so one panel of a shared chart
# can be scaled independently of its neighbour. the component part selects a
# class and is matched loosely; the field part names data, where 'D' and 'd'
# are different quantities and a silent fold between them mis-styles the plot.


def test_a_field_qualified_key_keeps_the_data_name_verbatim() -> None:
    props = load_component_props(overrides=["QUAD:D.cmap=magma"])

    assert set(props) == {"quad:D"}
    assert props["quad:D"].cmap == "magma"


def test_a_field_qualified_key_records_only_what_was_asked() -> None:
    """the override is layered over the shared props, so a value nobody set
    must not travel with it and overwrite the shared one."""
    props = load_component_props(overrides=["quad:u.log_scale=false"])

    assert props["quad:u"].model_fields_set == {"log_scale"}


def test_a_field_qualified_key_validates_against_its_component() -> None:
    with pytest.raises(ValueError, match=r"did you mean 'cmap'"):
        load_component_props(overrides=["quad:u.cma=magma"])


def test_a_mistyped_component_is_caught_even_when_field_qualified() -> None:
    with pytest.raises(ValueError, match=r"did you mean 'quad'"):
        load_component_props(overrides=["qaud:u.cmap=magma"])
