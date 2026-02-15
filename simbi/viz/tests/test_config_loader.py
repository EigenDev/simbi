# =============================================================================
# test_config_loader.py
#
# tests for per-file override parsing and resolution in config_loader.
# =============================================================================
import pytest

from simbi.viz.components.shared import ColormappedProps
from simbi.viz.config_loader import (
    _split_file_prefix,
    parse_overrides,
    resolve_per_file_props,
)


# =========================================================================
# _split_file_prefix
# =========================================================================
class TestSplitFilePrefix:
    def test_no_prefix(self):
        idx, remainder = _split_file_prefix("quad.cmap=viridis")
        assert idx is None
        assert remainder == "quad.cmap=viridis"

    def test_with_prefix(self):
        idx, remainder = _split_file_prefix("0:quad.cmap=viridis")
        assert idx == 0
        assert remainder == "quad.cmap=viridis"

    def test_multi_digit(self):
        idx, remainder = _split_file_prefix("12:quad.cmap=plasma")
        assert idx == 12
        assert remainder == "quad.cmap=plasma"

    def test_non_digit_passthrough(self):
        idx, remainder = _split_file_prefix("quad:cmap=viridis")
        assert idx is None
        assert remainder == "quad:cmap=viridis"

    def test_empty_prefix(self):
        idx, remainder = _split_file_prefix(":quad.cmap=viridis")
        assert idx is None
        assert remainder == ":quad.cmap=viridis"


# =========================================================================
# parse_overrides
# =========================================================================
class TestParseOverrides:
    def test_global_only(self):
        global_cfg, per_file = parse_overrides(["quad.cmap=inferno"])
        assert global_cfg == {"quad": {"cmap": "inferno"}}
        assert per_file == {}

    def test_per_file_only(self):
        global_cfg, per_file = parse_overrides(["0:quad.cmap=inferno"])
        assert global_cfg == {}
        assert per_file == {0: {"quad": {"cmap": "inferno"}}}

    def test_mixed(self):
        global_cfg, per_file = parse_overrides([
            "quad.log_scale=true",
            "0:quad.cmap=inferno",
            "1:quad.cmap=plasma",
        ])
        assert global_cfg == {"quad": {"log_scale": True}}
        assert per_file[0] == {"quad": {"cmap": "inferno"}}
        assert per_file[1] == {"quad": {"cmap": "plasma"}}

    def test_empty(self):
        global_cfg, per_file = parse_overrides([])
        assert global_cfg == {}
        assert per_file == {}

    def test_multiple_fields_same_file(self):
        global_cfg, per_file = parse_overrides([
            "0:coordinate_profile.normalization=1e-3",
            "0:coordinate_profile.label=sim_a",
        ])
        assert global_cfg == {}
        assert per_file[0] == {
            "coordinate_profile": {
                "normalization": 1e-3,
                "label": "sim_a",
            }
        }


# =========================================================================
# resolve_per_file_props
# =========================================================================
class TestResolvePerFileProps:
    def test_no_overrides(self):
        base = {"quad": ColormappedProps(cmap="viridis")}
        result = resolve_per_file_props(base, None, 0)
        assert result["quad"].cmap == "viridis"

    def test_with_override(self):
        base = {"quad": ColormappedProps(cmap="viridis")}
        overrides = {0: {"quad": {"cmap": "inferno"}}}
        result = resolve_per_file_props(base, overrides, 0)
        assert result["quad"].cmap == "inferno"

    def test_no_override_for_index(self):
        base = {"quad": ColormappedProps(cmap="viridis")}
        overrides = {1: {"quad": {"cmap": "inferno"}}}
        result = resolve_per_file_props(base, overrides, 0)
        assert result["quad"].cmap == "viridis"

    def test_none_base(self):
        result = resolve_per_file_props(None, None, 0)
        assert result == {}

    def test_merge_preserves_unset_fields(self):
        base = {"quad": ColormappedProps(cmap="viridis", log_scale=True)}
        overrides = {0: {"quad": {"cmap": "plasma"}}}
        result = resolve_per_file_props(base, overrides, 0)
        assert result["quad"].cmap == "plasma"
        assert result["quad"].log_scale is True
