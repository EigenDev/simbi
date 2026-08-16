# =============================================================================
# test_colormaps.py
#
# the composite-colormap builders: each returns a matplotlib Colormap and, when
# asked, registers it under its name so it is addressable as a string everywhere a
# colormap name is accepted. the tests pin the registration contract and the
# palette math (segment tiling, alpha ramp endpoints, truncation, blend anchors).
# =============================================================================
import matplotlib
import numpy as np

import pytest
from matplotlib.colors import Colormap, LogNorm

from simbi.viz.colormaps import (
    alpha_ramp,
    blend_cmaps,
    join_cmaps,
    resolve_cmap,
    stack_cmaps,
    truncate_cmap,
)


def test_resolve_cmap_passes_plain_names_and_reverse() -> None:
    assert isinstance(resolve_cmap("viridis"), Colormap)
    # an `_r` suffix reverses any base map (covers third-party maps lacking a registered reverse).
    rev = resolve_cmap("viridis_r")
    assert np.allclose(rev(0.0)[:3], matplotlib.colormaps["viridis"](1.0)[:3], atol=1e-2)


def test_resolve_cmap_builds_a_join_spec() -> None:
    cmap = resolve_cmap("join:magma,Greys,at=0.5,blend=0.2")
    lo = np.asarray(cmap(0.25))[:3]
    hi = np.asarray(cmap(0.95))[:3]
    assert lo[0] > lo[2]  # magma (warm) on the low end
    assert abs(hi[0] - hi[1]) < 0.06 and abs(hi[1] - hi[2]) < 0.06  # grayscale on the high end
    assert _max_adjacent_jump(cmap) < 0.1  # blended, no cliff


def test_resolve_cmap_at_data_value_uses_the_norm() -> None:
    # `at=@DATA` maps a data value through the plot norm, so the split follows the data scale.
    norm = LogNorm(vmin=1e-6, vmax=1.0)
    cmap = resolve_cmap("join:Blues,Greys,at=@1e-5", norm=norm)
    split = float(norm(1e-5))  # log10(1e-5/1e-6)/log10(1/1e-6) = 1/6
    below = np.asarray(cmap(split * 0.4))[:3]
    above = np.asarray(cmap(min(1.0, split + 0.5)))[:3]
    assert below[2] > below[0]  # bluish below the split
    assert abs(above[0] - above[1]) < 0.08 and abs(above[1] - above[2]) < 0.08  # gray above


def test_resolve_cmap_at_data_value_needs_a_norm() -> None:
    with pytest.raises(ValueError, match="no norm"):
        resolve_cmap("join:Blues,Greys,at=@1e-5", norm=None)


def test_resolve_cmap_builds_a_stack_spec() -> None:
    cmap = resolve_cmap("stack:Greys_r@0:0.5,inferno@0.5:1,blend=0.12")
    assert isinstance(cmap, Colormap)
    # a blended stack has no full black<->white cliff at the seam.
    assert _max_adjacent_jump(cmap) < 0.5


def test_resolve_cmap_rejects_a_malformed_spec() -> None:
    with pytest.raises(ValueError, match="two colormaps"):
        resolve_cmap("join:onlyone")
    with pytest.raises(ValueError, match="cmap@lo:hi"):
        resolve_cmap("stack:viridis")


def _max_adjacent_jump(cmap, n: int = 256) -> float:
    """the largest sum-of-|dRGB| between neighboring colormap entries; a hard seam spikes
    this toward 3 (a full black<->white step), a smooth map keeps it small."""
    rgb = np.asarray(cmap(np.linspace(0.0, 1.0, n)))[:, :3]
    return float(np.abs(np.diff(rgb, axis=0)).sum(axis=1).max())


def test_join_cmaps_blends_without_a_discontinuity() -> None:
    # a raw concatenation of grayscale (ends white) and inferno (starts black) has a cliff;
    # join_cmaps crossfades the seam so neighboring entries step smoothly across it.
    hard = stack_cmaps([("Greys_r", 0.0, 0.5), ("inferno", 0.5, 1.0)], "test_hard", blend=0.0)
    soft = join_cmaps("Greys_r", "inferno", "test_join", at=0.5, blend=0.2)
    assert _max_adjacent_jump(hard) > 1.0  # the white->black cliff
    assert _max_adjacent_jump(soft) < 0.1  # smooth
    assert _max_adjacent_jump(soft) < _max_adjacent_jump(hard) / 5


def test_join_cmaps_orders_lo_below_hi() -> None:
    # `lo` fills the low end, `hi` the high end, with the split at `at`.
    cmap = join_cmaps("Reds", "Blues", "test_order", at=0.5, blend=0.1)
    lo = np.asarray(cmap(0.1))[:3]
    hi = np.asarray(cmap(0.9))[:3]
    assert lo[0] > lo[2]  # low end is reddish
    assert hi[2] > hi[0]  # high end is bluish


def test_stack_blend_smooths_the_seam() -> None:
    hard = stack_cmaps([("Greys_r", 0.0, 0.5), ("inferno", 0.5, 1.0)], "test_sh", blend=0.0)
    soft = stack_cmaps([("Greys_r", 0.0, 0.5), ("inferno", 0.5, 1.0)], "test_ss", blend=0.15)
    assert _max_adjacent_jump(soft) < _max_adjacent_jump(hard) / 2


def test_stack_cmaps_registers_and_tiles_segments() -> None:
    cmap = stack_cmaps([("Greys_r", 0.0, 0.5), ("inferno", 0.5, 1.0)], "test_stack", n=256)
    # registered under its name -> usable as a string anywhere a cmap name is accepted.
    # (the registry returns a copy on lookup, so compare by name.)
    assert "test_stack" in matplotlib.colormaps
    assert matplotlib.colormaps["test_stack"].name == "test_stack"
    # low end comes from Greys_r, high end from inferno; the two differ.
    lo = np.asarray(cmap(0.0))
    hi = np.asarray(cmap(1.0))
    assert np.allclose(lo[:3], matplotlib.colormaps["Greys_r"](0.0)[:3], atol=1e-2)
    assert np.allclose(hi[:3], matplotlib.colormaps["inferno"](1.0)[:3], atol=1e-2)


def test_alpha_ramp_endpoints_and_constant_hue() -> None:
    cmap = alpha_ramp("red", "test_ramp", n=256)
    assert cmap(0.0)[3] == 0.0  # transparent at the low end
    assert cmap(1.0)[3] == 1.0  # opaque at the high end
    # the hue is constant across the ramp; only alpha changes.
    assert np.allclose(cmap(0.2)[:3], cmap(0.9)[:3])
    assert np.allclose(cmap(1.0)[:3], (1.0, 0.0, 0.0))  # "red"


def test_alpha_ramp_gamma_pushes_opacity_to_the_top() -> None:
    lin = alpha_ramp("blue", "test_ramp_lin", gamma=1.0, register=False)
    steep = alpha_ramp("blue", "test_ramp_steep", gamma=3.0, register=False)
    # a higher gamma keeps the midrange more transparent (only the densest gas tints).
    assert steep(0.5)[3] < lin(0.5)[3]


def test_truncate_cmap_rescales_a_subrange() -> None:
    full = matplotlib.colormaps["viridis"]
    cut = truncate_cmap("viridis", 0.2, 0.8, "test_trunc", n=256)
    assert np.allclose(cut(0.0)[:3], full(0.2)[:3], atol=1e-2)
    assert np.allclose(cut(1.0)[:3], full(0.8)[:3], atol=1e-2)


def test_blend_cmaps_hits_the_anchor_colors() -> None:
    cmap = blend_cmaps(["black", "red", "white"], "test_blend", n=256)
    assert np.allclose(cmap(0.0)[:3], (0.0, 0.0, 0.0), atol=1e-2)
    assert np.allclose(cmap(1.0)[:3], (1.0, 1.0, 1.0), atol=1e-2)
    assert np.allclose(cmap(0.5)[:3], (1.0, 0.0, 0.0), atol=2e-2)  # red midpoint


def test_register_false_does_not_pollute_the_registry() -> None:
    alpha_ramp("green", "test_unregistered", register=False)
    assert "test_unregistered" not in matplotlib.colormaps


def test_presets_are_registered_at_import() -> None:
    for name in ("simbi_ember", "simbi_ash", "simbi_tint_hot", "simbi_tint_cold"):
        assert name in matplotlib.colormaps


def test_simbi_ember_is_ember_low_grayscale_high_and_smooth() -> None:
    # the disk aesthetic: low-mid density (minidisks) burns warm ember, the dense bulk (the
    # circumbinary disk) is grayscale, and the two bleed with no seam.
    ember = matplotlib.colormaps["simbi_ember"]
    lo = np.asarray(ember(0.30))[:3]  # minidisk / low-mid density
    hi = np.asarray(ember(0.97))[:3]  # circumbinary disk / high density
    assert lo[0] > lo[2] + 0.2  # warm (red > blue)
    assert abs(hi[0] - hi[1]) < 0.06 and abs(hi[1] - hi[2]) < 0.06  # grayscale
    assert _max_adjacent_jump(ember) < 0.15  # no discontinuity
