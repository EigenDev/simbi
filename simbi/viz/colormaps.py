# =============================================================================
# colormaps.py
#
# composite / on-the-fly colormap construction for the viz backend. every builder
# returns a matplotlib Colormap and, by default, registers it under `name` so the
# result is usable anywhere a colormap NAME is accepted -- the `quad.cmap=<name>`
# override, a yaml theme, the tui, an animation -- with no renderer change. this is
# the extensibility contract: a user composes a colormap once and refers to it by
# string everywhere thereafter.
#
# usage:
#  from simbi.viz.colormaps import join_cmaps, stack_cmaps, alpha_ramp, blend_cmaps
#  join_cmaps("magma", "Greys", "simbi_ember", at=0.5, blend=0.2)  # smooth two-cmap bleed
#  stack_cmaps([("Greys_r", 0.0, 0.7), ("inferno", 0.7, 1.0)], "hard_stack", blend=0.15)
#  alpha_ramp("red", "minidisk_red")            # transparent -> red overlay ramp
#  truncate_cmap("viridis", 0.2, 0.9, "viridis_mid")
#  blend_cmaps(["black", "crimson", "gold"], "ember_line")
# =============================================================================
from __future__ import annotations

from typing import Sequence, Union

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, to_rgba

# a color argument is anything matplotlib accepts: a name, hex, or rgb(a) tuple.
ColorLike = Union[str, tuple]
# a named colormap OR an already-built Colormap instance.
CmapLike = Union[str, Colormap]


def _resolve(cmap: CmapLike) -> Colormap:
    """a colormap name or instance -> a Colormap instance."""
    if isinstance(cmap, Colormap):
        return cmap
    return matplotlib.colormaps[cmap]


def _finalize(cmap: Colormap, name: str, register: bool) -> Colormap:
    """name the colormap and, when asked, (re)register it globally so it is
    addressable by string. `force=True` keeps re-imports idempotent."""
    cmap.name = name
    if register:
        matplotlib.colormaps.register(cmap, name=name, force=True)
    return cmap


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """the cubic smoothstep 3x^2 - 2x^3 on x already clamped to [0, 1]; flat-tangent at
    both ends so a crossfade begins and ends without a visible kink."""
    return x * x * (3.0 - 2.0 * x)


def join_cmaps(
    lo: CmapLike,
    hi: CmapLike,
    name: str,
    at: float = 0.5,
    blend: float = 0.15,
    n: int = 256,
    register: bool = True,
) -> ListedColormap:
    """smoothly BLEND two colormaps into one: `lo` fills the output below `at`, `hi` above,
    and the two CROSSFADE across a band of half-width `blend` (in [0, 1] units) centered on
    `at`, so the seam bleeds instead of jumping. this is the fix for a `stack_cmaps` seam
    that looks abrupt, and the way to fuse a neutral bulk with a colorful highlight -- e.g.
    an ember minidisk burning out of a grayscale disk, `join_cmaps('magma', 'Greys', ...)`,
    whose bright magma tail (near white) meets the white end of Greys so the blend is
    seamless. pick `lo`/`hi` whose meeting ends are close in color for the cleanest bleed."""
    lo_c, hi_c = _resolve(lo), _resolve(hi)
    t = np.linspace(0.0, 1.0, n)
    lo_col = lo_c(np.clip(t / at, 0.0, 1.0)) if at > 0.0 else hi_c(t)
    hi_col = hi_c(np.clip((t - at) / (1.0 - at), 0.0, 1.0)) if at < 1.0 else lo_c(t)
    if blend > 0.0:
        w = _smoothstep(np.clip((t - (at - blend)) / (2.0 * blend), 0.0, 1.0))
    else:
        w = (t >= at).astype(float)
    col = (1.0 - w)[:, None] * lo_col + w[:, None] * hi_col
    return _finalize(ListedColormap(col, name=name), name, register)


def stack_cmaps(
    specs: Sequence[tuple[CmapLike, float, float]],
    name: str,
    n: int = 256,
    blend: float = 0.0,
    register: bool = True,
) -> ListedColormap:
    """stitch several colormaps over fractions of [0, 1], so one map paints different data
    regimes with different palettes. each spec is `(cmap, lo, hi)` with `lo < hi` naming the
    output fraction that segment fills; segments should tile [0, 1] without gaps. `blend > 0`
    crossfades each internal seam over +/- `blend` (in [0, 1] units) so adjacent palettes
    bleed rather than jump (a hard `blend = 0` reproduces the raw concatenation, which is
    discontinuous wherever one segment's end color differs from the next's start color)."""
    t = np.linspace(0.0, 1.0, n)
    col = np.zeros((n, 4))
    for cmap, lo, hi in specs:
        sel = (t >= lo) & (t <= hi)
        local = np.zeros(int(sel.sum())) if hi <= lo else (t[sel] - lo) / (hi - lo)
        col[sel] = _resolve(cmap)(local)
    if blend > 0.0:
        half = max(1, int(round(blend * n)))
        for _, _, hi in specs[:-1]:
            c = int(round(hi * n))
            a, z = max(0, c - half), min(n - 1, c + half)
            if z > a:
                for k in range(4):
                    col[a:z, k] = np.linspace(col[a, k], col[z, k], z - a)
    return _finalize(ListedColormap(col, name=name), name, register)


def alpha_ramp(
    color: ColorLike,
    name: str,
    n: int = 256,
    gamma: float = 1.0,
    register: bool = True,
) -> ListedColormap:
    """a single color whose OPACITY ramps transparent -> opaque across the value range,
    for use as an overlay: mapped over a field it reveals only the high end (dense gas)
    in `color` while low values stay transparent and show whatever is underneath.
    `gamma > 1` pushes the opaque band toward the very top (only the densest gas tints)."""
    rgb = to_rgba(color)[:3]
    arr = np.empty((n, 4))
    arr[:, :3] = rgb
    arr[:, 3] = np.linspace(0.0, 1.0, n) ** gamma
    return _finalize(ListedColormap(arr, name=name), name, register)


def truncate_cmap(
    cmap: CmapLike,
    lo: float,
    hi: float,
    name: str,
    n: int = 256,
    register: bool = True,
) -> ListedColormap:
    """a sub-range `[lo, hi]` of an existing colormap, rescaled to fill [0, 1] -- to drop
    a palette's washed-out or too-dark tail (e.g. `truncate_cmap('viridis', 0.15, 0.95)`)."""
    return _finalize(
        ListedColormap(_resolve(cmap)(np.linspace(lo, hi, n)), name=name), name, register
    )


def blend_cmaps(
    colors: Sequence[ColorLike],
    name: str,
    n: int = 256,
    register: bool = True,
) -> LinearSegmentedColormap:
    """a smooth colormap interpolating through an arbitrary list of anchor colors, evenly
    spaced across [0, 1] -- the most open-ended builder: any sequence of colors -> a map."""
    cmap = LinearSegmentedColormap.from_list(name, [to_rgba(c) for c in colors], N=n)
    return _finalize(cmap, name, register)


# a small set of ready-made composites registered at import, so `cmap="simbi_ember"` and
# friends work out of the box. `simbi_ember`/`simbi_ash` burn the low-to-mid density range
# (the minidisks) in ember while the dense bulk (the circumbinary disk) fades to neutral
# grays, blended so the two bleed together with no seam. the cold/hot ramps are transparent
# overlays for per-body tinting (see bodies.tint_bodies).
def _register_presets() -> None:
    # magma's bright (near-white) tail meets the white end of Greys, so the blend is seamless;
    # low rho burns ember, high rho (the outer disk) is grayscale.
    join_cmaps("magma", "Greys", "simbi_ember", at=0.5, blend=0.22)
    join_cmaps("inferno", "bone", "simbi_ash", at=0.5, blend=0.22)
    alpha_ramp("crimson", "simbi_tint_hot")
    alpha_ramp("dodgerblue", "simbi_tint_cold")


_register_presets()


__all__ = [
    "join_cmaps",
    "stack_cmaps",
    "alpha_ramp",
    "truncate_cmap",
    "blend_cmaps",
]
