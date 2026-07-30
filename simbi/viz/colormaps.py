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

import re
from typing import Sequence, Union

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, to_rgba

# a color argument is anything matplotlib accepts: a name, hex, or rgb(a) tuple.
ColorLike = Union[str, tuple]
# a named colormap OR an already-built Colormap instance.
CmapLike = Union[str, Colormap]


def _resolve(cmap: CmapLike) -> Colormap:
    """a colormap name or instance -> a Colormap instance. an `_r` suffix reverses any base
    map (matplotlib built-ins register their own reverse; this covers third-party maps -- e.g.
    cmasher's `cmr.neutral_r` -- that may not)."""
    if isinstance(cmap, Colormap):
        return cmap
    try:
        return matplotlib.colormaps[cmap]
    except KeyError:
        if cmap.endswith("_r"):
            return matplotlib.colormaps[cmap[:-2]].reversed()
        raise


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


# =============================================================================
# spec strings: build a composite from a single colormap NAME, so the whole thing is
# expressible inline wherever a cmap name is accepted (e.g. `--props quad.cmap=...`).
# grammar (everything else falls through to a plain matplotlib lookup, incl `_r` reverse):
#   join:LO,HI[,at=F|@V][,blend=F]   crossfade LO (low values) into HI (high values); `at`
#                                    is a [0,1] fraction, or `@DATA` resolved through the
#                                    plot's norm (so `at=@1e-5` splits at that data value)
#   stack:C@lo:hi,C@lo:hi[,blend=F]  segmented palette over [0,1] fractions, seams blended
# =============================================================================
def _spec_name(spec: str) -> str:
    """a stable, readable registry name derived from a spec string (so repeated resolves of
    the same spec reuse one registered colormap)."""
    return "dyn_" + re.sub(r"[^0-9a-zA-Z]+", "_", spec).strip("_")[:48]


def _split_fields(body: str) -> tuple[list[str], dict[str, str]]:
    """split a comma list into positional items and `k=v` options."""
    positional: list[str] = []
    options: dict[str, str] = {}
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, val = part.split("=", 1)
            options[key.strip()] = val.strip()
        else:
            positional.append(part)
    return positional, options


def _resolve_at(raw: str, norm) -> float:
    """an `at` option -> a [0, 1] split fraction. `@DATA` maps a data value through the plot's
    norm (LogNorm etc.), so the split follows the data scale; a bare number is already a
    fraction. `@DATA` needs the render norm; without one, pass a fraction instead."""
    if raw.startswith("@"):
        if norm is None:
            raise ValueError(
                f"cmap split '{raw}' is a data value but no norm is available; "
                "use a [0, 1] fraction (e.g. at=0.3) outside the render path"
            )
        return float(np.clip(float(norm(float(raw[1:]))), 0.0, 1.0))
    return float(raw)


def resolve_cmap(spec: CmapLike, norm=None) -> Colormap:
    """resolve a colormap SPEC to a Colormap. a plain name (including `_r` reverse and any
    registered composite) passes straight through; a `join:`/`stack:` spec builds -- and
    caches by name -- a composite on the fly. `norm` (the plot's data normalization) lets a
    `@DATA` split be given in data units. unknown plain names raise the usual lookup error."""
    if isinstance(spec, Colormap):
        return spec
    s = str(spec).strip()
    if s.startswith("join:"):
        pos, opts = _split_fields(s[len("join:") :])
        if len(pos) != 2:
            raise ValueError(f"join spec needs exactly two colormaps: {spec!r}")
        at = _resolve_at(opts["at"], norm) if "at" in opts else 0.5
        blend = float(opts.get("blend", 0.15))
        return join_cmaps(pos[0], pos[1], _spec_name(s), at=at, blend=blend)
    if s.startswith("stack:"):
        pos, opts = _split_fields(s[len("stack:") :])
        specs: list[tuple[str, float, float]] = []
        for item in pos:
            m = re.fullmatch(r"(.+)@([0-9.]+):([0-9.]+)", item)
            if not m:
                raise ValueError(f"stack segment must be 'cmap@lo:hi': {item!r}")
            specs.append((m.group(1), float(m.group(2)), float(m.group(3))))
        blend = float(opts.get("blend", 0.0))
        return stack_cmaps(specs, _spec_name(s), blend=blend)
    return _resolve(s)


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
    "resolve_cmap",
    "join_cmaps",
    "stack_cmaps",
    "alpha_ramp",
    "truncate_cmap",
    "blend_cmaps",
]
