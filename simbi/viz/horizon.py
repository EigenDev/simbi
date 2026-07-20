# =============================================================================
# horizon.py
#
# draw the black-hole surfaces of a curved-spacetime run on a 2D field plot: the
# event horizon r_+ = M + sqrt(M^2 - a^2) (a circle centered on the chart origin),
# and, when supplied, the excision surface r_exc. the surfaces are read from the
# checkpoint metadata (spacetime, schwarzschild_mass, kerr_spin), so a flat
# (minkowski) run draws nothing.
#
# usage:
#  from simbi.viz.horizon import overlay_horizon_on_slice
#  overlay_horizon_on_slice(ax, sim_data.metadata, slice_spec, coord_system)
# =============================================================================
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from simbi.viz.bodies import slice_to_plane


def horizon_radius(mass: float, spin: float = 0.0) -> float:
    """the outer event-horizon radius r_+ = M + sqrt(M^2 - a^2); r_+ = 2M at a = 0."""
    return mass + math.sqrt(max(mass * mass - spin * spin, 0.0))


def _cross_section(radius: float, at: float) -> float:
    """the radius of the circle where a sphere of radius `radius` meets a slice plane
    offset `at` from its center: sqrt(radius^2 - at^2), or 0 once the plane clears it."""
    cut2 = radius * radius - at * at
    return math.sqrt(cut2) if cut2 > 0.0 else 0.0


def _draw_disk(
    ax, radius: float, facecolor: str, edgecolor: str, lw: float, alpha: float, zorder: float
) -> list:
    """a filled disk of the given radius about the chart origin. a cartesian axis fills a
    circle patch; a polar-projected axis fills r in [0, radius] across its angular span."""
    if getattr(ax, "name", "") == "polar":
        t = np.linspace(*ax.get_xlim(), 256)
        coll = ax.fill_between(
            t, 0.0, radius, facecolor=facecolor, edgecolor=edgecolor,
            linewidth=lw, alpha=alpha, zorder=zorder,
        )
        return [coll]
    import matplotlib.patches as mpatches

    circ = mpatches.Circle(
        (0.0, 0.0), radius, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(circ)
    return [circ]


def _draw_ring(
    ax, radius: float, linestyle: str, color: str, lw: float, alpha: float, zorder: float
) -> list:
    """an unfilled curve at constant radius about the chart origin (the excision surface,
    drawn on top of the horizon disk so it stays visible against the black fill)."""
    if getattr(ax, "name", "") == "polar":
        t = np.linspace(*ax.get_xlim(), 256)
        (line,) = ax.plot(
            t, np.full_like(t, radius), linestyle=linestyle, color=color,
            linewidth=lw, alpha=alpha, zorder=zorder,
        )
    else:
        t = np.linspace(0.0, 2.0 * np.pi, 256)
        (line,) = ax.plot(
            radius * np.cos(t), radius * np.sin(t), linestyle=linestyle,
            color=color, linewidth=lw, alpha=alpha, zorder=zorder,
        )
    return [line]


def draw_horizon(
    ax,
    mass: float,
    spin: float = 0.0,
    r_exc: Optional[float] = None,
    at: float = 0.0,
    facecolor: str = "black",
    edgecolor: str = "white",
    lw: float = 1.0,
    alpha: float = 1.0,
    zorder: float = 5.0,
) -> list:
    """draw the event horizon as a filled black disk (a black hole is opaque -- no light
    escapes r < r_+), outlined for definition against a dark colormap; and, if `r_exc` is
    given, the excision surface as a dashed ring on top. `at` is the slice-plane offset
    along its fixed axis: the horizon sphere meets that plane in a circle of radius
    sqrt(r_+^2 - at^2), which vanishes once the slice clears the sphere. returns the
    created matplotlib artists so an animation can remove them before the next frame."""
    artists: list = []
    r_cut = _cross_section(horizon_radius(mass, spin), at)
    if r_cut > 0.0:
        artists += _draw_disk(ax, r_cut, facecolor, edgecolor, lw, alpha, zorder)
    if r_exc is not None and r_exc > 0.0:
        exc_cut = _cross_section(r_exc, at)
        if exc_cut > 0.0:
            artists += _draw_ring(ax, exc_cut, ":", edgecolor, lw, 0.7, zorder + 1.0)
    return artists


def overlay_horizon_on_slice(
    ax,
    metadata: Any,
    slice_spec: Optional[dict[str, float]],
    coord_system: str,
    r_exc: Optional[float] = None,
    **kwargs,
) -> list:
    """overlay the black-hole surfaces on `ax`, read from the checkpoint `metadata`.
    a flat (minkowski) background or a massless run draws nothing. on a cartesian chart
    the horizon centers on the origin and follows the field `--slice` offset; on a
    spherical / cylindrical chart it is the r = r_+ coordinate surface. the single gated
    entry point for both the static plot and the per-frame animation overlay."""
    if ax is None:
        return []
    spacetime = str(getattr(metadata, "spacetime", "minkowski"))
    if spacetime == "minkowski":
        return []
    mass = float(getattr(metadata, "schwarzschild_mass", 0.0) or 0.0)
    if mass <= 0.0:
        return []
    spin = float(getattr(metadata, "kerr_spin", 0.0) or 0.0)

    # the slice offset shrinks a cartesian horizon sphere's cross section; a
    # spherical / cylindrical chart already plots the r = r_+ surface directly.
    at = 0.0
    if coord_system == "cartesian":
        plane_at = slice_to_plane(slice_spec)
        if plane_at is not None:
            _, at = plane_at

    return draw_horizon(ax, mass, spin, r_exc=r_exc, at=at, **kwargs)


__all__ = ["horizon_radius", "draw_horizon", "overlay_horizon_on_slice"]
