# =============================================================================
# bodies.py
#
# load + draw immersed-boundary bodies from a checkpoint. reads the `bodies`
# group (position / orientation / angular velocity + the per-body CSG shape
# wire) and shades each body's silhouette on a 2D field plot AT ITS POSE, so a
# spinning / tumbling body tracks its rotation across frames.
#
# usage:
#  from simbi.viz.bodies import load_bodies, overlay_bodies
#  overlay_bodies(ax, "data/ibm/wind_tunnel/chkpt.0100.h5")   # after plotting a field
# =============================================================================
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import h5py
import numpy as np

from simbi.types.shape import Shape


@dataclass
class BodyPose:
    """a body's persisted state, enough to draw it: position, orientation (3x3), angular
    velocity, mass, and the CSG shape (None = the analytic sphere, drawn as a marker)."""

    position: np.ndarray  # (D,)
    orientation: np.ndarray  # (3, 3) row-major, body-local -> world
    omega: np.ndarray  # (3,)
    mass: float
    shape: Optional[Shape]


def load_bodies(checkpoint_path: str) -> list[BodyPose]:
    """read every immersed body's pose + shape from a checkpoint's `bodies` group."""
    with h5py.File(checkpoint_path, "r") as f:
        if "bodies" not in f:
            return []
        g = f["bodies"]
        nb = int(g.attrs["n_bodies"])
        pos = np.asarray(g["position"])
        orient = np.asarray(g["orientation"])
        omega = np.asarray(g["omega"])
        mass = np.asarray(g["mass"])
        bodies: list[BodyPose] = []
        for b in range(nb):
            wire = g.attrs.get(f"shape_{b}", "")
            if isinstance(wire, bytes):
                wire = wire.decode("utf-8")
            shape = Shape.from_wire(json.loads(wire)) if wire else None
            bodies.append(
                BodyPose(
                    position=pos[b],
                    orientation=orient[b],
                    omega=omega[b],
                    mass=float(mass[b]),
                    shape=shape,
                )
            )
        return bodies


def _sdf_dist_np(node: dict[str, Any], x, y, z):
    # the vectorized (numpy) CSG distance, mirroring symbi-ib/src/sdf.rs::dist.
    kind = node["kind"]
    if kind == "sphere":
        c, r = node["center"], node["radius"]
        return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - r
    if kind == "box":
        c, h = node["center"], node["half_extents"]
        qx, qy, qz = np.abs(x - c[0]) - h[0], np.abs(y - c[1]) - h[1], np.abs(z - c[2]) - h[2]
        outside = np.sqrt(
            np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2
        )
        return outside + np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    if kind == "union":
        return np.minimum(_sdf_dist_np(node["a"], x, y, z), _sdf_dist_np(node["b"], x, y, z))
    if kind == "intersect":
        return np.maximum(_sdf_dist_np(node["a"], x, y, z), _sdf_dist_np(node["b"], x, y, z))
    if kind == "complement":
        return -_sdf_dist_np(node["inner"], x, y, z)
    if kind == "translated":
        o = node["offset"]
        return _sdf_dist_np(node["inner"], x - o[0], y - o[1], z - o[2])
    if kind == "rotated":
        r = node["rot"]  # map the world point into the inner frame via R^T
        xr = r[0][0] * x + r[1][0] * y + r[2][0] * z
        yr = r[0][1] * x + r[1][1] * y + r[2][1] * z
        zr = r[0][2] * x + r[1][2] * y + r[2][2] * z
        return _sdf_dist_np(node["inner"], xr, yr, zr)
    raise ValueError(f"unknown shape kind {kind!r}")


# world-axis name -> index, accepting both the cartesian (x/y/z) and the coordinate
# (x1/x2/x3) spellings the field `--slice` uses.
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "x1": 0, "x2": 1, "x3": 2}


def slice_to_plane(
    slice_spec: Optional[dict[str, float]],
) -> Optional[tuple[tuple[str, str], float]]:
    """map a field `--slice` spec to the body-overlay cut plane. a single fixed axis
    (e.g. {"x3": 0.0}) leaves the two remaining world axes -- in ascending index order,
    matching the field's own slice reduction -- as the plotted plane; returns
    `((horizontal, vertical), fixed_value)`. no slice -> the x-y plane at z = 0. returns
    None when the spec fixes two axes (the field reduces to a 1-D line, so a body
    silhouette is undefined)."""
    if not slice_spec:
        return ("x", "y"), 0.0
    fixed = [(_AXIS_INDEX[k], v) for k, v in slice_spec.items()]
    if len(fixed) != 1:
        return None
    fixed_axis, at = fixed[0]
    names = ("x", "y", "z")
    plane = tuple(names[a] for a in (0, 1, 2) if a != fixed_axis)
    return plane, at  # type: ignore[return-value]


def body_mask(
    body: BodyPose,
    us: np.ndarray,
    vs: np.ndarray,
    plane: tuple[str, str] = ("x", "y"),
    at: float = 0.0,
) -> np.ndarray:
    """the inside-body indicator on the 2-D cut `plane` (a pair of world-axis names;
    `us` runs the horizontal axis, `vs` the vertical), with the third world axis held at
    `at`, evaluated at the body's pose. shape `(len(vs), len(us))`. a world point maps
    into the body frame as `x_local = R^T (x - position)`."""
    if body.shape is None:
        return np.zeros((len(vs), len(us)), dtype=bool)
    ua, va = _AXIS_INDEX[plane[0]], _AXIS_INDEX[plane[1]]
    fixed_axis = 3 - ua - va  # the remaining index of {0, 1, 2}
    ug, vg = np.meshgrid(us, vs)
    world = [np.zeros_like(ug), np.zeros_like(ug), np.zeros_like(ug)]
    world[ua] = ug
    world[va] = vg
    world[fixed_axis] = np.full_like(ug, at)
    pos = [body.position[k] if k < len(body.position) else 0.0 for k in range(3)]
    d = [world[k] - pos[k] for k in range(3)]
    r = body.orientation
    xl = r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2]
    yl = r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2]
    zl = r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2]
    return _sdf_dist_np(body.shape.wire, xl, yl, zl) <= 0.0


def draw_body(
    ax,
    body: BodyPose,
    extent: Optional[tuple[float, float, float, float]] = None,
    nu: int = 240,
    nv: int = 240,
    plane: tuple[str, str] = ("x", "y"),
    at: float = 0.0,
    facecolor: str = "0.15",
    edgecolor: str = "white",
    alpha: float = 0.9,
) -> list:
    """shade one body's silhouette on `ax` for the cut `plane` (world-axis names), with
    the third axis held at `at`. `extent = (umin, umax, vmin, vmax)` defaults to the axis
    limits. a shapeless body (analytic sphere) is drawn as a marker at its projected
    position. returns the matplotlib artists created (so an animation can remove them
    before the next frame)."""
    ua, va = _AXIS_INDEX[plane[0]], _AXIS_INDEX[plane[1]]
    pos = [body.position[k] if k < len(body.position) else 0.0 for k in range(3)]
    if body.shape is None:
        (marker,) = ax.plot(pos[ua], pos[va], "o", color=facecolor, markersize=6)
        return [marker]
    if extent is None:
        umin, umax = ax.get_xlim()
        vmin, vmax = ax.get_ylim()
    else:
        umin, umax, vmin, vmax = extent
    us = np.linspace(umin, umax, nu)
    vs = np.linspace(vmin, vmax, nv)
    mask = body_mask(body, us, vs, plane=plane, at=at).astype(float)
    fill = ax.contourf(us, vs, mask, levels=[0.5, 1.5], colors=[facecolor], alpha=alpha)
    outline = ax.contour(us, vs, mask, levels=[0.5], colors=[edgecolor], linewidths=1.0)
    return [fill, outline]


def tint_bodies(
    ax,
    field2d: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    checkpoint_path: str,
    colors: Sequence[str],
    radius: float,
    norm=None,
    plane: tuple[str, str] = ("x", "y"),
    at: float = 0.0,
    alpha: float = 1.0,
    gamma: float = 1.0,
    region: Optional[Callable[[BodyPose, np.ndarray, np.ndarray, int, int], np.ndarray]] = None,
) -> list:
    """re-color the gas NEAR each immersed body with a per-body color, layered over an
    already-rendered field. `field2d` is the same 2-D array the background shows, on the
    (`us`, `vs`) cell grid; `colors` gives one color per body (cycled if shorter). each
    body's gas is mapped through a transparent -> opaque `alpha_ramp` of its color, so only
    the dense gas lights up while faint gas stays transparent and shows the background
    palette; sharing the background `norm` keeps color tied to value. the tinted REGION
    defaults to a disk of `radius` around the body's projected position (the minidisk
    neighborhood); pass `region(body, ug, vg, ua, va) -> bool array` to tint an arbitrary
    set instead (e.g. gravitationally-bound gas). returns the created artists so an
    animation can remove them before the next frame."""
    from .colormaps import alpha_ramp

    ua, va = _AXIS_INDEX[plane[0]], _AXIS_INDEX[plane[1]]
    ug, vg = np.meshgrid(us, vs)
    artists: list = []
    for ii, body in enumerate(load_bodies(checkpoint_path)):
        color = colors[ii % len(colors)]
        if region is not None:
            near = np.asarray(region(body, ug, vg, ua, va), dtype=bool)
        else:
            px = body.position[ua] if ua < len(body.position) else 0.0
            py = body.position[va] if va < len(body.position) else 0.0
            near = (ug - px) ** 2 + (vg - py) ** 2 <= radius**2
        masked = np.ma.masked_where(~near, field2d)
        cmap = alpha_ramp(color, f"_tint_body_{ii}", gamma=gamma, register=False)
        artists.append(
            ax.pcolormesh(us, vs, masked, cmap=cmap, norm=norm, alpha=alpha, shading="auto")
        )
    return artists


def tint_bodies_on_slice(
    ax,
    field2d: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    checkpoint_path: str,
    slice_spec: Optional[dict[str, float]],
    coord_system: str,
    colors: Sequence[str],
    radius: float,
    **kwargs,
) -> list:
    """tint every body's gas neighborhood on `ax`, matching a field `--slice` (via
    `slice_to_plane`). cartesian only -- the radial region is cartesian -- and skipped for a
    1-D (double-sliced) field. the gated entry point mirroring `overlay_bodies_on_slice`."""
    if ax is None or coord_system != "cartesian":
        return []
    plane_at = slice_to_plane(slice_spec)
    if plane_at is None:
        return []
    plane, at = plane_at
    return tint_bodies(
        ax, field2d, us, vs, checkpoint_path, colors, radius, plane=plane, at=at, **kwargs
    )


def overlay_bodies(
    ax,
    checkpoint_path: str,
    plane: tuple[str, str] = ("x", "y"),
    at: float = 0.0,
    **kwargs,
) -> list:
    """load every body from `checkpoint_path` and shade its silhouette on `ax` for the
    cut `plane` at `at` (call after plotting a field). returns the matplotlib artists
    created, flattened across bodies, for removal in an animation. extra kwargs pass to
    `draw_body`."""
    artists: list = []
    for body in load_bodies(checkpoint_path):
        artists.extend(draw_body(ax, body, plane=plane, at=at, **kwargs))
    return artists


def overlay_bodies_on_slice(
    ax,
    checkpoint_path: str,
    slice_spec: Optional[dict[str, float]],
    coord_system: str,
    **kwargs,
) -> list:
    """overlay every body's silhouette on `ax`, matching a field `--slice` (via
    `slice_to_plane`). cartesian only -- the body signed-distance is cartesian, so it
    does not align with a polar/spherical field plot -- and skipped for a 1-D
    (double-sliced) field, which has no silhouette. returns the created artists (empty
    when nothing is drawn), the single gated entry point for both the static plot and
    the per-frame animation overlay."""
    if ax is None or coord_system != "cartesian":
        return []
    plane_at = slice_to_plane(slice_spec)
    if plane_at is None:
        return []
    plane, at = plane_at
    return overlay_bodies(ax, checkpoint_path, plane=plane, at=at, **kwargs)


__all__ = [
    "BodyPose",
    "load_bodies",
    "body_mask",
    "draw_body",
    "overlay_bodies",
    "overlay_bodies_on_slice",
    "tint_bodies",
    "tint_bodies_on_slice",
    "slice_to_plane",
]
