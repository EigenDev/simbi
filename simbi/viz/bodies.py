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
from typing import Any, Optional

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


def body_mask(body: BodyPose, xs: np.ndarray, ys: np.ndarray, z: float = 0.0) -> np.ndarray:
    """the inside-body indicator on the `(xs, ys)` grid, sliced at `z`, at the body's pose. shape
    `(len(ys), len(xs))`. a world point maps into the body frame as `x_local = R^T (x - position)`."""
    if body.shape is None:
        return np.zeros((len(ys), len(xs)), dtype=bool)
    xg, yg = np.meshgrid(xs, ys)
    zg = np.full_like(xg, z)
    px, py = body.position[0], body.position[1]
    pz = body.position[2] if len(body.position) > 2 else 0.0
    dx, dy, dz = xg - px, yg - py, zg - pz
    r = body.orientation
    xl = r[0][0] * dx + r[1][0] * dy + r[2][0] * dz
    yl = r[0][1] * dx + r[1][1] * dy + r[2][1] * dz
    zl = r[0][2] * dx + r[1][2] * dy + r[2][2] * dz
    return _sdf_dist_np(body.shape.wire, xl, yl, zl) <= 0.0


def draw_body(
    ax,
    body: BodyPose,
    extent: Optional[tuple[float, float, float, float]] = None,
    nx: int = 240,
    ny: int = 240,
    z: float = 0.0,
    facecolor: str = "0.15",
    edgecolor: str = "white",
    alpha: float = 0.9,
) -> None:
    """shade one body's silhouette on `ax`. `extent = (xmin, xmax, ymin, ymax)` (defaults to the
    axis limits); `z` is the slice plane for a 3D shape. a shapeless body (analytic sphere) is drawn
    as a marker at its position."""
    if body.shape is None:
        ax.plot(body.position[0], body.position[1], "o", color=facecolor, markersize=6)
        return
    if extent is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    else:
        xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    mask = body_mask(body, xs, ys, z=z).astype(float)
    ax.contourf(xs, ys, mask, levels=[0.5, 1.5], colors=[facecolor], alpha=alpha)
    ax.contour(xs, ys, mask, levels=[0.5], colors=[edgecolor], linewidths=1.0)


def overlay_bodies(ax, checkpoint_path: str, **kwargs) -> list[BodyPose]:
    """load every body from `checkpoint_path` and shade its silhouette on `ax` (call after plotting a
    field). returns the loaded poses (e.g. to annotate omega). extra kwargs pass to `draw_body`."""
    bodies = load_bodies(checkpoint_path)
    for body in bodies:
        draw_body(ax, body, **kwargs)
    return bodies


__all__ = ["BodyPose", "load_bodies", "body_mask", "draw_body", "overlay_bodies"]
