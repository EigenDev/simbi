# =============================================================================
# tracers.py
#
# load + scatter lagrangian tracer particles from a checkpoint. reads the
# `tracers` group (cell/reservoir ownership, derived position, exact identity,
# provenance flags, and mass weight) and scatters the derived positions on a
# 2D field plot.
# the eulerian complement is a `chi` field plot (see the passive scalar); this is
# the lagrangian view of the same transport.
#
# usage:
#  from simbi.viz.tracers import load_tracers, overlay_tracers
#  overlay_tracers(ax, "data/traced_kh/512x512.chkpt.10.00.h5")  # after a field
# =============================================================================
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import h5py
import matplotlib
import numpy as np


@dataclass
class TracerCloud:
    """the persisted mass-transport population. owner is the authoritative
    cell or reservoir address; position is derived display state."""

    position: np.ndarray  # (n, D)
    id: np.ndarray  # (n,) uint64
    cohort: np.ndarray  # (n,) uint16 immutable initial-material label
    owner: np.ndarray  # (n,) uint64 cell or reservoir address
    escaped: np.ndarray  # (n,) bool -- left the domain, frozen at exit
    crossed_sink: np.ndarray  # (n,) bool -- crossed an accretion radius, frozen
    crossing_time: np.ndarray  # (n,) float -- time of the sink crossing (0 if none)
    weight: float
    run_seed: int
    next_id: int
    injection_remainder: float

    def __len__(self) -> int:
        return len(self.id)

    def accretion_body(self) -> np.ndarray:
        """body index for body-owned accretion reservoirs; -1 otherwise."""
        prefix = np.uint64((1 << 62) | (1 << 61))
        body_mask = (self.owner & prefix) == prefix
        result = np.full(len(self), -1, dtype=np.int64)
        result[body_mask] = (
            self.owner[body_mask] & np.uint64((1 << 61) - 1)
        ).astype(np.int64)
        return result


@dataclass(frozen=True)
class TracerProjection:
    """logical checkpoint axes used by a tracer-only chart."""

    plane: tuple[int, int]
    collapsed_axis: Optional[int]
    projection: str
    labels: tuple[str, str]


def tracer_projection(
    coord_system: str,
    ndim: int,
    collapsed_axis: Optional[int] = None,
) -> TracerProjection:
    """select the native two-dimensional chart and projected column axis."""
    if ndim not in {2, 3}:
        raise ValueError("tracer projection requires two or three dimensions")
    if collapsed_axis is not None:
        if ndim != 3 or collapsed_axis not in range(3):
            raise ValueError("a projected axis requires a three-dimensional checkpoint")
        remaining = tuple(axis for axis in range(3) if axis != collapsed_axis)
        if coord_system == "spherical" and collapsed_axis in {1, 2}:
            angular = 2 if collapsed_axis == 1 else 1
            return TracerProjection(
                plane=(angular, 0),
                collapsed_axis=collapsed_axis,
                projection="polar",
                labels=(r"$\phi$" if angular == 2 else r"$\theta$", "$r$"),
            )
        if coord_system == "cylindrical" and collapsed_axis == 2:
            return TracerProjection(
                plane=(1, 0),
                collapsed_axis=2,
                projection="polar",
                labels=(r"$\phi$", "$R$"),
            )
        labels = {
            "cartesian": ("x", "y", "z"),
            "spherical": ("r", r"$\theta$", r"$\phi$"),
            "cylindrical": ("R", r"$\phi$", "z"),
        }.get(coord_system)
        if labels is None:
            raise ValueError(f"unsupported tracer coordinate system '{coord_system}'")
        return TracerProjection(
            plane=(remaining[0], remaining[1]),
            collapsed_axis=collapsed_axis,
            projection="cartesian",
            labels=(labels[remaining[0]], labels[remaining[1]]),
        )
    if coord_system in {"spherical", "planar_cylindrical"}:
        return TracerProjection(
            plane=(1, 0),
            collapsed_axis=2 if ndim == 3 else None,
            projection="polar",
            labels=(r"$\theta$", "$r$"),
        )
    if coord_system in {"cylindrical", "axis_cylindrical"}:
        return TracerProjection(
            plane=(0, 2 if ndim == 3 else 1),
            collapsed_axis=1 if ndim == 3 else None,
            projection="cartesian",
            labels=("R", "z"),
        )
    if coord_system == "cartesian":
        return TracerProjection(
            plane=(0, 1),
            collapsed_axis=2 if ndim == 3 else None,
            projection="cartesian",
            labels=("x", "y"),
        )
    raise ValueError(f"unsupported tracer coordinate system '{coord_system}'")


def _smooth_grid(values: np.ndarray, sigma: float) -> np.ndarray:
    """apply a normalized separable gaussian display kernel."""
    if sigma <= 0.0:
        return values
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    result = values
    for axis in range(2):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="edge")
        result = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"),
            axis,
            padded,
        )
    return result


def tracer_concentration(
    cloud: TracerCloud,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    plane: tuple[str, str] | tuple[int, int] = ("x", "y"),
    smoothing: Optional[float] = None,
    cohort: Optional[int] = None,
) -> np.ndarray:
    """estimate projected tracer mass per area on the supplied display mesh."""
    axes = tuple(_AXIS[name] if isinstance(name, str) else name for name in plane)
    reservoir_bits = np.uint64((1 << 63) | (1 << 62))
    live = (cloud.owner & reservoir_bits) == 0
    if cohort is not None:
        live &= cloud.cohort == cohort
    positions = cloud.position[live]
    weights = np.full(len(positions), cloud.weight, dtype=float)
    mass, _, _ = np.histogram2d(
        positions[:, axes[1]],
        positions[:, axes[0]],
        bins=(y_edges, x_edges),
        weights=weights,
    )
    if smoothing is None:
        particles_per_cell = len(positions) / mass.size
        # a gaussian kernel needs a useful local sample before its surface
        # density stops looking like individual particle shot noise. target
        # roughly 16 particles inside a radius of two sigma.
        smoothing = np.sqrt(
            16.0 / max(4.0 * np.pi * particles_per_cell, np.finfo(float).tiny)
        )
        smoothing = min(8.0, max(1.0, smoothing))
    if not np.isfinite(smoothing) or smoothing < 0.0:
        raise ValueError("tracer smoothing must be finite and non-negative")
    smoothed_mass = _smooth_grid(mass, smoothing)
    area = np.outer(np.diff(y_edges), np.diff(x_edges))
    return smoothed_mass / area


def cohort_to_gas_ratio(
    cohort_concentration: np.ndarray,
    gas_column_density: np.ndarray,
    cell_area: np.ndarray,
) -> np.ndarray:
    """ratio of mean-normalized cohort and gas column densities."""
    if cohort_concentration.shape != gas_column_density.shape:
        raise ValueError("cohort and gas concentration shapes differ")
    if cell_area.shape != cohort_concentration.shape:
        raise ValueError("cell-area and concentration shapes differ")
    area_total = np.sum(cell_area)
    cohort_mean = np.sum(cohort_concentration * cell_area) / area_total
    gas_mean = np.sum(gas_column_density * cell_area) / area_total
    tiny = np.finfo(float).tiny
    return (cohort_concentration / max(cohort_mean, tiny)) / (
        gas_column_density / max(gas_mean, tiny)
    )


def projected_gas_concentration(
    density: np.ndarray,
    edges: tuple[np.ndarray, ...],
    coord_system: str,
    projection: TracerProjection,
) -> np.ndarray:
    """integrate gas mass over the collapsed chart axis per display-coordinate area."""
    ndim = len(edges)
    logical_density = np.asarray(density).transpose(tuple(reversed(range(ndim))))
    factors = [np.diff(edge) for edge in edges]
    if coord_system == "spherical":
        factors[0] = np.diff(edges[0] ** 3) / 3.0
        factors[1] = np.cos(edges[1][:-1]) - np.cos(edges[1][1:])
    elif coord_system == "planar_cylindrical":
        factors[0] = np.diff(edges[0] ** 2) / 2.0
    elif coord_system in {"cylindrical", "axis_cylindrical"}:
        factors[0] = np.diff(edges[0] ** 2) / 2.0
    volume = np.ones(tuple(len(edge) - 1 for edge in edges))
    for axis, factor in enumerate(factors):
        shape = [1] * ndim
        shape[axis] = len(factor)
        volume *= factor.reshape(shape)
    mass = logical_density * volume
    if projection.collapsed_axis is not None:
        mass = mass.sum(axis=projection.collapsed_axis)
        remaining = [
            axis for axis in range(ndim) if axis != projection.collapsed_axis
        ]
    else:
        remaining = list(range(ndim))
    order = (
        remaining.index(projection.plane[1]),
        remaining.index(projection.plane[0]),
    )
    projected_mass = mass.transpose(order)
    x_edges = edges[projection.plane[0]]
    y_edges = edges[projection.plane[1]]
    return projected_mass / np.outer(np.diff(y_edges), np.diff(x_edges))


def load_tracers(checkpoint_path: str) -> Optional[TracerCloud]:
    """read the tracer population from a checkpoint's `tracers` group, or None when the
    run carried no tracers."""
    with h5py.File(checkpoint_path, "r") as f:
        if "tracers" not in f:
            return None
        g = f["tracers"]
        return TracerCloud(
            position=np.asarray(g["position"], dtype=float),
            id=np.asarray(g["id"], dtype=np.uint64),
            cohort=np.asarray(g["cohort"], dtype=np.uint16),
            owner=np.asarray(g["owner"], dtype=np.uint64),
            escaped=np.asarray(g["escaped"], dtype=float) > 0.5,
            crossed_sink=np.asarray(g["crossed_sink"], dtype=float) > 0.5,
            crossing_time=np.asarray(g["crossing_time"], dtype=float),
            weight=float(np.asarray(g["weight"], dtype=float)[0]),
            run_seed=int(g.attrs["run_seed"]),
            next_id=int(g.attrs["next_id"]),
            injection_remainder=float(g.attrs["injection_remainder"]),
        )


_AXIS = {"x": 0, "y": 1, "z": 2}


def overlay_tracers(
    ax,
    checkpoint_path: str,
    plane: tuple[str, str] | tuple[int, int] = ("x", "y"),
    at: Optional[float] = None,
    slab: Optional[float] = None,
    color_by: str = "flag",
    **kwargs,
):
    """scatter the tracer particles from `checkpoint_path` on `ax` for the two axes in
    `plane` (call after plotting a field). in 3D, `at` + `slab` keep only particles whose
    out-of-plane coordinate is within `slab` of `at` (a thin sheet); with `slab` None
    every particle projects onto the plane. `color_by`:
      - "flag": crossed-sink crimson, escaped grey, live blue (provenance at a glance)
      - "reservoir": accreted particles colored by body index
      - "cohort": immutable initial-material cohort
      - "id":   a stable per-particle color (follow a particle across frames)
      - "none": a single color (pass `c=` / `color=` through kwargs)
    returns the scatter artist, or None when there is nothing to draw -- in which case it
    WARNS to stderr (a checkpoint with no `tracers` group, or an empty one) rather than
    drawing nothing silently, so `--draw-tracers` on a run that carried no tracers tells
    you why the plot is bare."""
    cloud = load_tracers(checkpoint_path)
    if cloud is None:
        print(
            f"--draw-tracers: '{checkpoint_path}' has no 'tracers' group "
            "(the run carried no tracers; set n_tracers > 0 to seed them)",
            file=sys.stderr,
        )
        return None
    if len(cloud) == 0:
        print(f"--draw-tracers: '{checkpoint_path}' tracer group is empty", file=sys.stderr)
        return None

    ai, aj = (
        _AXIS[plane[0]] if isinstance(plane[0], str) else plane[0],
        _AXIS[plane[1]] if isinstance(plane[1], str) else plane[1],
    )
    pos = cloud.position
    keep = np.ones(len(cloud), dtype=bool)
    # thin-sheet filter in 3D: drop particles off the slice so the scatter matches the
    # field slab beneath it.
    if at is not None and slab is not None and pos.shape[1] == 3:
        ak = ({0, 1, 2} - {ai, aj}).pop()
        keep = np.abs(pos[:, ak] - at) <= slab
    x, y = pos[keep, ai], pos[keep, aj]

    # a thin dark outline makes any fill legible on top of ANY field colormap, and a high
    # zorder keeps the particles above the mesh; sized to read as points, not a haze.
    opts = {
        "s": 9,
        "alpha": 0.9,
        "edgecolors": "black",
        "linewidths": 0.3,
        "zorder": 5,
    }
    if color_by == "flag":
        # live particles white (pops on viridis/inferno/magma alike); crossed-sink and
        # escaped keep their provenance colors.
        opts["c"] = np.where(
            cloud.crossed_sink[keep],
            "crimson",
            np.where(cloud.escaped[keep], "0.5", "white"),
        )
    elif color_by == "id":
        opts["c"] = cloud.id[keep]
        opts.setdefault("cmap", "twilight")
    elif color_by == "reservoir":
        body = cloud.accretion_body()[keep]
        palette = matplotlib.colormaps["tab20"]
        opts["c"] = [
            palette(int(index) % palette.N)
            if index >= 0
            else ("0.5" if escaped else "white")
            for index, escaped in zip(body, cloud.escaped[keep])
        ]
    elif color_by == "cohort":
        opts["c"] = cloud.cohort[keep]
        opts.setdefault("cmap", "tab20")
    opts.update(kwargs)
    return ax.scatter(x, y, **opts)
