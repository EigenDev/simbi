# =============================================================================
# tracers.py
#
# load + scatter lagrangian tracer particles from a checkpoint. reads the
# `tracers` group (position / id / provenance flags / mass weight) and scatters
# the particle positions on a 2D field plot, so the winding of a shear layer or
# the infall onto a sink is visible as the particle cloud at its current state.
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
import numpy as np


@dataclass
class TracerCloud:
    """the persisted tracer population: positions (n, D), ids, provenance flag masks over
    the n particles, per-tracer crossing time, and the shared mass weight one tracer
    represents (sampled mass / population)."""

    position: np.ndarray  # (n, D)
    id: np.ndarray  # (n,) int
    escaped: np.ndarray  # (n,) bool -- left the domain, frozen at exit
    crossed_sink: np.ndarray  # (n,) bool -- crossed an accretion radius, frozen
    crossing_time: np.ndarray  # (n,) float -- time of the sink crossing (0 if none)
    weight: float

    def __len__(self) -> int:
        return len(self.id)


def load_tracers(checkpoint_path: str) -> Optional[TracerCloud]:
    """read the tracer population from a checkpoint's `tracers` group, or None when the
    run carried no tracers."""
    with h5py.File(checkpoint_path, "r") as f:
        if "tracers" not in f:
            return None
        g = f["tracers"]
        return TracerCloud(
            position=np.asarray(g["position"], dtype=float),
            id=np.asarray(g["id"], dtype=float).astype(np.int64),
            escaped=np.asarray(g["escaped"], dtype=float) > 0.5,
            crossed_sink=np.asarray(g["crossed_sink"], dtype=float) > 0.5,
            crossing_time=np.asarray(g["crossing_time"], dtype=float),
            weight=float(np.asarray(g["weight"], dtype=float)[0]),
        )


_AXIS = {"x": 0, "y": 1, "z": 2}


def overlay_tracers(
    ax,
    checkpoint_path: str,
    plane: tuple[str, str] = ("x", "y"),
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

    ai, aj = _AXIS[plane[0]], _AXIS[plane[1]]
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
    opts.update(kwargs)
    return ax.scatter(x, y, **opts)
