# =============================================================================
# mesh_overlay.py
#
# cell-edge geometry for 2d field renders.
#
# builds the polylines that trace a logically-rectangular mesh's cell edges,
# expressed in the coordinates the axes draws in. the vertex arrays are the
# ones carried by the checkpoint, so the drawn grid is the grid the solver
# used: a graded (logarithmic / geometric) spacing shows its grading, and a
# homologously expanding mesh moves its edges between checkpoints.
#
# on a polar axes a constant-radius edge is an arc and is sampled in angle;
# a constant-angle edge is a radial ray and needs only its two endpoints.
#
# usage:
#   segs = mesh_segments(theta_edges, r_edges, curved=True)
#   collection.set_segments(segs)
# =============================================================================
from typing import Sequence

import numpy as np

from ..types import Array

# an axis with more coordinate lines than this is decimated before drawing.
# past roughly this count the overlay saturates into a solid block and hides
# the field it is drawn over.
DEFAULT_MAX_LINES = 64

# samples along a constant-radius edge on a polar axes. each sample pair is
# drawn as a straight chord, so the sampling sets the chord error; 128 chords
# over a half-plane wedge keeps that error below a pixel at print dpi.
ARC_SAMPLES = 129


def edge_stride(n_edges: int, max_lines: int) -> int:
    """decimation step keeping at most max_lines of n_edges coordinate lines."""
    if max_lines <= 0 or n_edges <= max_lines:
        return 1
    return int(np.ceil((n_edges - 1) / max(max_lines - 1, 1)))


def select_edges(edges: Array, stride: int) -> Array:
    """every stride-th edge, with the two domain boundaries always retained.

    decimating a graded mesh must keep its bounds: dropping the outermost
    radial edge draws a grid that stops short of the field it annotates."""
    if stride <= 1:
        return edges
    kept = edges[::stride]
    if kept[-1] != edges[-1]:
        kept = np.append(kept, edges[-1])
    return kept


def mesh_segments(
    x_edges: Sequence[float] | Array,
    y_edges: Sequence[float] | Array,
    curved: bool = False,
    stride: int = 0,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list[Array]:
    """cell-edge polylines of the mesh spanned by (x_edges, y_edges).

    x is the axes' horizontal coordinate and y its vertical one; on a polar
    axes those are angle and radius respectively, and `curved` must be set so
    constant-radius edges are drawn as arcs rather than chords.

    a stride of 0 decimates each axis to at most max_lines lines; a stride of
    1 draws every edge.

    returns a list of (n, 2) point arrays, the form LineCollection takes.
    """
    xs_all = np.asarray(x_edges, dtype=float)
    ys_all = np.asarray(y_edges, dtype=float)

    if xs_all.ndim != 1 or ys_all.ndim != 1:
        raise ValueError(
            "mesh edges must be 1d vertex arrays, got shapes "
            f"{xs_all.shape} and {ys_all.shape}"
        )

    # fewer than two vertices on an axis spans no cell, so there is no grid
    if xs_all.size < 2 or ys_all.size < 2:
        return []

    x_step = stride if stride > 0 else edge_stride(xs_all.size, max_lines)
    y_step = stride if stride > 0 else edge_stride(ys_all.size, max_lines)

    # constant-x edges are straight in both charts: a vertical line in
    # cartesian, a radial ray in polar
    y_span = np.array([ys_all[0], ys_all[-1]])
    segments = [
        np.column_stack([np.full(2, xx), y_span])
        for xx in select_edges(xs_all, x_step)
    ]

    # a constant-y edge sweeps an arc on a polar axes and a straight line in
    # cartesian
    x_span = (
        np.linspace(xs_all[0], xs_all[-1], ARC_SAMPLES)
        if curved
        else np.array([xs_all[0], xs_all[-1]])
    )
    segments += [
        np.column_stack([x_span, np.full(x_span.size, yy)])
        for yy in select_edges(ys_all, y_step)
    ]

    return segments
