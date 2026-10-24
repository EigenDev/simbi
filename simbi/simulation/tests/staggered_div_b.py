# =============================================================================
# staggered_div_b.py
#
# the divergence of the staggered magnetic field a 2D cartesian checkpoint
# stores, computed from its face datasets and its mesh geometry alone, apart
# from any diagnostic the solver writes.
# - reads B1 (ny, nx + 1) and B2 (ny + 1, nx) and the linear spacing of each axis
# - returns max over cells |div B| * min(dx, dy) / max over faces |B|: the
#   discrete divergence in units of the largest field per cell width
# constrained transport changes every face by the curl of edge emfs, whose
# discrete divergence cancels up to rounding: each step adds at most a few ulp
# of max |B| per cell, so a run of N steps stays below about 4 N 2.2e-16.
# DIV_B_TOLERANCE = 1e-12 holds that bound to N ~ 1000 steps.
# usage:
#   assert normalized_div_b(path) <= DIV_B_TOLERANCE
# =============================================================================

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

DIV_B_TOLERANCE = 1e-12


def _spacing(geometry: h5py.Group, axis: int, cells: int) -> float:
    dim = geometry[f"dim_{axis}"].attrs
    assert str(dim["type"]) == "linear", f"axis {axis} is spaced {dim['type']!r}; the check takes linear axes"
    return (float(dim["end"]) - float(dim["start"])) / cells


def normalized_div_b(path: Path) -> float:
    """max over cells |div B| min(dx, dy) / max over faces |B| of the checkpoint's level 0"""
    with h5py.File(path, "r") as f:
        level = f["level_0"]
        geometry = level["mesh/geometry"]
        assert str(geometry.attrs["metric"]) == "cartesian", f"metric {geometry.attrs['metric']!r}"
        b1 = level["partition_0/hydro/magnetic/B1/data"][()]
        b2 = level["partition_0/hydro/magnetic/B2/data"][()]
        ny, nx = b1.shape[0], b1.shape[1] - 1
        assert b2.shape == (ny + 1, nx), f"face shapes {b1.shape}, {b2.shape}"
        dx, dy = _spacing(geometry, 0, nx), _spacing(geometry, 1, ny)
    scale = max(float(np.abs(b1).max()), float(np.abs(b2).max()))
    assert scale > 0.0, f"{path} carries no magnetic field; the divergence check is vacuous"
    div = (b1[:, 1:] - b1[:, :-1]) / dx + (b2[1:, :] - b2[:-1, :]) / dy
    return float(np.abs(div).max()) * min(dx, dy) / scale
