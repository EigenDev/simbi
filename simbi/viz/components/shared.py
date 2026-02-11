# =============================================================================
# shared.py
#
# shared utilities and base classes for plot components.
# - ColormappedProps: base props for 2d color-mapped components
# - create_color_normalization: log/power norm from data + settings
# - draw_bodies: immersed body rendering
# =============================================================================
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.axes import Axes
from pydantic import ValidationInfo, field_validator

from ..types import Array, ColorRange
from .interface import ComponentProps

try:
    from simbi.reader.io import BodyCollection
except ImportError:
    BodyCollection = None  # type: ignore

LOGICAL_AXIS_MAP: dict[str, int] = {"x1": 0, "x2": 1, "x3": 2}


# =========================================================================
# shared props for 2d color-mapped components (quad, polygon)
# =========================================================================
class ColormappedProps(ComponentProps):
    """base props shared by quad and polygon components."""

    cmap: str = "viridis"
    color_range: ColorRange = ColorRange(min=None, max=None)
    log_scale: bool = False
    power: float = 1.0
    alpha: float = 1.0

    show_mesh_grid: bool = False
    mesh_color: str = "white"
    mesh_alpha: float = 0.3
    mesh_linewidth: float = 0.1

    @field_validator("power")
    @classmethod
    def validate_power(cls, v: float, _: ValidationInfo) -> float:
        if v <= 0:
            raise ValueError(f"Power must be positive, got {v}")
        return v

    @field_validator("alpha", "mesh_alpha")
    @classmethod
    def validate_alpha(cls, v: float, _: ValidationInfo) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {v}")
        return v


# =========================================================================
# color normalization
# =========================================================================
def create_color_normalization(
    values: Array,
    color_range: ColorRange,
    log_scale: bool = False,
    power: float = 1.0,
) -> mcolors.Normalize:
    """create color normalization based on data and settings."""
    vmin = color_range.min if color_range.min is not None else np.nanmin(values)
    vmax = color_range.max if color_range.max is not None else np.nanmax(values)

    if np.allclose(vmin, vmax, rtol=1e-10):
        eps = max(float(abs(vmin) * 1e-2), 0.1)
        vmin -= eps
        vmax += eps

    if log_scale:
        if vmin <= 0:
            pos_min = (
                np.nanmin(values[values > 0]) if np.any(values > 0) else 1e-10
            )
            vmin = pos_min * 0.9
        return mcolors.LogNorm(vmin=float(vmin), vmax=float(vmax))
    else:
        return mcolors.PowerNorm(
            gamma=power, vmin=float(vmin), vmax=float(vmax)
        )


# =========================================================================
# body drawing
# =========================================================================
def draw_bodies(
    ax: Axes,
    body_collection: "BodyCollection",
    zorder: int,
    axes: Sequence[str],
) -> None:
    """draw immersed bodies on a 2d plot as circles."""
    for patch in ax.patches:
        patch.remove()

    n_i = LOGICAL_AXIS_MAP[axes[0]]
    n_j = LOGICAL_AXIS_MAP[axes[1]]

    for body in body_collection.bodies:
        radius = body.radius
        if body.accretion is not None:
            radius = body.accretion.accretion_radius
        position = (body.position[n_i], body.position[n_j])

        circle = mpatches.Circle(
            position,
            radius,
            color="black",
            linestyle="--",
            alpha=0.5,
            zorder=zorder,
        )
        ax.add_patch(circle)
