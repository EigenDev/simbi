# =============================================================================
# shared.py
#
# shared utilities for plot components.
# =============================================================================
from typing import Sequence

import matplotlib.patches as mpatches
from matplotlib.axes import Axes

from simbi.reader.io import BodyCollection

LOGICAL_AXIS_MAP: dict[str, int] = {"x1": 0, "x2": 1, "x3": 2}


def draw_bodies(
    ax: Axes,
    body_collection: BodyCollection,
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
