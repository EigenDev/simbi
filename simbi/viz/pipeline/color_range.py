# =============================================================================
# color_range.py
#
# the colour scale a movie is drawn on.
#
# a scale taken from each frame's own extremes gives a decaying shock the same
# colours in every frame. colour is read as an absolute quantity, so such a
# movie reports a constant where the data has a decay of decades -- and it is
# the default a plotting library falls into, because one frame at a time has no
# way to know better.
#
# the extremes are swept over the whole sequence once instead and pinned before
# the first frame. reading a checkpoint's fields costs milliseconds against the
# tens a frame takes to draw.
#
# usage:
#   ranges = sequence_color_range(files, ["rho", "u"], config)
#   props = props.model_copy(update={"color_range": ranges["rho"]})
# =============================================================================
from typing import Sequence

import numpy as np

from ..config import VisualizationConfig
from ..types import ColorRange
from .panels import base_field_name


def sequence_color_range(
    files: Sequence[str],
    fields: Sequence[str],
    config: VisualizationConfig,
) -> dict[str, ColorRange]:
    """The extremes of each quantity across every checkpoint in `files`.

    The fields are prepared exactly as they will be drawn -- same slice, same
    levels, same normalization -- so the pinned scale is the scale of what
    ends up on screen.

    Keyed by the quantity's base name, so every refinement level of a field
    shares one scale: levels drawn on scales of their own would put a seam at
    every refinement boundary.
    """
    from .plot_data import create_plot_data
    from .transforms import load_data

    extremes: dict[str, tuple[float, float]] = {}

    for path in files:
        plot_data = create_plot_data(load_data(path), fields, config)

        for field in plot_data.fields:
            finite = field.values[np.isfinite(field.values)]
            if finite.size == 0:
                continue

            name = base_field_name(field.name)
            low, high = extremes.get(name, (np.inf, -np.inf))
            extremes[name] = (
                min(low, float(finite.min())),
                max(high, float(finite.max())),
            )

    return {
        name: ColorRange(min=low, max=high)
        for name, (low, high) in extremes.items()
        if low <= high
    }
