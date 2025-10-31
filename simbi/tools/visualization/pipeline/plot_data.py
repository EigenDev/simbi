from typing import Sequence

from ....reader.lazy import SimData
from ..core.config import VisualizationConfig
from ..core.types import CoordSystem, FieldData, PlotData
from .fmr import (
    transform_composite_field,
)
from .transforms import (
    get_effective_dimensions,
    get_slice_config,
    transform_field,
)


def create_plot_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """Create plot data with support for both individual and composite FMR views"""
    ndim = get_effective_dimensions(data, config)
    slice_config = get_slice_config(config)

    # Determine which levels to include
    active_levels = {0}  # Always include base level
    if data.has_refinement() and getattr(config.multidim, "active_levels"):
        active_levels.update(config.multidim.active_levels)

    # Determine if we should create composite view
    use_composite = data.has_refinement() and getattr(
        config.multidim, "composite_view", False
    )

    all_fields: list[FieldData] = []
    for field_name in field_names:
        if use_composite:
            # Create single composite view
            field_data = transform_composite_field(
                data, field_name, active_levels, ndim, slice_config
            )
            all_fields.append(field_data)
        else:
            # Create individual level views
            for level in sorted(active_levels):
                if level >= data.num_levels:
                    continue

                try:
                    field_data = transform_field(
                        data, field_name, level, ndim, slice_config
                    )
                    all_fields.append(field_data)
                except KeyError:
                    continue

    return PlotData(
        fields=all_fields,
        bodies=data.bodies,
        time=data.metadata.time,
        dimensions=ndim,
        coord_system=CoordSystem(data.metadata.coord_system),
        hierarchy=data.hierarchy() if data.has_refinement() else None,
    )
