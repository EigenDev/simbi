from typing import Optional, Sequence

from simbi.reader.lazy import SimData

from ..config import VisualizationConfig
from ..types import CoordSystem, FieldData, PlotData
from .fmr import (
    prepare_composite_field,
)
from .transforms import (
    execute_slice,
    get_effective_dimensions,
    plan_slice,
    prepare_field_level,
)


def prepare_fields(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> list[FieldData]:
    """
    Handles FMR logic to produce a list of full-dimensional fields.
    """
    ndim = get_effective_dimensions(
        data, config
    )  # Assuming you have this helper

    # Get FMR settings from the config
    active_levels = {0}
    if hasattr(config, "fmr") and config.fmr.active_levels is not None:
        active_levels.update(config.fmr.active_levels)

    use_composite = False
    if hasattr(config, "fmr"):
        use_composite = data.has_refinement() and getattr(
            config.fmr, "composite_view", False
        )

    all_fields: list[FieldData] = []
    for field_name in field_names:
        if use_composite:
            print(f"Preparing composite field for '{field_name}'")
            # Prepare one composite field
            field_data = prepare_composite_field(
                data, field_name, active_levels, ndim
            )
            all_fields.append(field_data)
        else:
            # Prepare one field for each active level
            for level in sorted(active_levels):
                if level >= data.num_levels:
                    continue
                try:
                    field_data = prepare_field_level(
                        data, field_name, level, ndim
                    )
                    all_fields.append(field_data)
                except KeyError:
                    continue

    return all_fields


def apply_slicing(
    fields: list[FieldData], slice_spec: Optional[dict[str, float]]
) -> list[FieldData]:
    """
    Applies a slice_spec to a list of full-dimensional fields.
    """
    if not slice_spec:
        # No slicing requested, return the fields as-is
        return fields

    sliced_fields: list[FieldData] = []
    for field in fields:
        print(f"Applying slicing to field '{field.name}'")
        # Plan and execute the slice
        plan = plan_slice(field.domain, slice_spec)
        sliced_values, sliced_domain = execute_slice(
            field.values, field.domain, plan
        )

        # Create a new FieldData object with the sliced data
        sliced_fields.append(
            FieldData(
                name=field.name,
                values=sliced_values,
                domain=list(sliced_domain),
                axis_names=plan.final_axis_names,
            )
        )

    return sliced_fields


def create_plot_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """
    The main pipeline function.

    > Prepares full-dim FMR levels (respecting composite_view).
    > Applies optional slicing.
    """
    # Handles FMR logic (gets full-dim fields)
    full_dim_fields = prepare_fields(data, field_names, config)

    # Get slice spec from config
    slice_spec: Optional[dict[str, float]] = None
    if hasattr(config, "plot") and hasattr(config.plot, "slice"):
        slice_spec = config.plot.slice

    # Apply Slicing
    sliced_fields = apply_slicing(full_dim_fields, slice_spec)

    # Package and return the (potentially un-stitched) fields
    return PlotData(
        fields=sliced_fields,
        bodies=data.bodies,
        time=data.metadata.time,
        dimensions=sliced_fields[0].ndim if sliced_fields else 0,
        coord_system=CoordSystem(data.metadata.coord_system),
        hierarchy=data.hierarchy() if data.has_refinement() else None,
    )
