from typing import Optional, Sequence

from simbi.reader.adapter import SimData
from simbi.reader.computation import FieldComputationError

from ..config import VisualizationConfig
from ..types import CoordSystem, FieldData, PlotData
from .refinement import (
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
    Handles refinement logic to produce a list of full-dimensional fields.
    """
    ndim = get_effective_dimensions(data, config)

    # get refinement settings from config
    # None signals "all levels", empty set means just base level
    active_levels_raw = None
    if hasattr(config, "refinement"):
        active_levels_raw = config.refinement.active_levels

    # expand "all" (None) to actual set of all levels
    if active_levels_raw is None:
        # "all" was requested - use all available levels
        active_levels = set(range(data.num_levels))
    elif len(active_levels_raw) == 0:
        # no levels specified - default to base level only
        active_levels = {0}
    else:
        # specific levels requested
        active_levels = active_levels_raw

    use_composite = False
    if hasattr(config, "refinement"):
        use_composite = data.has_refinement() and getattr(
            config.refinement, "composite_view", False
        )

    # determine whether to crop to the owned region
    # crop when visualizing a single refined level only
    single_level = len(active_levels) == 1
    crop_to_owned = single_level and not use_composite

    all_fields: list[FieldData] = []
    for field_name in field_names:
        if use_composite:
            # prepare one composite field
            field_data = prepare_composite_field(
                data, field_name, active_levels, ndim
            )
            all_fields.append(field_data)
        else:
            # prepare one field for each active level. a field genuinely
            # absent from one level is skipped; a field that comes up empty
            # across every level raises here by name, since dropping it
            # silently yields an empty PlotData whose validation error
            # ("dimensions = 0") points nowhere near the cause.
            produced = 0
            last_missing: KeyError | None = None
            for level in sorted(active_levels):
                if level >= data.num_levels:
                    continue
                try:
                    # crop to owned region when showing single level
                    should_crop = crop_to_owned and level > 0
                    field_data = prepare_field_level(
                        data, field_name, level, ndim, crop_to_owned=should_crop
                    )
                    all_fields.append(field_data)
                    produced += 1
                except KeyError as exc:
                    last_missing = exc
                    continue
            if produced == 0:
                raise FieldComputationError(
                    f"field '{field_name}' is not available in this"
                    f" checkpoint (tried levels {sorted(active_levels)});"
                    f" available: {sorted(data.available_fields())}"
                ) from last_missing

    return all_fields


def apply_slicing(
    fields: list[FieldData], slice_spec: Optional[dict[str, float]]
) -> list[FieldData]:
    """
    Applies a slice_spec to a list of full-dimensional fields.

    Each field may have different resolution (different levels),
    so slice planning happens per-field using physical coordinates.
    """
    if not slice_spec:
        # no slicing requested, return the fields as-is
        return fields

    sliced_fields: list[FieldData] = []
    for field in fields:
        # plan and execute the slice for THIS field's domain
        # find_slice_index uses physical position, so works across levels
        plan = plan_slice(field.domain, slice_spec)
        sliced_values, sliced_domain = execute_slice(
            field.values, field.domain, plan
        )

        # preserve spacing_types through the slice
        sliced_spacing_types = None
        if field.spacing_types and plan.final_domain_indices:
            sliced_spacing_types = [
                field.spacing_types[i] for i in plan.final_domain_indices
            ]

        # create a new FieldData object with the sliced data
        sliced_fields.append(
            FieldData(
                name=field.name,
                values=sliced_values,
                domain=list(sliced_domain),
                spacing_types=sliced_spacing_types,
                axis_names=plan.final_axis_names,
                time=field.time,
            )
        )

    return sliced_fields


def create_plot_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """
    The main pipeline function.

    > Prepares full-dim refinement levels (respecting composite_view).
    > Applies optional slicing.
    """
    # handles refinement logic (gets full-dim fields)
    full_dim_fields = prepare_fields(data, field_names, config)

    # get slice spec from config
    slice_spec: Optional[dict[str, float]] = None
    if hasattr(config, "plot") and hasattr(config.plot, "slice"):
        slice_spec = config.plot.slice

    # apply slicing
    sliced_fields = apply_slicing(full_dim_fields, slice_spec)

    # field-value normalization: divide each field by a constant, or its own "max"/"min"
    # (M&DZ-style "relative to max value": pass the number, e.g., max(initial)).
    norm = getattr(getattr(config, "plot", None), "norm", None)
    if norm is not None and sliced_fields:
        import numpy as np

        def _denom(arr: "np.ndarray") -> float:
            if norm == "max":
                d = float(np.nanmax(arr))
            elif norm == "min":
                d = float(np.nanmin(arr))
            else:
                try:
                    d = float(norm)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"--norm must be a number, 'max', or 'min'; got '{norm}'"
                    )
            return d if d != 0.0 else 1.0

        sliced_fields = [
            fd.model_copy(update={"values": fd.values / _denom(fd.values)})
            for fd in sliced_fields
        ]

    # an empty field list would fail PlotData validation with a message
    # ("dimensions = 0") far from the cause; name the inputs instead.
    if not sliced_fields:
        raise FieldComputationError(
            f"no plottable fields were produced for {list(field_names)}"
            + (f" with slice {slice_spec}" if slice_spec else "")
        )

    # package and return the (potentially un-stitched) fields
    return PlotData(
        fields=sliced_fields,
        time=data.metadata.time,
        dimensions=sliced_fields[0].ndim if sliced_fields else 0,
        coord_system=CoordSystem(data.metadata.coord_system),
        hierarchy=data.hierarchy() if data.has_refinement() else None,
    )
