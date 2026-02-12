# =============================================================================
# coord_binning.py
#
# thin wrapper around simbi.analysis.radial_profiles.
# unpacks PlotData into plain arrays, calls pure functions,
# packages results into FieldData/PlotData.
# =============================================================================
from typing import Sequence

import numpy as np

from simbi.analysis import (
    mass_flux_profile,
    momentum_equation_terms,
    spherical_profile,
    stitch_leaf_cells,
)
from simbi.reader.adapter import SimData

from ..config import VisualizationConfig
from ..types import CoordSystem, FieldData, PlotData
from .plot_data import prepare_fields


def _unpack_to_stitch_args(
    plot_data: PlotData, field_names: list[str]
) -> tuple[list[list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """
    unpack PlotData fields into the (level_domains, level_values) format
    expected by stitch_leaf_cells.
    """
    level_fields_map: dict[str, list[FieldData]] = {}
    all_levels = set()

    for name in field_names:
        level_fields_map[name] = [
            f for f in plot_data.fields if f.name.startswith(name)
        ]
        if not level_fields_map[name]:
            raise ValueError(f"no fields found for base name: {name}")
        level_fields_map[name].sort(key=lambda f: f.name)
        all_levels.update(range(len(level_fields_map[name])))

    num_levels = len(all_levels)
    if num_levels == 0:
        raise ValueError("no fields found for any requested name")

    # build per-level domain lists and value arrays
    level_domains: list[list[np.ndarray]] = []
    level_values: dict[str, list[np.ndarray]] = {n: [] for n in field_names}

    for level_idx in range(num_levels):
        ref_field = level_fields_map[field_names[0]][level_idx]
        domain_arrays = list(ref_field.domain)
        level_domains.append(domain_arrays)

        for name in field_names:
            level_values[name].append(
                level_fields_map[name][level_idx].values
            )

    return level_domains, level_values


def create_coordinate_profile_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """
    pipeline for coordinate profile analysis.

    stitches 3D refined data and computes spherically averaged profiles.
    """
    # determine prerequisite raw fields
    prerequisite_fields = set()
    for name in field_names:
        if name == "mdot":
            prerequisite_fields.update(["rho", "v1", "v2", "v3"])
        elif name == "momentum_terms":
            prerequisite_fields.update(["rho", "v1", "v2", "v3", "p"])
        else:
            prerequisite_fields.add(name)

    # load all refinement levels for prerequisites
    refined_plot_data = PlotData(
        fields=prepare_fields(data, list(prerequisite_fields), config),
    )

    # unpack into plain arrays and stitch
    level_domains, level_values = _unpack_to_stitch_args(
        refined_plot_data, list(prerequisite_fields)
    )
    stitched_data = stitch_leaf_cells(level_domains, level_values)

    # run requested analyses
    final_fields: list[FieldData] = []
    n_bins = getattr(config.coordinate, "n_bins", 100)

    for name in field_names:
        if name == "mdot":
            bin_centers, mdot_vals = mass_flux_profile(stitched_data, n_bins)
            final_fields.append(
                FieldData(
                    name="mdot_vs_r",
                    values=mdot_vals,
                    domain=[bin_centers],
                    spacing_types=["linear"],
                    time=data.metadata.time,
                )
            )
        elif name == "momentum_terms":
            terms = momentum_equation_terms(
                stitched_data, n_bins, data.metadata.gamma
            )
            for term_name, (bin_centers, vals) in terms.items():
                final_fields.append(
                    FieldData(
                        name=f"term_{term_name}",
                        values=vals,
                        domain=[bin_centers],
                        spacing_types=["linear"],
                    )
                )
        else:
            bin_centers, mean_vals = spherical_profile(
                stitched_data, name, n_bins
            )
            final_fields.append(
                FieldData(
                    name=f"{name}_vs_r",
                    values=mean_vals,
                    domain=[bin_centers],
                    spacing_types=["linear"],
                    time=data.metadata.time,
                )
            )

    return PlotData(
        fields=final_fields,
        body_collection=data.body_collection,
        time=data.metadata.time,
        dimensions=1,
        coord_system=CoordSystem(data.metadata.coord_system),
        hierarchy=data.hierarchy() if data.has_refinement() else None,
    )
