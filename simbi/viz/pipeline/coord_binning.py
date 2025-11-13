from typing import Sequence

import numpy as np
from scipy.stats import binned_statistic

from simbi.reader.lazy import SimData

from ..config import VisualizationConfig
from ..types import Array, CoordSystem, FieldData, PlotData
from .plot_data import prepare_fields


def _get_stitched_leaf_data(
    data: PlotData, field_names: list[str]
) -> dict[str, np.ndarray]:
    """
    Stitch all FMR levels into high-resolution flat arrays of leaf cells.

    This is the core of the level-aware 3D analysis.
    """
    level_fields_map: dict[str, list[FieldData]] = {}
    all_levels = set()

    # Group the fields by their base name (e.g., 'rho_L0', 'rho_L1')
    for name in field_names:
        level_fields_map[name] = [
            f for f in data.fields if f.name.startswith(name)
        ]
        if not level_fields_map[name]:
            raise ValueError(f"No fields found for base name: {name}")

        level_fields_map[name].sort(key=lambda f: f.name)
        all_levels.update(range(len(level_fields_map[name])))

    num_levels = len(all_levels)
    if num_levels == 0:
        raise ValueError("No fields found for any requested name")

    # Find all refined regions from L1+
    is_3d = False
    refined_regions = []
    for i in range(1, num_levels):
        field_L_i = level_fields_map[field_names[0]][i]
        domain = field_L_i.domain
        is_3d = len(domain) == 3

        region = {
            "xmin": domain[0][0],
            "xmax": domain[0][-1],
            "ymin": domain[1][0],
            "ymax": domain[1][-1],
        }
        if is_3d:
            region["zmin"] = domain[2][0]
            region["zmax"] = domain[2][-1]
        refined_regions.append(region)

    # Prepare output lists
    stitched_data: dict[str, list] = {
        f"{name}_flat": [] for name in field_names
    }
    stitched_data["x_flat"] = []
    stitched_data["y_flat"] = []
    if is_3d:
        stitched_data["z_flat"] = []

    for level_idx in range(num_levels):
        current_level_fields = {
            name: level_fields_map[name][level_idx] for name in field_names
        }
        domain = current_level_fields[field_names[0]].domain
        values_map = {
            name: current_level_fields[name].values for name in field_names
        }

        x_verts, y_verts = domain[0], domain[1]
        z_verts = domain[2] if is_3d else np.array([0.0, 1.0])

        x_centers = 0.5 * (x_verts[1:] + x_verts[:-1])
        y_centers = 0.5 * (y_verts[1:] + y_verts[:-1])
        z_centers = (
            0.5 * (z_verts[1:] + z_verts[:-1]) if is_3d else np.array([0.0])
        )

        nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)

        for k in range(nz):
            zc = z_centers[k]
            for j in range(ny):
                yc = y_centers[j]
                for i in range(nx):
                    xc = x_centers[i]

                    # Check if this cell is covered by a *finer* level
                    is_covered = False
                    for region in refined_regions[level_idx:]:
                        covered_2d = (
                            region["xmin"] <= xc <= region["xmax"]
                            and region["ymin"] <= yc <= region["ymax"]
                        )
                        if not is_3d:
                            if covered_2d:
                                is_covered = True
                                break
                        else:
                            covered_3d = (
                                covered_2d
                                and region["zmin"] <= zc <= region["zmax"]
                            )
                            if covered_3d:
                                is_covered = True
                                break

                    if is_covered:
                        continue

                    # This is a leaf cell. Add its data.
                    stitched_data["x_flat"].append(xc)
                    stitched_data["y_flat"].append(yc)
                    if is_3d:
                        stitched_data["z_flat"].append(zc)

                    for name in field_names:
                        val = (
                            values_map[name][k, j, i]
                            if is_3d
                            else values_map[name][j, i]
                        )
                        stitched_data[f"{name}_flat"].append(val)

    return {key: np.array(val) for key, val in stitched_data.items()}


def _calculate_coordinate_profile(
    stitched_data: dict[str, Array], field_name: str, n_bins: int
) -> FieldData:
    """Calculates a generic spherically averaged profile."""
    x_flat = stitched_data["x_flat"]
    y_flat = stitched_data["y_flat"]
    z_flat = stitched_data.get("z_flat", np.zeros_like(x_flat))

    val_flat = stitched_data[f"{field_name}_flat"]

    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    max_radius = np.max(r_flat)
    bins = np.linspace(0, max_radius, n_bins + 1)

    mean_val, bin_edges, _ = binned_statistic(
        r_flat, val_flat, statistic="mean", bins=bins
    )
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    name = field_name + "_vs_r"
    return FieldData(name=name, values=mean_val, domain=[bin_centers])


def _calculate_mass_flux_profile(
    stitched_data: dict[str, Array], n_bins: int
) -> FieldData:
    """Calculates the M-dot(r) profile."""
    x_flat = stitched_data["x_flat"]
    y_flat = stitched_data["y_flat"]
    z_flat = stitched_data.get("z_flat", np.zeros_like(x_flat))
    rho_flat = stitched_data["rho_flat"]
    vx_flat = stitched_data["v1_flat"]
    vy_flat = stitched_data["v2_flat"]
    vz_flat = stitched_data.get("v3_flat", np.zeros_like(x_flat))

    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    vr_flat = (vx_flat * x_flat + vy_flat * y_flat + vz_flat * z_flat) / (
        r_flat + 1e-10
    )
    flux_density_flat = rho_flat * vr_flat

    max_radius = np.max(r_flat)
    bins = np.linspace(0, max_radius, n_bins + 1)

    # We need the *mean* flux density in each shell
    mean_flux_density, bin_edges, _ = binned_statistic(
        r_flat, flux_density_flat, statistic="mean", bins=bins
    )

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    shell_area = 4.0 * np.pi * bin_centers**2
    mass_flux_profile = mean_flux_density * shell_area
    label = "mdot_vs_r"
    return FieldData(
        name=label,
        values=mass_flux_profile,
        domain=[bin_centers],
    )


def create_coordinate_profile_data(
    data: SimData, field_names: Sequence[str], config: VisualizationConfig
) -> PlotData:
    """
    The pipeline for coordinate profile analysis.

    Stitches 3D FMR data and computes spherically averaged profiles.
    """

    # This is our "contract":
    # What raw fields do we need for the requested analyses?
    prerequisite_fields = set()
    for name in field_names:
        if name == "mdot":
            prerequisite_fields.update(["rho", "v1", "v2", "v3"])
        else:
            prerequisite_fields.add(name)

    # Load *all* full-dim FMR levels for the prerequisites
    fmr_plot_data = PlotData(
        fields=prepare_fields(data, list(prerequisite_fields), config),
        # ... (other PlotData fields) ...
    )

    # Stitch all prerequisite fields into flat arrays
    stitched_data = _get_stitched_leaf_data(
        fmr_plot_data, list(prerequisite_fields)
    )

    # Run the requested analyses
    final_fields: list[FieldData] = []
    n_bins = getattr(config.coordinate, "n_bins", 100)

    for name in field_names:
        if name == "mdot":
            profile_data = _calculate_mass_flux_profile(stitched_data, n_bins)
            final_fields.append(profile_data)
        else:
            profile_data = _calculate_coordinate_profile(
                stitched_data, name, n_bins
            )
            final_fields.append(profile_data)

    return PlotData(
        fields=final_fields,
        bodies=data.bodies,
        time=data.metadata.time,
        dimensions=1,  # The result is always 1D
        coord_system=CoordSystem(data.metadata.coord_system),
        hierarchy=data.hierarchy() if data.has_refinement() else None,
    )
