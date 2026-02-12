# =============================================================================
# radial_profiles.py
#
# spherically-averaged radial profile analysis for AMR simulation data.
# leaf-cell stitching, generic profiles, mass flux, and momentum equation
# decomposition. pure numpy/scipy — no viz dependency.
#
# usage:
#   from simbi.analysis import stitch_leaf_cells, spherical_profile
#   stitched = stitch_leaf_cells(level_domains, level_values)
#   centers, vals = spherical_profile(stitched, "rho", n_bins=100)
# =============================================================================
import numpy as np
from scipy.stats import binned_statistic


def stitch_leaf_cells(
    level_domains: list[list[np.ndarray]],
    level_values: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """
    stitch all refined levels into flat arrays of leaf cells.

    for each cell on each level, checks whether it is covered by a finer
    level. only leaf (uncovered) cells are kept.

    args:
        level_domains: per-level coordinate arrays. each entry is a list of
            1d vertex arrays [x_verts, y_verts] or [x_verts, y_verts, z_verts].
        level_values: mapping from field name to per-level value arrays.
            e.g. {"rho": [rho_L0, rho_L1], "v1": [v1_L0, v1_L1]}

    returns:
        dict with keys like "rho_flat", "v1_flat", "x_flat", "y_flat",
        "volume_flat", and optionally "z_flat".
    """
    num_levels = len(level_domains)
    if num_levels == 0:
        raise ValueError("no levels provided")

    field_names = list(level_values.keys())
    is_3d = len(level_domains[0]) == 3

    # find refined regions from L1+
    refined_regions = []
    for ii in range(1, num_levels):
        domain = level_domains[ii]
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

    # prepare output lists
    stitched_data: dict[str, list] = {f"{name}_flat": [] for name in field_names}
    stitched_data["x_flat"] = []
    stitched_data["y_flat"] = []
    stitched_data["volume_flat"] = []
    if is_3d:
        stitched_data["z_flat"] = []

    for level_idx in range(num_levels):
        domain = level_domains[level_idx]
        x_verts, y_verts = domain[0], domain[1]
        z_verts = domain[2] if is_3d else np.array([0.0, 1.0])

        x_centers = 0.5 * (x_verts[1:] + x_verts[:-1])
        y_centers = 0.5 * (y_verts[1:] + y_verts[:-1])
        z_centers = (
            0.5 * (z_verts[1:] + z_verts[:-1]) if is_3d else np.array([0.0])
        )

        dx = x_verts[1] - x_verts[0]
        dy = y_verts[1] - y_verts[0]
        dz = (z_verts[1] - z_verts[0]) if is_3d else 1.0
        cell_volume = dx * dy * dz

        nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)

        for kk in range(nz):
            zc = z_centers[kk]
            for jj in range(ny):
                yc = y_centers[jj]
                for ii in range(nx):
                    xc = x_centers[ii]

                    # check if covered by a finer level
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
                            if (
                                covered_2d
                                and region["zmin"] <= zc <= region["zmax"]
                            ):
                                is_covered = True
                                break

                    if is_covered:
                        continue

                    # leaf cell — keep it
                    stitched_data["x_flat"].append(xc)
                    stitched_data["y_flat"].append(yc)
                    stitched_data["volume_flat"].append(cell_volume)
                    if is_3d:
                        stitched_data["z_flat"].append(zc)

                    for name in field_names:
                        val = (
                            level_values[name][level_idx][kk, jj, ii]
                            if is_3d
                            else level_values[name][level_idx][jj, ii]
                        )
                        stitched_data[f"{name}_flat"].append(val)

    return {key: np.array(val) for key, val in stitched_data.items()}


def spherical_profile(
    stitched_data: dict[str, np.ndarray],
    field_name: str,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute spherically-averaged radial profile.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        field_name: base field name (looked up as "{field_name}_flat")
        n_bins: number of radial bins

    returns:
        (bin_centers, mean_values)
    """
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

    return bin_centers, mean_val


def mass_flux_profile(
    stitched_data: dict[str, np.ndarray],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute radial mass flux profile with proper volume weighting for AMR.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        n_bins: number of radial bins

    returns:
        (bin_centers, mdot_profile)
    """
    x_flat = stitched_data["x_flat"]
    y_flat = stitched_data["y_flat"]
    z_flat = stitched_data.get("z_flat", np.zeros_like(x_flat))
    volume_flat = stitched_data["volume_flat"]
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

    weighted_flux, bin_edges, _ = binned_statistic(
        r_flat,
        flux_density_flat * volume_flat,
        statistic="sum",
        bins=bins,
    )
    total_volume, _, _ = binned_statistic(
        r_flat, volume_flat, statistic="sum", bins=bins
    )

    mean_flux_density = weighted_flux / (total_volume + 1e-10)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    shell_area = 4.0 * np.pi * bin_centers**2
    mdot_profile = mean_flux_density * shell_area

    return bin_centers, mdot_profile


def momentum_equation_terms(
    stitched_data: dict[str, np.ndarray],
    n_bins: int,
    gamma: float,
    GM: float = 1.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    compute terms of the radial momentum equation.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        n_bins: number of radial bins
        gamma: adiabatic index
        GM: gravitational parameter (default 1.0)

    returns:
        dict mapping term name to (bin_centers, values) tuples.
        keys: "advection", "pressure", "gravity", "residual"
    """
    x_flat = stitched_data["x_flat"]
    y_flat = stitched_data["y_flat"]
    z_flat = stitched_data.get("z_flat", np.zeros_like(x_flat))
    rho_flat = stitched_data["rho_flat"]

    if "p_flat" in stitched_data:
        p_flat = stitched_data["p_flat"]
    else:
        # isothermal dimensionless units where cs=1, P = rho
        p_flat = rho_flat

    vx = stitched_data["v1_flat"]
    vy = stitched_data["v2_flat"]
    vz = stitched_data.get("v3_flat", np.zeros_like(x_flat))

    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    vr_flat = (vx * x_flat + vy * y_flat + vz * z_flat) / (r_flat + 1e-10)

    max_radius = np.max(r_flat)
    bins = np.linspace(0, max_radius, n_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    mean_vr, _, _ = binned_statistic(
        r_flat, vr_flat, statistic="mean", bins=bins
    )
    mean_rho, _, _ = binned_statistic(
        r_flat, rho_flat, statistic="mean", bins=bins
    )
    mean_p, _, _ = binned_statistic(r_flat, p_flat, statistic="mean", bins=bins)

    # advection: rho * v_r * dv_r/dr
    dvr_dr = np.gradient(mean_vr, bin_centers)
    term_advection = mean_rho * mean_vr * dvr_dr

    # pressure gradient: -dP/dr
    dp_dr = np.gradient(mean_p, bin_centers)
    term_pressure = -dp_dr

    # gravity: -rho * GM / r^2
    term_gravity = -mean_rho * GM / (bin_centers**2 + 1e-10)

    # residual
    term_residual = term_advection - (term_pressure + term_gravity)

    return {
        "advection": (bin_centers, term_advection),
        "pressure": (bin_centers, term_pressure),
        "gravity": (bin_centers, term_gravity),
        "residual": (bin_centers, term_residual),
    }
