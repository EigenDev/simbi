# =============================================================================
# radial_profiles.py
#
# spherically-averaged radial profile analysis for AMR simulation data.
# leaf-cell stitching, volume-weighted profiles, mass flux, and momentum
# equation decomposition. pure numpy/scipy -- no viz dependency.
#
# all profiles use logarithmic radial bins and volume-weighted averages:
#   <q>(r) = sum(q_i * dV_i) / sum(dV_i)   for cells i in shell
#
# usage:
#   from simbi.analysis import stitch_leaf_cells, spherical_profile
#   stitched = stitch_leaf_cells(level_domains, level_values)
#   centers, vals = spherical_profile(stitched, "rho", n_bins=100)
# =============================================================================
import numpy as np
from scipy.stats import binned_statistic


def _cell_radii(stitched_data: dict[str, np.ndarray]) -> np.ndarray:
    """compute radial distance of each leaf cell from the origin."""
    x = stitched_data["x_flat"]
    y = stitched_data["y_flat"]
    z = stitched_data.get("z_flat", np.zeros_like(x))
    return np.sqrt(x**2 + y**2 + z**2)


def _log_bins(r_flat: np.ndarray, n_bins: int) -> np.ndarray:
    """logarithmically-spaced radial bin edges."""
    r_pos = r_flat[r_flat > 0]
    if len(r_pos) == 0:
        return np.linspace(0, 1, n_bins + 1)
    r_min = np.min(r_pos)
    r_max = np.max(r_flat)
    return np.geomspace(r_min, r_max, n_bins + 1)


def _volume_weighted_mean(
    r_flat: np.ndarray,
    values: np.ndarray,
    volume: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    volume-weighted shell average of a scalar field.

    returns (bin_centers, weighted_mean, binnumber).
    """
    weighted_sum, bin_edges, binnumber = binned_statistic(
        r_flat, values * volume, statistic="sum", bins=bins,
    )
    total_vol, _, _ = binned_statistic(
        r_flat, volume, statistic="sum", bins=bins,
    )
    mean = weighted_sum / np.where(total_vol > 0, total_vol, 1.0)
    # geometric bin centers for log-spaced bins
    centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    return centers, mean, binnumber


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

                    # leaf cell -- keep it
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
    volume-weighted spherically-averaged radial profile with log bins.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        field_name: base field name (looked up as "{field_name}_flat")
        n_bins: number of radial bins

    returns:
        (bin_centers, mean_values)
    """
    r_flat = _cell_radii(stitched_data)
    volume = stitched_data["volume_flat"]
    val_flat = stitched_data[f"{field_name}_flat"]
    bins = _log_bins(r_flat, n_bins)

    centers, mean_val, _ = _volume_weighted_mean(r_flat, val_flat, volume, bins)
    return centers, mean_val


def mass_flux_profile(
    stitched_data: dict[str, np.ndarray],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute radial mass flux profile with volume weighting and log bins.

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

    bins = _log_bins(r_flat, n_bins)
    centers, mean_flux_density, _ = _volume_weighted_mean(
        r_flat, flux_density_flat, volume_flat, bins,
    )

    shell_area = 4.0 * np.pi * centers**2
    mdot_profile = mean_flux_density * shell_area

    return centers, mdot_profile


def turbulent_velocity_sq_profile(
    stitched_data: dict[str, np.ndarray],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    volume-weighted spherically-averaged turbulent velocity squared profile.

    for each radial bin, computes <|v - <v>|^2> where <v> is the
    volume-weighted bin-averaged velocity vector and the outer average
    is volume-weighted over cells in the bin.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        n_bins: number of radial bins

    returns:
        (bin_centers, mean_v_turb_sq)
    """
    r_flat = _cell_radii(stitched_data)
    volume = stitched_data["volume_flat"]
    vx = stitched_data["v1_flat"]
    vy = stitched_data["v2_flat"]
    vz = stitched_data.get("v3_flat", np.zeros_like(r_flat))

    bins = _log_bins(r_flat, n_bins)

    # volume-weighted mean velocity per shell
    centers, mean_vx, binnumber = _volume_weighted_mean(
        r_flat, vx, volume, bins,
    )
    _, mean_vy, _ = _volume_weighted_mean(r_flat, vy, volume, bins)
    _, mean_vz, _ = _volume_weighted_mean(r_flat, vz, volume, bins)

    # compute |v - <v>|^2 per cell using its bin's mean velocity
    idx = np.clip(binnumber - 1, 0, n_bins - 1)
    dvx = vx - mean_vx[idx]
    dvy = vy - mean_vy[idx]
    dvz = vz - mean_vz[idx]
    v_turb_sq = dvx**2 + dvy**2 + dvz**2

    # volume-weighted average of squared fluctuation per shell
    _, mean_v_turb_sq, _ = _volume_weighted_mean(
        r_flat, v_turb_sq, volume, bins,
    )

    return centers, mean_v_turb_sq


def radial_velocity_profile(
    stitched_data: dict[str, np.ndarray],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    volume-weighted spherically-averaged radial velocity profile.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        n_bins: number of radial bins

    returns:
        (bin_centers, mean_vr)
    """
    x_flat = stitched_data["x_flat"]
    y_flat = stitched_data["y_flat"]
    z_flat = stitched_data.get("z_flat", np.zeros_like(x_flat))

    vx = stitched_data["v1_flat"]
    vy = stitched_data["v2_flat"]
    vz = stitched_data.get("v3_flat", np.zeros_like(x_flat))

    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    vr_flat = (vx * x_flat + vy * y_flat + vz * z_flat) / (r_flat + 1e-10)

    volume = stitched_data["volume_flat"]
    bins = _log_bins(r_flat, n_bins)
    centers, mean_vr, _ = _volume_weighted_mean(r_flat, vr_flat, volume, bins)

    return centers, mean_vr


def sound_speed_profile(
    stitched_data: dict[str, np.ndarray],
    n_bins: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    volume-weighted spherically-averaged sound speed profile.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        n_bins: number of radial bins
        gamma: adiabatic index

    returns:
        (bin_centers, mean_cs)
    """
    r_flat = _cell_radii(stitched_data)
    volume = stitched_data["volume_flat"]

    rho_flat = stitched_data["rho_flat"]
    p_flat = stitched_data.get("p_flat", rho_flat)
    cs_flat = np.sqrt(gamma * p_flat / (rho_flat + 1e-10))

    bins = _log_bins(r_flat, n_bins)
    centers, mean_cs, _ = _volume_weighted_mean(r_flat, cs_flat, volume, bins)

    return centers, mean_cs


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
    volume = stitched_data["volume_flat"]

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

    bins = _log_bins(r_flat, n_bins)

    centers, mean_vr, _ = _volume_weighted_mean(r_flat, vr_flat, volume, bins)
    _, mean_rho, _ = _volume_weighted_mean(r_flat, rho_flat, volume, bins)
    _, mean_p, _ = _volume_weighted_mean(r_flat, p_flat, volume, bins)

    # advection: rho * v_r * dv_r/dr
    dvr_dr = np.gradient(mean_vr, centers)
    term_advection = mean_rho * mean_vr * dvr_dr

    # pressure gradient: -dP/dr
    dp_dr = np.gradient(mean_p, centers)
    term_pressure = -dp_dr

    # gravity: -rho * GM / r^2
    term_gravity = -mean_rho * GM / (centers**2 + 1e-10)

    # residual
    term_residual = term_advection - (term_pressure + term_gravity)

    return {
        "advection": (centers, term_advection),
        "pressure": (centers, term_pressure),
        "gravity": (centers, term_gravity),
        "residual": (centers, term_residual),
    }


def time_average_profiles(
    snapshots: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    time-average a sequence of per-snapshot radial profiles.

    each snapshot is a (bin_centers, values) pair from any profile function.
    all snapshots must share the same bin_centers (same n_bins + same domain).

    args:
        snapshots: list of (bin_centers, values) tuples

    returns:
        (bin_centers, time_averaged_values)
    """
    if not snapshots:
        raise ValueError("no snapshots to average")
    centers = snapshots[0][0]
    stacked = np.column_stack([vals for _, vals in snapshots])
    return centers, np.nanmean(stacked, axis=1)
