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
    level. only leaf (uncovered) cells are kept. uses vectorized numpy
    operations over entire levels instead of per-cell python loops.

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

    # collect refined region bounds for vectorized coverage checks
    refined_bounds: list[tuple] = []
    for ii in range(1, num_levels):
        domain = level_domains[ii]
        if is_3d:
            refined_bounds.append((
                domain[0][0], domain[0][-1],
                domain[1][0], domain[1][-1],
                domain[2][0], domain[2][-1],
            ))
        else:
            refined_bounds.append((
                domain[0][0], domain[0][-1],
                domain[1][0], domain[1][-1],
            ))

    # accumulate chunks per level, concatenate once at the end
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    vol_chunks: list[np.ndarray] = []
    field_chunks: dict[str, list[np.ndarray]] = {n: [] for n in field_names}

    for level_idx in range(num_levels):
        domain = level_domains[level_idx]
        x_verts, y_verts = domain[0], domain[1]

        x_centers = 0.5 * (x_verts[1:] + x_verts[:-1])
        y_centers = 0.5 * (y_verts[1:] + y_verts[:-1])
        dx = x_verts[1] - x_verts[0]
        dy = y_verts[1] - y_verts[0]

        if is_3d:
            z_verts = domain[2]
            z_centers = 0.5 * (z_verts[1:] + z_verts[:-1])
            dz = z_verts[1] - z_verts[0]
            cell_volume = dx * dy * dz

            # 3d meshgrid: shape (nz, ny, nx)
            ZZ, YY, XX = np.meshgrid(
                z_centers, y_centers, x_centers, indexing="ij",
            )

            # vectorized coverage: OR across all finer regions
            covered = np.zeros(XX.shape, dtype=bool)
            for bounds in refined_bounds[level_idx:]:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds
                covered |= (
                    (XX >= xmin) & (XX <= xmax)
                    & (YY >= ymin) & (YY <= ymax)
                    & (ZZ >= zmin) & (ZZ <= zmax)
                )

            leaf = ~covered
            x_chunks.append(XX[leaf])
            y_chunks.append(YY[leaf])
            z_chunks.append(ZZ[leaf])
            vol_chunks.append(np.full(int(leaf.sum()), cell_volume))

            for name in field_names:
                field_chunks[name].append(level_values[name][level_idx][leaf])
        else:
            cell_volume = dx * dy

            # 2d meshgrid: shape (ny, nx)
            YY, XX = np.meshgrid(y_centers, x_centers, indexing="ij")

            covered = np.zeros(XX.shape, dtype=bool)
            for bounds in refined_bounds[level_idx:]:
                xmin, xmax, ymin, ymax = bounds
                covered |= (
                    (XX >= xmin) & (XX <= xmax)
                    & (YY >= ymin) & (YY <= ymax)
                )

            leaf = ~covered
            x_chunks.append(XX[leaf])
            y_chunks.append(YY[leaf])
            vol_chunks.append(np.full(int(leaf.sum()), cell_volume))

            for name in field_names:
                field_chunks[name].append(level_values[name][level_idx][leaf])

    result: dict[str, np.ndarray] = {
        "x_flat": np.concatenate(x_chunks),
        "y_flat": np.concatenate(y_chunks),
        "volume_flat": np.concatenate(vol_chunks),
    }
    if is_3d:
        result["z_flat"] = np.concatenate(z_chunks)
    for name in field_names:
        result[f"{name}_flat"] = np.concatenate(field_chunks[name])

    return result


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
    volume-weighted spherically-averaged tangential velocity squared profile.

    computes v_t^2 = |v|^2 - v_r^2 per cell, then shell-averages.
    this correctly measures non-radial motion without the cancellation
    bug of shell-averaging cartesian components (where <v_x> = 0 by
    symmetry even for coherent radial inflow).

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        n_bins: number of radial bins

    returns:
        (bin_centers, mean_v_t_sq)
    """
    x = stitched_data["x_flat"]
    y = stitched_data["y_flat"]
    z = stitched_data.get("z_flat", np.zeros_like(x))

    vx = stitched_data["v1_flat"]
    vy = stitched_data["v2_flat"]
    vz = stitched_data.get("v3_flat", np.zeros_like(x))

    r_flat = np.sqrt(x**2 + y**2 + z**2)
    volume = stitched_data["volume_flat"]

    # radial velocity: v_r = v . r_hat
    vr = (vx * x + vy * y + vz * z) / (r_flat + 1e-10)

    # tangential component: v_t^2 = |v|^2 - v_r^2
    v_t_sq = (vx**2 + vy**2 + vz**2) - vr**2
    v_t_sq = np.maximum(v_t_sq, 0.0)

    bins = _log_bins(r_flat, n_bins)
    centers, mean_v_t_sq, _ = _volume_weighted_mean(
        r_flat, v_t_sq, volume, bins,
    )

    return centers, mean_v_t_sq


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


def reynolds_delta_v_profile(
    stitched_data: dict[str, np.ndarray],
    mean_vx: np.ndarray,
    mean_vy: np.ndarray,
    mean_vz: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    volume-weighted shell-averaged velocity fluctuation profile.

    uses reynolds decomposition: delta_v = sqrt(<|v - <v>_t|^2>_shell)
    where <v>_t is the time-mean velocity at each cell, passed in as
    mean_vx/vy/vz.

    args:
        stitched_data: flat arrays from stitch_leaf_cells()
        mean_vx, mean_vy, mean_vz: time-averaged velocity per cell
        n_bins: number of radial bins

    returns:
        (bin_centers, delta_v_profile)
    """
    r_flat = _cell_radii(stitched_data)
    volume = stitched_data["volume_flat"]

    vx = stitched_data["v1_flat"]
    vy = stitched_data["v2_flat"]
    vz = stitched_data.get("v3_flat", np.zeros_like(r_flat))

    dv_sq = (vx - mean_vx) ** 2 + (vy - mean_vy) ** 2 + (vz - mean_vz) ** 2

    bins = _log_bins(r_flat, n_bins)
    centers, mean_dv_sq, _ = _volume_weighted_mean(r_flat, dv_sq, volume, bins)

    return centers, np.sqrt(np.maximum(mean_dv_sq, 0.0))


def time_average_profiles(
    snapshots: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    time-average a sequence of per-snapshot radial profiles.

    each snapshot is a (bin_centers, values) pair from any profile function.
    all snapshots must share the same bin_centers (same n_bins + same domain).

    args:
        snapshots: list of (bin_centers, values) tuples

    returns:
        (bin_centers, time_averaged_values, temporal_std)
    """
    if not snapshots:
        raise ValueError("no snapshots to average")
    centers = snapshots[0][0]
    stacked = np.column_stack([vals for _, vals in snapshots])
    return centers, np.nanmean(stacked, axis=1), np.nanstd(stacked, axis=1)
