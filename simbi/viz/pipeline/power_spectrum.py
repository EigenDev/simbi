# =============================================================================
# power_spectrum.py
#
# computes shell-averaged kinetic energy power spectrum E(k) from
# 3D velocity field data. uses base level only (no AMR stitching).
#
# usage:
#   from simbi.viz.pipeline.power_spectrum import create_power_spectrum_data
#   data = load_data("checkpoint.h5")
#   plot_data = create_power_spectrum_data(data, config)
# =============================================================================
from typing import Sequence

import numpy as np
from scipy.stats import binned_statistic

from simbi.reader.adapter import SimData

from ..config import VisualizationConfig
from ..types import FieldData, PlotData


def _compute_shell_averaged_spectrum(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute shell-averaged kinetic energy spectrum from 3D velocity fields.

    args:
        vx, vy, vz: 3D velocity component arrays (same shape)
        dx: uniform cell spacing

    returns:
        (k_centers, E_k): wavenumber bin centers and spectrum values
    """
    nx, ny, nz = vx.shape
    n_cells = nx * ny * nz

    # 3D real FFT of each velocity component
    vx_hat = np.fft.rfftn(vx)
    vy_hat = np.fft.rfftn(vy)
    vz_hat = np.fft.rfftn(vz)

    # kinetic energy density in fourier space
    energy = 0.5 * (
        np.abs(vx_hat) ** 2 + np.abs(vy_hat) ** 2 + np.abs(vz_hat) ** 2
    )

    # parseval normalization
    energy /= n_cells**2

    # wavenumber grid
    kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
    kz = np.fft.rfftfreq(nz, d=dx) * 2.0 * np.pi

    k_mag = np.sqrt(
        kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    )

    # shell-average into radial wavenumber bins
    dk = 2.0 * np.pi / (nx * dx)
    k_max = k_mag.max()
    bin_edges = np.arange(dk, k_max + dk, dk)

    result = binned_statistic(
        k_mag.ravel(),
        energy.ravel(),
        statistic="sum",
        bins=bin_edges,
    )

    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    e_k = result.statistic

    # drop bins with no data
    valid = ~np.isnan(e_k) & (e_k > 0)
    return k_centers[valid], e_k[valid]


def create_power_spectrum_data(
    data: SimData,
    config: VisualizationConfig,
    fields: Sequence[str] = ("v1", "v2", "v3"),
) -> PlotData:
    """
    compute kinetic energy power spectrum from simulation checkpoint.

    uses base level (level 0) only. for AMR data, the spectrum reflects
    the coarsest resolution — fine-level structure is not captured.

    args:
        data: loaded simulation data
        config: visualization configuration
        fields: velocity field names (default: v1, v2, v3)

    returns:
        PlotData with a single 1D FieldData containing E(k) vs k
    """
    # load velocity fields from base level
    vx = data.get_field(fields[0], level=0)
    vy = data.get_field(fields[1], level=0)
    vz = data.get_field(fields[2], level=0)

    # compute uniform cell spacing from mesh
    mesh = data.mesh
    x1v = mesh.x1v
    dx = float(x1v[1] - x1v[0])

    k_centers, e_k = _compute_shell_averaged_spectrum(vx, vy, vz, dx)

    spectrum_field = FieldData(
        name="E_k",
        values=e_k,
        domain=[k_centers],
        time=data.metadata.time,
        axis_names=["k"],
    )

    return PlotData(
        fields=[spectrum_field],
        time=data.metadata.time,
        dimensions=1,
    )
