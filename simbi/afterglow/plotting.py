# =============================================================================
# plotting.py
#
# visualization functions for photon event analysis.
# matplotlib-based plotting for lightcurves, skymaps, polarization, spectra.
# =============================================================================

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .postprocess import lightcurve_t, polarization_t, skymap_t, spectrum_t


def plot_lightcurve(lc: lightcurve_t, save: Optional[str] = None) -> None:
    """
    plot observer lightcurve.

    args:
        lc: lightcurve data
        save: filename to save (None=show)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for nu in lc.frequencies:
        flux = lc.fluxes[nu]
        label = f"{nu:.2e} Hz"
        ax.plot(lc.times, flux, label=label, marker='o', markersize=3)

    ax.set_xlabel("Observer Time [day]")
    ax.set_ylabel("Flux Density [mJy]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Afterglow Lightcurve")

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"saved figure to {save}")
    else:
        plt.show()


def plot_skymap(skymap: skymap_t, save: Optional[str] = None) -> None:
    """
    plot sky intensity map.

    args:
        skymap: skymap data
        save: filename to save (None=show)
    """
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "mollweide"})

    # convert to longitude/latitude for mollweide
    phi_grid, theta_grid = np.meshgrid(
        skymap.phi - np.pi,  # shift to [-pi, pi]
        skymap.theta - np.pi / 2  # shift to [-pi/2, pi/2]
    )

    # plot
    im = ax.pcolormesh(
        phi_grid,
        theta_grid,
        skymap.intensity,
        cmap="viridis",
        shading="auto"
    )

    ax.set_xlabel("Azimuth [rad]")
    ax.set_ylabel("Elevation [rad]")
    ax.set_title(f"Sky Map at t = {skymap.time:.3f} day")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Intensity [arbitrary]")

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"saved figure to {save}")
    else:
        plt.show()


def plot_polarization(pol: polarization_t, save: Optional[str] = None) -> None:
    """
    plot polarization evolution.

    args:
        pol: polarization data
        save: filename to save (None=show)
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # top panel: polarization degree
    ax1 = axes[0]
    ax1.plot(pol.times, pol.polarization_degree * 100, marker='o', markersize=3)
    ax1.set_ylabel("Polarization Degree [%]")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Polarization Evolution")

    # bottom panel: polarization angle
    ax2 = axes[1]
    ax2.plot(pol.times, np.rad2deg(pol.polarization_angle), marker='o', markersize=3, color='C1')
    ax2.set_xlabel("Observer Time [day]")
    ax2.set_ylabel("Polarization Angle [deg]")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"saved figure to {save}")
    else:
        plt.show()


def plot_spectrum(spec: spectrum_t, save: Optional[str] = None) -> None:
    """
    plot spectral flux.

    args:
        spec: spectrum data
        save: filename to save (None=show)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(spec.frequencies, spec.fluxes, marker='o', markersize=4)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Flux Density [mJy]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Spectrum at t = {spec.time:.3f} day")

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print(f"saved figure to {save}")
    else:
        plt.show()
