# =============================================================================
# plotting.py
#
# visualization functions for photon event analysis.
# matplotlib-based plotting for lightcurves, skymaps, polarization, spectra.
# =============================================================================

from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from .postprocess import lightcurve_t, polarization_t, skymap_t, spectrum_t


def convolve_with_beam(
    intensity: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    beam_fwhm_mas: float,
) -> np.ndarray:
    """
    convolve intensity map with gaussian telescope beam.

    args:
        intensity: 2D intensity array
        x_grid: x coordinates [mas]
        y_grid: y coordinates [mas]
        beam_fwhm_mas: beam FWHM in milliarcseconds

    returns:
        convolved intensity array
    """
    from scipy.ndimage import gaussian_filter

    # compute pixel scale from grid
    dx = np.abs(x_grid[0, 1] - x_grid[0, 0]) if x_grid.shape[1] > 1 else 1.0
    dy = np.abs(y_grid[1, 0] - y_grid[0, 0]) if y_grid.shape[0] > 1 else 1.0

    # convert FWHM to sigma: FWHM = 2 * sqrt(2 * ln(2)) * sigma
    sigma_mas = beam_fwhm_mas / 2.355

    # sigma in pixels
    sigma_x_pix = sigma_mas / dx
    sigma_y_pix = sigma_mas / dy

    return gaussian_filter(intensity, sigma=[sigma_y_pix, sigma_x_pix])


# configure matplotlib for publication-quality plots
def _setup_plot_style():
    """setup latex and times new roman if available"""
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times"],
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )
    except Exception:
        # fallback if latex not available
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            }
        )


_setup_plot_style()


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
        ax.plot(lc.times, flux, label=label, marker="o", markersize=3)

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


def plot_skymap(
    skymap: skymap_t,
    save: Optional[str] = None,
    beam_fwhm_arcsec: Optional[float] = None,
) -> None:
    """
    plot sky intensity map as telescope observer view.

    args:
        skymap: skymap data (must have .metadata with d_L)
        save: filename to save (None=show)
        beam_fwhm_arcsec: telescope beam FWHM [arcsec] for PSF convolution
    """
    theta = np.array(skymap.theta)
    phi = np.array(skymap.phi)
    intensity = np.array(skymap.intensity)

    # validate: check if skymap has any flux
    max_intensity = intensity.max()
    if max_intensity <= 0:
        raise ValueError(
            "skymap has zero intensity everywhere. "
            "check time window and photon energy range."
        )

    # get luminosity distance from metadata if available
    # default to 300 Mpc if not present
    d_L = getattr(skymap, "d_L", 3e26)  # cm

    # typical blast wave radius at few days: ~10^17 cm
    # angular size = R / d_L radians
    # convert radians to milliarcseconds (mas)
    rad_to_mas = u.radian.to(u.mas)

    # create cartesian grid in angular coordinates
    # project spherical angles to sky plane
    n_theta, n_phi = len(theta), len(phi)

    # compute cartesian coordinates (vectorized)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    X_raw = THETA * np.cos(PHI) * rad_to_mas
    Y_raw = THETA * np.sin(PHI) * rad_to_mas

    # create regular grid for clean plotting
    x_min, x_max = X_raw.min(), X_raw.max()
    y_min, y_max = Y_raw.min(), Y_raw.max()

    x_grid = np.linspace(x_min, x_max, n_phi)
    y_grid = np.linspace(y_min, y_max, n_theta)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # interpolate intensity onto regular grid
    from scipy.interpolate import griddata

    points = np.column_stack([X_raw.ravel(), Y_raw.ravel()])
    values = intensity.ravel()

    # check if we have meaningful data
    if values.max() == 0 or np.allclose(values, values[0]):
        print(
            f"warning: no significant emission in skymap (max intensity = {values.max():.2e})"
        )
        print("  try adjusting --time or --time-window")
        intensity_grid = np.zeros_like(X_grid)
    else:
        try:
            intensity_grid = griddata(
                points, values, (X_grid, Y_grid), method="cubic", fill_value=0.0
            )
        except Exception:
            # fallback to nearest neighbor if cubic fails
            intensity_grid = griddata(
                points,
                values,
                (X_grid, Y_grid),
                method="nearest",
                fill_value=0.0,
            )

    # apply beam convolution if requested
    if beam_fwhm_arcsec is not None and beam_fwhm_arcsec > 0:
        # convert arcsec to mas (1 arcsec = 1000 mas)
        beam_fwhm_mas = beam_fwhm_arcsec * 1000.0
        intensity_grid = convolve_with_beam(
            intensity_grid, X_grid, Y_grid, beam_fwhm_mas
        )
        print(f"  convolved with {beam_fwhm_arcsec:.2f} arcsec beam")

    # compute flux centroid and emission extent
    total_flux = intensity_grid.sum()
    if total_flux > 0:
        x_centroid = (X_grid * intensity_grid).sum() / total_flux
        y_centroid = (Y_grid * intensity_grid).sum() / total_flux
    else:
        x_centroid, y_centroid = 0, 0

    # find emission region extent for auto-zoom
    # locate pixels with significant emission (>1% of peak)
    threshold = intensity_grid.max() * 0.01
    emission_mask = intensity_grid > threshold
    if emission_mask.any():
        y_indices, x_indices = np.where(emission_mask)
        x_emit = X_grid[y_indices, x_indices]
        y_emit = Y_grid[y_indices, x_indices]

        x_min_emit = x_emit.min()
        x_max_emit = x_emit.max()
        y_min_emit = y_emit.min()
        y_max_emit = y_emit.max()

        # add 50% padding around emission region
        x_range = x_max_emit - x_min_emit
        y_range = y_max_emit - y_min_emit
        max_range = max(x_range, y_range)

        if max_range > 0:
            padding = max_range * 0.5
            x_center = (x_max_emit + x_min_emit) / 2
            y_center = (y_max_emit + y_min_emit) / 2
            zoom_extent = [
                x_center - max_range / 2 - padding,
                x_center + max_range / 2 + padding,
                y_center - max_range / 2 - padding,
                y_center + max_range / 2 + padding,
            ]
        else:
            zoom_extent = None
    else:
        zoom_extent = None

    # plot with dark background (telescope-like view)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    # logarithmic intensity
    # only plot where intensity > 0
    intensity_positive = intensity_grid[intensity_grid > 0]
    if len(intensity_positive) == 0:
        raise ValueError("no positive intensity after interpolation")

    # set dynamic range: 3-4 decades below peak for better contrast
    vmax_linear = intensity_positive.max()
    vmin_linear = vmax_linear * 1e-4

    # mask values below threshold (set to zero for dark background)
    intensity_plot = np.where(
        intensity_grid > vmin_linear, intensity_grid, vmin_linear * 0.1
    )
    intensity_log = np.log10(intensity_plot)

    vmax = np.log10(vmax_linear)
    vmin = np.log10(vmin_linear)

    # use hot colormap for telescope-like appearance (black->red->yellow->white)
    im = ax.pcolormesh(
        X_grid,
        Y_grid,
        intensity_log,
        cmap="hot",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    # mark centroid (filled orange circle)
    ax.plot(
        x_centroid,
        y_centroid,
        "o",
        color="orange",
        markersize=8,
        markeredgewidth=0,
        label=r"Centroid $x_{\rm c}$",
    )

    # mark origin/merger site (red plus)
    ax.plot(0, 0, "+", color="red", markersize=10, markeredgewidth=2)

    ax.set_xlabel(r"$x$ [mas]", fontsize=12, color="white")
    ax.set_ylabel(r"$y$ [mas]", fontsize=12, color="white")

    # title with optional beam info
    title = rf"$I_{{\nu}}(x,y)$ @ $t = {skymap.time:.2f}$ day"
    if beam_fwhm_arcsec is not None and beam_fwhm_arcsec > 0:
        title += rf" (beam: {beam_fwhm_arcsec:.1f}$''$)"
    ax.set_title(title, fontsize=14, color="white")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--", color="gray")
    ax.tick_params(colors="white", which="both")
    ax.legend(
        loc="upper right",
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
    )

    # add colorbar with dark background styling
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        r"$\log_{10}(I_{\nu})$ [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        fontsize=11,
        color="white",
    )
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")

    # auto-zoom to emission region if detected
    if zoom_extent is not None:
        ax.set_xlim(zoom_extent[0], zoom_extent[1])
        ax.set_ylim(zoom_extent[2], zoom_extent[3])

    # draw beam size indicator in lower left corner
    if beam_fwhm_arcsec is not None and beam_fwhm_arcsec > 0:
        beam_fwhm_mas = beam_fwhm_arcsec * 1000.0
        # position beam circle in lower-left corner
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        beam_x = xlim[0] + 0.1 * x_range
        beam_y = ylim[0] + 0.1 * y_range
        beam_circle = plt.Circle(
            (beam_x, beam_y),
            beam_fwhm_mas / 2,
            fill=False,
            color="white",
            linewidth=1,
            linestyle="-",
        )
        ax.add_patch(beam_circle)
        ax.text(
            beam_x,
            beam_y - beam_fwhm_mas * 0.8,
            "beam",
            color="white",
            fontsize=8,
            ha="center",
        )

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", facecolor="black")
        print(f"saved figure to {save}")
    else:
        plt.show()


def plot_skymap_animation(
    skymaps: List[skymap_t],
    save: Optional[str] = None,
    fps: int = 5,
    show: bool = False,
) -> None:
    """
    create animation of skymap time evolution.

    args:
        skymaps: list of skymap_t at different times
        save: output filename (.mp4 or .gif)
        fps: frames per second
        show: display animation interactively
    """
    if len(skymaps) == 0:
        raise ValueError("no skymaps provided")

    # use first skymap to setup figure
    first_map = skymaps[0]
    d_L = first_map.d_L
    rad_to_mas = u.radian.to(u.mas)

    fig, ax = plt.subplots(figsize=(8, 8))

    # precompute grids for all frames
    from scipy.interpolate import griddata

    def prepare_frame(skymap):
        theta = np.array(skymap.theta)
        phi = np.array(skymap.phi)
        intensity = np.array(skymap.intensity)
        n_theta, n_phi = len(theta), len(phi)

        # compute cartesian coordinates (vectorized)
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
        X_raw = THETA * np.cos(PHI) * rad_to_mas
        Y_raw = THETA * np.sin(PHI) * rad_to_mas

        # regular grid
        x_min, x_max = X_raw.min(), X_raw.max()
        y_min, y_max = Y_raw.min(), Y_raw.max()
        x_grid = np.linspace(x_min, x_max, n_phi)
        y_grid = np.linspace(y_min, y_max, n_theta)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # interpolate
        points = np.column_stack([X_raw.ravel(), Y_raw.ravel()])
        values = intensity.ravel()
        intensity_grid = griddata(
            points, values, (X_grid, Y_grid), method="cubic", fill_value=0.0
        )

        # logarithmic intensity
        intensity_log = np.log10(intensity_grid + 1e-100)

        # compute centroid
        total_flux = intensity_grid.sum()
        if total_flux > 0:
            x_c = (X_grid * intensity_grid).sum() / total_flux
            y_c = (Y_grid * intensity_grid).sum() / total_flux
        else:
            x_c, y_c = 0, 0

        return X_grid, Y_grid, intensity_log, x_c, y_c

    frames = [prepare_frame(sm) for sm in skymaps]

    # find global vmin/vmax for consistent colorscale
    all_intensities = [f[2] for f in frames]
    vmax = max(arr.max() for arr in all_intensities)
    vmin = vmax - 2.0

    # animation update function
    def update(frame_idx):
        ax.clear()
        X_grid, Y_grid, intensity_log, x_c, y_c = frames[frame_idx]

        im = ax.pcolormesh(
            X_grid,
            Y_grid,
            intensity_log,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        # mark centroid
        ax.plot(
            x_c,
            y_c,
            "o",
            color="orange",
            markersize=8,
            markeredgewidth=0,
        )

        # mark origin
        ax.plot(0, 0, "+", color="red", markersize=10, markeredgewidth=2)

        ax.set_xlabel(r"$x$ [mas]", fontsize=12)
        ax.set_ylabel(r"$y$ [mas]", fontsize=12)
        ax.set_title(
            rf"$I_{{\nu}}(x,y)$ @ $t = {skymaps[frame_idx].time:.2f}$ day",
            fontsize=14,
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")

        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )

    plt.tight_layout()

    if save:
        print(f"saving animation to {save}...")
        if save.endswith(".gif"):
            anim.save(save, writer="pillow", fps=fps)
        else:
            anim.save(save, writer="ffmpeg", fps=fps, dpi=150)
        print(f"saved animation to {save}")

    if show:
        plt.show()
    else:
        plt.close()


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
    ax1.plot(pol.times, pol.polarization_degree * 100, marker="o", markersize=3)
    ax1.set_ylabel("Polarization Degree [%]")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Polarization Evolution")

    # bottom panel: polarization angle
    ax2 = axes[1]
    ax2.plot(
        pol.times,
        np.rad2deg(pol.polarization_angle),
        marker="o",
        markersize=3,
        color="C1",
    )
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

    ax.plot(spec.frequencies, spec.fluxes, marker="o", markersize=4)
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
