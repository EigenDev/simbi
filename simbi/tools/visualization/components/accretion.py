"""Component for analyzing accretion disk properties."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic

from ..core.config import StyleConfig
from ..core.types import PlotData
from ..formatters.line import format_line_plot_axes
from ..formatters.multidim import format_multidim_plot_axes
from .interface import Component, ComponentProps


class AnalysisType(Enum):
    """Types of accretion analysis."""

    ANGULAR_MOMENTUM = auto()  # Specific angular momentum profile
    MASS_FLUX = auto()  # Mass flux through shells
    DENSITY_PROFILE = auto()  # Density vs radius/angle
    QUIVER = auto()  # Density with velocity quiver overlay
    STREAMLINES = auto()  # Density with velocity streamline overlay


@dataclass(frozen=True)
class RadialConfig:
    """Configuration for radial analysis."""

    n_bins: int = 50


@dataclass(frozen=True)
class AngularConfig:
    """Configuration for angular analysis."""

    radius: float  # Analysis radius
    n_angles: int = 50  # Number of angular bins
    average_width: float = 0.1  # Fractional width for radial averaging


@dataclass(frozen=True)
class AccretionAnalysisProps(ComponentProps):
    """Properties for accretion analysis visualization."""

    analysis_type: AnalysisType
    radial_config: Optional[RadialConfig] = None
    angular_config: Optional[AngularConfig] = None
    level: int = 0  # Which refinement level to analyze
    time_average: bool = False  # Whether to average over time series
    normalize: bool = False  # Whether to normalize quantities


class AccretionAnalysisComponent(Component[AccretionAnalysisProps]):
    """Component for visualizing accretion properties."""

    def __init__(self, props: AccretionAnalysisProps):
        self._validate_props(props)
        self.props = props
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self._initialized = False
        self._lines: list[Line2D] = []

    def _validate_props(self, props: AccretionAnalysisProps) -> None:
        """Validate component properties."""
        if props.analysis_type in (
            AnalysisType.ANGULAR_MOMENTUM,
            AnalysisType.MASS_FLUX,
        ):
            if props.radial_config is None:
                raise ValueError(
                    f"Radial configuration required for {props.analysis_type}"
                )

        if props.analysis_type == AnalysisType.DENSITY_PROFILE:
            if props.angular_config is None:
                raise ValueError(
                    "Angular configuration required for density profile"
                )

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    def update(self, props: AccretionAnalysisProps) -> None:
        self._validate_props(props)
        self.props = props

    def cleanup(self) -> None:
        if self.ax is not None:
            for line in self._lines:
                if line in self.ax.lines:
                    line.remove()
            self._lines = []
            self.ax.cla()

    def render(self, data: PlotData, style: StyleConfig) -> None:
        """Render the appropriate analysis."""
        if not self._initialized or self.ax is None:
            raise RuntimeError("Component not initialized")

        match self.props.analysis_type:
            case AnalysisType.ANGULAR_MOMENTUM:
                self._plot_angular_momentum(data, style)
            case AnalysisType.MASS_FLUX:
                self._plot_mass_flux_profile(data, style)
            case AnalysisType.QUIVER:
                self._plot_density_with_quiver(data, style)
            case AnalysisType.STREAMLINES:
                self._plot_density_with_streamlines(data, style)
            case AnalysisType.DENSITY_PROFILE:
                self._plot_density_profile(data, style)

    def _plot_angular_momentum(
        self, data: PlotData, style: StyleConfig
    ) -> None:
        """Plot the MASS-WEIGHTED specific angular momentum profile."""
        if self.ax is None or self.props.radial_config is None:
            return

        ell = data.fields[0].values
        sigma = data.fields[1].values

        x_verts_1d = data.fields[0].domain[-2]
        y_verts_1d = data.fields[0].domain[-1]

        x_centers_1d = 0.5 * (x_verts_1d[1:] + x_verts_1d[:-1])
        y_centers_1d = 0.5 * (y_verts_1d[1:] + y_verts_1d[:-1])
        xv_2d, yv_2d = np.meshgrid(x_centers_1d, y_centers_1d, indexing="ij")

        r_2d = np.sqrt(xv_2d**2 + yv_2d**2)  # shape (Nx, Ny)

        Lz = ell * sigma  # (l_z * Sigma)

        r_flat = r_2d.flat
        Lz_flat = Lz.flat
        sigma_flat = sigma.flat

        max_radius = np.max(r_flat)
        num_bins = 100
        bins = np.linspace(0, max_radius, num_bins + 1)

        Lz_sum_in_bin, bin_edges, _ = binned_statistic(
            r_flat, Lz_flat, statistic="sum", bins=bins
        )
        sigma_sum_in_bin, _, _ = binned_statistic(
            r_flat, sigma_flat, statistic="sum", bins=bins
        )

        mean_ell_mass_weighted = Lz_sum_in_bin / (
            sigma_sum_in_bin + np.finfo(float).tiny
        )
        q = 1
        mu = q / (1 + q) ** 2
        G = 1.0
        M = 1.0
        a = 1
        jref = mu * np.sqrt(G * M / a) / M
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        good_bins = ~np.isnan(mean_ell_mass_weighted)
        self.ax.plot(
            bin_centers[good_bins], mean_ell_mass_weighted[good_bins] / jref
        )
        self.ax.axvline(1, linestyle="dashed", color="k", alpha=0.5)
        self.ax.axvline(10, linestyle="dashed", color="k", alpha=0.5)
        format_line_plot_axes(
            self.ax,
            data,
            0,
            style,
        )

    def _plot_density_with_quiver(
        self, data: PlotData, style: StyleConfig
    ) -> None:
        """Plot Log(Density) with a velocity quiver overlay."""
        if self.ax is None:
            return

        # --- 1. Get 1D VERTEX arrays ---
        x_verts_1d = data.fields[0].domain[-2]  # shape (Nx + 1,)
        y_verts_1d = data.fields[0].domain[-1]  # shape (Ny + 1,)

        # --- 2. Get 2D Data Fields ---
        # (Assuming sigma is field 0, vx is 1, vy is 2)
        sigma_2d = data.fields[0].values
        vx_2d = data.fields[1].values
        vy_2d = data.fields[2].values

        # --- 3. Plot the Density Heatmap ---
        # Use pcolormesh for vertex-based data
        # We use log-density for better contrast
        log_sigma = np.log10(sigma_2d + 1e-10)

        mesh = self.ax.pcolormesh(
            x_verts_1d,
            y_verts_1d,
            log_sigma.T,  # .T to match pcolormesh (x, y, C) convention
            cmap="viridis",
            shading="auto",
        )

        # --- 4. Downsample Velocity for Quiver ---

        # We need the 2D *centers* for the quiver locations
        x_centers_1d = 0.5 * (x_verts_1d[1:] + x_verts_1d[:-1])
        y_centers_1d = 0.5 * (y_verts_1d[1:] + y_verts_1d[:-1])
        xv_2d_cen, yv_2d_cen = np.meshgrid(
            x_centers_1d, y_centers_1d, indexing="xy"
        )

        skip = 10  # Adjust this to your liking

        x_coords_sparse = xv_2d_cen[::skip, ::skip]
        y_coords_sparse = yv_2d_cen[::skip, ::skip]
        vx_sparse = vx_2d[::skip, ::skip]
        vy_sparse = vy_2d[::skip, ::skip]

        # --- 5. Plot the Quiver Overlay ---
        self.ax.quiver(
            x_coords_sparse,
            y_coords_sparse,
            vx_sparse,
            vy_sparse,
            color="white",  # White arrows stand out on a heatmap
            scale=20.0,  # Experiment with this value
            width=0.002,  # (Optional: arrow width)
        )

        self.ax.set_aspect("equal")
        format_multidim_plot_axes(self.ax, self.fig, mesh, data, 0, style)

    def _plot_density_with_streamlines(
        self, data: PlotData, style: StyleConfig
    ) -> None:
        """Plot Log(Density) heatmap with a velocity streamline overlay."""
        if self.ax is None:
            return

        # --- 1. Get 1D VERTEX arrays (for pcolormesh) ---
        x_verts_1d = data.fields[0].domain[-2]  # e.g., shape (Nx + 1,)
        y_verts_1d = data.fields[0].domain[-1]  # e.g., shape (Ny + 1,)

        # --- 2. Get 1D CENTER arrays (for streamplot) ---
        x_centers_1d = 0.5 * (x_verts_1d[1:] + x_verts_1d[:-1])  # shape (Nx,)
        y_centers_1d = 0.5 * (y_verts_1d[1:] + y_verts_1d[:-1])  # shape (Ny,)

        # --- 3. Get 2D Data Fields ---
        # (Assuming sigma is field 0, vx is 1, vy is 2)
        sigma_2d = data.fields[0].values  # shape (Nx, Ny)
        vx_2d = data.fields[1].values  # shape (Nx, Ny)
        vy_2d = data.fields[2].values  # shape (Nx, Ny)

        # --- 4. Plot the Density Heatmap (Background) ---
        # Use log-density for better contrast
        log_sigma = np.log10(sigma_2d + 1e-10)  # 1e-10 avoids log(0)

        mesh = self.ax.pcolormesh(
            x_verts_1d,
            y_verts_1d,
            log_sigma.T,  # .T to match pcolormesh (x, y, C) convention
            cmap="inferno",  # 'inferno' or 'viridis' look good
            shading="auto",
            vmin=np.min(log_sigma),  # Optional: set color limits
            vmax=np.max(log_sigma),
        )

        # --- 5. Plot the Streamlines (Overlay) ---
        # streamplot expects (M, N) data, where M=len(y) and N=len(x).
        # This requires transposing the velocity data.
        self.ax.streamplot(
            x_centers_1d,  # 1D array of x-coords (shape Nx)
            y_centers_1d,  # 1D array of y-coords (shape Ny)
            vx_2d.T,  # 2D array of u-velocity (shape Ny, Nx)
            vy_2d.T,  # 2D array of v-velocity (shape Ny, Nx)
            # --- Styling as requested ---
            color="white",  # "nice... white lines"
            linewidth=0.5,  # "thin"
            density=2.0,  # A bit denser to see the pattern
            arrowsize=0.7,  # Small arrows to show direction
            arrowstyle="->",  # Standard arrow
        )

        # --- 6. Final Touches ---
        self.ax.set_aspect("equal")

        # Set limits to the cell centers
        self.ax.set_xlim(x_centers_1d[0], x_centers_1d[-1])
        self.ax.set_ylim(y_centers_1d[0], y_centers_1d[-1])
        format_multidim_plot_axes(self.ax, self.fig, mesh, data, 0, style)

    def _plot_angular_momentum_rays(
        self, data: PlotData, style: StyleConfig
    ) -> None:
        """Plot the MASS-WEIGHTED specific angular momentum profile."""
        if self.ax is None or self.props.radial_config is None:
            return
        ell = data.fields[0].values

        x_verts_1d = data.fields[0].domain[-2]
        y_verts_1d = data.fields[0].domain[-1]

        x_centers_1d = 0.5 * (x_verts_1d[1:] + x_verts_1d[:-1])
        # y_centers_1d = 0.5 * (y_verts_1d[1:] + y_verts_1d[:-1])
        self.ax.plot(x_centers_1d, ell, linestyle="dotted", alpha=0.5)
        self.ax.axvline(1, linestyle="dashed")
        self.ax.axvline(-1, linestyle="dashed")

    def _plot_mass_flux_profile(
        self, data: PlotData, style: StyleConfig
    ) -> None:
        """Plot the mass flux profile M_dot(r) in spherical shells."""
        if self.ax is None:
            return

        # --- 1. Get 1D CENTER arrays (for all 3 dimensions) ---
        # Assumes domain has x, y, and z vertices
        x_verts_1d = data.fields[0].domain[0]  # x-vertices
        y_verts_1d = data.fields[0].domain[1]  # y-vertices
        z_verts_1d = x_verts_1d  # data.fields[0].domain[2]  # z-vertices
        x_centers_1d = 0.5 * (x_verts_1d[1:] + x_verts_1d[:-1])
        y_centers_1d = 0.5 * (y_verts_1d[1:] + y_verts_1d[:-1])
        z_centers_1d = 0.5 * (z_verts_1d[1:] + z_verts_1d[:-1])

        # --- 2. Create 3D CENTER-coordinate grids ---
        xv_3d, yv_3d, zv_3d = np.meshgrid(
            x_centers_1d, y_centers_1d, z_centers_1d, indexing="ij"
        )

        # --- 3. Get 3D Data Fields ---
        # (Assuming field 0:rho, 1:vx, 2:vy, 3:vz)
        rho_3d = data.fields[0].values
        vx_3d = data.fields[1].values
        vy_3d = data.fields[2].values
        vz_3d = data.fields[3].values

        # --- 4. Calculate 3D maps of r, v_r, and (rho * v_r) ---
        r_3d = np.sqrt(xv_3d**2 + yv_3d**2 + zv_3d**2)

        # Radial velocity: v_r = (v . r) / |r|
        vr_3d = (vx_3d * xv_3d + vy_3d * yv_3d + vz_3d * zv_3d) / (r_3d + 1e-10)

        # Radial mass flux density
        flux_density_3d = rho_3d * vr_3d

        # --- 5. Flatten arrays for binning ---
        r_flat = r_3d.flatten()
        flux_density_flat = flux_density_3d.flatten()

        # --- 6. Bin the flux density by radius ---
        max_radius = np.max(r_flat)
        num_bins = 100
        bins = np.linspace(0, max_radius, num_bins + 1)

        mean_flux_density, bin_edges, _ = binned_statistic(
            r_flat, flux_density_flat, statistic="mean", bins=bins
        )

        # --- 7. Calculate Mass Flux M_dot ---
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Area of each spherical shell
        shell_area = 4.0 * np.pi * bin_centers**2

        # M_dot = <rho * v_r> * Area
        mass_flux_profile = mean_flux_density * shell_area

        # --- 8. Plot the result ---
        xi = 10
        rho_infty = 1.0
        c = xi ** (0.5)
        vorb = 1
        rbh = 10
        mdot_bh = 4.0 * np.pi * rbh**2 * rho_infty * c
        good_bins = ~np.isnan(mass_flux_profile)
        self.ax.plot(
            bin_centers[good_bins], mass_flux_profile[good_bins] / mdot_bh
        )

        self.ax.set_xlabel("Radius (r/a)")
        self.ax.set_ylabel(r"Mass Flux ($\dot{M}$)")
        self.ax.set_title(r"Mass Flux Profile $\dot{M}(r)$")

        # Add a line at M_dot=0 for reference
        self.ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

    @property
    def initialized(self) -> bool:
        return self._initialized
