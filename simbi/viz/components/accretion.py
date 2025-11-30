"""Component for analyzing accretion properties."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic

from simbi.functional.helpers import find_nearest
from simbi.viz.utility import get_field_str

from ..config import StyleConfig
from ..formatters.line import format_line_plot_axes
from ..formatters.multidim import format_multidim_plot_axes
from ..types import FieldData, PlotData, RenderResult
from .interface import Component, ComponentProps


class AnalysisType(Enum):
    """Types of accretion analysis."""

    ANGULAR_MOMENTUM = auto()  # Specific angular momentum profile
    MASS_FLUX = auto()  # Mass flux through shells
    RADIAL_PROFILE = auto()  # Field vs radius
    QUIVER = auto()  # Density with velocity quiver overlay
    STREAMLINES = auto()  # Density with velocity streamline overlay


@dataclass(frozen=True)
class RadialConfig:
    """Configuration for radial analysis."""

    n_bins: int = 100


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
    time_average: bool = False  # Whether to average over time series
    normalize: bool = False  # Whether to normalize quantities


class AccretionAnalysisComponent(Component):
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
            AnalysisType.RADIAL_PROFILE,
        ):
            if props.radial_config is None:
                raise ValueError(
                    f"Radial configuration required for {props.analysis_type}"
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

    def render(self, data: PlotData, style: StyleConfig) -> RenderResult:
        """Render the appropriate analysis and return a RenderResult describing created artists."""
        if not self._initialized or self.ax is None:
            raise RuntimeError("Component not initialized")

        # Dispatch to the selected analysis and collect returned RenderResult
        result: RenderResult | None = None
        match self.props.analysis_type:
            case AnalysisType.ANGULAR_MOMENTUM:
                result = self._plot_angular_momentum(data, style)
            case AnalysisType.MASS_FLUX:
                result = self._plot_mass_flux_profile(data, style)
            case AnalysisType.QUIVER:
                # Quiver/Streamlines operate on a heatmap + overlay; return mesh + vector artists
                result = self._plot_density_with_quiver(data, style)
            case AnalysisType.STREAMLINES:
                result = self._plot_density_with_streamlines(data, style)
            case AnalysisType.RADIAL_PROFILE:
                result = self._plot_radial_profile(data, style)

        # Ensure a RenderResult is always returned
        if isinstance(result, RenderResult):
            return result
        else:
            return RenderResult(artists={}, metadata={})

    def _get_stitched_leaf_data(
        self, data: PlotData, field_names: list[str]
    ) -> dict[str, np.ndarray]:
        """
        Stitch all FMR levels into high-resolution flat arrays of leaf cells.

        This is the core of the level-aware analysis. It adapts the logic
        from your _render_polygons to build analysis arrays instead of patches.
        """
        level_fields_map: dict[str, list[FieldData]] = {}
        all_levels = set()
        for name in field_names:
            level_fields_map[name] = [
                f for f in data.fields if f.name.startswith(name)
            ]
            if not level_fields_map[name]:
                raise ValueError(f"No fields found for base name: {name}")

            # Sort fields by level, e.g., _L0, _L1, ...
            level_fields_map[name].sort(key=lambda f: f.name)
            all_levels.update(range(len(level_fields_map[name])))

        num_levels = len(all_levels)
        if num_levels == 0:
            raise ValueError("No fields found for any requested name")

        is_3d = False
        refined_regions = []
        for i in range(1, num_levels):  # All levels except base
            # Use the domain from the first field's L1, L2, etc.
            field_L_i = level_fields_map[field_names[0]][i]
            domain = field_L_i.domain

            # Check for 2D or 3D
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
            # Get the fields for *this* level for all requested names
            current_level_fields = {
                name: level_fields_map[name][level_idx] for name in field_names
            }

            # Get domain info from the first field
            domain = current_level_fields[field_names[0]].domain
            values_map = {
                name: current_level_fields[name].values for name in field_names
            }

            x_verts = domain[0]
            y_verts = domain[1]
            z_verts = (
                domain[2] if is_3d else np.array([0.0, 1.0])
            )  # Dummy for 2D

            x_centers = 0.5 * (x_verts[1:] + x_verts[:-1])
            y_centers = 0.5 * (y_verts[1:] + y_verts[:-1])
            z_centers = (
                0.5 * (z_verts[1:] + z_verts[:-1]) if is_3d else np.array([0.0])
            )

            nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)

            # Iterate through all cells in this level
            for k in range(nz):
                zc = z_centers[k]
                for j in range(ny):
                    yc = y_centers[j]
                    for i in range(nx):
                        xc = x_centers[i]

                        # Check if this cell is covered by a *finer* level
                        is_covered = False
                        for region in refined_regions[level_idx:]:
                            # Check 2D coverage
                            covered_2d = (
                                region["xmin"] <= xc <= region["xmax"]
                                and region["ymin"] <= yc <= region["ymax"]
                            )
                            if not is_3d:
                                if covered_2d:
                                    is_covered = True
                                    break
                            else:
                                # Check 3D coverage
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

        # Convert all lists to numpy arrays and return
        return {key: np.array(val) for key, val in stitched_data.items()}

    def _plot_angular_momentum(
        self, data: PlotData, style: StyleConfig
    ) -> RenderResult:
        """Plot the MASS-WEIGHTED specific angular momentum profile and return artists."""
        if self.ax is None or self.props.radial_config is None:
            return RenderResult(artists={}, metadata={})

        # Get stitched leaf cell data for all levels
        try:
            stitched_data = self._get_stitched_leaf_data(
                data, ["j_spec", "Sigma"]
            )
        except ValueError as e:
            print(f"Error stitching data: {e}")
            return RenderResult(artists={}, metadata={})

        x_flat = stitched_data["x_flat"]
        y_flat = stitched_data["y_flat"]
        ell_flat = stitched_data["j_spec_flat"]
        sigma_flat = stitched_data["Sigma_flat"]

        # Calculate r and Lz for *all leaf cells*
        r_flat = np.sqrt(x_flat**2 + y_flat**2)
        Lz_flat = ell_flat * sigma_flat  # (l_z * Sigma)

        # Bin the stitched data
        max_radius = np.max(r_flat)
        num_bins = self.props.radial_config.n_bins
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

        G = 1.0
        M = 1.0
        a = 1.0
        q = 1.0
        mu = q / (1.0 + q) ** 2
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        jref = (G * M * bin_centers) ** (0.5)

        good_bins = ~np.isnan(mean_ell_mass_weighted)
        main_lines = self.ax.plot(
            bin_centers[good_bins], (mean_ell_mass_weighted / jref)[good_bins]
        )
        vline1 = self.ax.axvline(
            1.0, color="gray", linestyle="--", linewidth=0.5
        )
        vline2 = self.ax.axvline(
            5.0, color="gray", linestyle="--", linewidth=0.5
        )

        # format and return artists
        format_line_plot_axes(self.ax, data, 0, style)
        return RenderResult(
            artists={"lines": main_lines, "vlines": [vline1, vline2]},
            metadata={"label": "j_spec_mass_weighted"},
        )

    def _plot_mass_flux_profile(
        self, data: PlotData, style: StyleConfig
    ) -> RenderResult:
        """Plot the mass flux profile M_dot(r) in spherical shells and return artists."""
        if self.ax is None or self.props.radial_config is None:
            return RenderResult(artists={}, metadata={})

        try:
            stitched_data = self._get_stitched_leaf_data(
                data, ["rho", "v1", "v2", "v3"]
            )
        except ValueError as e:
            print(f"Error stitching data: {e}")
            return RenderResult(artists={}, metadata={})

        x_flat = stitched_data["x_flat"]
        y_flat = stitched_data["y_flat"]
        z_flat = stitched_data["z_flat"]
        rho_flat = stitched_data["rho_flat"]
        vx_flat = stitched_data["v1_flat"]
        vy_flat = stitched_data["v2_flat"]
        vz_flat = stitched_data["v3_flat"]

        # Calculate r, v_r, and (rho * v_r) for *all leaf cells*
        r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
        vr_flat = (vx_flat * x_flat + vy_flat * y_flat + vz_flat * z_flat) / (
            r_flat + 1e-10
        )
        flux_density_flat = rho_flat * vr_flat

        # Bin the stitched data
        max_radius = np.max(r_flat)
        num_bins = self.props.radial_config.n_bins
        bins = np.linspace(0, max_radius, num_bins + 1)

        mean_flux_density, bin_edges, _ = binned_statistic(
            r_flat, flux_density_flat, statistic="mean", bins=bins
        )

        # Calculate Mass Flux M_dot
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        shell_area = 4.0 * np.pi * bin_centers**2
        mass_flux_profile = mean_flux_density * shell_area

        # Plotting
        good_bins = ~np.isnan(mass_flux_profile)
        norm = 14  # 445
        main_lines = self.ax.plot(
            bin_centers[good_bins], mass_flux_profile[good_bins] / norm
        )

        self.ax.set_xlabel("Radius (r/a)")
        self.ax.set_ylabel("Mass Flux $\\dot{M}$ (normalized)")
        vline0 = self.ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        vline1 = self.ax.axvline(
            1.0, color="gray", linestyle="--", linewidth=0.5
        )
        vline2 = self.ax.axvline(
            0.5, color="gray", linestyle="--", linewidth=0.5
        )

        format_line_plot_axes(self.ax, data, 0, style)
        return RenderResult(
            artists={"lines": main_lines, "vlines": [vline0, vline1, vline2]},
            metadata={"label": "mdot"},
        )

    def _plot_radial_profile(
        self, data: PlotData, style: StyleConfig
    ) -> RenderResult:
        """Plot the spherically-averaged volume density profile and return artists."""
        if self.ax is None or self.props.radial_config is None:
            return RenderResult(artists={}, metadata={})

        field_name = data.fields[0].name.split("_L")[0]
        try:
            stitched_data = self._get_stitched_leaf_data(data, [field_name])
        except ValueError as e:
            print(f"Error stitching data: {e}")
            return RenderResult(artists={}, metadata={})

        x_flat = stitched_data["x_flat"]
        y_flat = stitched_data["y_flat"]
        z_flat = stitched_data.get("z_flat", np.zeros_like(x_flat))
        rho_flat = stitched_data[f"{field_name}_flat"]

        # calc r for *all leaf cells*
        r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)

        # bin the stitched data
        max_radius = np.max(r_flat)
        num_bins = self.props.radial_config.n_bins
        bins = np.linspace(0, max_radius, num_bins + 1)

        mean_var, bin_edges, _ = binned_statistic(
            r_flat, rho_flat, statistic="mean", bins=bins
        )

        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        good_bins = ~np.isnan(mean_var)
        main_lines = self.ax.plot(bin_centers[good_bins], mean_var[good_bins])

        field_str = get_field_str(field_name)
        if field_str.startswith("$") and field_str.endswith("$"):
            field_str = field_str[1:-1]
        self.ax.set_xlabel("Radius (r/a)")
        self.ax.set_ylabel(f"$\\langle {field_str} \\rangle$")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        r_sonic = 0.5
        vline1 = self.ax.axvline(
            1.0, color="gray", linestyle="--", linewidth=0.5
        )
        vline2 = self.ax.axvline(
            r_sonic, color="gray", linestyle="--", linewidth=0.5
        )
        ref_lines = []
        if field_name in ["rho", "v"]:
            ref_distance = 0.1
            ref_max = 0.5
            power = -1.5 if field_name == "rho" else -0.5

            r_ref = bin_centers[good_bins]
            ref_idx = find_nearest(r_ref, ref_distance)[0] + 1
            ren_idx = find_nearest(r_ref, ref_max)[0] + 1
            var_ref = mean_var[good_bins][ref_idx] * (
                r_ref / r_ref[ref_idx]
            ) ** (power)
            ref_lines = self.ax.plot(
                r_ref[ref_idx:ren_idx],
                var_ref[ref_idx:ren_idx] * (1.5),
                linestyle="--",
                color="red",
                label=r"$r^{-3/2}$",
            )

        return RenderResult(
            artists={
                "lines": main_lines,
                "vlines": [vline1, vline2],
                "refs": ref_lines,
            },
            metadata={"label": field_name},
        )

    def _plot_density_with_quiver(
        self, data: PlotData, style: StyleConfig
    ) -> RenderResult:
        """Plot Log(Density) with a velocity quiver overlay and return mesh+quiver artists."""
        if self.ax is None:
            return RenderResult(artists={}, metadata={})

        x_verts_1d = data.fields[0].domain[-2]  # shape (Nx + 1,)
        y_verts_1d = data.fields[0].domain[-1]  # shape (Ny + 1,)

        sigma_2d = data.fields[0].values
        vx_2d = data.fields[1].values
        vy_2d = data.fields[2].values

        log_sigma = np.log10(sigma_2d + 1e-10)

        mesh = self.ax.pcolormesh(
            x_verts_1d,
            y_verts_1d,
            log_sigma.T,  # .T to match pcolormesh (x, y, C) convention
            cmap="viridis",
            shading="auto",
        )

        # Downsample Velocity for Quiver
        x_centers_1d = 0.5 * (x_verts_1d[1:] + x_verts_1d[:-1])
        y_centers_1d = 0.5 * (y_verts_1d[1:] + y_verts_1d[:-1])
        xv_2d_cen, yv_2d_cen = np.meshgrid(
            x_centers_1d, y_centers_1d, indexing="xy"
        )

        skip = 10

        x_coords_sparse = xv_2d_cen[::skip, ::skip]
        y_coords_sparse = yv_2d_cen[::skip, ::skip]
        vx_sparse = vx_2d[::skip, ::skip]
        vy_sparse = vy_2d[::skip, ::skip]

        quiv = self.ax.quiver(
            x_coords_sparse,
            y_coords_sparse,
            vx_sparse,
            vy_sparse,
            color="white",
            scale=20.0,
            width=0.002,
        )

        self.ax.set_aspect("equal")
        format_multidim_plot_axes(self.ax, self.fig, mesh, data, 0, style)
        return RenderResult(
            artists={"mesh": mesh, "quiver": quiv}, metadata={"mappable": mesh}
        )

    def _plot_density_with_streamlines(
        self, data: PlotData, style: StyleConfig
    ) -> RenderResult:
        """Plot Log(Density) heatmap with a velocity streamline overlay and return artists."""
        if self.ax is None:
            return RenderResult(artists={}, metadata={})

        x_verts_1d = data.fields[0].domain[-2]  # e.g., shape (Nx + 1,)
        y_verts_1d = data.fields[0].domain[-1]  # e.g., shape (Ny + 1,)

        x_centers_1d = 0.5 * (x_verts_1d[1:] + x_verts_1d[:-1])  # shape (Nx,)
        y_centers_1d = 0.5 * (y_verts_1d[1:] + y_verts_1d[:-1])  # shape (Ny,)

        sigma_2d = data.fields[0].values  # shape (Nx, Ny)
        vx_2d = data.fields[1].values  # shape (Nx, Ny)
        vy_2d = data.fields[2].values  # shape (Nx, Ny)

        log_sigma = np.log10(sigma_2d + 1e-10)  # 1e-10 avoids log(0)

        mesh = self.ax.pcolormesh(
            x_verts_1d,
            y_verts_1d,
            log_sigma.T,  # .T to match pcolormesh (x, y, C) convention
            cmap="inferno",
            shading="auto",
            vmin=np.min(log_sigma),  # Optional: set color limits
            vmax=np.max(log_sigma),
        )

        # streamplot expects (M, N) data, where M=len(y) and N=len(x).
        # This requires transposing the velocity data.
        sp = self.ax.streamplot(
            x_centers_1d,  # 1D array of x-coords (shape Nx)
            y_centers_1d,  # 1D array of y-coords (shape Ny)
            vx_2d.T,  # 2D array of u-velocity (shape Ny, Nx)
            vy_2d.T,  # 2D array of v-velocity (shape Ny, Nx)
            color="white",
            linewidth=0.5,
            density=2.0,
            arrowsize=0.7,  # Small arrows to show direction
            arrowstyle="->",  # Standard arrow
        )

        self.ax.set_aspect("equal")
        self.ax.set_xlim(x_centers_1d[0], x_centers_1d[-1])
        self.ax.set_ylim(y_centers_1d[0], y_centers_1d[-1])
        format_multidim_plot_axes(self.ax, self.fig, mesh, data, 0, style)
        return RenderResult(
            artists={"mesh": mesh, "streamplot": sp},
            metadata={"mappable": mesh},
        )

    @property
    def initialized(self) -> bool:
        return self._initialized
