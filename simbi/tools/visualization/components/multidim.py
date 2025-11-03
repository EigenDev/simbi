"""Multidimensional plot component for visualization."""

from typing import Any, Literal, Optional

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from pydantic import ValidationInfo, field_validator

from simbi.core.types import Body, FieldData
from simbi.tools.visualization.formatters.multidim import (
    format_multidim_plot_axes,
)

from ..core.config import StyleConfig
from ..core.types import Array, ColorRange, PlotData
from .interface import Component, ComponentProps


def create_color_normalization(
    values: Array,
    color_range: ColorRange,
    log_scale: bool = False,
    power: float = 1.0,
) -> mcolors.Normalize:
    """Create color normalization based on data and settings."""
    vmin = color_range.min or np.nanmin(values)
    vmax = color_range.max or np.nanmax(values)

    # Handle uniform data
    if np.allclose(vmin, vmax, rtol=1e-10):
        eps = max(float(abs(vmin) * 1e-2), 0.1)
        vmin -= eps
        vmax += eps

    if log_scale:
        if vmin <= 0:
            pos_min = (
                np.nanmin(values[values > 0]) if np.any(values > 0) else 1e-10
            )
            vmin = pos_min * 0.9
        return mcolors.LogNorm(vmin=float(vmin), vmax=float(vmax))
    else:
        return mcolors.PowerNorm(
            gamma=power, vmin=float(vmin), vmax=float(vmax)
        )


def compose_amr_leaf_cells(
    fields: list[FieldData],
) -> tuple[Array, Array, Array]:
    """
    Compose AMR hierarchy into single leaf-cell grid.

    Assumes fields are ordered by level (L0, L1, L2, ...) and that
    refined regions are spatial subsets of coarser levels.

    Returns:
        (x_edges, y_edges, leaf_values): Unified grid with finest available data
    """
    if not fields:
        raise ValueError("No fields provided for composition")

    # Start with base level (assumed to be first field)
    base_field = fields[0]
    x_base, y_base = base_field.domain
    leaf_values = base_field.values.copy()

    # Single level case
    if len(fields) == 1:
        return x_base, y_base, leaf_values

    # Multi-level: overwrite coarse cells with refined data
    for field in fields[1:]:
        x_fine, y_fine = field.domain

        # Find spatial overlap region
        x_overlap = (x_fine[0] >= x_base[0]) and (x_fine[-1] <= x_base[-1])
        y_overlap = (y_fine[0] >= y_base[0]) and (y_fine[-1] <= y_base[-1])

        if not (x_overlap and y_overlap):
            continue  # No overlap, skip this level

        # Find indices in base grid that overlap with refined region
        i_start = np.searchsorted(x_base, x_fine[0])
        i_end = np.searchsorted(x_base, x_fine[-1])
        j_start = np.searchsorted(y_base, y_fine[0])
        j_end = np.searchsorted(y_base, y_fine[-1])

        # Determine refinement ratio
        n_fine_x = len(x_fine)
        n_fine_y = len(y_fine)
        n_coarse_x = i_end - i_start
        n_coarse_y = j_end - j_start

        # If dimensions match, direct replacement
        if n_fine_x == n_coarse_x and n_fine_y == n_coarse_y:
            leaf_values[j_start:j_end, i_start:i_end] = field.values
        else:
            # Refinement exists: downsample or reconstruct
            # For now, simple averaging (TODO: use more sophisticated methods)
            refined_region = _downsample_to_coarse(
                field.values, n_coarse_x, n_coarse_y
            )
            leaf_values[j_start:j_end, i_start:i_end] = refined_region

    return x_base, y_base, leaf_values


def _downsample_to_coarse(
    fine_data: Array, nx_coarse: int, ny_coarse: int
) -> Array:
    """Downsample refined data to coarse grid resolution via averaging."""
    ny_fine, nx_fine = fine_data.shape

    # Calculate refinement ratios
    rx = nx_fine // nx_coarse
    ry = ny_fine // ny_coarse

    # Reshape and average
    downsampled = (
        fine_data[: ny_coarse * ry, : nx_coarse * rx]
        .reshape(ny_coarse, ry, nx_coarse, rx)
        .mean(axis=(1, 3))
    )

    return downsampled


class MultidimPlotProps(ComponentProps):
    """Properties for multidimensional plot component."""

    cmap: str = "viridis"
    color_range: ColorRange = ColorRange(min=None, max=None)
    field_index: int = 0
    log_scale: bool = False
    power: float = 1.0
    shading: Literal["auto", "nearest", "gouraud", "flat"] = "auto"
    alpha: float = 1.0
    projection: tuple[int, int, int] = (1, 2, 3)
    plot_type: Literal["polar", "cartesian"] = "cartesian"

    # Rendering mode
    render_mode: Literal["pcolormesh", "polygons"] = "pcolormesh"

    # Mesh visualization
    show_mesh_grid: bool = False
    mesh_color: str = "white"
    mesh_alpha: float = 0.3
    mesh_linewidth: float = 0.1

    @field_validator("field_index")
    @classmethod
    def validate_field_index(cls, v: int, info: ValidationInfo) -> int:
        if v < 0:
            raise ValueError(f"Field index must be non-negative, got {v}")
        return v

    @field_validator("power")
    @classmethod
    def validate_power(cls, v: float, info: ValidationInfo) -> float:
        if v <= 0:
            raise ValueError(f"Power must be positive, got {v}")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float, info: ValidationInfo) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {v}")
        return v

    @field_validator("shading")
    @classmethod
    def validate_shading(cls, v: str, info: ValidationInfo) -> str:
        valid_options = ["auto", "flat", "gouraud", "nearest"]
        if v not in valid_options:
            raise ValueError(f"Shading must be one of {valid_options}, got {v}")
        return v


class MultidimPlotComponent(Component):
    """Multidimensional plot visualization component."""

    def __init__(self, props: MultidimPlotProps):
        self.props = props
        self._mesh: Optional[QuadMesh] = None
        self._poly_collection = None
        self._mirror_mesh: Optional[QuadMesh] = None
        self._initialized: bool = False
        self.last_x = np.array([])
        self.last_y = np.array([])
        self._mesh_lines: list = []

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: MultidimPlotProps) -> None:
        """Update component properties."""
        prev_props = self.props
        self.props = props

        if self._mesh and hasattr(self, "ax"):
            if prev_props.cmap != props.cmap:
                self._mesh.set_cmap(props.cmap)
            if prev_props.alpha != props.alpha:
                self._mesh.set_alpha(props.alpha)

            # Handle mesh grid toggle
            if prev_props.show_mesh_grid != props.show_mesh_grid:
                if props.show_mesh_grid:
                    self._draw_mesh_grid(self.last_x, self.last_y)
                else:
                    self._clear_mesh_grid()

    def draw_bodies(
        self, bodies: dict[str, Body], proj: tuple[int, ...], zorder: int
    ) -> None:
        """Draw immersed bodies on the plot."""
        for patch in self.ax.patches:
            patch.remove()

        import matplotlib.patches as mpatches

        for _, body in bodies.items():
            radius = (
                body.accretion.accretion_radius
                if body.accretion is not None
                else body.radius
            )

            # Project body position onto 2D plane
            if proj == (1, 2, 3):
                projected_position = (body.position[0], body.position[1])
            elif proj == (2, 3, 1):
                projected_position = (body.position[1], body.position[2])
            else:
                projected_position = (body.position[0], body.position[2])

            circle = mpatches.Circle(
                projected_position,
                radius,
                color="black",
                linestyle="--",
                alpha=0.5,
                zorder=zorder,
            )
            self.ax.add_patch(circle)

    def render(self, data: PlotData, style: StyleConfig) -> dict[str, Any]:
        """Render the multidimensional plot."""
        if not self._initialized or not hasattr(self, "ax"):
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        # Get field(s) for specified index
        field = data.fields[self.props.field_index]

        # Check if we have multi-level data for this field
        field_base_name = (
            field.name.split("_L")[0] if "_L" in field.name else field.name
        )
        level_fields = [
            f for f in data.fields if f.name.startswith(field_base_name)
        ]

        # Route to appropriate rendering method
        if self.props.render_mode == "polygons":
            result = self._render_polygons(level_fields, data, style)
        else:
            result = self._render_pcolormesh(level_fields, data, style)

        # Format axes
        mesh_or_collection = result.get("mesh") or result.get("collection")
        format_multidim_plot_axes(
            self.ax,
            self.fig,
            mesh_or_collection,
            data,
            self.props.field_index,
            style,
            self.props.plot_type == "polar",
        )

        return result

    def _render_pcolormesh(
        self, level_fields: list[FieldData], data: PlotData, style: StyleConfig
    ) -> dict[str, Any]:
        """Render using pcolormesh with composed leaf cells."""
        # Compose leaf cells if multiple levels exist
        if len(level_fields) > 1:
            x, y, values = compose_amr_leaf_cells(level_fields)
        else:
            field = level_fields[0]
            x, y = field.domain
            values = field.values

        # Handle polar coordinate transformation
        if self.props.plot_type == "polar":
            x, y = y, x
            values = values.T

        # Create or update mesh
        if self._mesh is None:
            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=self.props.cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
            )
        else:
            self._update_mesh(x, y, values)

        # Apply color normalization
        norm = create_color_normalization(
            values,
            self.props.color_range,
            self.props.log_scale,
            self.props.power,
        )
        self._mesh.set_norm(norm)

        # Draw mesh grid if enabled
        if self.props.show_mesh_grid:
            self._draw_mesh_grid(x, y)

        # Draw bodies if configured
        if style.draw_bodies:
            self.draw_bodies(data.bodies, self.props.projection, zorder=10)

        self.last_x = x
        self.last_y = y
        return {"mesh": self._mesh}

    def _render_polygons(
        self, level_fields: list[FieldData], data: PlotData, style: StyleConfig
    ) -> dict[str, Any]:
        """Render using PolyCollection to show AMR structure."""
        from matplotlib.collections import PolyCollection

        # clear old collection
        if self._poly_collection is not None:
            self._poly_collection.remove()
            self._poly_collection = None

        # build polygon patches from all levels (leaf cells only)
        patches = []
        values = []

        # track refined regions to skip covered coarse cells
        refined_regions = []
        for field in level_fields[1:]:  # All levels except base
            x, y = field.domain
            refined_regions.append(
                {"xmin": x[0], "xmax": x[-1], "ymin": y[0], "ymax": y[-1]}
            )

        for level_idx, field in enumerate(level_fields):
            x, y = field.domain

            # handle polar coordinate transformation
            if self.props.plot_type == "polar":
                x, y = y, x
                field_values = field.values.T
            else:
                field_values = field.values

            # Create cell rectangles
            for i in range(len(x) - 1):
                for j in range(len(y) - 1):
                    cell_x_center = (x[i] + x[i + 1]) / 2
                    cell_y_center = (y[j] + y[j + 1]) / 2

                    # Skip if this cell is covered by a finer level
                    is_covered = False
                    for region in refined_regions[level_idx:]:
                        if (
                            region["xmin"] <= cell_x_center <= region["xmax"]
                            and region["ymin"]
                            <= cell_y_center
                            <= region["ymax"]
                        ):
                            is_covered = True
                            break

                    if is_covered:
                        continue

                    cell = [
                        (x[i], y[j]),
                        (x[i + 1], y[j]),
                        (x[i + 1], y[j + 1]),
                        (x[i], y[j + 1]),
                    ]
                    patches.append(cell)
                    values.append(field_values[j, i])

        edge_color = (
            self.props.mesh_color if self.props.show_mesh_grid else "none"
        )
        edge_width = (
            self.props.mesh_linewidth if self.props.show_mesh_grid else 0
        )

        self._poly_collection = PolyCollection(
            patches,
            array=np.array(values),
            cmap=self.props.cmap,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=self.props.alpha,
        )

        norm = create_color_normalization(
            np.array(values),
            self.props.color_range,
            self.props.log_scale,
            self.props.power,
        )
        self._poly_collection.set_norm(norm)

        self.ax.add_collection(self._poly_collection)
        self.ax.autoscale_view()

        if style.draw_bodies:
            self.draw_bodies(data.bodies, self.props.projection, zorder=10)

        return {"collection": self._poly_collection}

    def _update_mesh(self, x: Array, y: Array, values: Array) -> None:
        """Update existing mesh with new data."""
        if self._mesh is None:
            raise RuntimeError("Mesh is not initialized. Call render() first.")

        fast_mesh = self.props.render_mode == "pcolormesh"
        if fast_mesh and (
            not np.allclose(x, self.last_x) or not np.allclose(y, self.last_y)
        ):
            # Coordinates changed: create new mesh
            if self._mesh in self.ax.collections:
                self._mesh.remove()

            self._mesh = self.ax.pcolormesh(
                x,
                y,
                values,
                cmap=self.props.cmap,
                shading=self.props.shading,
                alpha=self.props.alpha,
            )

            if self.ax.name == "polar":
                self._mirror_mesh = self.ax.pcolormesh(
                    -x[::-1],
                    y,
                    values,
                    cmap=self.props.cmap,
                    shading=self.props.shading,
                    alpha=self.props.alpha,
                )

            self.last_x = x
            self.last_y = y
        else:
            # Just update values
            self._mesh.set_array(values.ravel())
            if self._mirror_mesh:
                self._mirror_mesh.set_array(values.ravel())

    def _draw_mesh_grid(self, x: Array, y: Array) -> None:
        """Draw cell boundaries on the mesh."""
        self._clear_mesh_grid()

        # Draw vertical lines
        for xi in x:
            line = self.ax.axvline(
                xi,
                color=self.props.mesh_color,
                alpha=self.props.mesh_alpha,
                linewidth=self.props.mesh_linewidth,
                zorder=5,
            )
            self._mesh_lines.append(line)

        # Draw horizontal lines
        for yi in y:
            line = self.ax.axhline(
                yi,
                color=self.props.mesh_color,
                alpha=self.props.mesh_alpha,
                linewidth=self.props.mesh_linewidth,
                zorder=5,
            )
            self._mesh_lines.append(line)

    def _clear_mesh_grid(self) -> None:
        """Remove all mesh grid lines."""
        for line in self._mesh_lines:
            line.remove()
        self._mesh_lines.clear()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._clear_mesh_grid()
        if hasattr(self, "ax"):
            if self._mesh is not None and self._mesh in self.ax.collections:
                self._mesh.remove()
                self._mesh = None
            if self._poly_collection is not None:
                self._poly_collection.remove()
                self._poly_collection = None
