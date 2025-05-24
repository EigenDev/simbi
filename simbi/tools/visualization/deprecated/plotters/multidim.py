import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any
from numpy.typing import NDArray
from matplotlib.collections import QuadMesh
import matplotlib.patches as mpatches
from itertools import cycle
from ..utils.formatting import PlotFormatter
from ..utils.io import DataManager
from ..core.mixins import DataHandlerMixin, AnimationMixin, CoordinatesMixin
from ..core.base import BasePlotter
from ....functional.helpers import find_nearest
from ....core.types.constants import BodyCapability, has_capability
from ..core.constants import LINEAR_FIELDS
from ...utility import get_field_str


class MultidimPlotter(BasePlotter, DataHandlerMixin, AnimationMixin, CoordinatesMixin):
    """Handles 2D and 3D visualization"""

    def __init__(self, parser) -> None:
        super().__init__(parser)
        self.formatter = PlotFormatter()
        self.data_manager = DataManager(
            self.config["plot"].files, movie_mode=self.config["plot"].kind == "movie"
        )
        self.setup_plot()
        self.patch_idx = 0

    def setup_plot(self) -> None:
        """Initialize plot settings"""
        self.cartesian = super().check_cartesian()
        self.patches = self._calculate_patches()

    def _calculate_patches(self) -> None:
        """Calculate number of required patches"""
        n_patches = len(self.config["plot"].fields)
        if len(self.config["plot"].fields) == 1:
            n_patches += 1
        if self.config["multidim"].bipolar:
            n_patches *= 2
        return n_patches

    def create_figure(self) -> None:
        """Create figure with appropriate projection"""
        self.fig, self.axes = plt.subplots(
            1,
            1,
            subplot_kw={"projection": "polar"} if not self.cartesian else None,
            figsize=self.config["style"].fig_dims,
            constrained_layout=not self.cartesian,
        )
        if not self.cartesian:
            self.formatter.setup_polar_axes(self.axes, show_ticks=False)

    def plot(self):
        """Main plotting method"""
        self.frames = []
        field_cycle = cycle(self.config["plot"].fields)
        nfields = len(self.config["plot"].fields)
        labels = get_field_str(self.config["plot"].fields)
        if isinstance(labels, str):
            labels = cycle([labels])
        else:
            labels = cycle(labels)

        for data in self.data_manager.iter_files():
            if not data.setup["is_cartesian"]:
                # create a cycle of theta values that shift backwards by 2*pi/patches
                thetas = self._generate_theta_values(data.setup["x2max"], self.patches)
                theta_cycle = cycle(thetas)

            for idx in range(self.patches):
                field = next(field_cycle)
                is_distinct_field = idx < nfields
                color_range = next(self.config["style"].color_range)
                var = self.get_variable(data.fields, field)

                if var.ndim == 3:
                    var = self._handle_3d_projection(var, data.mesh)

                xx, yy = self.transform_coordinates(data.mesh, data.setup)

                if not self.cartesian:
                    xx, yy, var = self._place_at_patch(xx, yy, var, idx, theta_cycle)

                plot = self._create_plot(xx, yy, var, field, color_range)
                self.frames.append(plot)
                # if axissymmetric cylindrical coordinates,
                # mirror the plot across the axis
                if data.setup["coord_system"] == "axis_cylindrical":
                    self.frames.append(
                        self._create_plot(
                            -xx,
                            yy,
                            var,
                            field,
                            color_range,
                        )
                    )

                if self.config["style"].show_colorbar:
                    if is_distinct_field:
                        if self.cartesian:
                            self.formatter.format_cartesian_colorbar(
                                self.fig,
                                plot,
                                self.axes,
                                label=next(labels),
                                orientation="vertical",
                                side="right" if idx == 0 else "left",
                            )
                        else:
                            self.formatter.format_polar_colorbar(
                                self.fig,
                                plot,
                                nfields,
                                data.setup["x2max"],
                                idx,
                                label=next(labels),
                            )

            # plot any immersed bodies that might exist,
            # on top of the current plot
            if data.immersed_bodies and self.config["style"].draw_immersed_bodies:
                self._plot_immersed_bodies(
                    self.axes,
                    data.immersed_bodies,
                    data.setup,
                )

            self.formatter.set_axes_properties(
                self.fig, self.axes, data.setup, self.config
            )

    def _handle_3d_projection(self, var: np.ndarray, mesh: dict) -> np.ndarray:
        """Handle 3D data projection"""
        depth = (
            np.deg2rad(self.config["multidim"].box_depth)
            if not self.cartesian
            else self.config["multidim"].box_depth
        )
        coord_idx = find_nearest(
            mesh[f"x{self.config['multidim'].projection[2]}v"], depth
        )[0]
        return (
            var[coord_idx]
            if self.config["multidim"].projection[2] == 3
            else (
                var[:, coord_idx, :]
                if self.config["multidim"].projection[2] == 2
                else var[:, :, coord_idx]
            )
        )

    def _place_at_patch(
        self,
        xx: NDArray[np.floating[Any]],
        yy: NDArray[np.floating[Any]],
        var: NDArray[np.floating[Any]],
        idx: int,
        theta_cycle: cycle,
    ) -> tuple[
        NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
    ]:
        """Position data in angular sectors"""

        def theta_sign(quadrant: int) -> int:
            return 1 if quadrant in [0, 3] else -1

        xx = xx[:: theta_sign(idx)] + next(theta_cycle)
        # if idx in [2, 3]:
        #     xx = xx[::-1]
        return xx, yy, var

    def _create_plot(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        var: np.ndarray,
        field: str,
        color_range: tuple,
    ) -> QuadMesh:
        """Create plot with appropriate normalization"""
        vmin = color_range[0] or var.min()
        vmax = color_range[1] or var.max()

        # Handle uniform/near-uniform data
        if np.allclose(vmin, vmax, rtol=1e-10):
            if self.config["style"].log and field not in LINEAR_FIELDS:
                if vmin <= 0:
                    vmin = 0.0
                    vmax = 1.0
                else:
                    vmin = 0.9 * vmax
            else:
                eps = max(abs(vmin) * 1e-2, 0.1)
                vmin -= eps
                vmax += eps

        if self.config["style"].log and field not in LINEAR_FIELDS:
            if any(x < 1e-10 for x in var.flat):
                var = np.where(var < 1e-10, 1e-10, var)
                vmin = max(vmin, 1e-10)

        norm = (
            mcolors.LogNorm(vmin=vmin, vmax=vmax)
            if self.config["style"].log and field not in LINEAR_FIELDS
            else mcolors.PowerNorm(
                gamma=self.config["style"].power,
                vmin=vmin,
                vmax=vmax,
            )
        )
        return self.axes.pcolormesh(
            xx,
            yy,
            var,
            cmap=next(self.config["style"].color_maps),
            shading="auto",
            norm=norm,
        )

    def _generate_theta_values(self, x2max: float, npatches: int) -> np.ndarray:
        """Generate theta values that shift backwards by 2*x2max

        Args:
            x2max: Maximum theta value
            npatches: Number of patches to generate

        Returns:
            Array of theta values
        """
        base = np.zeros(npatches)
        for i in range(npatches):
            if i == 0:
                base[i] = 0
            else:
                base[i] = -x2max * i
                if base[i] < -2 * x2max:  # Wrap around if needed
                    base[i] += 4 * x2max
        return base

    def _plot_circle(
        self,
        axes: Any,
        center: tuple[float, float],
        radius: float,
        color: str,
        linestyle: str,
    ) -> None:
        """Plot a circle on the given axes"""
        circle = mpatches.Circle(
            center, radius, color=color, linestyle=linestyle, alpha=1
        )
        axes.add_patch(circle)
        axes.set_aspect("equal", adjustable="box")
        axes.autoscale_view()

    def _plot_immersed_bodies(
        self,
        axes: Any,
        immersed_bodies: dict[str, Any],
        setup: dict[str, Any],
    ) -> None:
        """Plot immersed bodies on top of the current plot"""
        for body in immersed_bodies.values():
            if has_capability(body["type"], BodyCapability.ACCRETION):
                radius = body["accretion_radius"]
            else:
                radius = body["radius"]
            self._plot_circle(
                axes,
                body["position"],
                radius,
                "red",
                "--",
            )
