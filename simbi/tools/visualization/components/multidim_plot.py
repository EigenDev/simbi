from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import QuadMesh
from .base import Component
from ..bridge import SimbiDataBridge
from ..constants.fields import LINEAR_FIELDS
from ..styling import colorbar_formatter, axis_formatter


class MultidimPlotComponent(Component):
    """2D/3D visualization component"""

    def setup(self) -> None:
        """Initialize multidim plot resources"""
        # Create bridge to access data
        self.bridge = SimbiDataBridge(self.state)
        file_mesh = self.state.data.mesh
        file_setup = self.state.data.setup
        x1, x2 = self.bridge.transform_coordinates(file_mesh, file_setup)
        lx1 = len(x1)
        lx2 = len(x2)
        # Create empty pcolormesh plot
        self.mesh = self.ax.pcolormesh(
            x1,
            x2,
            np.zeros((lx2 - 1, lx1 - 1)),
            cmap=self.props.get("cmap", "viridis"),
            shading="auto",
        )

        # Store reference
        self.state.plot_elements[f"{self.id}_mesh"] = self.mesh

    def _add_colorbar(self, field: str) -> plt.colorbar:
        """Add colorbar to the plot with appropriate positioning"""
        # Determine if we're using polar coordinates
        is_cartesian = self.state.data.setup.get("is_cartesian", True)

        if is_cartesian:
            # Use centralized formatter for Cartesian colorbar
            self.colorbar = colorbar_formatter.add_cartesian_colorbar(
                self.fig, self.ax, self.mesh, field, self.state.config
            )
        else:
            # Use centralized formatter for polar colorbar
            self.colorbar = colorbar_formatter.add_polar_colorbar(
                self.fig,
                self.ax,
                self.mesh,
                field,
                self.state.config,
                self.state.data.setup,
            )

    def _draw_immersed_bodies(self) -> None:
        """Draw immersed bodies on the plot"""
        # Skip if no immersed bodies
        if not self.state.data or not self.state.data.immersed_bodies:
            return

        # Clear any existing patches
        for patch in self.ax.patches:
            patch.remove()

        import matplotlib.patches as mpatches

        proj = self.state.config["multidim"]["projection"]
        # Draw each immersed body
        for body_id, body in self.state.data.immersed_bodies.items():
            # Determine radius based on body type
            from ....core.types.constants import BodyCapability, has_capability

            if has_capability(body.get("type", 0), BodyCapability.ACCRETION):
                radius = body.get("accretion_radius", 0.0)
            else:
                radius = body.get("radius", 0.0)

            # if we're in 3D, then we must be
            # mindful to plot the circles projected
            # along the 2D plane. If the projection
            # is (1,2,3), then we are plotting in the
            # x1-x2 plane, and the z coordinate
            # is the third coordinate. If we are plotting
            # in the x2-x3 plane, then we need to
            # onto the x1. The body position is
            # always in cartesian coordinates.
            if proj == (1, 2, 3):
                projected_position = (
                    body["position"][0],
                    body["position"][1],
                )
            elif proj == (2, 3, 1):
                projected_position = (
                    body["position"][1],
                    body["position"][2],
                )
            else:
                projected_position = (
                    body["position"][0],
                    body["position"][2],
                )

            # Create circle
            circle = mpatches.Circle(
                projected_position,
                radius,
                color="red",
                linestyle="--",
                alpha=1.0,
            )

            # Add to plot
            self.ax.add_patch(circle)

        # Set aspect and autoscale
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.autoscale_view()

    def _place_at_patch(
        self, xx: np.ndarray, yy: np.ndarray, var: np.ndarray, idx: int, theta: float
    ) -> tuple:
        """Position data in angular sectors for polar plots"""

        def theta_sign(quadrant: int) -> int:
            """Determine sign for patch direction"""
            return 1 if quadrant in [0, 3] else -1

        # Apply theta shift
        xx_shifted = xx[:: theta_sign(idx)] + theta

        return xx_shifted, yy, var

    def _generate_theta_values(self, x2max: float, npatches: int) -> np.ndarray:
        """Generate theta values for patches"""
        base = np.zeros(npatches)
        for i in range(npatches):
            if i == 0:
                base[i] = 0
            else:
                base[i] = -x2max * i
                if base[i] < -2 * x2max:  # Wrap around if needed
                    base[i] += 4 * x2max
        return base

    def _project_3d_data(self, var: np.ndarray) -> np.ndarray:
        """Project 3D data to 2D based on configuration"""
        depth = self.state.config.get("multidim", {}).get("box_depth", 0)
        projection = self.state.config.get("multidim", {}).get("projection", [1, 2, 3])

        # Find nearest depth index
        from ....functional.helpers import find_nearest

        mesh = self.state.data.mesh
        depth_idx = find_nearest(mesh[f"x{projection[2]}v"], depth)[0]

        # Project data
        if projection[2] == 3:
            return var[depth_idx]
        elif projection[2] == 2:
            return var[:, depth_idx, :]
        else:
            return var[:, :, depth_idx]

    def update(self, props: dict[str, Any]) -> None:
        """Update component properties"""
        super().update(props)

        # Apply updates to the mesh if it exists
        if hasattr(self, "mesh"):
            if "cmap" in props:
                self.mesh.set_cmap(props["cmap"])

    def render(self) -> QuadMesh:
        """Render the 2D/3D plot with current data"""
        if not self.state.data:
            return self.mesh

        # Get field data
        field = self.props["field"]
        var = self.bridge.get_variable(field)

        # For 3D data, handle projection
        if var.ndim == 3:
            var = self._project_3d_data(var)

        # Get coordinate meshes for plotting
        mesh = self.state.data.mesh
        setup = self.state.data.setup

        # Determine if we're using polar coordinates
        is_cartesian = setup.get("is_cartesian", True)

        # Handle polar patches if needed
        if not is_cartesian and self.state.config.get("multidim", {}).get(
            "bipolar", False
        ):
            # Get number of patches
            nfields = len(self.state.config.get("plot", {}).get("fields", ["rho"]))
            npatches = nfields
            if nfields == 1:
                npatches += 1
            if self.state.config.get("multidim", {}).get("bipolar", False):
                npatches *= 2

            # Get patch index
            patch_idx = int(self.id.split("_")[-1]) % npatches

            # Generate theta values
            thetas = self._generate_theta_values(setup.get("x2max", np.pi), npatches)

            # Get coordinates
            xx, yy = self.bridge.transform_coordinates(mesh, setup)

            # Apply patch position
            xx, yy, var = self._place_at_patch(
                xx, yy, var, patch_idx, thetas[patch_idx]
            )
        else:
            # Get standard coordinates
            xx, yy = self.bridge.transform_coordinates(mesh, setup)

        # Create or update the mesh
        if isinstance(xx, np.ndarray) and isinstance(yy, np.ndarray):
            if xx.ndim == 1 and yy.ndim == 1:
                X, Y = np.meshgrid(xx, yy)
            else:
                X, Y = xx, yy

            # Update mesh with new data
            self.mesh.set_array(var.ravel())

            # Update normalization
            color_range = self.props.get("color_range", (None, None))
            vmin, vmax = color_range

            # Auto calculate limits if not provided
            if vmin is None:
                vmin = var.min()
            if vmax is None:
                vmax = var.max()

            # Handle uniform/near-uniform data
            if np.allclose(vmin, vmax, rtol=1e-10):
                if (
                    self.state.config.get("style", {}).get("log", False)
                    and field not in LINEAR_FIELDS
                ):
                    if vmin <= 0:
                        vmin = 0.0
                        vmax = 1.0
                    else:
                        vmin = 0.9 * vmax
                else:
                    eps = max(abs(vmin) * 1e-2, 0.1)
                    vmin -= eps
                    vmax += eps

            # Apply appropriate normalization
            if (
                self.state.config.get("style", {}).get("log", False)
                and field not in LINEAR_FIELDS
            ):
                if any(x < 1e-10 for x in var.flat):
                    var = np.where(var < 1e-10, 1e-10, var)
                    vmin = max(vmin, 1e-10)
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                # Use power norm with configurable gamma
                gamma = self.state.config.get("style", {}).get("power", 1.0)
                norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

            self.mesh.set_norm(norm)

            # Add colorbar if needed
            if self.state.config.get("style", {}).get(
                "show_colorbar", True
            ) and not hasattr(self, "colorbar"):
                self._add_colorbar(field)

            # Draw immersed bodies if configured
            if self.state.config.get("style", {}).get("draw_immersed_bodies", False):
                self._draw_immersed_bodies()

            # Handle axisymmetric cylindrical coordinates by mirroring
            if (
                setup.get("coord_system") == "axis_cylindrical"
                and not hasattr(self, "mirror_mesh")
                and not self.props.get("is_mirror", False)
            ):
                self._add_mirror_mesh(X, Y, var, field, color_range)

        self.format_axis()

        return self.mesh

    def _add_mirror_mesh(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        var: np.ndarray,
        field: str,
        color_range: tuple,
    ) -> None:
        """Add a mirrored mesh for axisymmetric cylindrical coordinates"""
        # Create mirrored mesh
        self.mirror_mesh = self.ax.pcolormesh(
            -xx, yy, var, cmap=self.mesh.get_cmap(), shading="auto", norm=self.mesh.norm
        )

        # Store reference
        self.state.plot_elements[f"{self.id}_mirror_mesh"] = self.mirror_mesh
