import math
from typing import Any

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from simbi.functional.reader import BodyCapability, has_capability

from ....functional.helpers import calc_any_mean
from ... import utility as util
from ...utility import get_field_str
from ..core.constants import FIELD_ALIASES


class DataHandlerMixin:
    """Mixin for data handling operations"""

    def get_variable(self, fields: dict[str, np.ndarray], field: str) -> np.ndarray:
        """Extract and transform variable data"""
        if field in FIELD_ALIASES:
            var = fields[FIELD_ALIASES[field]]
        else:
            var = fields[field]

        if self.config["style"].units:
            self._apply_units(var, field)

        return var

    def get_label(self, field_name: str, user_defined_label: str | None = None) -> str:
        if user_defined_label:
            return user_defined_label
        if field_name in FIELD_ALIASES:
            field_name = FIELD_ALIASES[field_name]
        return str(get_field_str(field_name))

    def _apply_units(self, var: np.ndarray, field: str) -> None:
        """Apply physical units to variable"""
        if field in ["p", "energy", "energy_rst"]:
            var *= util.edens_scale.value
        elif field in ["rho", "D"]:
            var *= util.rho_scale.value


class AnimationMixin:
    """Mixin for animation capabilities"""

    def update_frame(self, frame: int) -> tuple:
        """Update plot for animation frame"""
        fields, metadata, mesh, immersed_bodies = util.read_file(
            self.data_manager.file_list[frame]
        )

        # Update plot title
        self._update_title(metadata)

        # Update data
        for idx, field in enumerate(self.config["plot"].fields):
            var = self.get_variable(fields, field)
            self._update_plot_data(idx, var, mesh, metadata, immersed_bodies)

        return (self.frames,)

    def animate(self) -> None:
        """Create animation"""
        self.plot()

        self.animation = FuncAnimation(
            self.fig,
            self.update_frame,
            np.arange(self.data_manager.frame_count),
            interval=1000 / self.config["animation"].frame_rate,
        )

    def _update_title(self, setup: dict) -> None:
        """Update plot title"""
        time = setup["time"] * (util.time_scale if self.config["style"].units else 1.0)
        time_unit = ""
        if self.config["style"].orbital_params is not None:
            p = self.config["style"].orbital_params
            time = setup["time"] / (
                2.0
                * math.pi
                * math.sqrt(float(p["separation"]) ** 3 / float(p["mass"]))
            )
            time_unit = "orbit(s)"

        setup_name = self.config["plot"].setup
        title = rf"{setup_name} t = {time:.1f} {time_unit}"
        if setup["is_cartesian"] or self.config["multidim"].slice_along:
            if isinstance(self.axes, list):
                self.axes[0].set_title(title)
            else:
                self.axes.set_title(title)
        else:
            kwargs = {
                "y": 0.95 if setup["x2max"] == np.pi else 0.8,
            }
            self.fig.suptitle(title, **kwargs)

    def _update_plot_data(
        self,
        idx: int,
        var: np.ndarray,
        mesh: dict[str, NDArray[np.floating[Any]]],
        setup: dict[str, Any],
        immersed_bodies: dict[str, Any] | None = None,
    ) -> None:
        """Update plot data"""
        if self.config["plot"].ndim == 1:
            self.axes[0].set_xlim(setup["x1min"], setup["x1max"])
            x = calc_any_mean(mesh["x1v"], setup["x1_spacing"])
            self.frames[idx].set_data(x, var)
        elif self.config["multidim"].slice_along:
            x = calc_any_mean(
                mesh[f"{self.config['multidim'].slice_along}v"],
                setup[f"{self.config['multidim'].slice_along}_spacing"],
            )
            sliced_var = self.get_slice_data(var, mesh, setup)
            self.frames[idx].set_data(x, sliced_var)
        else:
            if len(self.config["plot"].fields) == 1:
                for drawing in self.frames:
                    drawing.set_array(var.ravel())
            else:
                drawing = self.frames[idx]
                # if a user has not defined colorbar limits,
                # let the colorbar auto adjust
                if all(x is None for x in next(self.config["style"].color_range)):
                    vmin, vmax = np.min(var), np.max(var)
                    if isinstance(drawing.norm, mcolors.LogNorm):
                        drawing.norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                    else:
                        drawing.norm = mcolors.PowerNorm(
                            vmin=vmin, vmax=vmax, gamma=self.config["style"].power
                        )
                drawing.set_array(var.ravel())
            if immersed_bodies:
                # Clear previous patches
                for patch in self.axes.patches:
                    patch.remove()

                for body in immersed_bodies.values():
                    if has_capability(body["type"], BodyCapability.ACCRETION):
                        radius = body["accretion_radius"]
                    else:
                        radius = body["radius"]
                    circle = mpatches.Circle(
                        body["position"],
                        radius,
                        color="black",
                        linestyle="--",
                        alpha=0.5,
                    )
                    self.axes.add_patch(circle)
                    self.axes.set_aspect("equal", adjustable="box")
                    self.axes.autoscale_view()


class CoordinatesMixin:
    """Mixin for coordinate system operations"""

    def get_slice_data(
        self,
        var: NDArray[np.floating[Any]],
        mesh: dict[str, NDArray[np.floating[Any]]],
        setup: dict[str, Any],
    ) -> NDArray[np.floating[Any]]:
        """ "Get data for a 1D slice of a higher-dimensional field"""
        slices = self._get_slice_indices(mesh, setup)
        return var[slices]

    def check_cartesian(self) -> bool:
        """Check if the mesh is Cartesian"""
        data = self.data_manager.read_file(self.config["plot"].files[0])
        return data.setup["is_cartesian"]

    def transform_coordinates(self, mesh: dict, setup: dict) -> tuple:
        """Transform coordinates based on system"""
        if self.config["multidim"].slice_along:
            return self._transform_slice_coords(mesh, setup)
        elif not setup["is_cartesian"]:
            return self._transform_polar(mesh, setup)
        return self._transform_cartesian(mesh, setup)

    def _transform_slice_coords(self, mesh: dict, setup: dict) -> tuple:
        """Transform coordinates for 1D slice"""
        x = calc_any_mean(
            mesh[f"{self.config['multidim'].slice_along}v"],
            setup[f"{self.config['multidim'].slice_along}_spacing"],
        )
        return x, self._get_slice_indices(mesh, setup)

    def _get_permuted_indices(
        self, mesh: dict[str, NDArray[np.floating[Any]]], setup: dict[str, Any]
    ) -> tuple:
        """Get the permuted indices for slice through higher dimensions"""
        if self.config["multidim"].slice_along == "x1":
            return "x2v", "x3v"
        elif self.config["multidim"].slice_along == "x2":
            return "x3v", "x1v"
        else:
            return "x1v", "x2v"

    def _get_slice_indices(
        self, mesh: dict[str, NDArray[np.floating[Any]]], setup: dict[str, Any]
    ) -> Any:
        """Get indices for slice through higher dimensions"""
        for xkcoord in map(float, self.config["multidim"].coords["xk"].split(",")):
            for xjcoord in map(float, self.config["multidim"].coords["xj"].split(",")):
                if setup["coord_system"] == "spherical":
                    xjcoord = np.deg2rad(xjcoord)
                    xkcoord = np.deg2rad(xkcoord)
                xj, xk = self._get_permuted_indices(mesh, setup)
                jidx = util.find_nearest(mesh.get(xj, np.linspace(0, 1)), xjcoord)[0]
                kidx = util.find_nearest(mesh.get(xk, np.linspace(0, 1)), xkcoord)[0]
                if mesh["effective_dimensions"] == 1:
                    return (None, None, None)
                elif mesh["effective_dimensions"] == 2:
                    return jidx
                else:
                    if self.config["multidim"].slice_along == "x1":
                        slice = np.s_[kidx, jidx, :]
                    elif self.config["multidim"].slice_along == "x2":
                        slice = np.s_[jidx, :, kidx]
                    else:
                        slice = np.s_[:, kidx, jidx]
                    return slice

    def _transform_polar(self, mesh: dict, setup: dict[str, Any]) -> tuple:
        """Handle polar coordinate transforms"""
        xx, yy = np.meshgrid(mesh["x1v"], mesh["x2v"])[::-1]
        if self.config["multidim"].bipolar:
            xx *= -1
        return xx, yy

    def _transform_cartesian(
        self, mesh: dict[str, NDArray[np.floating[Any]]], setup: dict[str, Any]
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Transform coordinates for standard plot"""
        dim = self.config["plot"].ndim
        if dim == 1:
            return (
                calc_any_mean(mesh["x1v"], setup["x1_spacing"]),
                range(len(mesh["x1v"])),
            )
        elif dim == 2:
            return (
                mesh["x1v"],
                mesh["x2v"],
            )
        else:
            xj = f"x{self.config['multidim'].projection[0]}"
            xk = f"x{self.config['multidim'].projection[1]}"
            return (
                mesh[f"{xj}v"],
                mesh[f"{xk}v"],
            )
