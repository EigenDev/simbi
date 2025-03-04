from matplotlib.animation import FuncAnimation
from typing import Dict, Any
from matplotlib import colors as mcolors
from ..core.constants import DERIVED
from ....functional.helpers import calc_any_mean
from ... import utility as util
import numpy as np
from numpy.typing import NDArray


class DataHandlerMixin:
    """Mixin for data handling operations"""

    def get_variable(self, fields: Dict[str, np.ndarray], field: str) -> np.ndarray:
        """Extract and transform variable data"""
        var = (
            util.prims2var(fields, field)
            if field in DERIVED
            else fields.get(field, fields["v1"] if field == "v" else fields[field])
        )

        if self.config["style"].units:
            self._apply_units(var, field)
        return var

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
        fields, setup, mesh = util.read_file(self.data_manager.file_list[frame])

        # Update plot title
        self._update_title(setup)

        # Update data
        for idx, field in enumerate(self.config["plot"].fields):
            var = self.get_variable(fields, field)
            self._update_plot_data(idx, var, mesh, setup)

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
        setup_name = self.config["plot"].setup
        title = rf"{setup_name} t = {time:.1f}"
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
        mesh: dict[str, NDArray[np.float64]],
        setup: dict[str, Any],
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
            # vmin, vmax = np.nanmin(var), np.nanmax(var)
            for drawing in self.frames:
                # Update norm if using LogNorm
                # if isinstance(drawing.norm, mcolors.LogNorm):
                #     drawing.norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                # else:
                #     drawing.norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                drawing.set_array(var.ravel())


class CoordinatesMixin:
    """Mixin for coordinate system operations"""

    def get_slice_data(
        self,
        var: NDArray[np.float64],
        mesh: dict[str, NDArray[np.float64]],
        setup: dict[str, Any],
    ) -> NDArray[np.float64]:
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
        self, mesh: dict[str, NDArray[np.float64]], setup: dict[str, Any]
    ) -> tuple:
        """Get the permuted indices for slice through higher dimensions"""
        if self.config["multidim"].slice_along == "x1":
            return "x2v", "x3v"
        elif self.config["multidim"].slice_along == "x2":
            return "x3v", "x1v"
        else:
            return "x1v", "x2v"

    def _get_slice_indices(
        self, mesh: dict[str, NDArray[np.float64]], setup: dict[str, Any]
    ) -> tuple:
        """Get indices for slice through higher dimensions"""
        for xkcoord in map(float, self.config["multidim"].coords["xk"].split(",")):
            for xjcoord in map(float, self.config["multidim"].coords["xj"].split(",")):
                if setup["coord_system"] == "spherical":
                    xjcoord = np.deg2rad(xjcoord)
                    xkcoord = np.deg2rad(xkcoord)
                xj, xk = self._get_permuted_indices(mesh, setup)
                jidx = util.find_nearest(mesh.get(xj, np.linspace(0, 1)), xjcoord)[0]
                kidx = util.find_nearest(mesh.get(xk, np.linspace(0, 1)), xkcoord)[0]
                if self.config["plot"].ndim == 2:
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
        x1c = calc_any_mean(mesh["x1v"], setup["x1_spacing"])
        x2c = calc_any_mean(mesh["x2v"], setup["x2_spacing"])
        xx, yy = np.meshgrid(x1c, x2c)[::-1]
        if self.config["multidim"].bipolar:
            xx *= -1
        return xx, yy

    def _transform_cartesian(
        self, mesh: dict[str, NDArray[np.float64]], setup: dict[str, Any]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
