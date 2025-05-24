from typing import Any, Sequence, Optional
import numpy as np
from .state.core import VisualizationState, SimulationData
from ...functional.reader import read_file
from ..utility import get_field_str
from .constants.alias import FIELD_ALIASES


class SimbiDataBridge:
    """Bridge between existing data processing logic and new components"""

    def __init__(self, state: VisualizationState):
        self.state = state

    def load_file(self, file_path: str) -> SimulationData:
        """Load data from file into SimulationData structure"""
        fields, setup, mesh, immersed_bodies = read_file(file_path)
        return SimulationData(fields, setup, mesh, immersed_bodies)

    def get_variable(self, field: str) -> np.ndarray:
        """Extract variable using existing logic"""
        if not self.state.data:
            return np.array([])

        # Handle field aliases
        if field in FIELD_ALIASES:
            field_name = FIELD_ALIASES[field]
        else:
            field_name = field

        var = self.state.data.fields[field_name]

        # Apply units if configured
        if self.state.config.get("style", {}).get("units", False):
            self._apply_units(var, field)

        return var

    def _apply_units(self, var: np.ndarray, field: str) -> None:
        """Apply physical units to variable"""
        from ...tools import utility as util

        if field in ["p", "energy", "energy_rst"]:
            var *= util.edens_scale.value
        elif field in ["rho", "D"]:
            var *= util.rho_scale.value

    def get_slice_data(
        self,
        var: np.ndarray,
        mesh: dict[str, np.ndarray],
        setup: dict[str, Any],
        label: Optional[str] = None,
    ) -> tuple[Sequence[np.ndarray], Sequence[str]]:
        """Get data for a 1D slice of a higher-dimensional field"""
        # Get slice configuration
        slice_along = self.state.config.get("multidim", {}).get("slice_along")
        if not slice_along:
            return [var], [label or ""]

        # Get coordinates for slicing
        coords = self.state.config["multidim"]["coords"]

        slice_arr = []
        sliced_labels = []

        # Process each coordinate combination
        for xkcoord in map(float, coords.get("xk", ["0.0"])):
            for xjcoord in map(float, coords.get("xj", ["0.0"])):
                # Create label if provided
                if label:
                    sliced_label = label + f" $x_j={xjcoord:.0f}$"
                    if "xk" in coords:
                        sliced_label += f" $x_k={xkcoord:.0f}$"
                    sliced_labels.append(sliced_label)
                else:
                    sliced_labels.append("")

                # Handle spherical coordinates
                if setup.get("coord_system") == "spherical":
                    xjcoord = np.deg2rad(xjcoord)
                    xkcoord = np.deg2rad(xkcoord)

                # Get variable names based on slice direction
                if slice_along == "x1":
                    xj, xk = "x2v", "x3v"
                elif slice_along == "x2":
                    xj, xk = "x3v", "x1v"
                else:
                    xj, xk = "x1v", "x2v"

                # Find nearest indices
                from ...functional.helpers import find_nearest

                jidx = find_nearest(mesh.get(xj, np.linspace(0, 1)), xjcoord)[0]
                kidx = find_nearest(mesh.get(xk, np.linspace(0, 1)), xkcoord)[0]

                # Adjust indices
                if jidx > 0:
                    jidx -= 1
                if kidx > 0:
                    kidx -= 1

                # Create slice based on dimensionality
                if mesh.get("effective_dimensions", var.ndim) == 1:
                    slice_arr.append(var)
                elif mesh.get("effective_dimensions", var.ndim) == 2:
                    slice_arr.append(var[jidx] if slice_along == "x1" else var[:, jidx])
                else:
                    if slice_along == "x1":
                        slice_index = np.s_[kidx, jidx, :]
                    elif slice_along == "x2":
                        slice_index = np.s_[jidx, :, kidx]
                    else:
                        slice_index = np.s_[:, kidx, jidx]
                    slice_arr.append(var[slice_index])

        return slice_arr, sliced_labels

    def get_field_label(self, field: str) -> str:
        """Get formatted field label"""
        if field in FIELD_ALIASES:
            field = FIELD_ALIASES[field]
        return str(get_field_str(field))

    def transform_coordinates(
        self, mesh: dict[str, np.ndarray], setup: dict[str, Any]
    ) -> tuple:
        """Transform coordinates based on coordinate system"""
        from ...functional.helpers import calc_any_mean

        # Handle slices if configured
        slice_along = self.state.config.get("multidim", {}).get("slice_along")
        if slice_along:
            x = calc_any_mean(mesh[f"{slice_along}v"], setup[f"{slice_along}_spacing"])
            # Get slice indices (simplified for now)
            slice_indices = np.s_[0]
            return x, slice_indices

        # Handle polar coordinates
        if not setup.get("is_cartesian", True):
            xx, yy = np.meshgrid(mesh["x1v"], mesh["x2v"])[::-1]

            # Handle bipolar configuration
            if self.state.config.get("multidim", {}).get("bipolar", False):
                xx *= -1

            return xx, yy

        # Handle Cartesian coordinates
        ndim = self.state.config.get("plot", {}).get("ndim", 1)
        if ndim == 1:
            return (
                calc_any_mean(mesh["x1v"], setup["x1_spacing"]),
                range(len(mesh["x1v"])),
            )
        elif ndim == 2:
            return mesh["x1v"], mesh["x2v"]
        else:
            projection = self.state.config.get("multidim", {}).get(
                "projection", [1, 2, 3]
            )
            xj = f"x{projection[0]}"
            xk = f"x{projection[1]}"
            return mesh[f"{xj}v"], mesh[f"{xk}v"]
