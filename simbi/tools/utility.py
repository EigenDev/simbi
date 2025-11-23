# Utility functions for visualization scripts
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import astropy.constants as const
import astropy.units as units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from simbi.core.types.input import CoordSystem
from simbi.reader.io import read_raw_data

from ..functional.helpers import find_nearest

# FONT SIZES
SMALL_SIZE = 6
DEFAULT_SIZE = 10
BIGGER_SIZE = 12

logically_curvlinear = ["spherical", "planar_cylindrical"]
logically_cartesian = ["cartesian", "axis_cylindrical", "cylindrical"]
# ================================
#   constants of nature
# ================================
R_0 = const.R_sun.cgs
c = const.c.cgs
m = const.M_sun.cgs

rho_scale = m / R_0**3
e_scale = m * c**2
edens_scale = e_scale / R_0**3
time_scale = R_0 / c
mass_scale = m

e_scale_bmk = 1e53 * units.erg
rho_scale_bmk = 1.0 * const.m_p.cgs / units.cm**3
ell_scale = (e_scale_bmk / rho_scale_bmk / const.c.cgs**2) ** (1 / 3)
t_scale = const.c.cgs * ell_scale


class FieldType(Enum):
    DENSITY = "density"
    ENERGY = "energy"
    VELOCITY = "velocity"
    TEMPERATURE = "temperature"
    MAGNETIC = "magnetic"
    OTHER = "other"


FIELD_MAP: dict[str, str] = {
    "rho": r"\rho",
    "D": "D",
    "gamma_beta": r"$\Gamma \beta$",
    "u": r"$\Gamma \beta$",
    "gamma_beta_1": r"$\Gamma \beta_1$",
    "u1": r"$\Gamma \beta_1$",
    "gamma_beta_2": r"$\Gamma \beta_2$",
    "u2": r"$\Gamma \beta_2$",
    "gamma_beta_3": r"$\Gamma \beta_3$",
    "u3": r"$\Gamma \beta_3$",
    "energy": r"\tau",
    "p": r"p",
    "energy_rst": r"$E$",
    "chi": r"$\chi$",
    "chi_dens": r"$\rho \cdot \chi$",
    "T_eV": "T [eV]",
    "temperature": "T",
    "mach": r"$\mathcal{M}$",
    "v1": r"$v_1 / v_0$",
    "v": r"$v / v_0$",
    "v2": r"$v_2 / v_0$",
    "v3": r"$v_3 / v_0$",
    "tau-s": r"$\tau_s$",
    "pmag": r"$p_{\rm mag}$",
    "ptot": r"$p_{\rm tot}$",
    "sigma": r"$\sigma$",
    "Sigma": r"\Sigma",
    "enthalpy_density": r"$w$",
    "b1": r"$B_1$",
    "b2": r"$B_2$",
    "b3": r"$B_3$",
    "b1_mean": r"$B_1$",
    "b2_mean": r"$B_2$",
    "b3_mean": r"$B_3$",
    "accretion_rate": r"$\dot{M} / \dot{M}_0$",
    "accreted_mass": r"$M_{\rm acc}$",
    "mdot": r"$\dot{M} / \dot{M_0}$",
    "maccr": r"$M_{\rm acc}$",
    "j": r"$L_z / L_{z,0}$",
    "vr": r"$v_r / v_0$",
    "vphi": r"$v_\phi / v_0$",
    "vtheta": r"$v_\theta / v_0$",
    "j_spec": r"$j / j_{0}$",
    "div_v": r"$\nabla \cdot \mathbf{v}$",
    "vorticity": r"$(\nabla \times \mathbf{v})_z$",
    "term_advection": r"$\rho \mathbf{v} \cdot \nabla \mathbf{v}$",
    "term_gravity": r"$-\rho \nabla \Phi$",
    "term_pressure": r"$-\nabla p$",
    "term_residual": r"$\mathbf{R}$",
}

UNITS: dict[str, str] = {
    "energy": r"\rm erg \ cm^{-3}",
    "density": r"\rm g \ cm^{-3}",
}


@dataclass
class FieldMapper:
    """Maps field names to LaTeX strings"""

    field_map: dict[str, str] = field(default_factory=lambda: FIELD_MAP)
    units: dict[str, str] = field(default_factory=lambda: UNITS)

    def get_field_str(
        self,
        field: str,
        units: bool = False,
        normalized: bool = True,
    ) -> str:
        """Get LaTeX string for field(s)"""
        return self._format_field(field, units, normalized)

    def _format_field(self, field: str, units: bool, normalized: bool) -> str:
        """Format single field with optional units"""
        if field not in self.field_map:
            return self._format_unknown_field(field)

        var = self.field_map[field]
        field_type = self._get_field_type(field)

        return self._format_by_type(var, field_type, units, normalized)

    def _format_unknown_field(self, field: str) -> str:
        """Format unknown field"""
        return f"${field}$"

    def _get_field_type(self, field: str) -> FieldType:
        """Determine field type"""
        if field in ["rho", "D", "Sigma"]:
            return FieldType.DENSITY
        if field in ["energy", "p"]:
            return FieldType.ENERGY
        if field == "temperature":
            return FieldType.TEMPERATURE
        if field.startswith("b"):
            return FieldType.MAGNETIC
        return FieldType.OTHER

    def _format_by_type(
        self, var: str, field_type: FieldType, units: bool, normalized: bool
    ) -> str:
        """Format field based on its type"""
        if field_type in [FieldType.DENSITY, FieldType.ENERGY]:
            if units:
                return f"{var} [{self.units[field_type.value]}]"
            elif normalized:
                return f"${var} / {var}_0$"
            else:
                return f"${var}$"
        return var


# Usage remains the same
def get_field_str(
    fields: str,
    units: bool = False,
    normalized: bool = True,
) -> str:
    """Get LaTeX string for field(s)"""
    mapper = FieldMapper()
    return mapper.get_field_str(fields, units, normalized)


def calc_enthalpy(fields: dict[str, NDArray[np.floating[Any]]]) -> Any:
    return 1.0 + fields["p"] * fields["adiabatic_index"] / (
        fields["rho"] * (fields["adiabatic_index"] - 1.0)
    )


def calc_lorentz_factor(fields: dict[str, NDArray[np.floating[Any]]]) -> Any:
    return (1.0 + fields["gamma_beta"] ** 2) ** 0.5


def calc_beta(fields: dict[str, NDArray[np.floating[Any]]]) -> Any:
    W = calc_lorentz_factor(fields)
    return (1.0 - 1.0 / W**2) ** 0.5


def unpad(
    arr: NDArray[np.floating[Any]], pad_width: tuple[tuple[Any, ...], ...]
) -> Any:
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return arr[tuple(slices)]


def flatten_fully(
    x: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]] | Any:
    if any(dim == 1 for dim in x.shape):
        x = np.vstack(x)  # type: ignore
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flat
        return flatten_fully(x)
    else:
        return np.asanyarray(x)


def get_dimensionality(files: Union[list[str], dict[int, list[str]]]) -> int:
    import h5py

    dims = []

    def all_equal(x: list[int]) -> bool:
        return x.count(x[0]) == len(x)

    def extract_shape(raw: Any, attrs: dict[str, Any]) -> tuple[int, ...]:
        """Extract shape from mesh_config, handling both v1 and v2 formats"""
        mesh_data = raw.groups.get("mesh_config", {})

        # v2 format uses global_cells
        if "global_cells" in mesh_data:
            return tuple(int(x) for x in mesh_data["global_cells"])

        # v1 format uses shape directly
        if "shape" in mesh_data:
            return tuple(int(x) for x in mesh_data["shape"])

        # fallback to resolution from attributes
        res = attrs.get("resolution", (1,))
        if isinstance(res, str):
            return tuple(int(x) for x in res.split(","))
        return tuple(int(x) for x in res)

    ndim: int = 0
    if isinstance(files, dict):
        import itertools

        files = list(itertools.chain(*files.values()))

    files = list(filter(bool, files))
    for file in files:
        with h5py.File(file, "r") as hf:
            dat = read_raw_data(hf)
            if dat.is_ok:
                raw = dat.unwrap()
                print("mesh config", raw.groups.get("mesh_config", {}))
                shape = extract_shape(raw, raw.attributes)
                dims.append(sum(int(r) > 1 for r in shape))

    if dims and all_equal(dims):
        ndim = dims[0]
    else:
        raise ValueError("Inconsistent dimensionality across files.")

    return ndim


def get_colors(
    interval: NDArray[np.floating[Any]],
    cmap: matplotlib.colors.ListedColormap,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> NDArray[Any]:
    """
    Return array of rgba colors for a given matplotlib colormap

    Parameters
    -------------------------
    interval: interval range for colormarp min and max
    cmap: the matplotlib colormap instance
    vmin: minimum for colormap
    vmax: maximum for colormap

    Returns
    -------------------------
    arr: the colormap array generate by the user conditions
    """
    matplotlib.colors.Normalize(vmin, vmax)
    return np.asarray(cmap(interval), dtype=np.float64)


def fill_below_intersec(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    constraint: float,
    color: float,
    axis: str,
) -> None:
    if axis == "x":
        ind: int = find_nearest(x, constraint)[0]
    else:
        ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:], y[ind:], color=color, alpha=0.1, interpolate=True)


def get_file_list(
    inputs: str, sort: bool = False
) -> Union[tuple[list[str], int], tuple[dict[int, list[str]], bool]]:
    from pathlib import Path

    files: Union[list[str], dict[int, list[str]]]
    dirs = list(filter(lambda x: Path(x).is_dir(), inputs))
    multidir = len(dirs) > 1

    if multidir:
        files = {
            key: sorted(
                [str(f) for f in Path(fdir).glob("*.h5") if f.is_file()]
            )
            for key, fdir in enumerate(inputs)
        }
    else:
        files = []
        if dirs:
            files = sorted(
                [
                    str(f)
                    for d in dirs
                    for f in Path(d).glob("*.h5")
                    if f.is_file()
                ]
            )
        files += [file for file in filter(lambda x: x not in dirs, inputs)]

    if not isinstance(files, dict):
        # sort by length of strings now
        if sort:
            files.sort(key=len, reverse=False)
        return files, len(files)
    else:
        any(files[key].sort(key=len, reverse=False) for key in files.keys())
        return files, multidir


def map_coordinate_label(
    coord_name: str,
    coord_system: CoordSystem,
) -> str:
    """Maps a logical coordinate name to a human-readable label."""
    label_map_cartesian = {
        "x1": "$x$",
        "x2": "$y$",
        "x3": "$z$",
    }
    label_map_polar = {
        "x1": "$r$",
        "x2": r"$\theta$",
        "x3": r"$\phi$",
    }
    label_map_cylindrical = {
        "x1": "$r$",
        "x2": "$z$",
        "x3": r"$\phi$",
    }

    if coord_system == CoordSystem.SPHERICAL:
        return label_map_polar.get(coord_name, coord_name)
    elif coord_system == CoordSystem.CARTESIAN:
        return label_map_cartesian.get(coord_name, coord_name)
    else:
        return label_map_cylindrical.get(coord_name, coord_name)
