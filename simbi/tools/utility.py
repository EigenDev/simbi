# Utility functions for visualization scripts
import h5py
import astropy.constants as const
import matplotlib
import astropy.units as units
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Any, Optional
from numpy.typing import NDArray
from ..functional.helpers import find_nearest
from dataclasses import dataclass, field
from enum import Enum

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
    "accretion_rate": r"$\dot{M}$",
    "accreted_mass": r"$M_{\rm acc}$",
    "mdot": r"$\dot{M}$",
    "maccr": r"$M_{\rm acc}$",
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
        fields: Union[str, list[str], dict[str, Any]],
        units: bool = False,
        normalized: bool = True,
    ) -> Union[str, list[str]]:
        """Get LaTeX string for field(s)"""
        field_list = self._normalize_fields(fields)
        field_strings = [self._format_field(f, units, normalized) for f in field_list]
        return field_strings[0] if len(field_strings) == 1 else field_strings

    def _normalize_fields(
        self, fields: Union[str, list[str], dict[str, Any]]
    ) -> list[str]:
        """Convert input to list of field names"""
        if isinstance(fields, str):
            return [fields]
        if isinstance(fields, dict):
            return list(fields.keys())
        return fields

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
    fields: Union[str, list[str], dict[str, Any]],
    units: bool = False,
    normalized: bool = True,
) -> Union[str, list[str]]:
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


def flatten_fully(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]] | Any:
    if any(dim == 1 for dim in x.shape):
        x = np.vstack(x)  # type: ignore
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flat
        return flatten_fully(x)
    else:
        return np.asanyarray(x)


def get_dimensionality(files: Union[list[str], dict[int, list[str]]]) -> int:
    dims = []

    def all_equal(x: list[int]) -> bool:
        return x.count(x[0]) == len(x)

    ndim: int = 0
    if isinstance(files, dict):
        import itertools

        files = list(itertools.chain(*files.values()))

    files = list(filter(bool, files))
    for file in files:
        with h5py.File(file, "r") as hf:
            ds = dict(hf["sim_info"].attrs)
            effective_dim = sum(q > 1 for q in [ds["nx"], ds["ny"], ds["nz"]])
            ndim = ds["dimensions"]
            dims += [ndim]
            if not all_equal(dims):
                raise ValueError(
                    "All simulation files require identical dimensionality"
                )

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
    from typing import cast

    files: Union[list[str], dict[int, list[str]]]
    dirs = list(filter(lambda x: Path(x).is_dir(), inputs))
    multidir = len(dirs) > 1

    if multidir:
        files = {
            key: sorted([str(f) for f in Path(fdir).glob("*.h5") if f.is_file()])
            for key, fdir in enumerate(inputs)
        }
    else:
        files = []
        if dirs:
            files = sorted(
                [str(f) for d in dirs for f in Path(d).glob("*.h5") if f.is_file()]
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
