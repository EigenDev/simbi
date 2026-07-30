# utility functions for visualization scripts
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import astropy.constants as const
import astropy.units as units
import numpy as np
from numpy.typing import NDArray

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
    "W": r"$\Gamma$",
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
    "v2": r"$v_2 / v_0$",
    "v3": r"$v_3 / v_0$",
    "m1": r"$M_1 / M_0$",
    "m2": r"$M_2 / M_0$",
    "m3": r"$M_3 / M_0$",
    "v": r"$v / v_0$",
    "tau-s": r"$\tau_s$",
    "pmag": r"$p_{\rm mag}$",
    "ptot": r"$p_{\rm tot}$",
    "sigma": r"$\sigma$",
    "Sigma": r"\Sigma",
    "enthalpy": r"$h$",
    "enthalpy_density": r"$w$",
    "b1": r"$B_1$",
    "b2": r"$B_2$",
    "b3": r"$B_3$",
    "b1_mean": r"$B_1$",
    "b2_mean": r"$B_2$",
    "b3_mean": r"$B_3$",
    "bmu1": r"$b^1$",
    "bmu2": r"$b^2$",
    "bmu3": r"$b^3$",
    "emag": r"$E_{\rm mag}$",
    "accretion_rate": r"$\dot{M} / \dot{M}_0$",
    "accreted_mass": r"$M_{\rm acc}$",
    "mdot": r"$\dot{M} / \dot{M_0}$",
    "maccr": r"$M_{\rm acc}$",
    "j": r"$L_z / L_{z,0}$",
    "vr": r"$v_r / v_0$",
    "vphi": r"$v_\phi / v_0$",
    "vtheta": r"$v_\theta / v_0$",
    "j_spec": r"$j / j_{0}$",
    "mass_flux": r"$4 \pi r^2 \rho v_r$",
    "div_v": r"$\nabla \cdot \mathbf{v}$",
    "vorticity": r"$(\nabla \times \mathbf{v})_z$",
    "vorticity_magnitude": r"$|\nabla \times \mathbf{v}|$",
    "q_criterion": r"$Q$",
    "okubo_weiss": r"$s_n^2 + s_s^2 - \omega^2$",
    "term_advection": r"$\rho \mathbf{v} \cdot \nabla \mathbf{v}$",
    "term_gravity": r"$-\rho \nabla \Phi$",
    "term_pressure": r"$-\nabla p$",
    "term_residual": r"$\mathbf{R}$",
    "schlieren": r"$|\nabla \ln \rho|$",
    "torque x": r"$\tau_x$",
    "torque y": r"$\tau_y$",
    "torque z": r"$\tau_z$",
    "radial force": r"$F_{\rm rad}$",
    "tangential force": r"$F_{\rm \perp}$",
    "decay rate": r"\dot{a}",
    "power": r"$\dot{E}$",
    "drag force": r"$F_{\rm drag}$",
    "tracer_concentration": (
        r"$\Sigma_{\rm tr} / \langle \Sigma_{\rm tr} \rangle$"
    ),
    "tracer_cohort_concentration": (
        r"$\Sigma_{{\rm tr},c} / \langle \Sigma_{{\rm tr},c} \rangle$"
    ),
    "tracer_cohort_ratio": (
        r"$\log_{10}\left[(\Sigma_{{\rm tr},c}/"
        r"\langle\Sigma_{{\rm tr},c}\rangle)/"
        r"(\Sigma_{\rm gas}/\langle\Sigma_{\rm gas}\rangle)\right]$"
    ),
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


def get_field_str(
    field: str,
    units: bool = False,
    normalized: bool = True,
) -> str:
    """Get LaTeX string for field(s)"""
    mapper = FieldMapper()
    if "$" in field:
        return field  # already formatted
    return mapper.get_field_str(field, units, normalized)


def get_tracer_field_str(field: str, cohort: Optional[int] = None) -> str:
    """get a tracer label and identify the selected initial-material cohort."""
    label = get_field_str(field)
    if cohort is None:
        return label
    return label.replace(",c}", rf",c={cohort}}}")


def calc_lorentz_factor(fields: dict[str, NDArray[np.floating[Any]]]) -> Any:
    return (1.0 + fields["gamma_beta"] ** 2) ** 0.5


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
    """get effective dimensionality from checkpoint files using io."""
    from simbi.reader import read_checkpoint

    dims = []

    def all_equal(x: list[int]) -> bool:
        return x.count(x[0]) == len(x)

    if isinstance(files, dict):
        import itertools

        files = list(itertools.chain(*files.values()))

    files = list(filter(bool, files))
    for file in files:
        result = read_checkpoint(file)
        if result.is_ok():
            checkpoint = result.value
            # get shape from mesh geometry - global_cells is (nz, ny, nx)
            mesh = checkpoint.levels[0].mesh
            shape = mesh.global_cells
            dims.append(sum(int(r) > 1 for r in shape))
        else:
            raise ValueError(f"failed to read {file}: {result.error}")

    if dims and all_equal(dims):
        return dims[0]
    else:
        raise ValueError("inconsistent dimensionality across files.")
