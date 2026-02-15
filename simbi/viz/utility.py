# =============================================================================
# utility.py
#
# field name formatting and dimensionality detection.
# =============================================================================
from typing import Union

# field name -> latex string
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
    "schlieren": r"$|\nabla \ln \rho|$",
    "entropy-gradient": r"$|\nabla (p / \rho^\gamma)|$",
    "entropy-measure": r"$p / \rho^\gamma$",
    "v_turb": r"$|\mathbf{v} - \langle \mathbf{v} \rangle|$",
    "torque x": r"$\tau_x$",
    "torque y": r"$\tau_y$",
    "torque z": r"$\tau_z$",
    "radial force": r"$F_{\rm rad}$",
    "tangential force": r"$F_{\rm \perp}$",
    "decay rate": r"\dot{a}",
    "power": r"$\dot{E}$",
    "drag force": r"$F_{\rm drag}$",
}

UNITS: dict[str, str] = {
    "energy": r"\rm erg \ cm^{-3}",
    "density": r"\rm g \ cm^{-3}",
}

# fields that get normalized display (e.g., ρ / ρ₀)
_DENSITY_FIELDS = {"rho", "D", "Sigma"}
_ENERGY_FIELDS = {"energy", "p"}


def get_field_str(
    field: str,
    units: bool = False,
    normalized: bool = True,
) -> str:
    """get latex string for a field name."""
    if "$" in field:
        return field

    if field not in FIELD_MAP:
        return f"${field}$"

    var = FIELD_MAP[field]

    if field in _DENSITY_FIELDS or field in _ENERGY_FIELDS:
        unit_key = "density" if field in _DENSITY_FIELDS else "energy"
        if units:
            return f"{var} [{UNITS[unit_key]}]"
        elif normalized:
            return f"${var} / {var}_0$"
        else:
            return f"${var}$"

    return var


def get_dimensionality(files: Union[list[str], dict[int, list[str]]]) -> int:
    """get effective dimensionality from checkpoint files."""
    from simbi.reader import read_checkpoint

    if isinstance(files, dict):
        import itertools

        files = list(itertools.chain(*files.values()))

    files = list(filter(bool, files))
    dims = []
    for file in files:
        result = read_checkpoint(file)
        if result.is_ok():
            checkpoint = result.value
            if checkpoint.levels:
                mesh = checkpoint.levels[0].mesh
                dims.append(sum(int(r) > 1 for r in mesh.global_cells))
            else:
                # diagnostic-only file, no grid data
                dims.append(1)
        else:
            raise ValueError(f"failed to read {file}: {result.error}")

    if dims and dims.count(dims[0]) == len(dims):
        return dims[0]

    raise ValueError("inconsistent dimensionality across files.")
