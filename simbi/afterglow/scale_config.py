# =============================================================================
# scale_config.py
#
# scale configuration system for converting dimensionless simulation units
# to physical CGS units for afterglow radiation calculations.
#
# provides standard scale models (grb, kilonova, agn) and supports
# custom yaml-based configurations.
#
# usage:
#   scales = load_scale_config("grb")  # standard
#   scales = load_scale_config("my_scales.yaml")  # custom
# =============================================================================

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml
from astropy import constants as const
from astropy import units as u


@dataclass(frozen=True)
class scale_config_t:
    """
    physical scales for converting dimensionless units to cgs.

    all scales in cgs:
        length_scale: [cm]
        time_scale: [s]
        rho_scale: [g/cm^3]
        pre_scale: [erg/cm^3]
        v_scale: dimensionless (fraction of c)
    """

    name: str
    length_scale: float  # stored as float (cgs value)
    time_scale: float
    rho_scale: float
    pre_scale: float
    v_scale: float = 1.0

    # physical context (for documentation)
    description: str = ""
    reference: str = ""

    def to_dict(self) -> dict:
        """convert to dictionary for yaml export"""
        return {
            "name": self.name,
            "length_scale": self.length_scale,
            "time_scale": self.time_scale,
            "rho_scale": self.rho_scale,
            "pre_scale": self.pre_scale,
            "v_scale": self.v_scale,
            "description": self.description,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "scale_config_t":
        """create from dictionary (yaml load)"""
        length_scale = float(data["length_scale"])
        time_scale = float(data["time_scale"])
        rho_scale = float(data["rho_scale"])
        pre_scale = float(data["pre_scale"])
        v_scale = float(data.get("v_scale", 1.0))

        # validate scale consistency
        _validate_scales(
            length_scale, time_scale, rho_scale, pre_scale, v_scale
        )

        return cls(
            name=data["name"],
            length_scale=length_scale,
            time_scale=time_scale,
            rho_scale=rho_scale,
            pre_scale=pre_scale,
            v_scale=v_scale,
            description=data.get("description", ""),
            reference=data.get("reference", ""),
        )

    def print_info(self) -> None:
        """print human-readable scale information"""
        day_cgs = u.day.to(u.s)
        print(f"Scale Configuration: {self.name}")
        print(f"  Description: {self.description}")
        if self.reference:
            print(f"  Reference: {self.reference}")
        print("\nScales (CGS):")
        print(f"  length_scale: {self.length_scale:.4e} cm")
        print(
            f"  time_scale:   {self.time_scale:.4e} s ({self.time_scale / day_cgs:.4e} day)"
        )
        print(f"  rho_scale:    {self.rho_scale:.4e} g/cm^3")
        print(f"  pre_scale:    {self.pre_scale:.4e} erg/cm^3")
        print(f"  v_scale:      {self.v_scale:.4e} (fraction of c)")


# =============================================================================
# internal validation
# =============================================================================


def _validate_scales(
    length_scale: float,
    time_scale: float,
    rho_scale: float,
    pre_scale: float,
    v_scale: float = 1.0,
) -> None:
    """
    validate scale configuration for physical consistency.

    checks:
        - all scales are positive
        - velocity scale is dimensionless and <= 1
        - time_scale ~ length_scale / c
        - pre_scale ~ rho_scale * c^2
    """
    c_cgs = const.c.cgs.value

    # positivity
    if length_scale <= 0:
        raise ValueError(f"length_scale must be positive, got {length_scale}")
    if time_scale <= 0:
        raise ValueError(f"time_scale must be positive, got {time_scale}")
    if rho_scale <= 0:
        raise ValueError(f"rho_scale must be positive, got {rho_scale}")
    if pre_scale <= 0:
        raise ValueError(f"pre_scale must be positive, got {pre_scale}")

    # velocity scale
    if not (0 < v_scale <= 1):
        raise ValueError(f"v_scale must be in (0, 1], got {v_scale}")

    # time ~ length / c
    expected_time = length_scale / (v_scale * c_cgs)
    time_ratio = time_scale / expected_time
    if not (0.5 < time_ratio < 2.0):
        raise ValueError(
            f"time_scale inconsistent with length_scale:\n"
            f"  expected: {expected_time:.4e} s\n"
            f"  got:      {time_scale:.4e} s\n"
            f"  ratio:    {time_ratio:.2f}"
        )

    # pressure ~ rho * c^2
    expected_pre = rho_scale * (v_scale * c_cgs) ** 2
    pre_ratio = pre_scale / expected_pre
    if not (0.1 < pre_ratio < 10.0):
        raise ValueError(
            f"pre_scale inconsistent with rho_scale:\n"
            f"  expected: {expected_pre:.4e} erg/cm^3\n"
            f"  got:      {pre_scale:.4e} erg/cm^3\n"
            f"  ratio:    {pre_ratio:.2f}"
        )


# =============================================================================
# standard scale configurations
# =============================================================================

# compute solar scales
_SOLAR_LENGTH = const.R_sun.cgs.value
_SOLAR_RHO = const.M_sun.cgs.value / (4.0 / 3.0 * math.pi * _SOLAR_LENGTH**3)
_SOLAR_E = const.M_sun.cgs.value * const.c.cgs.value**2
_SOLAR_PRE = _SOLAR_E / (4.0 / 3.0 * math.pi * _SOLAR_LENGTH**3)
_SOLAR_TIME = _SOLAR_LENGTH / const.c.cgs.value

# compute blandford-mckee scales
_BMK_E = 1e53  # 10^53 erg (canonical grb energy)
_BMK_RHO = const.m_p.cgs.value  # 1 proton/cm^3 ism
_BMK_LENGTH = (_BMK_E / (_BMK_RHO * const.c.cgs.value**2)) ** (1.0 / 3.0)
_BMK_TIME = _BMK_LENGTH / const.c.cgs.value
_BMK_PRE = _BMK_E / _BMK_LENGTH**3

STANDARD_SCALES = {
    "solar": scale_config_t(
        name="solar",
        length_scale=_SOLAR_LENGTH,
        time_scale=_SOLAR_TIME,
        rho_scale=_SOLAR_RHO,
        pre_scale=_SOLAR_PRE,
        v_scale=1.0,
        description="solar scales: R_sun, M_sun, c",
        reference="legacy compatibility scale",
    ),
    "blandford_mckee": scale_config_t(
        name="blandford_mckee",
        length_scale=_BMK_LENGTH,
        time_scale=_BMK_TIME,
        rho_scale=_BMK_RHO,
        pre_scale=_BMK_PRE,
        v_scale=1.0,
        description="blandford-mckee blast wave: E=10^53 erg, n=1 cm^-3",
        reference="canonical grb afterglow scales",
    ),
    "grb_standard": scale_config_t(
        name="grb_standard",
        length_scale=1e16,  # 10^16 cm ~ 10^13 km
        time_scale=1e16 / const.c.cgs.value,  # light crossing time
        rho_scale=1e-24,  # ism density ~ 1 proton/cm^3
        pre_scale=1e-24 * const.c.cgs.value**2,  # rho * c^2
        v_scale=1.0,
        description="standard grb jet in ism",
        reference="typical grb afterglow parameters",
    ),
    "grb_high_density": scale_config_t(
        name="grb_high_density",
        length_scale=1e15,  # 10^15 cm
        time_scale=1e15 / const.c.cgs.value,
        rho_scale=1e-20,  # dense circumburst medium
        pre_scale=1e-20 * const.c.cgs.value**2,
        v_scale=1.0,
        description="grb in dense circumburst medium",
        reference="grb in wind or stellar envelope",
    ),
    "kilonova": scale_config_t(
        name="kilonova",
        length_scale=1e13,  # 10^13 cm ~ 0.01 au
        time_scale=1e13 / const.c.cgs.value,  # light crossing time
        rho_scale=1e-10,  # ejecta density
        pre_scale=1e-10 * const.c.cgs.value**2,
        v_scale=1.0,
        description="neutron star merger ejecta",
        reference="kilonova afterglow",
    ),
    "agn_jet": scale_config_t(
        name="agn_jet",
        length_scale=1e18,  # 10^18 cm ~ 10 pc
        time_scale=1e18 / const.c.cgs.value,
        rho_scale=1e-27,  # igm density
        pre_scale=1e-27 * const.c.cgs.value**2,
        v_scale=1.0,
        description="agn jet in intergalactic medium",
        reference="large-scale relativistic jets",
    ),
    "pulsar_wind": scale_config_t(
        name="pulsar_wind",
        length_scale=1e17,  # pulsar wind nebula scale
        time_scale=1e17 / const.c.cgs.value,
        rho_scale=1e-24,
        pre_scale=1e-24 * const.c.cgs.value**2,
        v_scale=1.0,
        description="pulsar wind nebula",
        reference="crab-like systems",
    ),
    "tde": scale_config_t(
        name="tde",
        length_scale=1e14,  # tidal disruption scale
        time_scale=1e14 / const.c.cgs.value,
        rho_scale=1e-15,  # stellar debris density
        pre_scale=1e-15 * const.c.cgs.value**2,
        v_scale=1.0,
        description="tidal disruption event outflow",
        reference="relativistic tde jets",
    ),
}


# =============================================================================
# configuration loading
# =============================================================================


def load_scale_config(config: Union[str, Path, dict]) -> scale_config_t:
    """
    load scale configuration from standard name, yaml file, or dict.

    args:
        config: one of:
            - standard name: "grb_standard", "kilonova", etc.
            - path to yaml file: "my_scales.yaml"
            - dict with scale parameters

    returns:
        scale_config_t

    example:
        # standard scales
        scales = load_scale_config("grb_standard")

        # custom yaml
        scales = load_scale_config("my_grb_scales.yaml")

        # inline dict
        scales = load_scale_config({
            "name": "custom",
            "length_scale": 1e16,
            "time_scale": 333.0,
            "rho_scale": 1e-24,
            "pre_scale": 9e-4,
        })
    """
    # case 1: dict directly provided
    if isinstance(config, dict):
        return scale_config_t.from_dict(config)

    # case 2: standard scale name
    if isinstance(config, str) and config in STANDARD_SCALES:
        return STANDARD_SCALES[config]

    # case 3: yaml file path
    config_path = Path(config)
    if config_path.exists() and config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return scale_config_t.from_dict(data)

    # case 4: not found
    raise ValueError(
        f"scale config '{config}' not found.\n"
        f"available standard scales: {list(STANDARD_SCALES.keys())}\n"
        f"or provide a yaml file path."
    )


def list_standard_scales() -> None:
    """print all available standard scale configurations"""
    print("Available Standard Scale Configurations:\n")
    for name, scales in STANDARD_SCALES.items():
        print(f"  {name:20s} - {scales.description}")


def save_scale_config(
    scales: scale_config_t, filepath: Union[str, Path]
) -> None:
    """
    save scale configuration to yaml file.

    args:
        scales: scale configuration to save
        filepath: output yaml file path

    example:
        scales = load_scale_config("grb_standard")
        save_scale_config(scales, "my_grb_scales.yaml")
    """
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        yaml.dump(
            scales.to_dict(), f, default_flow_style=False, sort_keys=False
        )
    print(f"saved scale configuration to {filepath}")


def create_custom_scale_template(filepath: Union[str, Path]) -> None:
    """
    create a template yaml file for custom scales.

    args:
        filepath: output yaml file path

    example:
        create_custom_scale_template("my_scales.yaml")
        # then edit my_scales.yaml with your custom values
    """
    template = {
        "name": "my_custom_scales",
        "description": "custom scale configuration for my simulation",
        "reference": "citation or notes",
        "length_scale": 1.0e16,  # cm
        "time_scale": 333.56,  # s (example: 1e16 cm / c)
        "rho_scale": 1.0e-24,  # g/cm^3
        "pre_scale": 9.0e-4,  # erg/cm^3 (example: rho_scale * c^2)
        "v_scale": 1.0,  # dimensionless (fraction of c)
    }

    filepath = Path(filepath)
    with open(filepath, "w") as f:
        f.write("# Custom scale configuration for simbi afterglow\n")
        f.write("# Edit the values below to match your simulation units\n\n")
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
        f.write("\n# Physical scales in CGS:\n")
        f.write(
            "#   length_scale: [cm]  - converts dimensionless length to cm\n"
        )
        f.write("#   time_scale:   [s]   - converts dimensionless time to s\n")
        f.write(
            "#   rho_scale:    [g/cm^3] - converts dimensionless density to g/cm^3\n"
        )
        f.write(
            "#   pre_scale:    [erg/cm^3] - converts dimensionless pressure to erg/cm^3\n"
        )
        f.write(
            "#   v_scale:      dimensionless - usually 1.0 for relativistic sims\n"
        )

    print(f"created scale template: {filepath}")
    print("edit this file with your custom scale values")


# =============================================================================
# helper functions for common scale derivations
# =============================================================================


def derive_scales_from_energy_density(
    E0: float,  # energy scale [erg]
    n0: float,  # ambient number density [cm^-3]
    gamma0: float = 10.0,  # initial lorentz factor
) -> scale_config_t:
    """
    derive scales from blast wave energy and ambient density.

    uses sedov-taylor / blandford-mckee scaling:
        R = (E / n m_p c^2)^(1/3)
        T = R / c

    args:
        E0: total energy [erg]
        n0: ambient number density [cm^-3]
        gamma0: initial lorentz factor (optional, for name)

    returns:
        scale_config_t
    """
    m_p = const.m_p.cgs.value
    c_cgs = const.c.cgs.value
    rho0 = n0 * m_p
    R_scale = (E0 / (n0 * m_p * c_cgs**2)) ** (1.0 / 3.0)
    T_scale = R_scale / c_cgs
    pre_scale_val = rho0 * const.c.cgs.value**2

    # validate derived scales
    _validate_scales(R_scale, T_scale, rho0, pre_scale_val, 1.0)

    return scale_config_t(
        name=f"derived_E{E0:.1e}_n{n0:.1e}_gamma{gamma0:.0f}",
        length_scale=R_scale,
        time_scale=T_scale,
        rho_scale=rho0,
        pre_scale=pre_scale_val,
        v_scale=1.0,
        description=f"derived from E={E0:.1e} erg, n={n0:.1e} cm^-3",
        reference="sedov-taylor / blandford-mckee scaling",
    )


def derive_scales_from_simulation(
    ell: float,  # length scale from simulation
    rho0: float,  # density scale from simulation
    gamma0: float = 10.0,  # shock lorentz factor
) -> scale_config_t:
    """
    derive scales from simulation's internal parameters.

    for blandford-mckee simulations where:
        ell = ((17-4k) * E0 / (8pi * rho0))^(1/(3-k))

    args:
        ell: characteristic length scale (dimensionless units)
        rho0: ambient density scale (dimensionless units)
        gamma0: shock lorentz factor

    returns:
        scale_config_t
    """
    # assume scales are already in cgs from simulation setup
    T_scale = ell / const.c.cgs.value
    pre_scale_val = rho0 * const.c.cgs.value**2

    # validate derived scales
    _validate_scales(ell, T_scale, rho0, pre_scale_val, 1.0)

    return scale_config_t(
        name=f"sim_derived_ell{ell:.1e}_rho{rho0:.1e}",
        length_scale=ell,
        time_scale=T_scale,
        rho_scale=rho0,
        pre_scale=pre_scale_val,
        v_scale=1.0,
        description=f"derived from simulation ell={ell:.1e}, rho0={rho0:.1e}",
        reference="simulation internal scales",
    )


def make_blandford_mckee_scale(
    E0: float = 1e53,  # energy [erg]
    n0: float = 1.0,  # ambient number density [cm^-3]
    k: float = 0.0,  # density power law exponent
) -> scale_config_t:
    """
    create blandford-mckee scale configuration.

    computes length scale from blast wave energy and ambient density:
        ell = [(17-4k) * E0 / (8pi * rho_0 * c^2)]^(1/(3-k))

    for k=0 (uniform density):
        ell = (E0 / rho_0 c^2)^(1/3)

    args:
        E0: total energy [erg]
        n0: ambient number density [cm^-3]
        k: density power law exponent (rho propto r^-k)

    returns:
        scale_config_t

    example:
        # canonical grb (default)
        scales = make_blandford_mckee_scale()

        # high energy, low density
        scales = make_blandford_mckee_scale(E0=1e54, n0=0.1)

        # wind environment (k=2)
        scales = make_blandford_mckee_scale(E0=1e52, n0=10, k=2)
    """
    rho0 = n0 * const.m_p.cgs.value
    c_cgs = const.c.cgs.value

    # blandford-mckee characteristic length
    if abs(k) < 1e-6:
        # k=0: uniform medium
        ell = (E0 / (rho0 * c_cgs**2)) ** (1.0 / 3.0)
    else:
        # k!=0: power-law density profile
        ell = ((17.0 - 4.0 * k) * E0 / (8.0 * math.pi * rho0 * c_cgs**2)) ** (
            1.0 / (3.0 - k)
        )

    time_scale = ell / c_cgs
    pre_scale = E0 / ell**3

    # validate blandford-mckee scales
    _validate_scales(ell, time_scale, rho0, pre_scale, 1.0)

    return scale_config_t(
        name=f"bmk_E{E0:.1e}_n{n0:.1e}_k{k:.1f}",
        length_scale=ell,
        time_scale=time_scale,
        rho_scale=rho0,
        pre_scale=pre_scale,
        v_scale=1.0,
        description=f"blandford-mckee: E={E0:.1e} erg, n={n0:.1e} cm^-3, k={k}",
        reference="blandford & mckee (1976)",
    )
