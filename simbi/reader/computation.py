# =============================================================================
# computation.py
#
# derived field computation for simulation data.
# contains all physics calculations needed for post-processing.
# =============================================================================
from enum import Enum, IntEnum
from typing import Any, Callable, Sequence

import numpy as np

from ..types import Array, ProcessedData

ComputeFunc = Callable[[dict[str, Any]], Array]


# =============================================================================
# physics primitives
# =============================================================================
class VectorComponent(IntEnum):
    X1 = 0
    X2 = 1
    X3 = 2


class VectorMode(Enum):
    Magnitude = "magnitude"
    All = "all"


def _dot_product(a: Sequence[Array], b: Sequence[Array]) -> Array:
    """dot product of two vector fields."""
    return np.sum([a[ii] * b[ii] for ii in range(len(a))], axis=0)


def lorentz_factor(
    velocity: Sequence[Array], regime: str, using_gamma_beta: bool = False
) -> Array | float:
    """compute lorentz factor from velocity."""
    vsquared = _dot_product(velocity, velocity)
    if regime != "newtonian" and np.any(vsquared >= 1.0):
        raise ValueError("velocity exceeds speed of light")

    if regime == "newtonian":
        return 1.0
    elif not using_gamma_beta:
        return np.asarray(1.0 / np.sqrt(1.0 - vsquared))
    else:
        return np.asarray(np.sqrt(1.0 + vsquared))


def four_velocity(
    velocity: Sequence[Array], regime: str, component: int
) -> Array:
    """compute four-velocity component."""
    W = lorentz_factor(velocity, regime)
    return np.asarray(velocity[component] * W)


def spec_enthalpy(
    adiabatic_index: float, rho: Array, pressure: Array, regime: str
) -> Array | float:
    """compute specific enthalpy."""
    if regime == "newtonian":
        if adiabatic_index == 1.0:
            return 1.0 + pressure / rho
        return 1.0
    return 1.0 + adiabatic_index * pressure / (rho * (adiabatic_index - 1.0))


def _enthalpy(
    rho: Array, pre: Array, gamma: float, regime: str
) -> Array | float:
    """compute enthalpy per particle."""
    if regime == "newtonian":
        return 1.0
    return 1.0 + gamma * pre / (rho * (gamma - 1.0))


def labframe_density(
    rho: Array, velocity: Sequence[Array], regime: str
) -> Array:
    """compute lab-frame density."""
    return rho * lorentz_factor(velocity, regime)


def labframe_energy_density(
    rho: Array,
    pre: Array,
    vel: Sequence[Array],
    bfield: Sequence[Array],
    gamma: float,
    regime: str = "newtonian",
) -> Array:
    """compute lab-frame energy density."""
    vsq = sum(v**2 for v in vel)

    if regime == "newtonian":
        if gamma == 1.0:
            return pre / rho
        return pre / (gamma - 1.0) + 0.5 * rho * vsq

    W = 1.0 / np.sqrt(1.0 - vsq)
    h = _enthalpy(rho, pre, gamma, regime)

    if regime == "srhd":
        return np.asarray(rho * W**2 * h - pre - rho * W)

    if regime == "srmhd":
        bsq = sum(b**2 for b in bfield)
        vdb = _dot_product(vel, bfield)
        return np.asarray(
            rho * W**2 * h - pre - rho * W + 0.5 * (bsq + vsq * bsq - vdb**2)
        )

    raise NotImplementedError(f"regime '{regime}' not implemented")


def labframe_momentum(
    rho: Array,
    pre: Array,
    vel: Sequence[Array],
    bfield: Sequence[Array],
    gamma: float,
    regime: str = "newtonian",
    mode: VectorMode | VectorComponent = VectorMode.All,
) -> Array:
    """compute lab-frame momentum."""
    if regime == "newtonian":
        mom_vec = np.asarray(rho) * np.asarray(vel)
    elif "sr" in regime:
        h = _enthalpy(rho, pre, gamma, regime)
        D = labframe_density(rho, vel, regime)
        W = lorentz_factor(vel, regime)

        magnetic_part: Array | float = 0.0
        if regime == "srmhd":
            bsq = np.array(sum(b**2 for b in bfield), dtype=float)
            vdb = _dot_product(vel, bfield)
            magnetic_part = np.array(
                [bsq * v - vdb * b for v, b in zip(vel, bfield)]
            ).squeeze()

        mom_vec = D * h * W * np.asarray(vel) + magnetic_part
    else:
        raise NotImplementedError(f"regime '{regime}' not implemented")

    if mode == VectorMode.Magnitude:
        return np.asarray(np.sqrt(np.sum(mom_vec**2, axis=0)))
    elif mode == VectorMode.All:
        return mom_vec
    else:
        return np.asarray(mom_vec[int(mode.value)])


def magnetic_pressure(
    bfields: Sequence[Array], velocity: Sequence[Array], regime: str
) -> Array:
    """compute magnetic pressure."""
    bsq: Array = np.sum([b**2 for b in bfields], axis=0)
    if regime == "srmhd":
        W = 1.0 / np.sqrt(1.0 - _dot_product(velocity, velocity))
        vdb = _dot_product(velocity, bfields)
        bsq = bsq / W**2 + vdb**2
    return 0.5 * bsq


def total_pressure(
    pre: Array, bfields: Sequence[Array], velocity: Sequence[Array], regime: str
) -> Array:
    """compute total pressure (gas + magnetic)."""
    if "mhd" not in regime:
        return pre
    return pre + magnetic_pressure(bfields, velocity, regime)


def enthalpy_density(
    rho: Array,
    pre: Array,
    bfields: Sequence[Array],
    velocity: Sequence[Array],
    adiabatic_index: float,
    regime: str = "newtonian",
) -> Array:
    """compute enthalpy density."""
    if regime == "newtonian":
        return rho
    elif regime == "srhd":
        return rho * _enthalpy(rho, pre, adiabatic_index, regime)
    elif regime == "srmhd":
        return rho * _enthalpy(
            rho, pre, adiabatic_index, regime
        ) + 2.0 * magnetic_pressure(bfields, velocity, regime)
    raise NotImplementedError(f"regime '{regime}' not implemented")


def magnetization(rho: Array, bfields: Sequence[Array]) -> Array:
    """compute magnetization sigma = b^2 / rho."""
    return np.asarray(_dot_product(bfields, bfields) / rho)


# =============================================================================
# computation pipeline factory
# =============================================================================
def create_computation_pipeline(data: ProcessedData) -> dict[str, ComputeFunc]:
    """
    create a pipeline of derived field computations.

    returns a dict mapping field names to compute functions.
    each function takes fields dict and returns computed array.
    """
    ndim = data.metadata.dimensions
    regime = data.metadata.regime
    gamma = data.metadata.adiabatic_index

    def get_velocities(fields: dict[str, Array]) -> list[Array]:
        return [fields[f"v{ii}"] for ii in range(1, ndim + 1)]

    def get_b_fields(fields: dict[str, Array]) -> list[Array]:
        return [
            fields.get(f"b{ii}", np.zeros_like(fields["rho"]))
            for ii in range(1, ndim + 1)
        ]

    # basic derived fields
    def compute_W(fields: dict[str, Array]) -> Array | float:
        return lorentz_factor(get_velocities(fields), regime)

    def compute_D(fields: dict[str, Array]) -> Array:
        return fields["rho"] * lorentz_factor(get_velocities(fields), regime)

    def compute_velocity_magnitude(fields: dict[str, Array]) -> Array:
        if ndim == 1:
            return fields["v1"]
        return np.sqrt(sum(fields[f"v{ii}"] ** 2 for ii in range(1, ndim + 1)))

    def compute_four_velocity_magnitude(fields: dict[str, Array]) -> Array:
        if ndim == 1:
            return four_velocity(get_velocities(fields), regime, 0)
        v_mag = compute_velocity_magnitude(fields)
        W = lorentz_factor(get_velocities(fields), regime)
        return v_mag * W

    def compute_u_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        def _compute(fields: dict[str, Array]) -> Array:
            return four_velocity(get_velocities(fields), regime, component)

        return _compute

    def compute_momentum_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        def _compute(fields: dict[str, Array]) -> Array:
            return labframe_momentum(
                fields["rho"],
                fields["p"],
                get_velocities(fields),
                get_b_fields(fields),
                gamma,
                regime,
                VectorComponent(component),
            )

        return _compute

    def compute_b_mean_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        def _compute(fields: dict[str, Array]) -> Array:
            field_name = f"b{component}"
            if field_name not in fields:
                return np.zeros_like(fields["rho"])

            view = fields[field_name]
            if field_name == "b1":
                return 0.5 * (view[..., 1:] + view[..., :-1])
            elif field_name == "b2":
                return 0.5 * (view[:, 1:, :] + view[:, :-1, :])
            elif field_name == "b3":
                return 0.5 * (view[1:, :, :] + view[:-1, :, :])
            raise ValueError(f"invalid b-field component: {field_name}")

        return _compute

    def compute_energy(fields: dict[str, Array]) -> Array:
        return labframe_energy_density(
            fields["rho"],
            fields["p"],
            get_velocities(fields),
            get_b_fields(fields),
            gamma,
            regime,
        )

    def compute_enthalpy(fields: dict[str, Array]) -> Array | float:
        return spec_enthalpy(gamma, fields["rho"], fields["p"], regime)

    def compute_enthalpy_density(fields: dict[str, Array]) -> Array:
        return enthalpy_density(
            fields["rho"],
            fields["p"],
            get_b_fields(fields),
            get_velocities(fields),
            gamma,
            regime,
        )

    def compute_magnetization(fields: dict[str, Array]) -> Array:
        return magnetization(fields["rho"], get_b_fields(fields))

    def compute_total_pressure(fields: dict[str, Array]) -> Array:
        return total_pressure(
            fields["p"], get_b_fields(fields), get_velocities(fields), regime
        )

    def compute_magnetic_pressure(fields: dict[str, Array]) -> Array:
        return magnetic_pressure(
            get_b_fields(fields), get_velocities(fields), regime
        )

    def compute_mach_number(fields: dict[str, Array]) -> Array:
        v_mag = compute_velocity_magnitude(fields)
        cs = np.sqrt(gamma * fields["p"] / fields["rho"])
        return np.asanyarray(v_mag / cs)

    def compute_chi_density(fields: dict[str, Array]) -> Array:
        return (
            fields["rho"]
            * fields["chi"]
            * lorentz_factor(get_velocities(fields), regime)
        )

    def angular_momentum_density(level_data: dict[str, Array]) -> Array:
        """compute angular momentum density L = r x S."""
        mesh = getattr(level_data, "mesh")
        x, y = mesh.x1c, mesh.x2c
        Sx = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(0),
        )
        Sy = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(1),
        )
        return np.asarray(x * Sy - y * Sx)

    def specific_angular_momentum(level_data: dict[str, Array]) -> Array:
        """compute specific angular momentum."""
        mesh = getattr(level_data, "mesh")
        x, y = mesh.x1c, mesh.x2c
        Sx = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(0),
        )
        Sy = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(1),
        )
        Lz = x * Sy - y * Sx
        den = level_data["rho"]

        if den.ndim == 3:
            dz = mesh.x3v[1:] - mesh.x3v[:-1]
            Sigma = np.sum(den * dz, axis=0)
            Lz_int = np.sum(Lz * dz, axis=0)
            return np.asarray(Lz_int / (Sigma + np.finfo(float).tiny))
        return np.asarray(Lz / (den + np.finfo(float).tiny))

    def surface_density(level_data: dict[str, Any]) -> Array:
        den = level_data["rho"]
        if den.ndim == 3:
            mesh = getattr(level_data, "mesh")
            dz = mesh.x3v[1:] - mesh.x3v[:-1]
            return np.asarray(np.sum(den * dz, axis=0))
        return np.asarray(den)

    def mass_flux(level_data: dict[str, Any]) -> Array:
        """compute mass flux through spherical shells."""
        mesh = getattr(level_data, "mesh")
        x, y = mesh.x1c, mesh.x2c
        vx, vy = level_data["v1"], level_data["v2"]

        if vx.ndim == 3:
            z = mesh.x3c
            r = np.sqrt(x**2 + y**2 + z**2)
            vz = level_data["v3"]
            vr = (x * vx + y * vy + z * vz) / (r + np.finfo(float).tiny)
        else:
            r = np.sqrt(x**2 + y**2)
            vr = (x * vx + y * vy) / (r + np.finfo(float).tiny)

        return np.asarray(4.0 * np.pi * r**2 * level_data["rho"] * vr)

    def compute_divergence(level_data: dict[str, Array]) -> Array:
        """compute velocity divergence."""
        mesh = getattr(level_data, "mesh")
        vx, vy = level_data["v1"], level_data["v2"]
        x, y = mesh.x1c, mesh.x2c

        dvx_dx = np.gradient(vx, x, axis=ndim - 1)
        dvy_dy = np.gradient(vy, y, axis=ndim - 2)

        if ndim == 3:
            vz = level_data["v3"]
            z = mesh.x3c
            dvz_dz = np.gradient(vz, z, axis=0)
            return np.asarray(dvx_dx + dvy_dy + dvz_dz)
        return np.asarray(dvx_dx + dvy_dy)

    def compute_vorticity_z(level_data: dict[str, Array]) -> Array:
        """compute z-component of vorticity."""
        mesh = getattr(level_data, "mesh")
        vx, vy = level_data["v1"], level_data["v2"]
        x, y = mesh.x1c, mesh.x2c

        dvy_dx = np.gradient(vy, x, axis=ndim - 1)
        dvx_dy = np.gradient(vx, y, axis=ndim - 2)
        return np.asarray(dvy_dx - dvx_dy)

    # build pipeline
    pipeline: dict[str, Any] = {
        "W": compute_W,
        "D": compute_D,
        "v": compute_velocity_magnitude,
        "u": compute_four_velocity_magnitude,
        "energy": compute_energy,
        "enthalpy": compute_enthalpy,
        "enthalpy_density": compute_enthalpy_density,
        "sigma": compute_magnetization,
        "ptot": compute_total_pressure,
        "pmag": compute_magnetic_pressure,
        "mach": compute_mach_number,
        "chi_dens": compute_chi_density,
        "j": angular_momentum_density,
        "mass_flux": mass_flux,
        "j_spec": specific_angular_momentum,
        "Sigma": surface_density,
        "vorticity": compute_vorticity_z,
        "div_v": compute_divergence,
    }

    # add component fields
    for ii in range(1, ndim + 1):
        pipeline[f"u{ii}"] = compute_u_component(ii - 1)
        pipeline[f"m{ii}"] = compute_momentum_component(ii - 1)

    if data.metadata.is_mhd:
        for ii in range(1, ndim + 1):
            pipeline[f"b{ii}_mean"] = compute_b_mean_component(ii)

    return pipeline
