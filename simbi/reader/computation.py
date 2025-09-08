from typing import Any, Callable

import numpy as np

from ..core.types import Array, ProcessedData
from ..physics.calculations import (
    VectorComponent,
    enthalpy_density,
    four_velocity,
    labframe_energy_density,
    labframe_momentum,
    lorentz_factor,
    magnetic_pressure,
    magnetization,
    spec_enthalpy,
    total_pressure,
)

ComputeFunc = Callable[[dict[str, Array]], Array]


def create_computation_pipeline(data: ProcessedData) -> dict[str, ComputeFunc]:
    ndim = data.metadata.dimensions
    regime = data.metadata.regime
    gamma = data.metadata.adiabatic_index

    def get_velocities(fields: dict[str, Array]) -> list[Array]:
        return [fields[f"v{i}"] for i in range(1, ndim + 1)]

    def get_b_fields(fields: dict[str, Array]) -> list[Array]:
        return [
            fields.get(f"b{i}", np.zeros_like(fields["rho"]))
            for i in range(1, ndim + 1)
        ]

    # Basic derived fields
    def compute_W(fields: dict[str, Array]) -> Array | float:
        return lorentz_factor(get_velocities(fields), regime)

    def compute_D(fields: dict[str, Array]) -> Array:
        return fields["rho"] * lorentz_factor(get_velocities(fields), regime)

    def compute_velocity_magnitude(fields: dict[str, Array]) -> Array:
        if ndim == 1:
            return fields["v1"]
        return np.sqrt(sum(fields[f"v{i}"] ** 2 for i in range(1, ndim + 1)))

    def compute_four_velocity_magnitude(fields: dict[str, Array]) -> Array:
        if ndim == 1:
            return four_velocity(get_velocities(fields), regime, 0)
        v_mag = compute_velocity_magnitude(fields)
        W = lorentz_factor(get_velocities(fields), regime)
        return v_mag * W

    # Four-velocity components
    def compute_u_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        def _compute(fields: dict[str, Array]) -> Array:
            return four_velocity(get_velocities(fields), regime, component)

        return _compute

    # Momentum components
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
            else:
                raise ValueError(
                    f"Invalid magnetic field component: {field_name}"
                )

        return _compute

    # Energy and thermodynamics
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

    # Magnetic fields and pressures
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

    # Mach number
    def compute_mach_number(fields: dict[str, Array]) -> Array:
        v_mag = compute_velocity_magnitude(fields)
        sound_speed = np.sqrt(gamma * fields["p"] / fields["rho"])
        return np.asanyarray(v_mag / sound_speed)

    # Tracer density
    def compute_chi_density(fields: dict[str, Array]) -> Array:
        return (
            fields["rho"]
            * fields["chi"]
            * lorentz_factor(get_velocities(fields), regime)
        )

    # Build the pipeline
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
    }

    # Add component fields
    for i in range(1, ndim + 1):
        pipeline[f"u{i}"] = compute_u_component(i - 1)
        pipeline[f"m{i}"] = compute_momentum_component(i - 1)

    if data.metadata.is_mhd:
        for i in range(1, ndim + 1):
            pipeline[f"b{i}_mean"] = compute_b_mean_component(i)

    return pipeline
