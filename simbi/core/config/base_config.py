"""
Base configuration model for simbi simulations.

This module provides the foundational configuration model that defines
the structure and validation rules for simulation configurations.
"""

from pydantic import computed_field, model_validator
from typing import (
    Any,
    ClassVar,
    Optional,
    Union,
    Sequence,
    Callable,
    cast,
    get_args,
    get_origin,
)
import argparse
import math
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from ..types.typing import InitialStateType, ExpressionDict
from ..types.input import (
    CoordSystem,
    Regime,
    SpatialOrder,
    TimeStepping,
    CellSpacing,
    Solver,
)
from ..types.bodies import BodySystemConfig, ImmersedBodyConfig
from .parameters import CLIConfigurableModel
from .fields import SimbiField


class SimbiBaseConfig(CLIConfigurableModel):
    """Base configuration model for simbi simulations.

    This provides a structured, validated configuration for all simulations.
    Problem implementations should extend this class and provide problem-specific
    parameters and methods.
    """

    # Track CLI parser for global access
    cli_parser: ClassVar[Optional[argparse.ArgumentParser]] = None

    # Required fields - these must be provided in subclasses
    resolution: Union[int, Sequence[int], NDArray[np.int64]] = SimbiField(
        ..., description="Grid resolution"
    )

    coord_system: CoordSystem = SimbiField(..., description="Coordinate system")

    regime: Regime = SimbiField(..., description="Physics regime")

    bounds: Sequence[Sequence[float]] = SimbiField(..., description="Domain boundaries")

    adiabatic_index: float = SimbiField(..., description="Adiabatic index")

    # Optional fields with defaults
    data_directory: Path = SimbiField(
        Path("data/"), description="Output data directory"
    )

    cfl_number: float = SimbiField(0.1, description="CFL condition number")

    solver: Solver = SimbiField(Solver.HLLC, description="Numerical solver")

    spatial_order: SpatialOrder = SimbiField(
        SpatialOrder.PLM, description="Spatial order of accuracy"
    )

    temporal_order: TimeStepping = SimbiField(
        TimeStepping.RK2, description="Time stepping method"
    )

    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Spacing in x1 direction"
    )

    x2_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Spacing in x2 direction"
    )

    x3_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Spacing in x3 direction"
    )

    use_quirk_smoothing: bool = SimbiField(False, description="Use Quirk smoothing")

    checkpoint_interval: float = SimbiField(0.1, description="Checkpoint interval")
    checkpoint_index: int = SimbiField(
        0, description="Checkpoint index for resuming simulations"
    )

    checkpoint_file: Optional[str] = SimbiField(
        None, description="Checkpoint file to resume from"
    )

    boundary_conditions: Union[str, Sequence[str]] = SimbiField(
        "outflow", description="Boundary conditions"
    )

    plm_theta: float = SimbiField(1.5, description="PLM theta parameter")

    locally_isothermal: bool = SimbiField(
        False, description="Use locally isothermal equation of state"
    )

    default_start_time: float = SimbiField(0.0, description="Simulation start time")

    default_end_time: float = SimbiField(1.0, description="Simulation end time")

    viscosity: float = SimbiField(0.0, description="Viscosity coefficient")

    order_of_integration: Optional[str] = SimbiField(
        None, description="Order of integration for the simulation"
    )

    # Isothermal physics
    ambient_sound_speed: float = SimbiField(
        0.0, description="Ambient sound speed for isothermal simulations"
    )

    shakura_sunyaev_alpha: float = SimbiField(
        0.0, description="Shakura-Sunyaev alpha parameter for disk simulations"
    )

    buffer_parameters: dict[str, float] = SimbiField(
        default_factory=dict, description="Buffer zone parameters for disk simulations"
    )

    # Logging configuration
    log_output: bool = SimbiField(False, description="Enable logging to file")

    log_checkpoints_tuple: tuple[bool, int] = SimbiField(
        (False, 0), description="Logarithmic output settings (enabled, num_outputs)"
    )

    log_parameter_setup: bool = SimbiField(
        False, description="Log parameter setup information"
    )

    # Scale factor and derivative as callable
    scale_factor: Optional[Callable[[float], float]] = SimbiField(
        None, description="Scale factor function a(t)"
    )

    scale_factor_derivative: Optional[Callable[[float], float]] = SimbiField(
        None, description="Time derivative of scale factor"
    )

    # Boundary condition expressions
    bx1_inner_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Inner x1 boundary expressions"
    )

    bx1_outer_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Outer x1 boundary expressions"
    )

    bx2_inner_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Inner x2 boundary expressions"
    )

    bx2_outer_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Outer x2 boundary expressions"
    )

    bx3_inner_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Inner x3 boundary expressions"
    )

    bx3_outer_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Outer x3 boundary expressions"
    )

    # Source term expressions
    hydro_source_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Hydro source term expressions"
    )

    gravity_source_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Gravity source term expressions"
    )

    local_sound_speed_expressions: ExpressionDict = SimbiField(
        default_factory=dict, description="Local sound speed expressions"
    )

    # Body physics
    body_system: Optional[BodySystemConfig] = SimbiField(
        None, description="Body system configuration"
    )

    immersed_bodies: list[ImmersedBodyConfig] = SimbiField(
        default_factory=list, description="Immersed bodies configuration"
    )

    @computed_field
    @property
    def dimensionality(self) -> int:
        """Compute the dimensionality from resolution"""
        if isinstance(self.resolution, int):
            return 1
        return len(self.resolution)

    @computed_field
    @property
    def is_mhd(self) -> bool:
        """Check if the simulation involves MHD"""
        return self.regime in [Regime.SRMHD]

    @computed_field
    @property
    def isothermal(self) -> bool:
        """Check if the simulation is isothermal"""
        return self.adiabatic_index == 1.0

    @computed_field
    @property
    def nvars(self) -> int:
        """Get number of variables based on regime and dimensionality"""
        if self.is_mhd:
            return 9  # MHD has 9 primary variables
        return self.dimensionality + 3  # Hydro has density, momentum components, energy

    @computed_field
    @property
    def is_relativistic(self) -> bool:
        """Check if the simulation is relativistic"""
        return self.regime in [Regime.SRHD, Regime.SRMHD]

    @computed_field
    @property
    def mesh_motion(self) -> bool:
        """Check if the simulation involves mesh motion"""
        if self.scale_factor is None or self.scale_factor_derivative is None:
            return False
        elif self.scale_factor_derivative(1) / self.scale_factor(1) != 0:
            return True
        return False

    @computed_field
    @property
    def is_homlogous(self) -> bool:
        """Check if the simulation is homologous"""
        if self.mesh_motion and self.coord_system in [CoordSystem.SPHERICAL]:
            return True
        return False

    @computed_field
    @property
    def dlogt(self) -> float:
        """Return logarithmic time spacing"""
        log_enabled, num_outputs = self.log_checkpoints_tuple
        if log_enabled and num_outputs > 0:
            return (
                math.log10(self.default_end_time / self.default_start_time)
                / num_outputs
            )
        return 0.0

    @model_validator(mode="after")
    def validate_isothermal_settings(self) -> "SimbiBaseConfig":
        """Validate isothermal simulation settings"""
        if (
            self.isothermal
            and not self.locally_isothermal
            and self.ambient_sound_speed <= 0
        ):
            raise ValueError(
                "For isothermal simulations (gamma=1), the ambient sound speed *must* be defined "
                "unless locally_isothermal is set to True."
            )
        return self

    # Method to generate initial state
    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for the simulation.

        Returns:
            A generator function that produces the initial state values.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement initial_primitive_state")

    @model_validator(mode="before")
    @classmethod
    def validate_field_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure field types in subclasses match or are subtypes of base class types."""
        # For most use cases, we can let Pydantic handle the validation at runtime
        # We'll just do basic compatibility checks for simple cases
        if cls is not SimbiBaseConfig and issubclass(cls, SimbiBaseConfig):
            for field_name, field_info in cls.model_fields.items():
                # Check if this field exists in the base class
                if field_name in SimbiBaseConfig.model_fields:
                    base_field = SimbiBaseConfig.model_fields[field_name]

                    # Skip validation if annotations are identical
                    if field_info.annotation == base_field.annotation:
                        continue

                    # Special case for Union types - check if field type is in the Union
                    base_origin = get_origin(base_field.annotation)
                    if base_origin is Union:
                        base_args = get_args(base_field.annotation)
                        if field_info.annotation in base_args:
                            continue

                    # Special case for collections - basic compatibility check
                    # This handles cases like list being compatible with Sequence
                    if str(base_origin).startswith("typing.Sequence"):
                        field_origin = get_origin(field_info.annotation)
                        if field_origin in (list, tuple):
                            continue

        return data
