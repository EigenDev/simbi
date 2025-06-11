"""
Base configuration model for simbi simulations.

This module provides the foundational configuration model that defines
the structure and validation rules for simulation configurations.
"""

from pydantic import computed_field, model_validator, PrivateAttr
from typing import (
    Any,
    ClassVar,
    Optional,
    Union,
    Sequence,
    Callable,
    get_args,
    get_origin,
    final,
)
import argparse
import math
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from simbi.core.io.ib_load import load_immersed_bodies_or_body_system

from ..types.typing import InitialStateType, ExpressionDict
from ..types.input import (
    CoordSystem,
    Regime,
    SpatialOrder,
    TimeStepping,
    CellSpacing,
    Solver,
    BoundaryCondition,
)
from ..types.bodies import (
    BodySystemConfig,
    ImmersedBodyConfig,
)
from .parameters import CLIConfigurableModel
from .fields import SimbiField


class SimbiBaseConfig(CLIConfigurableModel):
    """Base configuration model for simbi simulations.

    This provides a structured, validated configuration for all simulations.
    Problem implementations should extend this class and provide problem-specific
    parameters and methods.
    """

    _from_checkpoint_called: ClassVar[bool] = False
    _body_system: BodySystemConfig = PrivateAttr(default_factory=BodySystemConfig)
    _immersed_bodies: list[ImmersedBodyConfig] = PrivateAttr(default_factory=list)

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
    use_fleischmann_limiter: bool = SimbiField(
        False,
        description="Use the Fleischmann et al. 2020 mechanism for low-Mach fixes tot eh HLLC solver",
    )

    checkpoint_interval: float = SimbiField(0.1, description="Checkpoint interval")
    checkpoint_index: int = SimbiField(
        0, description="Checkpoint index for resuming simulations"
    )

    checkpoint_file: Optional[str] = SimbiField(
        None, description="Checkpoint file to resume from"
    )

    boundary_conditions: Union[BoundaryCondition, Sequence[BoundaryCondition]] = (
        SimbiField("outflow", description="Boundary conditions")
    )

    plm_theta: float = SimbiField(1.5, description="PLM theta parameter")

    start_time: float = SimbiField(0.0, description="Simulation start time")

    end_time: float = SimbiField(1.0, description="Simulation end time")

    order_of_integration: Optional[str] = SimbiField(
        None, description="Order of integration for the simulation"
    )

    # Logging configuration
    log_output: bool = SimbiField(False, description="Enable logging to file")

    log_checkpoints_tuple: tuple[bool, int] = SimbiField(
        (False, 0), description="Logarithmic output settings (enabled, num_outputs)"
    )

    log_parameter_setup: bool = SimbiField(
        True, description="Log parameter setup information"
    )

    # Isothermal physics
    @computed_field
    @property
    def ambient_sound_speed(self) -> float:
        """Ambient sound speed for isothermal simulations"""
        return 0.0

    @computed_field
    @property
    def shakura_sunyaev_alpha(self) -> float:
        """Shakura-Sunyaev alpha parameter for accretion disk simulations"""
        return 0.0

    @computed_field
    @property
    def viscosity(self) -> float:
        """Viscosity coefficient for simulations"""
        return 0.0

    @computed_field
    @property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        """Scale factor for mesh motion, if applicable"""
        return None

    @computed_field
    @property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
        """Derivative of the scale factor for mesh motion, if applicable"""
        return None

    # Boundary condition expressions
    @computed_field
    @property
    def buffer_parameters(self) -> dict[str, float]:
        """Buffer parameters for boundary conditions"""
        return {}

    @computed_field
    @property
    def bx1_inner_expressions(self) -> ExpressionDict:
        """Inner x1 boundary expressions"""
        return {}

    @computed_field
    @property
    def bx1_outer_expressions(self) -> ExpressionDict:
        """Outer x1 boundary expressions"""
        return {}

    @computed_field
    @property
    def bx2_inner_expressions(self) -> ExpressionDict:
        """Inner x2 boundary expressions"""
        return {}

    @computed_field
    @property
    def bx2_outer_expressions(self) -> ExpressionDict:
        """Outer x2 boundary expressions"""
        return {}

    @computed_field
    @property
    def bx3_inner_expressions(self) -> ExpressionDict:
        """Inner x3 boundary expressions"""
        return {}

    @computed_field
    @property
    def bx3_outer_expressions(self) -> ExpressionDict:
        """Outer x3 boundary expressions"""
        return {}

    # Source term expressions
    @computed_field
    @property
    def hydro_source_expressions(self) -> ExpressionDict:
        """Hydro source term expressions"""
        return {}

    @computed_field
    @property
    def gravity_source_expressions(self) -> ExpressionDict:
        """Gravity source term expressions"""
        return {}

    @computed_field
    @property
    def local_sound_speed_expressions(self) -> ExpressionDict:
        """Local sound speed expressions"""
        return {}

    # Body physics
    @computed_field
    @property
    def body_system(self) -> Optional[BodySystemConfig]:
        """Get the body system configuration"""
        return None

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        """Get the list of immersed bodies configuration"""
        return []

    def set_body_system(self, body_system: BodySystemConfig) -> None:
        """Set the body system configuration from checkpoint data."""
        if not isinstance(body_system, BodySystemConfig):
            raise TypeError("body_system must be an instance of BodySystemConfig")
        self._body_system = body_system

    def set_immersed_bodies(
        self, immersed_bodies: Union[ImmersedBodyConfig, Sequence[ImmersedBodyConfig]]
    ) -> None:
        """Set the immersed bodies configuration from checkpoint data."""
        if isinstance(immersed_bodies, ImmersedBodyConfig):
            self._immersed_bodies = [immersed_bodies]
        elif isinstance(immersed_bodies, list):
            if not all(isinstance(b, ImmersedBodyConfig) for b in immersed_bodies):
                raise TypeError(
                    "All immersed bodies must be instances of ImmersedBodyConfig"
                )
            self._immersed_bodies = list(immersed_bodies)
        else:
            raise TypeError(
                "immersed_bodies must be an ImmersedBodyConfig or a list of them"
            )

    @computed_field
    @property
    def dimensionality(self) -> int:
        """Compute the dimensionality from resolution"""
        if self.regime in [Regime.SRMHD]:
            return 3  # MHD is always 3D

        if isinstance(self.resolution, int):
            return 1
        return sum(int(d > 1) for d in self.resolution)

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
    def is_homologous(self) -> bool:
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
            return math.log10(self.end_time / self.start_time) / num_outputs
        return 0.0

    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        """Check if the simulation is locally isothermal"""
        return False

    @model_validator(mode="after")
    def validate_isothermal_settings(self) -> "SimbiBaseConfig":
        """Validate isothermal simulation settings"""
        if self._from_checkpoint_called:
            # Skip validation if loading from checkpoint
            return self
        if (
            self.isothermal
            and self.ambient_sound_speed <= 0
            and not self.locally_isothermal
        ):
            raise ValueError(
                "Ambient sound speed must be positive / non-zero for isothermal simulations"
                " unless you explicity set locally_isothermal to True."
            )
        return self

    @model_validator(mode="after")
    def validate_plm_theta(self) -> "SimbiBaseConfig":
        """Validate PLM theta parameter."""
        if self.spatial_order == SpatialOrder.PLM and not (0.0 < self.plm_theta <= 2.0):
            raise ValueError(
                "PLM theta must be in the range (0, 2] when using PLM spatial order."
            )
        return self

    @model_validator(mode="after")
    def log_parameters_if_enabled(self) -> "SimbiBaseConfig":
        """Log parameters if log_parameter_setup is enabled."""
        if self.log_parameter_setup:
            self.print_parameters()
        return self

    def print_parameters(self) -> None:
        """
        Print parameters that are unique to this problem class and don't exist in the base class.

        This method focuses only on problem-specific parameters that extend beyond what's
        available in the base configuration.
        """
        import logging
        import sys
        from datetime import datetime

        # Set up logger if not already configured
        logger = logging.getLogger("simbi")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        # Set up file logging if enabled
        if self.log_output:
            timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = Path(self.data_directory) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            logfile = log_dir / f"simbi_{timestr}.log"

            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(file_handler)
            logger.info(f"Writing log file: {logfile}")

        # Get the class of the current instance and the base class
        current_class = self.__class__
        base_class = SimbiBaseConfig

        # Print header with class name
        logger.info(f"\n{current_class.__name__} Parameters:")
        logger.info("=" * 80)

        # Find fields that exist in the current class but not in the base class
        problem_fields = {}

        # Check each field in the current class
        for field_name, field_info in current_class.model_fields.items():
            # Skip private fields and CLI parser
            if field_name.startswith("_") or field_name == "cli_parser":
                continue

            # Only include fields that don't exist in the base class
            if field_name not in base_class.model_fields:
                value = getattr(self, field_name)
                description = field_info.description or ""
                problem_fields[field_name] = (value, description)

        # Print fields in a nicely formatted table
        for name, (value, description) in sorted(problem_fields.items()):
            # Format the value based on its type
            if isinstance(value, float):
                # Use scientific notation for large/small values
                if value != 0 and (abs(value) < 0.001 or abs(value) > 1000):
                    formatted_value = f"{value:.4e}"
                else:
                    formatted_value = f"{value:.4f}"
            elif isinstance(value, (np.ndarray, list, tuple)):
                if len(str(value)) > 20:  # Truncate long arrays
                    formatted_value = f"{str(value)[:20]}..."
                else:
                    formatted_value = str(value)
            elif callable(value) and not isinstance(value, (type, np.ndarray)):
                # Skip callable objects
                continue
            else:
                formatted_value = str(value)

            # Truncate long values
            if len(formatted_value) > 30:
                formatted_value = formatted_value[:27] + "..."

            # Print the parameter
            logger.info(f"{name:.<30} {formatted_value:<30} {description}")

        # Print a message if no unique fields were found
        if not problem_fields:
            logger.info("No unique parameters defined in this problem class.")

        logger.info("=" * 80)

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

    @final
    def merge_with_checkpoint(
        self,
        checkpoint_metadata: dict[str, Any],
        immersed_bodies_config: Optional[
            Union[BodySystemConfig, Sequence[ImmersedBodyConfig]]
        ] = None,
    ) -> "SimbiBaseConfig":
        """
        Create a new config using this (default) config as base,
        with checkpoint state data overlaid.
        """
        # Extract only the "state" fields that should come from checkpoint
        state_fields = {
            "resolution",
            "bounds",
            "start_time",
            "end_time",
            "checkpoint_index",
            "adiabatic_index",
            "cfl_number",
            "data_directory",
            "solver",
            # "boundary_conditions",
            "plm_theta",
            "spatial_order",
            "temporal_order",
        }

        # Build update dict with only the fields that exist in checkpoint
        updates = {
            field: checkpoint_metadata[field]
            for field in state_fields
            if field in checkpoint_metadata and field in self.model_fields
        }

        if isinstance(immersed_bodies_config, BodySystemConfig):
            self.set_body_system(immersed_bodies_config)
        elif isinstance(immersed_bodies_config, list):
            self.set_immersed_bodies(immersed_bodies_config)

        # Create new config with checkpoint state but keeping all computed fields
        return self.model_copy(update=updates)

    # Factory method to load from some checkpoint or configuration file
    @final
    @classmethod
    def from_checkpoint_and_default(
        cls,
        default_config: "SimbiBaseConfig",
        metadata: dict[str, Any],
        immersed_bodies_metadata: dict[str, Any],
    ) -> "SimbiBaseConfig":
        """Create config from checkpoint data."""
        # Process checkpoint metadata into the right format
        checkpoint_data = {
            "resolution": tuple(
                r
                for r, _ in zip(
                    [metadata["active_x"], metadata["active_y"], metadata["active_z"]],
                    range(default_config.dimensionality),
                )
            )
            if default_config.dimensionality > 1
            else metadata["active_x"],
            "start_time": float(metadata["time"]),
            "end_time": float(metadata["end_time"]),
            "adiabatic_index": float(metadata["adiabatic_index"]),
            "cfl_number": float(metadata["cfl_number"]),
            # "data_directory": Path(metadata["data_directory"]),
            "solver": Solver(metadata["solver"]),
            "boundary_conditions": (
                [BoundaryCondition(b) for b in metadata["boundary_conditions"]]
                if isinstance(metadata["boundary_conditions"], list)
                else [BoundaryCondition(metadata["boundary_conditions"])]
            ),
            "plm_theta": float(metadata["plm_theta"]),
            "spatial_order": SpatialOrder(metadata["spatial_order"]),
            "temporal_order": TimeStepping(metadata["temporal_order"]),
            "checkpoint_index": int(metadata["checkpoint_index"]),
            "checkpoint_interval": float(metadata["checkpoint_interval"]),
            "x1_spacing": CellSpacing(metadata["x1_spacing"]),
            "x2_spacing": CellSpacing(metadata["x2_spacing"]),
            "x3_spacing": CellSpacing(metadata["x3_spacing"]),
            "coord_system": CoordSystem(metadata["coord_system"]),
            "regime": Regime(metadata["regime"]),
        }

        ib_config = load_immersed_bodies_or_body_system(
            metadata, immersed_bodies_metadata
        )

        # Let the default config merge itself with checkpoint data
        return default_config.merge_with_checkpoint(checkpoint_data, ib_config)
