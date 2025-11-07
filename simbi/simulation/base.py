"""
Base problem configuration for all simulation problems.

This module defines the abstract base class that all problem configurations
must inherit from. It provides:
- Universal simulation parameters (time, output, solver settings)
- Automatic CLI argument generation via ProblemField
- Validation and constraint checking
- Auto-forwarding of fields to SimulationStateSpec
- Extensibility for new features

Design principles:
- Universal parameters live here with sensible defaults
- Optional features included with defaults (problems can override)
- Abstract methods enforce implementation contract
- Validation happens in stages (field → model → physics)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import model_validator

from simbi.core.config.parameters import CLIConfigurableModel
from simbi.core.types.input import (
    CellSpacing,
    Reconstruction,
    Solver,
    TimeStepping,
)
from simbi.core.types.typing import InitialStateType

from .fields import ProblemField
from .state import SimulationStateSpec


class BaseProblemConfig(CLIConfigurableModel, ABC):
    """
    Abstract base configuration for all simulation problems.

    This class provides:
    1. Universal parameters common to all simulations
    2. Automatic CLI argument generation (via ProblemField)
    3. Validation of universal constraints
    4. Helper utilities for problem setup
    5. Auto-forwarding to SimulationStateSpec

    Subclasses must implement:
    - build_state() -> SimulationStateSpec
    - initial_primitive_state() -> InitialStateType

    Subclasses may override:
    - validate_physics() for custom validation

    Example:
        >>> class MyProblem(BaseProblemConfig):
        ...     resolution: int = ProblemField(1000, description="Grid cells")
        ...
        ...     def build_state(self) -> SimulationStateSpec:
        ...         base_fields = self._get_state_fields_from_config()
        ...         return SimulationStateSpec(
        ...             **base_fields,
        ...             resolution=(self.resolution,),
        ...             bounds=[(0.0, 1.0)],
        ...             coord_system=CoordSystem.CARTESIAN,
        ...             regime=Regime.CLASSICAL,
        ...             adiabatic_index=5.0/3.0,
        ...             boundary_conditions=[BoundaryCondition.OUTFLOW],
        ...             source_config=self,
        ...         )
        ...
        ...     def initial_primitive_state(self) -> InitialStateType:
        ...         # Return generator for initial conditions
        ...         pass
    """

    # ========================================================================
    # TIME CONFIGURATION
    # ========================================================================

    start_time: float = ProblemField(
        default=0.0,
        description="Simulation start time",
    )

    end_time: float = ProblemField(
        ...,  # Required field
        description="Simulation end time (must be > start_time)",
    )

    checkpoint_interval: float = ProblemField(
        ...,  # Required field
        description="Time interval between checkpoints",
    )

    # ========================================================================
    # OUTPUT CONFIGURATION
    # ========================================================================

    data_directory: Path = ProblemField(
        default=Path("data/"),
        description="Output directory for simulation data",
    )

    log_output: bool = ProblemField(
        default=False,
        description="Enable logging to file",
    )

    log_checkpoints_tuple: tuple[bool, int] = ProblemField(
        default=(False, 0),
        description="Logarithmic checkpoint settings (enabled, num_outputs)",
        expose_cli=False,  # Too complex for CLI
    )

    checkpoint_index: int = ProblemField(
        default=0,
        description="Starting checkpoint index (for resume)",
    )

    # ========================================================================
    # SOLVER CONFIGURATION
    # ========================================================================

    solver: Solver = ProblemField(
        default=Solver.HLLC,
        description="Riemann solver to use",
    )

    reconstruction: Reconstruction = ProblemField(
        default=Reconstruction.PLM,
        description="Spatial reconstruction method",
    )

    timestepping: TimeStepping = ProblemField(
        default=TimeStepping.RK2,
        description="Time integration method",
    )

    order: int | None = ProblemField(
        default=None,
        description="Order of accuracy (1=first order, 2=second order). "
        "Convenience parameter that sets reconstruction and timestepping. "
        "Takes precedence over explicit reconstruction/timestepping settings.",
        choices=[1, 2],
    )

    # ========================================================================
    # NUMERICAL PARAMETERS
    # ========================================================================

    cfl_number: float = ProblemField(
        default=0.4,
        description="CFL number for timestep control (0 < CFL <= 1)",
    )

    plm_theta: float = ProblemField(
        default=1.5,
        description="PLM slope limiter parameter (0 < theta <= 2)",
    )

    # ========================================================================
    # OPTIONAL FEATURES (all problems get these with defaults)
    # ========================================================================

    # Grid spacing
    x1_spacing: CellSpacing = ProblemField(
        default=CellSpacing.LINEAR,
        description="Cell spacing type in x1 direction",
    )

    x2_spacing: CellSpacing = ProblemField(
        default=CellSpacing.LINEAR,
        description="Cell spacing type in x2 direction",
    )

    x3_spacing: CellSpacing = ProblemField(
        default=CellSpacing.LINEAR,
        description="Cell spacing type in x3 direction",
    )

    # Solver options
    use_quirk_smoothing: bool = ProblemField(
        default=False,
        description="Enable Quirk smoothing for carbuncle fix",
    )

    use_fleischmann_limiter: bool = ProblemField(
        default=False,
        description="Enable Fleischmann low-Mach fix for HLLC solver",
    )

    # ========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def build_state(self) -> SimulationStateSpec:
        """
        Build the complete simulation state specification.

        This is where ALL computation happens. The method should:
        1. Use self._get_state_fields_from_config() to get base fields
        2. Compute problem-specific fields (resolution, bounds, etc.)
        3. Create and return SimulationStateSpec with source_config=self

        Returns:
            Complete, immutable simulation state ready for execution

        Example:
            >>> def build_state(self) -> SimulationStateSpec:
            ...     base_fields = self._get_state_fields_from_config()
            ...     return SimulationStateSpec(
            ...         **base_fields,
            ...         resolution=(self.resolution,),
            ...         bounds=[(0.0, 1.0)],
            ...         coord_system=CoordSystem.CARTESIAN,
            ...         regime=Regime.CLASSICAL,
            ...         adiabatic_index=5.0/3.0,
            ...         boundary_conditions=[BoundaryCondition.OUTFLOW],
            ...         source_config=self,
            ...     )
        """
        pass

    @abstractmethod
    def initial_primitive_state(self) -> InitialStateType:
        """
        Generate initial primitive variable state.

        Returns a generator function that yields primitive variables
        (density, velocity_x, [velocity_y, velocity_z,] pressure)
        for each cell in the grid.

        Returns:
            Generator function for initial conditions

        Example:
            >>> def initial_primitive_state(self) -> InitialStateType:
            ...     def gas_state():
            ...         for i in range(self.resolution):
            ...             x = i * dx
            ...             if x < 0.5:
            ...                 yield (1.0, 0.0, 1.0)  # left state
            ...             else:
            ...                 yield (0.125, 0.0, 0.1)  # right state
            ...     return gas_state
        """
        pass

    # ========================================================================
    # HOOK METHODS - Optional overrides for customization
    # ========================================================================

    def validate_physics(self) -> None:
        """
        Override this method to add problem-specific validation.

        Called automatically after field validation but before build_state().
        Raise ValueError or other exceptions if validation fails.

        Example:
            >>> def validate_physics(self) -> None:
            ...     if self.mach_number < 0:
            ...         raise ValueError("Mach number must be non-negative")
            ...     if self.temperature > self.max_temperature:
            ...         raise ValueError("Temperature exceeds physical limits")
        """
        pass

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @model_validator(mode="before")
    @classmethod
    def apply_order_parameter(cls, data: Any) -> Any:
        """
        Apply 'order' convenience parameter before field validation.

        If 'order' is specified, it OVERRIDES any explicit reconstruction
        or timestepping settings. This gives the order parameter highest
        priority for controlling the numerical scheme.

        Priority (highest to lowest):
        1. order parameter (if specified)
        2. Explicit reconstruction/timestepping
        3. Field defaults

        Args:
            data: Input data (dict or model instance)

        Returns:
            Potentially modified data dict
        """
        # Only process dict inputs (initial construction)
        if not isinstance(data, dict):
            return data

        order = data.get("order")

        if order == 1:
            # First-order scheme: PCM + RK1
            data["reconstruction"] = "pcm"
            data["timestepping"] = "rk1"
        elif order == 2:
            # Second-order scheme: PLM + RK2
            data["reconstruction"] = "plm"
            data["timestepping"] = "rk2"
        # If order is None, reconstruction/timestepping use their defaults

        return data

    @model_validator(mode="after")
    def validate_universal_constraints(self) -> "BaseProblemConfig":
        """
        Validate universal constraints that apply to all problems.

        This runs after field validation but before build_state().
        Checks things like time consistency, numerical parameters, etc.
        Also calls the problem-specific validate_physics() hook.

        Returns:
            Self (for method chaining)

        Raises:
            ValueError: If any universal constraint is violated
        """
        # Time consistency
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than "
                f"start_time ({self.start_time})"
            )

        # Checkpoint interval sanity
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval ({self.checkpoint_interval}) must be positive"
            )

        duration = self.end_time - self.start_time
        if self.checkpoint_interval > duration:
            raise ValueError(
                f"checkpoint_interval ({self.checkpoint_interval}) is larger than "
                f"simulation duration ({duration}). No checkpoints would be written!"
            )

        # CFL number bounds
        if not (0.0 < self.cfl_number <= 1.0):
            raise ValueError(
                f"cfl_number ({self.cfl_number}) must be in range (0, 1]"
            )

        # PLM theta bounds (only check if using PLM)
        if self.reconstruction == Reconstruction.PLM:
            if not (0.0 < self.plm_theta <= 2.0):
                raise ValueError(
                    f"plm_theta ({self.plm_theta}) must be in range (0, 2] "
                    f"when using PLM reconstruction"
                )

        # Call problem-specific validation hook
        self.validate_physics()

        return self

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_state_fields_from_config(self) -> dict[str, Any]:
        """
        Extract fields that should be forwarded to SimulationStateSpec.

        This uses introspection to automatically forward any field from this
        config that also exists in SimulationStateSpec. This means adding a
        new optional feature only requires changes in TWO places:
        1. Add field to SimulationStateSpec
        2. Add field to BaseProblemConfig (with default)

        The forwarding happens automatically, so existing problem implementations
        don't need changes.

        Returns:
            Dictionary of field names to values that exist in both configs

        Example:
            >>> base_fields = self._get_state_fields_from_config()
            >>> # base_fields contains: solver, cfl_number, plm_theta, etc.
            >>> state = SimulationStateSpec(**base_fields, ...)
        """
        # Import here to avoid circular dependency
        from .state import SimulationStateSpec

        state_field_names = SimulationStateSpec.model_fields.keys()
        config_values = self.model_dump()

        # Forward any config field that exists in state spec
        # Exclude 'order' since it's just a convenience parameter
        return {
            key: value
            for key, value in config_values.items()
            if key in state_field_names and key != "order"
        }

    @property
    def duration(self) -> float:
        """
        Total simulation duration.

        Returns:
            end_time - start_time
        """
        return self.end_time - self.start_time

    def get_output_path(self, filename: str) -> Path:
        """
        Construct full path for output file.

        Args:
            filename: Name of output file

        Returns:
            Full path: data_directory / filename

        Example:
            >>> config.get_output_path("results.h5")
            Path("data/sod_problem/results.h5")
        """
        return self.data_directory / filename

    def print_summary(self) -> None:
        """
        Print a formatted summary of the problem configuration.

        Displays:
        - Problem class name
        - Key parameters (time, output, solver)
        - Problem-specific parameters
        - Computed properties

        Useful for verification before running simulation.
        """
        print(f"\n{'=' * 80}")
        print(f"Problem Configuration: {self.__class__.__name__}")
        print(f"{'=' * 80}")

        # Time configuration
        print("\nTime Configuration:")
        print(f"  Start time:           {self.start_time:.6e}")
        print(f"  End time:             {self.end_time:.6e}")
        print(f"  Duration:             {self.duration:.6e}")
        print(f"  Checkpoint interval:  {self.checkpoint_interval:.6e}")

        # Output configuration
        print("\nOutput Configuration:")
        print(f"  Data directory:       {self.data_directory}")
        print(f"  Log to file:          {self.log_output}")
        print(f"  Starting index:       {self.checkpoint_index}")

        # Solver configuration
        print("\nSolver Configuration:")
        print(f"  Riemann solver:       {self.solver.value}")
        print(f"  Reconstruction:       {self.reconstruction.value}")
        print(f"  Time stepping:        {self.timestepping.value}")
        if self.order is not None:
            print(f"  Order:                {self.order}")

        # Numerical parameters
        print("\nNumerical Parameters:")
        print(f"  CFL number:           {self.cfl_number:.3f}")
        print(f"  PLM theta:            {self.plm_theta:.3f}")

        # Optional features (if enabled)
        features = []
        if self.use_quirk_smoothing:
            features.append("Quirk smoothing")
        if self.use_fleischmann_limiter:
            features.append("Fleischmann limiter")

        if features:
            print("\nEnabled Features:")
            for feature in features:
                print(f"  - {feature}")

        # Problem-specific fields (exclude base class fields)
        base_fields = set(BaseProblemConfig.model_fields.keys())
        problem_fields = {
            name: value
            for name, value in self.model_dump().items()
            if name not in base_fields and not name.startswith("_")
        }

        if problem_fields:
            print("\nProblem-Specific Parameters:")
            for name, value in sorted(problem_fields.items()):
                # Format value nicely
                if isinstance(value, float):
                    if abs(value) < 1e-3 or abs(value) > 1e3:
                        value_str = f"{value:.4e}"
                    else:
                        value_str = f"{value:.6f}"
                else:
                    value_str = str(value)

                print(f"  {name:.<30} {value_str}")

        print(f"{'=' * 80}\n")
