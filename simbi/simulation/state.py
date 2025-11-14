"""
Simulation state specification - the fixed schema that all problems produce.

This module defines the complete simulation state that:
- All problem configs must produce via build_state()
- Gets serialized to C++ for execution
- Gets saved to/loaded from checkpoints by C++
- Is immutable once created

The state spec represents the "execution contract" - everything the simulation
engine needs to run, with no ambiguity or partial state.
"""

import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field

from simbi.core.types.bodies import BodySystemConfig, ImmersedBodyConfig
from simbi.core.types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Reconstruction,
    Regime,
    Solver,
    TimeStepping,
)
from simbi.core.types.typing import ExpressionDict

from .interfaces import ProblemInterface


class SimulationStateSpec(BaseModel):
    """
    Complete simulation state specification.

    This is the FIXED schema that:
    - All problem configs must produce via build_state()
    - Gets serialized to C++ for execution
    - Gets saved to/loaded from checkpoints by C++
    - Is immutable once created

    Every field here must be fully resolved - no computation happens
    in this class beyond simple @computed_field derivations.

    Construction paths:
    1. Normal: problem_config.build_state() → creates fresh state
    2. Resume: SimulationStateSpec.from_checkpoint() → loads from checkpoint
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable after creation
        arbitrary_types_allowed=True,  # For callables, custom types
    )

    # ========================================================================
    # REQUIRED FIELDS - Every simulation must specify these
    # ========================================================================

    # Grid configuration
    resolution: tuple[int, ...] = Field(
        ...,
        description="Grid resolution (nx,) or (nx, ny) or (nx, ny, nz)",
    )

    bounds: Sequence[Sequence[float]] = Field(
        ...,
        description="Domain boundaries [(xmin, xmax), (ymin, ymax), (zmin, zmax)]",
    )

    coord_system: CoordSystem = Field(
        ...,
        description="Coordinate system (Cartesian, cylindrical, spherical)",
    )

    # Physics configuration
    regime: Regime = Field(
        ...,
        description="Physics regime (newtonian, SRHD, SRMHD)",
    )

    adiabatic_index: float = Field(
        ...,
        description="Adiabatic index (gamma)",
        gt=0.0,  # Must be positive
    )

    # Solver configuration
    solver: Solver = Field(
        ...,
        description="Riemann solver (HLLE, HLLC, etc.)",
    )

    reconstruction: Reconstruction = Field(
        ...,
        description="Spatial reconstruction method (PCM, PLM, etc.)",
    )

    timestepping: TimeStepping = Field(
        ...,
        description="Time integration method (RK1, RK2, etc.)",
    )

    # Time configuration
    start_time: float = Field(
        ...,
        description="Simulation start time",
    )

    end_time: float = Field(
        ...,
        description="Simulation end time",
    )

    checkpoint_interval: float = Field(
        ...,
        description="Time between checkpoints",
        gt=0.0,  # Must be positive
    )

    # Boundary conditions
    boundary_conditions: list[BoundaryCondition] = Field(
        ...,
        description="Boundary conditions for each dimension",
    )

    # ========================================================================
    # OPTIONAL FIELDS - Problem-specific features with sensible defaults
    # ========================================================================

    # Grid spacing
    x1_spacing: CellSpacing = Field(
        default=CellSpacing.LINEAR,
        description="Cell spacing in x1 direction",
    )

    x2_spacing: CellSpacing = Field(
        default=CellSpacing.LINEAR,
        description="Cell spacing in x2 direction",
    )

    x3_spacing: CellSpacing = Field(
        default=CellSpacing.LINEAR,
        description="Cell spacing in x3 direction",
    )

    # Numerical parameters
    cfl_number: float = Field(
        default=0.4,
        description="CFL number for time step control",
        gt=0.0,
        le=1.0,
    )

    plm_theta: float = Field(
        default=1.5,
        description="PLM slope limiter parameter",
        gt=0.0,
        le=2.0,
    )

    # Solver options
    use_quirk_smoothing: bool = Field(
        default=False,
        description="Use Quirk smoothing for carbuncle fix",
    )

    use_fleischmann_limiter: bool = Field(
        default=False,
        description="Use Fleischmann low-Mach fix for HLLC solver",
    )

    # Fixed Mesh Refinement (FMR)
    fmr_enabled: bool = Field(
        default=False,
        description="Whether fixed mesh refinement is enabled",
    )

    fmr_max_levels: int = Field(
        default=1,
        description="Maximum number of refinement levels",
        ge=1,
    )

    fmr_regions: list[list[float]] = Field(
        default_factory=list,
        description="Refinement regions for each level (nested)",
    )

    fmr_ratios: list[int] = Field(
        default_factory=list,
        description="Refinement ratios for each level relative to parent",
    )

    # Body physics
    body_system: Optional[BodySystemConfig] = Field(
        default=None,
        description="Gravitational body system configuration (if any)",
    )

    immersed_bodies: list[ImmersedBodyConfig] = Field(
        default_factory=list,
        description="Immersed boundary bodies (if any)",
    )

    # Thermodynamics
    ambient_sound_speed: float = Field(
        default=0.0,
        description="Ambient sound speed (for isothermal simulations)",
        ge=0.0,
    )

    viscosity: float = Field(
        default=0.0,
        description="Kinematic viscosity coefficient",
        ge=0.0,
    )

    shakura_sunyaev_alpha: float = Field(
        default=0.0,
        description="Shakura-Sunyaev alpha viscosity parameter",
        ge=0.0,
    )

    # Mesh motion (for cosmological or expanding flows)
    scale_factor: Optional[Callable[[float], float]] = Field(
        default=None,
        description="Scale factor function a(t) for mesh motion",
    )

    scale_factor_derivative: Optional[Callable[[float], float]] = Field(
        default=None,
        description="Time derivative of scale factor da/dt",
    )

    # Source terms and boundary expressions (serialized expression trees)
    hydro_source_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Hydrodynamic source term expressions",
    )

    gravity_source_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Gravity source term expressions",
    )

    # Boundary expressions
    bx1_inner_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Inner x1 boundary expressions",
    )

    bx1_outer_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Outer x1 boundary expressions",
    )

    bx2_inner_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Inner x2 boundary expressions",
    )

    bx2_outer_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Outer x2 boundary expressions",
    )

    bx3_inner_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Inner x3 boundary expressions",
    )

    bx3_outer_expressions: ExpressionDict = Field(
        default_factory=dict,
        description="Outer x3 boundary expressions",
    )

    buffer_parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Buffer zone damping parameters",
    )

    # I/O configuration
    data_directory: Path = Field(
        default=Path("data/"),
        description="Output directory for simulation data",
    )

    log_output: bool = Field(
        default=False,
        description="Enable logging to file",
    )

    log_checkpoints_tuple: tuple[bool, int] = Field(
        default=(False, 0),
        description="Logarithmic checkpoint settings (enabled, num_outputs)",
    )
    checkpoint_file: str = Field(
        default="",
        description="Checkpoint file to load at start (if any)",
    )

    # Checkpoint metadata
    checkpoint_index: int = Field(
        default=0,
        description="Current checkpoint index",
        ge=0,
    )

    # ========================================================================
    # BACK-REFERENCE (runtime only, never serialized to C++)
    # ========================================================================

    source_config: Optional[Any] = Field(
        default=None,
        exclude=True,  # Never serialize to C++
        validation_alias=None,
        description="Original problem config that produced this state (runtime only)",
    )

    # ========================================================================
    # COMPUTED FIELDS (simple derivations only - no complex logic)
    # ========================================================================

    @computed_field
    @property
    def dimensionality(self) -> int:
        """
        Compute dimensionality from resolution.

        Returns:
            1, 2, or 3 depending on which dimensions are active
        """
        if self.regime in [Regime.SRMHD]:
            return 3  # MHD is always 3D
        return sum(int(d > 1) for d in self.resolution)

    @computed_field
    @property
    def nvars(self) -> int:
        """
        Number of conserved variables.

        Returns:
            Number of variables (density + momentum components + energy, or MHD)
        """
        if self.regime in [Regime.SRMHD]:
            return 9  # MHD has 9 primary variables
        # Hydro: density + momentum (per dimension) + energy
        return self.dimensionality + 2

    @computed_field
    @property
    def is_mhd(self) -> bool:
        """Whether this is an MHD simulation."""
        return self.regime in [Regime.SRMHD]

    @computed_field
    @property
    def is_relativistic(self) -> bool:
        """Whether this is a relativistic simulation."""
        return self.regime in [Regime.SRHD, Regime.SRMHD]

    @computed_field
    @property
    def isothermal(self) -> bool:
        """Whether this is an isothermal simulation (gamma = 1)."""
        return self.adiabatic_index == 1.0

    @computed_field
    @property
    def mesh_motion(self) -> bool:
        """
        Whether mesh motion is enabled.

        Returns:
            True if scale factor has non-zero time derivative
        """
        if self.scale_factor is None or self.scale_factor_derivative is None:
            return False
        # Check if derivative is non-zero at t=1
        try:
            return (
                abs(self.scale_factor_derivative(1.0) / self.scale_factor(1.0))
                > 1e-12
            )
        except (ZeroDivisionError, TypeError):
            return False

    @computed_field
    @property
    def is_homologous(self) -> bool:
        """Whether this is homologous expansion (spherical + mesh motion)."""
        return self.mesh_motion and self.coord_system == CoordSystem.SPHERICAL

    @computed_field
    @property
    def duration(self) -> float:
        """Total simulation duration (end_time - start_time)."""
        return self.end_time - self.start_time

    @computed_field
    @property
    def dlogt(self) -> float:
        """
        Logarithmic time spacing for checkpoints.

        Returns:
            Delta log(t) if logarithmic checkpointing is enabled, else 0
        """
        log_enabled, num_outputs = self.log_checkpoints_tuple
        if log_enabled and num_outputs > 0 and self.start_time > 0:
            return math.log10(self.end_time / self.start_time) / num_outputs
        return 0.0

    # ========================================================================
    # FACTORY METHODS
    # ========================================================================

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_metadata: dict,
        user_config: ProblemInterface,
    ) -> "SimulationStateSpec":
        """
        Create simulation state from checkpoint metadata and user config.

        This is used for resuming simulations. The checkpoint provides the
        immutable physics/grid state, while the user config provides runtime
        parameters that can be safely overridden (e.g., end_time, cfl_number).

        Args:
            checkpoint_metadata: Metadata loaded from C++ checkpoint file
            user_config: User's problem config with resume parameters

        Returns:
            New simulation state ready for resume

        Example:
            >>> checkpoint_meta = load_checkpoint_metadata("checkpoint_060.h5")
            >>> new_config = BinaryBondi(xi_bondi=2.0, end_time=120.0)
            >>> state = SimulationStateSpec.from_checkpoint(checkpoint_meta, new_config)
        """
        # Extract physics from checkpoint (immutable - can't change these)
        resolution = tuple(checkpoint_metadata["resolution"])
        bounds = checkpoint_metadata["bounds"]
        coord_system = CoordSystem(checkpoint_metadata["coord_system"])
        regime = Regime(checkpoint_metadata["regime"])
        adiabatic_index = float(checkpoint_metadata["adiabatic_index"])
        solver = Solver(checkpoint_metadata["solver"])
        reconstruction = Reconstruction(checkpoint_metadata["reconstruction"])
        timestepping = TimeStepping(checkpoint_metadata["timestepping"])

        # Boundary conditions
        boundary_conditions = [
            BoundaryCondition(bc)
            for bc in checkpoint_metadata["boundary_conditions"]
        ]

        # Time: start from checkpoint time
        start_time = float(checkpoint_metadata["time"])

        # User can override these runtime parameters
        end_time = user_config.end_time
        checkpoint_interval = user_config.checkpoint_interval
        cfl_number = user_config.cfl_number
        plm_theta = user_config.plm_theta
        data_directory = user_config.data_directory

        # Optional checkpoint fields
        x1_spacing = CellSpacing(
            checkpoint_metadata.get("x1_spacing", "linear")
        )
        x2_spacing = CellSpacing(
            checkpoint_metadata.get("x2_spacing", "linear")
        )
        x3_spacing = CellSpacing(
            checkpoint_metadata.get("x3_spacing", "linear")
        )

        fmr_enabled = checkpoint_metadata.get("fmr_enabled", False)
        fmr_max_levels = checkpoint_metadata.get("fmr_max_levels", 1)
        fmr_regions = checkpoint_metadata.get("fmr_regions", [])
        fmr_ratios = checkpoint_metadata.get("fmr_ratios", [])

        checkpoint_index = checkpoint_metadata.get("checkpoint_index", 0)

        # Create state with checkpoint physics + user overrides
        return cls(
            # From checkpoint (physics - immutable)
            resolution=resolution,
            bounds=bounds,
            coord_system=coord_system,
            regime=regime,
            adiabatic_index=adiabatic_index,
            solver=solver,
            reconstruction=reconstruction,
            timestepping=timestepping,
            boundary_conditions=boundary_conditions,
            x1_spacing=x1_spacing,
            x2_spacing=x2_spacing,
            x3_spacing=x3_spacing,
            # Time configuration
            start_time=start_time,
            end_time=end_time,
            checkpoint_interval=checkpoint_interval,
            checkpoint_index=checkpoint_index + 1,  # Increment for next run
            # From user config (runtime - mutable)
            cfl_number=cfl_number,
            plm_theta=plm_theta,
            data_directory=data_directory,
            # FMR from checkpoint
            fmr_enabled=fmr_enabled,
            fmr_max_levels=fmr_max_levels,
            fmr_regions=fmr_regions,
            fmr_ratios=fmr_ratios,
            # Attach the config for introspection
            source_config=user_config,
        )
