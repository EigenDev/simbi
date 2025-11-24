# =============================================================================
# problem.py
#
# base class for simbi simulation problems.
# combines configuration, cli handling, and checkpoint support in one place.
#
# usage:
#   class SodProblem(SimbiProblem):
#       resolution: int = ProblemParam(1000, cli=True)
#       # ...
#       def initial_primitive_state(self) -> InitialStateType:
#           ...
# =============================================================================
from __future__ import annotations

import argparse
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from simbi.types.bodies import BodySystemConfig, ImmersedBodyConfig

# re-export types that problem authors need
from simbi.types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Reconstruction,
    Regime,
    Solver,
    SubCycleMode,
    TimeStepping,
)
from simbi.types.typing import (
    ExpressionDict,
    InitialStateType,
)

from .param import ProblemParam, get_param_metadata


class SimbiProblem(BaseModel):
    """
    base class for simbi simulation problems.

    subclasses define physics problems by:
    1. setting required fields (resolution, bounds, etc.)
    2. implementing initial_primitive_state()
    3. optionally overriding computed properties for custom physics
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # class-level storage for cli parser
    _cli_parser: ClassVar[Optional[argparse.ArgumentParser]] = None

    # =========================================================================
    # required fields - must be provided by subclass or user
    # =========================================================================
    resolution: Union[int, Sequence[int], NDArray[np.int64]] = ProblemParam(
        ..., description="grid resolution"
    )
    coord_system: CoordSystem = ProblemParam(
        ..., description="coordinate system"
    )
    regime: Regime = ProblemParam(..., description="physics regime")
    bounds: Sequence[Sequence[float]] = ProblemParam(
        ..., description="domain bounds"
    )
    adiabatic_index: float = ProblemParam(..., description="adiabatic index")

    # =========================================================================
    # simulation control - checkpoint_safe=True (can override on restart)
    # =========================================================================
    end_time: float = ProblemParam(
        1.0, cli=True, checkpoint_safe=True, description="simulation end time"
    )
    cfl_number: float = ProblemParam(
        0.1, cli=True, checkpoint_safe=True, description="cfl condition number"
    )
    start_time: float = ProblemParam(
        0.0, checkpoint_safe=True, description="simulation start time"
    )

    # =========================================================================
    # output settings - checkpoint_safe=True
    # =========================================================================
    data_directory: Path = ProblemParam(
        Path("data/"),
        cli=True,
        checkpoint_safe=True,
        description="output directory",
    )
    checkpoint_interval: float = ProblemParam(
        0.1, cli=True, checkpoint_safe=True, description="checkpoint interval"
    )
    checkpoint_index: int = ProblemParam(
        0, checkpoint_safe=True, description="checkpoint index for resuming"
    )
    checkpoint_file: Optional[str] = ProblemParam(
        None,
        cli=True,
        checkpoint_safe=True,
        description="checkpoint file to resume from",
    )
    log_output: bool = ProblemParam(
        False, checkpoint_safe=True, description="enable logging to file"
    )
    log_checkpoints_tuple: tuple[bool, int] = ProblemParam(
        (False, 0),
        checkpoint_safe=True,
        description="logarithmic output (enabled, num)",
    )

    # =========================================================================
    # numerics - checkpoint_safe=False (must match checkpoint)
    # =========================================================================
    solver: Solver = ProblemParam(
        Solver.HLLE, cli=True, description="numerical solver"
    )
    reconstruction: Reconstruction = ProblemParam(
        Reconstruction.PLM, cli=True, description="spatial reconstruction"
    )
    timestepping: TimeStepping = ProblemParam(
        TimeStepping.RK2, cli=True, description="time stepping method"
    )
    order: Optional[int] = ProblemParam(
        None, cli=True, description="order of accuracy (1 or 2)"
    )
    plm_theta: float = ProblemParam(
        1.5, cli=True, description="plm theta parameter"
    )
    boundary_conditions: Union[
        BoundaryCondition, Sequence[BoundaryCondition]
    ] = ProblemParam("outflow", cli=True, description="boundary conditions")

    # =========================================================================
    # mesh spacing
    # =========================================================================
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="x1 cell spacing"
    )
    x2_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="x2 cell spacing"
    )
    x3_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="x3 cell spacing"
    )

    # =========================================================================
    # solver options
    # =========================================================================
    use_quirk_smoothing: bool = ProblemParam(
        False, cli=True, description="use quirk smoothing"
    )
    use_fleischmann_limiter: bool = ProblemParam(
        False, cli=True, description="use fleischmann low-mach fix"
    )

    # =========================================================================
    # fmr (fixed mesh refinement)
    # =========================================================================
    fmr_enabled: bool = ProblemParam(False, description="enable fmr")
    fmr_max_levels: int = ProblemParam(1, description="max refinement levels")
    fmr_regions: list[list[float]] = ProblemParam(
        [], description="refinement regions"
    )
    fmr_ratios: list[int] = ProblemParam([], description="refinement ratios")
    fmr_substeps: list[int] = ProblemParam(
        [1], description="substeps per level"
    )
    fmr_subcycling_mode: SubCycleMode = ProblemParam(
        SubCycleMode.NONE, description="subcycling mode"
    )

    # =========================================================================
    # computed properties
    # =========================================================================
    @computed_field
    @property
    def dimensionality(self) -> int:
        """compute dimensionality from resolution."""
        if self.regime in [Regime.SRMHD]:
            return 3
        if isinstance(self.resolution, int):
            return 1
        return sum(int(d > 1) for d in self.resolution)

    @computed_field
    @property
    def is_mhd(self) -> bool:
        """check if simulation involves mhd."""
        return self.regime in [Regime.SRMHD]

    @computed_field
    @property
    def isothermal(self) -> bool:
        """check if simulation is isothermal."""
        return self.adiabatic_index == 1.0

    @computed_field
    @property
    def nvars(self) -> int:
        """number of variables based on regime and dimensionality."""
        if self.is_mhd:
            return 9
        return self.dimensionality + 3

    @computed_field
    @property
    def is_relativistic(self) -> bool:
        """check if simulation is relativistic."""
        return self.regime in [Regime.SRHD, Regime.SRMHD]

    @computed_field
    @property
    def mesh_motion(self) -> bool:
        """check if simulation involves mesh motion."""
        if self.scale_factor is None or self.scale_factor_derivative is None:
            return False
        return self.scale_factor_derivative(1) / self.scale_factor(1) != 0

    @computed_field
    @property
    def is_homologous(self) -> bool:
        """check if simulation is homologous."""
        return self.mesh_motion and self.coord_system in [CoordSystem.SPHERICAL]

    @computed_field
    @property
    def dlogt(self) -> float:
        """logarithmic time spacing."""
        log_enabled, num_outputs = self.log_checkpoints_tuple
        if log_enabled and num_outputs > 0:
            import math

            return math.log10(self.end_time / self.start_time) / num_outputs
        return 0.0

    # =========================================================================
    # overridable computed properties for custom physics
    # =========================================================================
    @computed_field
    @property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        """scale factor for mesh motion."""
        return None

    @computed_field
    @property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
        """derivative of scale factor."""
        return None

    @computed_field
    @property
    def ambient_sound_speed(self) -> float:
        """ambient sound speed for isothermal simulations."""
        return 0.0

    @computed_field
    @property
    def shakura_sunyaev_alpha(self) -> float:
        """shakura-sunyaev alpha for accretion disks."""
        return 0.0

    @computed_field
    @property
    def viscosity(self) -> float:
        """viscosity coefficient."""
        return 0.0

    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        """check if locally isothermal."""
        return False

    # =========================================================================
    # boundary expressions (override in subclass for custom bcs)
    # =========================================================================
    @computed_field
    @property
    def buffer_parameters(self) -> dict[str, float]:
        return {}

    @computed_field
    @property
    def bx1_inner_expressions(self) -> ExpressionDict:
        return {}

    @computed_field
    @property
    def bx1_outer_expressions(self) -> ExpressionDict:
        return {}

    @computed_field
    @property
    def bx2_inner_expressions(self) -> ExpressionDict:
        return {}

    @computed_field
    @property
    def bx2_outer_expressions(self) -> ExpressionDict:
        return {}

    @computed_field
    @property
    def bx3_inner_expressions(self) -> ExpressionDict:
        return {}

    @computed_field
    @property
    def bx3_outer_expressions(self) -> ExpressionDict:
        return {}

    # =========================================================================
    # source terms (override in subclass)
    # =========================================================================
    @computed_field
    @property
    def hydro_source_expressions(self) -> ExpressionDict:
        return {}

    @computed_field
    @property
    def gravity_source_expressions(self) -> ExpressionDict:
        return {}

    # =========================================================================
    # body physics (override in subclass)
    # =========================================================================
    @computed_field
    @property
    def body_system(self) -> Optional[BodySystemConfig]:
        return None

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return []

    # =========================================================================
    # abstract method - must be implemented by subclass
    # =========================================================================
    @abstractmethod
    def initial_primitive_state(self) -> InitialStateType:
        """
        generate initial primitive state for the simulation.

        returns:
            for hydro: a callable that returns a generator yielding (rho, v1, [v2, v3,] p)
            for mhd: a tuple of (gas_gen, bx_gen, by_gen, bz_gen)
        """
        raise NotImplementedError(
            "subclasses must implement initial_primitive_state"
        )

    # =========================================================================
    # validators
    # =========================================================================
    @model_validator(mode="after")
    def _enforce_order_settings(self) -> SimbiProblem:
        """enforce reconstruction and timestepping based on order parameter."""
        if self.order is not None:
            if self.order == 1:
                object.__setattr__(self, "reconstruction", Reconstruction.PCM)
                object.__setattr__(self, "timestepping", TimeStepping.RK1)
            elif self.order == 2:
                object.__setattr__(self, "reconstruction", Reconstruction.PLM)
                object.__setattr__(self, "timestepping", TimeStepping.RK2)
            else:
                raise ValueError("order must be 1 or 2")
        return self

    @model_validator(mode="after")
    def _validate_isothermal(self) -> SimbiProblem:
        """validate isothermal simulation settings."""
        if (
            self.isothermal
            and self.ambient_sound_speed <= 0
            and not self.locally_isothermal
        ):
            raise ValueError(
                "ambient_sound_speed must be positive for isothermal simulations "
                "unless locally_isothermal is True"
            )
        return self

    @model_validator(mode="after")
    def _validate_plm_theta(self) -> SimbiProblem:
        """validate plm theta parameter."""
        if self.reconstruction == Reconstruction.PLM and not (
            0.0 < self.plm_theta <= 2.0
        ):
            raise ValueError("plm_theta must be in (0, 2] when using PLM")
        return self

    @model_validator(mode="after")
    def _validate_fmr(self) -> SimbiProblem:
        """validate fmr configuration."""
        if not self.fmr_enabled:
            return self

        if not self.fmr_regions:
            raise ValueError("fmr_regions required when fmr_enabled=True")

        expected_levels = self.fmr_max_levels - 1
        if len(self.fmr_regions) != expected_levels:
            raise ValueError(
                f"expected {expected_levels} fmr_regions, got {len(self.fmr_regions)}"
            )

        if len(self.fmr_ratios) != expected_levels:
            raise ValueError(
                f"expected {expected_levels} fmr_ratios, got {len(self.fmr_ratios)}"
            )

        expected_coords = 2 * self.dimensionality
        for ii, region in enumerate(self.fmr_regions):
            if len(region) != expected_coords:
                raise ValueError(
                    f"fmr_region[{ii}] has {len(region)} coords, expected {expected_coords}"
                )

        if self.fmr_subcycling_mode == SubCycleMode.ADAPTIVE:
            if len(self.fmr_substeps) != expected_levels:
                raise ValueError(
                    "fmr_substeps must match fmr_max_levels - 1 for adaptive mode"
                )

        return self

    # =========================================================================
    # cli integration
    # =========================================================================
    @classmethod
    def setup_cli(cls, parser: argparse.ArgumentParser) -> None:
        """register cli parameters from fields with cli=True."""
        cls._cli_parser = parser
        group = parser.add_argument_group(f"{cls.__name__} parameters")

        for field_name, field_info in cls.model_fields.items():
            if field_name.startswith("_"):
                continue

            metadata = get_param_metadata(field_info)
            if not metadata.cli:
                continue

            cli_name = metadata.cli_name or field_name.replace("_", "-")
            kwargs: dict[str, Any] = {
                "dest": field_name,
                "help": field_info.description or f"set {field_name}",
            }

            if field_info.default is not None and field_info.default is not ...:
                kwargs["default"] = field_info.default

            cls._add_type_info(kwargs, field_info)

            try:
                group.add_argument(f"--{cli_name}", **kwargs)
            except argparse.ArgumentError:
                pass  # already registered

    @classmethod
    def _add_type_info(cls, kwargs: dict[str, Any], field_info: Any) -> None:
        """add type information to argparse kwargs."""
        from typing import Union, get_args, get_origin

        if field_info.annotation is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
            return

        simple_types = {str, int, float, Path}
        if field_info.annotation in simple_types:
            kwargs["type"] = field_info.annotation
            return

        origin = get_origin(field_info.annotation)
        if origin is Union:
            args = get_args(field_info.annotation)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1 and non_none[0] in simple_types:
                kwargs["type"] = non_none[0]

    @classmethod
    def from_cli(
        cls,
        argv: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> SimbiProblem:
        """create instance from cli arguments."""
        # create a dedicated parser for problem-specific args only
        parser = argparse.ArgumentParser(add_help=False)
        cls.setup_cli(parser)

        # parse into existing namespace (if provided)
        parsed, _ = parser.parse_known_args(argv, namespace)
        return cls.from_namespace(parsed)

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> SimbiProblem:
        """create instance from parsed namespace."""
        data = {}
        for field_name in cls.model_fields:
            if hasattr(namespace, field_name):
                value = getattr(namespace, field_name)
                if value is not None:
                    data[field_name] = value
        return cls(**data)

    # =========================================================================
    # checkpoint support
    # =========================================================================
    def get_checkpoint_immutable_fields(self) -> set[str]:
        """get field names that cannot be overridden when loading from checkpoint."""
        immutable = set()
        for field_name, field_info in self.model_fields.items():
            metadata = get_param_metadata(field_info)
            if not metadata.checkpoint_safe:
                immutable.add(field_name)
        return immutable

    def get_checkpoint_safe_fields(self) -> set[str]:
        """get field names that can be overridden when loading from checkpoint."""
        safe = set()
        for field_name, field_info in self.model_fields.items():
            metadata = get_param_metadata(field_info)
            if metadata.checkpoint_safe:
                safe.add(field_name)
        return safe
