# =============================================================================
# problem.py
#
# base class for simbi simulation problems.
# combines configuration, cli handling, and checkpoint support in one place.
#
# usage:
#   class SodProblem(SimbiProblem):
#       resolution: Annotated[int, ProblemParam(1000, cli=True)]
#       # ...
#       def initial_primitive_state(self) -> InitialStateType:
#           ...
# =============================================================================
from __future__ import annotations

import argparse
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    computed_field,
    model_validator,
)

from simbi.types.bodies import BodySystemConfig, ImmersedBodyConfig

# re-export types that problem authors need
from simbi.types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Reconstruction,
    RefinementMode,
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


class ConfigError(Exception):
    """a user-facing configuration error.

    carries a pre-formatted, traceback-free message; the cli boundary prints it
    verbatim and exits non-zero instead of dumping a pydantic stack trace.
    """


def _to_enum(enum_type: Any, value: str, field_name: str) -> Any:
    """look up an enum member by case-insensitive name, raising a ValueError that
    names the field and LISTS the valid choices. a plain `enum_type[name]` raises
    a bare KeyError that pydantic does NOT wrap — it escapes as a raw traceback.
    """
    try:
        return enum_type[value.upper()]
    except KeyError:
        choices = ", ".join(e.name.lower() for e in enum_type)
        raise ValueError(
            f"{field_name}: '{value}' is not a valid choice; pick one of: {choices}"
        ) from None


def _error_hint(field: str, err: dict) -> Optional[str]:
    """a targeted, actionable hint for a single pydantic error, or None."""
    etype = err.get("type", "")
    flag = "--" + field.replace("_", "-")
    if field == "resolution":
        return (
            f"pass comma-separated axis sizes, e.g. `{flag} 256,256`; trailing "
            "axes default to 1, so a 2d run needs only nx,ny"
        )
    if field == "bounds":
        return f"pass `{flag} [[x0,x1],[y0,y1]]` — one [min,max] pair per axis"
    if etype == "missing":
        return f"this field is required — pass `{flag} <value>`"
    if etype in ("int_parsing", "float_parsing", "int_from_float"):
        return f"expected a number — check the value passed to `{flag}`"
    if etype in ("too_long", "too_short"):
        ctx = err.get("ctx", {})
        want = ctx.get("actual_length") or ctx.get("field_type")
        return f"wrong number of values{f' (need {want})' if want else ''} for `{flag}`"
    return None


def _humanize_validation_error(setup_name: str, exc: ValidationError) -> str:
    """render a pydantic ValidationError as a clinical, actionable report:
    one line per offending field with the bad value and a fix hint."""
    n = exc.error_count()
    header = f"{n} invalid setting{'s' if n != 1 else ''} for '{setup_name}':"
    lines = [header, ""]
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else ""
        # strip pydantic's wrapping prefix on validator-raised errors.
        msg = err["msg"]
        for prefix in ("Value error, ", "Assertion failed, "):
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
                break
        # field-scoped errors get a `field:` prefix; model-level errors (raised
        # in the before-validator) already name the field in the message.
        loc = ".".join(str(p) for p in err["loc"])
        line = f"  {loc}: {msg}" if loc else f"  {msg}"
        # echo the offending value only for FIELD-scoped errors; a model-level
        # error carries the whole input dict as `input`, which is noise.
        if "input" in err and err["loc"] and not isinstance(err["input"], dict):
            line += f"  (got: {err['input']!r})"
        lines.append(line)
        hint = _error_hint(field, err)
        if hint:
            lines.append(f"      hint: {hint}")
    lines.append("")
    lines.append("run with `--info` to list every parameter and its default.")
    return "\n".join(lines)


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
    resolution: Annotated[
        Union[int, Sequence[int], NDArray[np.int64]],
        ProblemParam(..., description="grid resolution"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(..., description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(..., description="physics regime")]
    bounds: Annotated[
        Sequence[Sequence[float]],
        ProblemParam(..., description="domain bounds"),
    ]
    adiabatic_index: Annotated[
        Optional[float],
        ProblemParam(
            None,
            ge=1.0,
            le=2.0,
            description="adiabatic index; required for energy-bearing regimes, "
            "irrelevant for isothermal ones (which use sound_speed)",
        ),
    ]
    sound_speed: Annotated[
        Optional[float],
        ProblemParam(
            None,
            gt=0.0,
            description="constant isothermal sound speed; required for isothermal "
            "regimes unless locally_isothermal (then cs^2(x) is derived per cell)",
        ),
    ]

    # =========================================================================
    # simulation control - checkpoint_safe=True (can override on restart)
    # =========================================================================
    end_time: Annotated[
        float,
        ProblemParam(
            1.0,
            gt=0.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]
    cfl_number: Annotated[
        float,
        ProblemParam(
            0.1,
            gt=0.0,
            le=1.0,
            cli=True,
            checkpoint_safe=True,
            description="cfl condition number",
        ),
    ]
    start_time: Annotated[
        float,
        ProblemParam(
            0.0,
            ge=0.0,
            checkpoint_safe=True,
            description="simulation start time",
        ),
    ]

    # =========================================================================
    # output settings - checkpoint_safe=True
    # =========================================================================
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            0.1,
            gt=0.0,
            cli=True,
            checkpoint_safe=True,
            description="checkpoint interval",
        ),
    ]
    time_unit: Annotated[
        float,
        ProblemParam(
            1.0,
            gt=0.0,
            cli=True,
            checkpoint_safe=True,
            description="natural time unit (code units per unit); checkpoint "
            "names + the live display report time / time_unit. e.g. set to the "
            "orbital period for a binary so output reads in orbits",
        ),
    ]
    time_unit_label: Annotated[
        str,
        ProblemParam(
            "t",
            cli=True,
            checkpoint_safe=True,
            description="label for the natural time unit (e.g. 'orbit'); 't' "
            "means code units and is omitted from checkpoint names",
        ),
    ]
    diagnostic_interval: Annotated[
        float,
        ProblemParam(
            0.0,
            ge=0.0,
            cli=True,
            checkpoint_safe=True,
            description="body-diagnostics output cadence (natural units, "
            "× time_unit). 0 disables. only emitted when the problem has "
            "immersed bodies; writes <data_dir>diagnostics.dat",
        ),
    ]
    checkpoint_index: Annotated[
        int,
        ProblemParam(
            0,
            ge=0,
            checkpoint_safe=True,
            description="checkpoint index for resuming",
        ),
    ]
    checkpoint_file: Annotated[
        Optional[str | Path],
        ProblemParam(
            None,
            cli=True,
            checkpoint_safe=True,
            description="checkpoint file to resume from",
        ),
    ]
    log_output: Annotated[
        bool,
        ProblemParam(
            False, checkpoint_safe=True, description="enable logging to file"
        ),
    ]
    log_checkpoints_tuple: Annotated[
        tuple[bool, int],
        ProblemParam(
            (False, 0),
            checkpoint_safe=True,
            description="logarithmic output (enabled, num)",
        ),
    ]

    # =========================================================================
    # numerics - checkpoint_safe=False (must match checkpoint)
    # =========================================================================
    solver: Annotated[
        Solver,
        ProblemParam(Solver.HLLE, cli=True, description="numerical solver"),
    ]
    reconstruction: Annotated[
        Reconstruction,
        ProblemParam(
            Reconstruction.PLM, cli=True, description="spatial reconstruction"
        ),
    ]
    timestepping: Annotated[
        TimeStepping,
        ProblemParam(
            TimeStepping.RK2, cli=True, description="time stepping method"
        ),
    ]
    order: Annotated[
        Optional[int],
        ProblemParam(
            None, ge=1, le=2, cli=True, description="order of accuracy (1 or 2)"
        ),
    ]
    plm_theta: Annotated[
        float,
        ProblemParam(
            1.5, gt=0.0, le=2.0, cli=True, description="plm theta parameter"
        ),
    ]
    boundary_conditions: Annotated[
        Union[BoundaryCondition, Sequence[BoundaryCondition]],
        ProblemParam(
            BoundaryCondition.OUTFLOW,
            cli=True,
            description="boundary conditions",
        ),
    ]

    # =========================================================================
    # mesh spacing
    # =========================================================================
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="x1 cell spacing"),
    ]
    x2_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="x2 cell spacing"),
    ]
    x3_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="x3 cell spacing"),
    ]

    # =========================================================================
    # solver options
    # =========================================================================
    use_quirk_smoothing: Annotated[
        bool, ProblemParam(False, cli=True, description="use quirk smoothing")
    ]
    use_fleischmann_limiter: Annotated[
        bool,
        ProblemParam(
            False, cli=True, description="use fleischmann low-mach fix"
        ),
    ]

    # =========================================================================
    # refinement / fmr settings (fixed mesh refinement only for now)
    # =========================================================================
    refinement_enabled: Annotated[
        bool, ProblemParam(False, description="enable mesh refinement")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(1, ge=1, description="max refinement levels")
    ]
    refinement_regions: Annotated[
        list[list[float]], ProblemParam([], description="refinement regions")
    ]
    refinement_ratios: Annotated[
        list[int], ProblemParam([], description="refinement ratios")
    ]
    refinement_substeps: Annotated[
        list[int], ProblemParam([1], description="substeps per level")
    ]
    refinement_subcycling_mode: Annotated[
        SubCycleMode,
        ProblemParam(SubCycleMode.NONE, description="subcycling mode"),
    ]
    refinement_mode: Annotated[
        RefinementMode,
        ProblemParam(RefinementMode.FIXED, description="refinement mode"),
    ]

    # =========================================================================
    # computed properties
    # =========================================================================
    @computed_field
    @property
    def dimensionality(self) -> int:
        """compute dimensionality from resolution.

        mhd is genuine 1.5d / 2.5d / 3d (spatial d in {1,2,3}, vector dof=3), so
        it no longer forces 3d — the spatial dimensionality is read from the
        resolution like every other regime.
        """
        if self.resolution is None:
            return 3  # default assumption for uninitialized resolution
        if isinstance(self.resolution, int):
            return 1
        return sum(int(d > 1) for d in self.resolution)

    @computed_field
    @property
    def is_mhd(self) -> bool:
        """check if simulation involves mhd."""
        return self.regime in [Regime.SRMHD, Regime.NMHD, Regime.IMHD]

    @computed_field
    @property
    def isothermal(self) -> bool:
        """isothermal is a REGIME, not a gamma value. the isothermal closure
        (p = cs^2 rho, no energy equation) is a structurally different eos — NOT
        the gamma->1 limit of the adiabatic path — so it is keyed on the regime,
        never on adiabatic_index == 1."""
        return self.regime in [Regime.ISOTHERMAL, Regime.IMHD]

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
        """ambient (constant) isothermal sound speed — derived from the
        `sound_speed` field. kept as the wire/checkpoint name the backends read;
        disk configs may still override it directly for a reference cs_0."""
        return self.sound_speed if self.sound_speed is not None else 0.0

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
    @model_validator(mode="before")
    @classmethod
    def _parse_cli_types(cls, data: dict) -> dict:
        """parse cli string inputs to proper types before validation."""
        if not isinstance(data, dict):
            return data

        # resolution: "256,256" -> (256, 256)
        if "resolution" in data and isinstance(data["resolution"], str):
            raw = data["resolution"]
            try:
                data["resolution"] = tuple(
                    int(p.strip()) for p in raw.split(",")
                )
            except ValueError:
                raise ValueError(
                    f"resolution: expected comma-separated integers "
                    f"(e.g. 256,256), got {raw!r}"
                ) from None

        # pad a SHORT resolution to the field's declared tuple arity with singleton
        # trailing axes: a 2d input (1024,1024) satisfies a 3-component field as
        # (1024,1024,1). this scrubs the requirement that a 2d problem stored as a
        # flat 3d slab (the mhd convention, nz=1) must spell out the unused axis.
        # an OVER-long input is left for validation to reject with a clear message.
        if "resolution" in data and isinstance(data["resolution"], tuple):
            arity = cls._tuple_field_arity("resolution")
            res = data["resolution"]
            if arity is not None and len(res) < arity:
                data["resolution"] = res + (1,) * (arity - len(res))

        # bounds: "[[0,1],[0,1]]" -> [[0, 1], [0, 1]]
        if "bounds" in data and isinstance(data["bounds"], str):
            bounds_str = data["bounds"].strip("[]")
            pairs = bounds_str.split("],[")
            data["bounds"] = [
                [float(x.strip()) for x in pair.strip("[]").split(",")]
                for pair in pairs
            ]

        # refinement_regions: "[[0,1,0,1],[2,3,2,3]]" -> [[0,1,0,1],[2,3,2,3]]
        if "refinement_regions" in data and isinstance(
            data["refinement_regions"], str
        ):
            regions_str = data["refinement_regions"].strip("[]")
            regions = regions_str.split("],[")
            data["refinement_regions"] = [
                [float(x.strip()) for x in region.strip("[]").split(",")]
                for region in regions
            ]

        # refinement_ratios: "2,2,4" -> [2, 2, 4]
        if "refinement_ratios" in data and isinstance(
            data["refinement_ratios"], str
        ):
            data["refinement_ratios"] = [
                int(x.strip()) for x in data["refinement_ratios"].split(",")
            ]

        # refinement_substeps: "1,2,4" -> [1, 2, 4]
        if "refinement_substeps" in data and isinstance(
            data["refinement_substeps"], str
        ):
            data["refinement_substeps"] = [
                int(x.strip()) for x in data["refinement_substeps"].split(",")
            ]

        # boundary_conditions: "periodic,outflow" -> [BoundaryCondition.PERIODIC, BoundaryCondition.OUTFLOW]
        if "boundary_conditions" in data and isinstance(
            data["boundary_conditions"], str
        ):
            bc_str = data["boundary_conditions"]
            if "," in bc_str:
                data["boundary_conditions"] = [
                    _to_enum(BoundaryCondition, bc.strip(), "boundary_conditions")
                    for bc in bc_str.split(",")
                ]
            else:
                data["boundary_conditions"] = _to_enum(
                    BoundaryCondition, bc_str.strip(), "boundary_conditions"
                )

        # enum fields: convert string to enum
        enum_fields = {
            "solver": Solver,
            "coord_system": CoordSystem,
            "regime": Regime,
            "reconstruction": Reconstruction,
            "timestepping": TimeStepping,
            "x1_spacing": CellSpacing,
            "x2_spacing": CellSpacing,
            "x3_spacing": CellSpacing,
            "refinement_subcycling_mode": SubCycleMode,
            "refinement_mode": RefinementMode,
        }

        for field_name, enum_type in enum_fields.items():
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = _to_enum(
                    enum_type, data[field_name], field_name
                )

        # path fields: convert string to Path
        path_fields = ["data_directory", "checkpoint_file"]
        for field_name in path_fields:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = Path(data[field_name])

        return data

    @model_validator(mode="after")
    def _coerce_refinement_types(self) -> SimbiProblem:
        """convert refinement_ratios to np.uint64 for c++ backend."""
        if self.refinement_ratios:
            object.__setattr__(
                self,
                "refinement_ratios",
                [np.uint64(r) for r in self.refinement_ratios],
            )
        if self.refinement_substeps:
            object.__setattr__(
                self,
                "refinement_substeps",
                [np.uint64(s) for s in self.refinement_substeps],
            )
        return self

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
                "isothermal runs require a positive sound_speed (set "
                "`sound_speed=...`), unless locally_isothermal is True (then "
                "cs^2(x) is derived per cell from the initial pressure profile)"
            )
        if not self.isothermal and self.adiabatic_index is None:
            raise ValueError(
                "energy-bearing (non-isothermal) regimes require an "
                "adiabatic_index (set `adiabatic_index=...`)"
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

    def validate_refinement_config(self) -> None:
        """
        validate refinement configuration.

        called automatically by _finalize(). subclasses should not need to
        call this directly - just override _finalize() and call super().
        """
        if not self.refinement_enabled:
            return

        if self.refinement_regions is None:
            raise ValueError(
                "refinement_regions required when refinement_enabled=True"
            )

        if self.refinement_ratios is None:
            raise ValueError(
                "refinement_ratios required when refinement_enabled=True"
            )

        expected_levels = self.refinement_max_levels - 1
        if len(self.refinement_regions) != expected_levels:
            raise ValueError(
                f"expected {expected_levels} refinement_regions, got {len(self.refinement_regions)}"
            )

        if len(self.refinement_ratios) != expected_levels:
            raise ValueError(
                f"expected {expected_levels} refinement_ratios, got {len(self.refinement_ratios)}"
            )

        expected_coords = 2 * self.dimensionality
        for ii, region in enumerate(self.refinement_regions):
            if len(region) != expected_coords:
                raise ValueError(
                    f"refinement_region[{ii}] has {len(region)} coords, expected {expected_coords}"
                )

        if self.refinement_subcycling_mode == SubCycleMode.MANUAL:
            if len(self.refinement_substeps) != expected_levels:
                raise ValueError(
                    "refinement_substeps must match refinement_max_levels - 1 for manual mode"
                )
            # insert a single substep for the base level
            object.__setattr__(
                self,
                "refinement_substeps",
                [np.uint64(1)] + self.refinement_substeps,
            )

        if self.refinement_mode == RefinementMode.ADAPTIVE:
            raise NotImplementedError(
                "adaptive refinement mode not yet supported"
            )

    def setup(self) -> None:
        """
        override to compute dynamic fields before validation.

        this hook is called automatically during model construction, before
        validation runs. use it to compute fields like bounds, resolution,
        or refinement_regions based on other parameters.

        if you override this method in a subclass, call super().setup() first
        to ensure the full setup chain executes:

            def setup(self) -> None:
                super().setup()
                self.bounds = self._calculate_bounds()
                self.refinement_regions = self._calculate_regions()
        """
        self.__setup_base_reached = True

    @model_validator(mode="after")
    def _finalize(self) -> SimbiProblem:
        """
        internal validator that runs setup hook then validates.

        do not override this method. override setup() instead.
        """
        self.__setup_base_reached = False
        self.setup()

        if not self.__setup_base_reached:
            warnings.warn(
                f"{type(self).__name__}.setup() did not call through to base class. "
                "Did you forget super().setup()?",
                stacklevel=2,
            )

        self.validate_refinement_config()
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
            if not cls._field_is_cli(field_name):
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
    def _field_is_cli(cls, field_name: str) -> bool:
        """whether a field is cli-exposed, inheriting the flag across the mro.

        a core knob declared `cli=True` in a base class (e.g. `solver`,
        `reconstruction`, `cfl_number`) stays exposed even when a subclass
        overrides the field ONLY to change its default — the common config
        pattern `solver: Annotated[Solver, ProblemParam(Solver.HLLC)]` would
        otherwise silently drop `--solver` (the override replaces the base's
        Annotated metadata wholesale, losing `cli=True`). the child's default /
        type / help still win; only the cli exposure is inherited.
        """
        for klass in cls.__mro__:
            fields = getattr(klass, "model_fields", None)
            if not fields or field_name not in fields:
                continue
            if get_param_metadata(fields[field_name]).cli:
                return True
        return False

    @classmethod
    def _tuple_field_arity(cls, field_name: str) -> Optional[int]:
        """the fixed arity of a tuple-typed field, or None when it is variadic
        (`tuple[int, ...]`) or not a tuple. resolves across the mro so a subclass
        that re-declares the field is honored. used to pad a short resolution
        input to the declared number of axes.
        """
        from typing import get_args, get_origin

        for klass in cls.__mro__:
            fields = getattr(klass, "model_fields", None)
            if not fields or field_name not in fields:
                continue
            ann = fields[field_name].annotation
            if get_origin(ann) is tuple:
                args = get_args(ann)
                if args and Ellipsis not in args:
                    return len(args)
            return None
        return None

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
        if parsed is None:
            raise ValueError("failed to parse cli arguments for problem")
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
        try:
            return cls(**data)
        except ValidationError as exc:
            # convert pydantic's raw traceback into a clinical, actionable report
            # the cli prints without a stack trace.
            raise ConfigError(
                _humanize_validation_error(cls.__name__, exc)
            ) from None

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
