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
import math
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    ValidationError,
    computed_field,
    model_validator,
)

from simbi.types.bodies import BodySystemConfig, BondedAssembly, ImmersedBodyConfig

# re-export types that problem authors need
from simbi.types.input import (
    BoundaryCondition,
    Neumann,
    Robin,
    CellSpacing,
    CoordSystem,
    Spacetime,
    CtMethod,
    Eos,
    Limiter,
    Reconstruction,
    RefinementMode,
    Regime,
    Solver,
    SubCycleMode,
    TimeStepping,
    TracerScheme,
)
from simbi.types.typing import (
    ExpressionDict,
    GasStateGenerator,
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
            f"pass comma-separated axis sizes, e.g., `{flag} 256,256`; trailing "
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

    # extra="forbid": a typo'd constructor kwarg (cfl_numbr=0.9) must fail loudly,
    # not vanish while the field keeps its default. assignment constraints are
    # enforced FIELD-ONLY via __setattr__ — pydantic's validate_assignment
    # would re-run every model validator per assignment, which recurses infinitely
    # for the common pattern of a validator that assigns fields.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # per-field TypeAdapter cache for the field-only assignment validation.
    _field_adapters: ClassVar[dict[str, Any]] = {}

    # the flags the user EXPLICITLY passed on the command line. from_cli fills it
    # (possibly with the empty set — no flags passed); None marks a problem that
    # never went through the cli, where model_fields_set carries the same fact.
    # the checkpoint merge uses it to distinguish "user demanded --solver hllc"
    # from "the class default happens to differ from the checkpoint".
    _cli_explicit: Optional[set[str]] = PrivateAttr(default=None)

    def __setattr__(self, name: str, value: Any) -> None:
        # enforce the FIELD's own constraints (type, ge/gt/le) on every assignment
        # — an out-of-range value written in setup() or user code must not reach
        # the backend — WITHOUT re-running the model validators (whole-model
        # consistency is _finalize's job, once, after setup).
        fields = type(self).model_fields
        if name in fields:
            key = f"{type(self).__qualname__}.{name}"
            adapter = SimbiProblem._field_adapters.get(key)
            if adapter is None:
                from pydantic import TypeAdapter

                info = fields[name]
                adapter = TypeAdapter(
                    Annotated[info.annotation, info],
                    config=None if info.annotation is None else ConfigDict(arbitrary_types_allowed=True),
                )
                SimbiProblem._field_adapters[key] = adapter
            try:
                value = adapter.validate_python(value)
            except ValidationError as exc:
                raise ValueError(
                    f"invalid assignment to {type(self).__name__}.{name}: "
                    + "; ".join(e["msg"] for e in exc.errors())
                    + f" (got: {value!r})"
                ) from None
        super().__setattr__(name, value)

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
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.MINKOWSKI,
            cli=True,
            description="background spacetime (flat minkowski, or schwarzschild for GR)",
        ),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            description="schwarzschild geometric mass M (G=c=1); only used when "
            "spacetime is schwarzschild",
        ),
    ]
    kerr_spin: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            description="kerr specific angular momentum a = J/M, |a| < M; only used "
            "when spacetime is kerr",
        ),
    ]
    max_dt: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            description="upper clamp on the CFL time step (dt = min(dt_cfl, max_dt)); "
            "0 disables. pins the dt sequence across runs whose CFL estimators differ",
        ),
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
            # cli-exposed HERE, as the core physics knob it is, alongside solver /
            # reconstruction / cfl_number. it was previously left to each config to
            # re-declare with cli=True, which most did -- so the flag appeared generic
            # right up until a config that did NOT declare it (one on the synge closure,
            # which owns no gamma) was asked to run a gamma-law arm, and `--adiabatic-index`
            # came back "unrecognized" for the first time. _field_is_cli inherits the
            # exposure across the mro, so a config that still declares its own keeps its
            # default and help text; only the flag itself is now guaranteed.
            cli=True,
            description="adiabatic index; required for energy-bearing regimes, "
            "irrelevant for isothermal ones (which use sound_speed), and refused "
            "by the parameter-free synge closure",
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
    eos: Annotated[
        Eos,
        ProblemParam(
            Eos.IDEAL,
            cli=True,
            description="equation-of-state closure: 'ideal' (gamma-law, uses "
            "adiabatic_index) or 'synge' (taub-mathews relativistic perfect gas, "
            "parameter-free; effective gamma walks 5/3 cold -> 4/3 hot; rhd on a "
            "flat spacetime only)",
        ),
    ]
    ppm_flatten_onset: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            ge=0.0,
            description="ppm convergence-gated flatten onset, in compression per "
            "cell crossing over the isothermal sound speed. 0 (default) = the "
            "pure parabola. gravity-sink runs set onset/full (e.g. 0.015/0.05) "
            "to close the smooth-infall entropy vent; trans-sonic turbulence "
            "leaves them off — an active flatten there degrades ppm to first "
            "order in every eddy collision",
        ),
    ]
    ppm_flatten_full: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            ge=0.0,
            description="compression strength at which the ppm flatten saturates "
            "to the full cell-average (first-order) face; must exceed "
            "ppm_flatten_onset when the flatten is on. 0 (default) = off",
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
    checkpoint_log_anchor: Annotated[
        float,
        ProblemParam(
            0.0,
            ge=0.0,
            description="positive reference time for LOG-spaced checkpoints; distinct from "
            "start_time (the physical/resume clock). 0 = use start_time. set this (not start_time) "
            "when the log anchor differs from the run start, so restarts resume at the checkpoint time.",
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
            "names + the live display report time / time_unit. e.g., set to the "
            "orbital period for a binary so output reads in orbits",
        ),
    ]
    time_unit_label: Annotated[
        str,
        ProblemParam(
            "t",
            cli=True,
            checkpoint_safe=True,
            description="label for the natural time unit (e.g., 'orbit'); 't' "
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
            "x time_unit). 0 disables. only emitted when the problem has "
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
    gpus: Annotated[
        int,
        ProblemParam(
            1,
            ge=1,
            cli=True,
            checkpoint_safe=True,
            description="number of gpus to decompose the domain across, intra-node "
            "(NVLink/peer). 1 = single device (default). >1 requires a gpu build "
            "(./dev.py install --cuda or --hip) and at least that many visible devices. "
            "the backend (cuda/hip) is a BUILD choice; this is purely how many devices "
            "to use at runtime",
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
    ct_method: Annotated[
        CtMethod,
        ProblemParam(
            CtMethod.CONTACT,
            cli=True,
            description="CT edge-EMF scheme (contact | uct); MHD only",
        ),
    ]
    reconstruction: Annotated[
        Reconstruction,
        ProblemParam(
            Reconstruction.PLM, cli=True, description="spatial reconstruction"
        ),
    ]
    limiter: Annotated[
        Limiter,
        ProblemParam(
            Limiter.MINMOD,
            cli=True,
            description="PLM slope limiter (minmod | vanleer); minmod uses plm_theta (1=minmod, 2=MC)",
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
            None,
            ge=1,
            le=3,
            cli=True,
            description="order of accuracy (1=pcm/rk1, 2=plm/rk2, 3=ppm/rk3)",
        ),
    ]
    mach_limit: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            ge=0.0,
            le=1.0,
            description="reference mach number the HLLC-LM acoustic ramp saturates at "
            "(solver=hllc_lm_plain only). the ramp reduces acoustic dissipation only BELOW "
            "this number, so it sets how much of the flow the reduction reaches: 0.1 is the "
            "value used throughout Fleischmann, Adami and Adams (2020), and a deeply subsonic "
            "problem whose whole range sits under it is left at classical HLLC unless the "
            "limit is raised to meet the flow. 0 reduces nothing; 1 reduces to the sonic "
            "point. the CLAMPED arm (solver=hllc_lm) ignores this and holds 0.1, because its "
            "incompressible pressure ceiling is derived from that value",
        ),
    ]
    plm_theta: Annotated[
        float,
        ProblemParam(
            1.5, gt=0.0, le=2.0, cli=True,
            description="minmod-MC compression in (0,2] (1=minmod, 2=MC); ignored for --limiter vanleer",
        ),
    ]
    boundary_conditions: Annotated[
        Union[
            BoundaryCondition,
            Neumann,
            Robin,
            Sequence[Union[BoundaryCondition, Neumann, Robin]],
        ],
        ProblemParam(
            BoundaryCondition.OUTFLOW,
            cli=True,
            description="boundary conditions; a face may be a Neumann/Robin gradient wall",
        ),
    ]

    # =========================================================================
    # mesh spacing
    # =========================================================================
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="x1 cell spacing"),
    ]
    x1_spacing_ratio: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            description="adjacent x1 cell-width ratio for geometric spacing",
        ),
    ]
    x2_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="x2 cell spacing"),
    ]
    x2_spacing_ratio: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            description="adjacent x2 cell-width ratio for geometric spacing",
        ),
    ]
    x3_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="x3 cell spacing"),
    ]
    x3_spacing_ratio: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            description="adjacent x3 cell-width ratio for geometric spacing",
        ),
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
    # refinement / fmr settings (fixed mesh refinement only)
    # =========================================================================
    refinement_enabled: Annotated[
        bool, ProblemParam(False, description="enable mesh refinement")
    ]
    # seeding from the declared stationary target rather than from a pointwise sample of it:
    # cells covered by a finer level carry the RESTRICTION of the finer target, which is what
    # the hierarchy's own restriction produces and re-produces every parent step. a pointwise
    # sample sits a truncation-order distance off the state the well-balancing preserves, and
    # that distance evolves like any other perturbation.
    seed_from_equilibrium: Annotated[
        bool,
        ProblemParam(
            False,
            description="seed every level from the declared equilibrium target",
        ),
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
        ProblemParam(
            SubCycleMode.NONE,
            description="refinement subcycling schedule. only the implemented fixed-ratio "
            "schedule is selectable: level l advances 2^l times per root step, with the root "
            "step limited by every level's own cfl. NONE and STANDARD both select it and are "
            "equivalent; ADAPTIVE and MANUAL are refused",
        ),
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
        its spatial dimensionality is read from the resolution like every other
        regime.
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
        return self.regime in [Regime.RMHD, Regime.NMHD, Regime.IMHD]

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

    def expected_primitive_arity(self) -> Optional[tuple[int, str]]:
        """the EXACT per-cell primitive tuple (length, signature) that
        `initial_primitive_state`'s gas generator must yield — or None when the
        width is not uniquely fixed by the regime. the reader maps the tuple
        POSITIONALLY, so a too-long tuple silently shifts a trailing field (e.g.
        pressure) into an ignored slot rather than erroring; pinning the length
        turns that into a loud failure.

        the layout is `rho`, then one velocity per DOF, then pressure. it is exact
        only when BOTH are fixed:
        - DOF is 3 for mhd (v3 couples to B, so it is carried even in 1.5d/2.5d).
          for pure hydro it is `dimensionality` ONLY on a cartesian chart; a
          curvilinear chart (spherical/cylindrical) carries transverse velocities
          for angular momentum with no matching spatial axis, and relativistic
          hydro (rhd) likewise carries an azimuthal v3 on a curved/rotating chart
          — so those DOFs are not fixed by dimensionality (width UNDETERMINED).
        - pressure is mandatory for an energy regime. an ISOTHERMAL run may or may
          not pass an explicit p/cs field (p = cs^2 rho is derivable), so its width
          is UNDETERMINED (None).
        callers fall back to a lower-bound check for the None cases."""
        if self.isothermal:
            return None  # optional trailing p/cs entry
        if self.is_mhd:
            dof = 3  # v3 couples to B on every chart
        elif (
            self.regime != Regime.RHD
            and self.coord_system == CoordSystem.CARTESIAN
        ):
            dof = self.dimensionality  # cartesian hydro carries no extra velocity
        else:
            return None  # curvilinear / relativistic: chart-dependent velocity dof
        fields = ["rho"] + [f"v{ii + 1}" for ii in range(dof)] + ["p"]
        return len(fields), "(" + ", ".join(fields) + ")"

    @computed_field
    @property
    def is_relativistic(self) -> bool:
        """check if simulation is relativistic."""
        return self.regime in [Regime.RHD, Regime.RMHD]

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

            anchor = self.checkpoint_log_anchor if self.checkpoint_log_anchor > 0.0 else self.start_time
            if anchor <= 0.0:
                raise ConfigError(
                    "log-spaced checkpoints need a positive time anchor: set "
                    "checkpoint_log_anchor (or a positive start_time) — the "
                    "cadence is log10(end_time / anchor), undefined at 0"
                )
            return math.log10(self.end_time / anchor) / num_outputs
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
    def resistivity(self) -> float:
        """bulk ohmic resistivity eta for non-ideal mhd. the induction equation
        gains a diffusive emf eta*J, dissipating magnetic energy into the gas.
        zero recovers ideal mhd."""
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
    def source_expressions(self) -> list[ExpressionDict]:
        """ordered source terms applied additively during each stage."""
        return []

    @computed_field
    @property
    def census_expressions(self) -> list[ExpressionDict]:
        """binned reductions over the grid, emitted as a time series in the checkpoint.

        each entry is a `simbi.expression.Census(...).serialize()`. a census is a
        pointwise map followed by a segmented reduce: bin axes cut the grid into
        buckets, and each registered accumulator is combined within its bucket. no
        axes at all is a global reduction, which is how a total mass or energy is
        expressed.

        the reduce combines SUMS or extrema, not statistics. a mean or a variance is
        not order-agnostic, so it cannot be reduced in parallel or combined across
        restart segments — register `m*v` and `m` and divide when reading.
        """
        return []

    @computed_field
    @property
    def perturbation_expressions(self) -> ExpressionDict:
        """an initial-condition PERTURBATION as a traced expression of position: a delta
        on each primitive component, applied at EVERY refinement level's own cell centers.

        `initial_primitive_state` fills the ROOT grid alone; fine levels inherit it by
        prolongation, which reproduces nothing below the root's nyquist. initial data whose
        content is genuinely finer than the root can represent — a turbulent seed with power
        at every level's own scale, say — therefore cannot be delivered by the cell
        generator at all. a declared expression is evaluated per level and can.

        override in a subclass returning
        `graph.compile([drho, dv1, ..., dp]).serialize_equilibrium(dim=...)`, with the
        components in that order (no pressure on an isothermal regime, whose slot discards
        it). default {} = no perturbation.

        the delta is applied AFTER the base state is seeded and prolonged, and covered
        coarse cells are re-derived as the restriction of the fine state afterwards, so the
        hierarchy stays consistent. unavailable on mhd regimes: a cell-centered primitive
        rewrite cannot update the staggered face field.
        """
        return {}

    @computed_field
    @property
    def equilibrium_expressions(self) -> ExpressionDict:
        """the run's STATIONARY TARGET state as a traced expression of position, which a
        well-balanced scheme then holds exactly.

        a steady state solves the continuum equations, not the discrete ones, so the scheme
        leaves a residual at truncation order and gas seeded on an exact hydrostatic profile
        starts moving — faster still across a coarse-fine interface, where the two grids
        reduce the same exact solution to different face values. declaring the target lets
        the backend measure that residual once per level and subtract it back at every
        stage, which makes the target a fixed point to roundoff and costs no conservation.

        override in a subclass returning
        `graph.compile([rho, v1, ..., p]).serialize_equilibrium(dim=...)`, with the
        primitive components in that order (no pressure on an isothermal regime). default
        {} = no declared target, and the scheme behaves exactly as before.

        THE DECLARED STATE MUST ACTUALLY BE STATIONARY. the backend holds whatever it is
        given, so a state that is not an equilibrium would be frozen in place and the run
        would report nothing. this is checked: the target's discrete imbalance has to shrink
        under refinement, which truncation error does and a continuum residual does not, and
        a target that fails is refused at setup.
        """
        return {}

    @computed_field
    @property
    def scale_factor_expressions(self) -> ExpressionDict:
        """mesh-motion scale factor a(t) + its derivative a_dot(t) as a TRACED expression pair,
        evaluated exactly in the rust time loop (no linearization, no python in the loop). override
        in a subclass returning `graph.compile([a, a_dot]).serialize_motion()`, with
        `a_dot = a.diff(variable('t'))` (autodiff). default {} = no expression motion (static / the
        legacy linear scale_factor callable)."""
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

    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        return None

    @computed_field
    @property
    def n_tracers(self) -> int:
        """mass-transport tracer count: deterministic mass-weighted seeding;
        accepted finite-volume mass transfers update authoritative cell or
        reservoir ownership. checkpoints also carry derived positions for
        visualization. 0 = none."""
        return 0

    @computed_field
    @property
    def tracer_scheme(self) -> TracerScheme:
        """passive tracer realization: discrete, ito2, or ito3."""
        return TracerScheme.DISCRETE

    def tracer_cohort(self) -> Optional[GasStateGenerator]:
        """initial-material provenance: one non-negative integer cohort per
        interior cell in the same traversal order as the gas state. labels are
        immutable on tracers; injected material uses the reserved label 65535."""
        return None

    # =========================================================================
    # passive scalar (override in subclass)
    # =========================================================================
    def passive_scalar(self) -> Optional[GasStateGenerator]:
        """the passive-scalar (dye) initial condition: a generator yielding one
        chi value per interior cell, axis-0-fastest (the same traversal as the
        gas state generator), or None for an undyed run. the dye advects with
        the mass flux and appears as the `chi` dataset in checkpoints, with
        `chi_dens` derivable in the viz."""
        return None

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
                    f"(e.g., 256,256), got {raw!r}"
                ) from None

        # pad a SHORT resolution to the field's declared tuple arity with singleton
        # trailing axes: a 2d input (1024,1024) satisfies a 3-component field as
        # (1024,1024,1), so a 2d problem stored as a flat 3d slab (the mhd
        # convention, nz=1) need not spell out the unused axis.
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
            "ct_method": CtMethod,
            "coord_system": CoordSystem,
            "spacetime": Spacetime,
            "regime": Regime,
            "reconstruction": Reconstruction,
            "eos": Eos,
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
        """convert refinement_ratios to np.uint64 for the rust backend."""
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
            elif self.order == 3:
                object.__setattr__(self, "reconstruction", Reconstruction.PPM)
                object.__setattr__(self, "timestepping", TimeStepping.RK3)
            else:
                raise ValueError("order must be 1, 2, or 3")
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
        if (
            not self.isothermal
            and self.adiabatic_index is None
            and self.eos != Eos.SYNGE
        ):
            raise ValueError(
                "energy-bearing (non-isothermal) regimes require an "
                "adiabatic_index (set `adiabatic_index=...`)"
            )
        return self

    @model_validator(mode="after")
    def _validate_eos(self) -> SimbiProblem:
        """the synge closure is parameter-free and relativistic: reject a declared
        adiabatic_index alongside it (dead configuration, never swallowed) and any
        non-rhd regime, then supply the inert placeholder gamma the plumbing
        carries (the backend binds it to the kernels but the taub-mathews closure
        never reads it)."""
        if self.eos == Eos.SYNGE:
            if self.regime != Regime.RHD:
                raise ValueError(
                    f"eos = 'synge' (taub-mathews) is a relativistic closure and "
                    f"applies to the rhd regime only; got '{self.regime.value}'"
                )
            if self.adiabatic_index is not None:
                raise ValueError(
                    "adiabatic_index applies to the ideal (gamma-law) closure only; "
                    "the synge (taub-mathews) closure is parameter-free — its "
                    "effective gamma is set by the temperature"
                )
            object.__setattr__(self, "adiabatic_index", 5.0 / 3.0)
        return self

    @model_validator(mode="after")
    def _validate_plm_theta(self) -> SimbiProblem:
        """validate plm theta parameter."""
        if (
            self.reconstruction == Reconstruction.PLM
            and self.limiter == Limiter.MINMOD
            and not (0.0 < self.plm_theta <= 2.0)
        ):
            raise ValueError(
                "plm_theta must be in (0, 2] when using PLM with the minmod limiter"
            )
        # the van leer limiter is spelled theta = -1 ONLY at the execution-dict
        # boundary (runner.py); the validated model keeps the user's positive
        # compression so the field's own gt=0 constraint holds on every path a
        # model round-trips (assignment validation, checkpoint restore).
        return self

    @model_validator(mode="after")
    def _validate_ppm(self) -> SimbiProblem:
        """reject slope-limiter knobs alongside ppm rather than silently ignoring them."""
        if self.reconstruction == Reconstruction.PPM:
            # the monotonized parabola carries its own constraint; a plm_theta or
            # limiter moved off its declared default would be dead configuration —
            # surface that, never swallow it. (the config plumbing passes every field
            # explicitly, so presence in model_fields_set cannot distinguish a user
            # choice from a passthrough default; a changed VALUE can.)
            fields = SimbiProblem.model_fields
            stray = [
                name
                for name in ("plm_theta", "limiter")
                if getattr(self, name) != fields[name].default
            ]
            if stray:
                raise ValueError(
                    f"{stray} apply to PLM reconstruction only; PPM carries its "
                    "own monotonicity constraint and takes no slope limiter"
                )
            if self.ppm_flatten_full > 0.0 and self.ppm_flatten_full <= self.ppm_flatten_onset:
                raise ValueError(
                    "ppm_flatten_full must exceed ppm_flatten_onset when the "
                    "flatten is on (equal or inverted dials silently disable it)"
                )
        else:
            # a config class may DECLARE its ppm pairing (nonzero dial defaults)
            # while defaulting to plm — the dials are inert there and activate
            # when the run selects ppm. what is rejected is a USER moving a dial
            # off the declaring class's own default on a non-ppm run: that is
            # dead configuration. subclass defaults, not the base's, are the
            # reference (type(self).model_fields carries the override).
            fields = type(self).model_fields
            moved = [
                name
                for name in ("ppm_flatten_onset", "ppm_flatten_full")
                if getattr(self, name) != fields[name].default
            ]
            if moved:
                raise ValueError(
                    f"{moved} apply to PPM reconstruction only; moved off their "
                    "defaults on another scheme they would be dead configuration"
                )
        return self

    @model_validator(mode="after")
    def _validate_geometric_spacing(self) -> SimbiProblem:
        """validate each geometric cell-width ratio."""
        for axis in range(1, 4):
            spacing = getattr(self, f"x{axis}_spacing")
            ratio = getattr(self, f"x{axis}_spacing_ratio")
            if spacing == CellSpacing.GEOMETRIC:
                if not math.isfinite(ratio) or ratio <= 0.0:
                    raise ValueError(
                        f"x{axis}_spacing_ratio must be positive and finite "
                        f"for geometric spacing"
                    )
            elif ratio != 1.0:
                raise ValueError(
                    f"x{axis}_spacing_ratio is only valid when "
                    f"x{axis}_spacing='geometric'"
                )
        return self

    def validate_refinement_config(self) -> None:
        """
        validate refinement configuration.

        called automatically by _finalize(). subclasses should not need to
        call this directly - just override _finalize() and call super().
        """
        if not self.refinement_enabled:
            return

        # mesh refinement is cartesian + uniform-spacing only: the coarse-fine prolong/restrict
        # transfer is geometry-agnostic (equal index-based sub-cells), correct solely for
        # uniform-volume cells. a curvilinear grid (variable r^2 / r cell volumes) or a non-linear
        # axis (unequal sub-cells) would get silently-wrong transfers.
        if self.coord_system != CoordSystem.CARTESIAN:
            raise ValueError(
                "mesh refinement is cartesian-only (the coarse-fine transfer ignores curvilinear "
                f"cell volumes); got coord_system={self.coord_system.value}"
            )
        nonlinear = [
            f"x{ax}"
            for ax, sp in enumerate(
                (self.x1_spacing, self.x2_spacing, self.x3_spacing), start=1
            )
            if sp != CellSpacing.LINEAR
        ]
        if nonlinear:
            raise ValueError(
                "mesh refinement requires uniform (linear) cell spacing (the coarse-fine transfer "
                f"assumes equal sub-cells); non-linear axes: {', '.join(nonlinear)}"
            )

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

        # the backend subcycles at a FIXED refinement ratio: level l advances 2^l times per root
        # step, and the root step is min over levels of (that level's own cfl limit) * 2^l, so every
        # level lands inside its own cfl. neither an adaptive substep count nor a hand-specified one
        # is implemented — `refinement_subcycling_mode` and `refinement_substeps` reach no backend
        # code at all.
        #
        # refused rather than ignored. a config that declares ADAPTIVE and silently receives the
        # fixed schedule invites reasoning built on a knob that does nothing, and the two are not
        # equivalent: under the fixed schedule the ROOT is throttled by the finest level's
        # requirement, taking around twenty times more steps than its own cfl would need on a deep
        # gravitational ladder. that is a bounded cost (the finest level dominates the work either
        # way, so an ideal schedule saves order twenty percent) but it is not nothing, and it is not
        # what the declaration says.
        if self.refinement_subcycling_mode in (
            SubCycleMode.ADAPTIVE,
            SubCycleMode.MANUAL,
        ):
            raise NotImplementedError(
                f"refinement_subcycling_mode={self.refinement_subcycling_mode.value!r} is not "
                "implemented: the backend subcycles level l exactly 2^l times per root step, and "
                "the mode reaches no backend code. use SubCycleMode.STANDARD (or NONE) to select "
                "the implemented fixed-ratio schedule."
            )

        if self.refinement_mode == RefinementMode.ADAPTIVE:
            raise NotImplementedError(
                "adaptive refinement mode not yet supported"
            )

    def setup(self) -> None:
        """
        override to compute dynamic fields from other parameters.

        this hook runs at the END of model validation (the last after-validator),
        so every declared field has already been validated when it executes. a
        field this hook computes (bounds, refinement_regions, ...) must therefore
        be declared Optional with a None default — a required field with no value
        fails validation before setup ever runs. assignments made here re-validate
        against the field's own constraints (ge/gt/le), and the cross-field checks
        re-run once setup returns.

        a subclass override must call super().setup() first so the full setup
        chain executes:

            bounds: Annotated[Optional[list], ProblemParam(None, ...)]

            def setup(self) -> None:
                super().setup()
                self.bounds = self._calculate_bounds()
                self.refinement_regions = self._calculate_regions()
        """
        self.__setup_base_reached = True

    def summary(self) -> list[tuple[str, str, str]]:
        """
        override to report DERIVED quantities (bondi radius, expected rates,
        computed grid facts, ...) as (group, label, value) rows. the runner
        collects these once, after setup(), into the live dashboard's grouped
        problem-setup panel alongside the declared parameters.

        this is the config's one reporting hook: never print from __del__ —
        a destructor fires on garbage-collection timing, on every transient
        instantiation (discovery, validation, tests), and races the live
        dashboard for the terminal.
        """
        return []

    @model_validator(mode="after")
    def _finalize(self) -> SimbiProblem:
        """
        internal validator that runs setup hook then validates.

        do not override this method. override setup() instead.
        """
        self.__setup_base_reached = False
        # field assignments inside setup() validate individually (the field-only
        # __setattr__ check); the explicit validator re-runs restore full cross-field
        # consistency once every setup mutation has landed.
        self.setup()

        if not self.__setup_base_reached:
            warnings.warn(
                f"{type(self).__name__}.setup() did not call through to base class. "
                "Did you forget super().setup()?",
                stacklevel=2,
            )

        self._coerce_refinement_types()
        self._enforce_order_settings()
        self._validate_isothermal()
        self._validate_plm_theta()
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
            help_text = field_info.description or f"set {field_name}"
            kwargs: dict[str, Any] = {
                "dest": field_name,
                "help": help_text.replace("%", "%%"),
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

        a core knob declared `cli=True` in a base class (e.g., `solver`,
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

        # parse into existing namespace (if provided). REJECT unrecognized flags; a typo'd
        # or unsupported flag must fail loudly here, since silently ignoring it would run
        # with the default and mislead the user.
        parsed, extras = parser.parse_known_args(argv, namespace)
        if extras:
            raise ConfigError(
                f"unrecognized argument(s): {' '.join(extras)}\n"
                f"run `simbi run <config> --help` to list the supported flags."
            )
        if parsed is None:
            raise ValueError("failed to parse cli arguments for problem")
        # record which flags the user EXPLICITLY passed: a second parse with every
        # default suppressed leaves only the argv-provided dests in the namespace.
        # the checkpoint merge reads this to tell a demanded override apart from a
        # class default that merely differs from the checkpoint.
        explicit_parser = argparse.ArgumentParser(add_help=False)
        cls.setup_cli(explicit_parser)
        for action in explicit_parser._actions:
            action.default = argparse.SUPPRESS
        explicit_ns, _ = explicit_parser.parse_known_args(argv)
        problem = cls.from_namespace(parsed)
        problem._cli_explicit = set(vars(explicit_ns))
        return problem

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
        for field_name, field_info in type(self).model_fields.items():
            metadata = get_param_metadata(field_info)
            if not metadata.checkpoint_safe:
                immutable.add(field_name)
        return immutable

    def get_checkpoint_safe_fields(self) -> set[str]:
        """get field names that can be overridden when loading from checkpoint."""
        safe = set()
        for field_name, field_info in type(self).model_fields.items():
            metadata = get_param_metadata(field_info)
            if metadata.checkpoint_safe:
                safe.add(field_name)
        return safe
