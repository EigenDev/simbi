# =============================================================================
# checkpoint.py
#
# checkpoint loading and config merging for simbi simulations.
# the actual state data is loaded by the rust backend - python just handles metadata
# and config merging with checkpoint_safe field validation.
#
# usage:
#   problem = SodProblem(end_time=10.0)  # user wants to extend runtime
#   merged = merge_with_checkpoint(problem, Path("data/checkpoint.h5"))
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from simbi.reader import read_simulation
from simbi.types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Eos,
    Limiter,
    Metadata,
    Reconstruction,
    Regime,
    Solver,
    TimeStepping,
    normalize_regime,
)

if TYPE_CHECKING:
    from .problem import SimbiProblem


def load_checkpoint_metadata(
    checkpoint_path: Path,
) -> tuple[Metadata, tuple[int, ...]]:
    """
    load metadata from checkpoint file.

    returns:
        tuple of (metadata, mesh_shape)
    """
    data = read_simulation(str(checkpoint_path), unpad=False)

    # also need mesh shape for resolution calculation
    mesh_shape = data.mesh.shape if hasattr(data, "mesh") else (1, 1, 1)

    return data.metadata, mesh_shape


def metadata_to_config_dict(
    metadata: Metadata, mesh_shape: tuple[int, ...]
) -> dict[str, Any]:
    """
    convert checkpoint metadata to config dictionary format.

    these are the fields that define the physics state and must be
    preserved from checkpoint.

    args:
        metadata: checkpoint metadata
        mesh_shape: mesh shape from data.mesh.shape (includes ghost zones)
    """
    # mesh_shape is in STORAGE order (x_n, ..., x2, x1) — REVERSED, matching the bounds/spacing
    # `[::-1]` the parser applies. un-reverse to the forward (x1, x2, x3) the config expects, and pad
    # a lower-dimensional run (2D / 2.5D) up to the 3-tuple resolution field with trailing 1s.
    resolution = tuple(int(n) for n in reversed(tuple(mesh_shape)))
    resolution = resolution + (1,) * (3 - len(resolution))

    # a checkpoint may carry a legacy regime slug (srhd, srmhd); map it to the current name so the
    # run restarts cleanly.
    regime_str = normalize_regime(str(metadata.regime))

    config = {
        "resolution": resolution,
        "start_time": float(metadata.time),
        "adiabatic_index": float(metadata.gamma),
        "coord_system": CoordSystem(metadata.coord_system),
        "regime": Regime(regime_str),
        "solver": Solver(metadata.solver),
        "reconstruction": Reconstruction(metadata.reconstruction),
        "timestepping": TimeStepping(metadata.timestepping),
        "cfl_number": float(metadata.cfl),
        "checkpoint_index": int(metadata.checkpoint_index),
    }

    # the backend stores the KERNEL spelling of the limiter choice: plm_theta < 0
    # means van leer (the model field itself is constrained to (0, 2], so the raw
    # -1 must map back to the limiter selection).
    if float(metadata.plm_theta) < 0.0:
        config["limiter"] = Limiter.VAN_LEER
    else:
        config["plm_theta"] = float(metadata.plm_theta)

    config.update({
        "checkpoint_interval": float(metadata.checkpoint_interval),
        "x1_spacing": CellSpacing(metadata.x1_spacing),
        "x1_spacing_ratio": float(getattr(metadata, "x1_spacing_ratio", 1.0)),
        "x2_spacing": CellSpacing(metadata.x2_spacing),
        "x2_spacing_ratio": float(getattr(metadata, "x2_spacing_ratio", 1.0)),
        "x3_spacing": CellSpacing(metadata.x3_spacing),
        "x3_spacing_ratio": float(getattr(metadata, "x3_spacing_ratio", 1.0)),
        "boundary_conditions": [
            BoundaryCondition(b) for b in metadata.boundary_conditions
        ],
    })

    # amr fields if present
    if metadata.level_dts:
        config["refinement_level_dts"] = list(metadata.level_dts)
    if metadata.level_substeps:
        config["refinement_level_substeps"] = list(metadata.level_substeps)
    # the closure the run was written with. only recorded on checkpoints new enough to
    # carry the attribute — an empty slug means "not recorded", and the restart then keeps
    # whatever closure the config declares (asserting "ideal" would silently swap the
    # equations of state on an older synge run).
    recorded_eos = str(getattr(metadata, "eos", "") or "")
    if recorded_eos:
        config["eos"] = Eos(recorded_eos)

    if metadata.subcycling_mode and metadata.subcycling_mode != "none":
        from simbi.types import SubCycleMode

        config["refinement_subcycling_mode"] = SubCycleMode(
            metadata.subcycling_mode
        )

    return config


def _values_agree(a: Any, b: Any) -> bool:
    """order- and container-insensitive equality for the restart conflict check:
    a CLI tuple must match a checkpoint list, an enum its value string, a numpy
    scalar its python twin. false only for a REAL disagreement."""
    av = getattr(a, "value", a)
    bv = getattr(b, "value", b)
    if isinstance(av, (list, tuple)) and isinstance(bv, (list, tuple)):
        return len(av) == len(bv) and all(_values_agree(x, y) for x, y in zip(av, bv))
    if isinstance(av, float) or isinstance(bv, float):
        try:
            return float(av) == float(bv)
        except (TypeError, ValueError):
            return False
    return av == bv


def _assert_same_equilibrium_target(
    problem: SimbiProblem, metadata: Metadata
) -> None:
    """refuse a restart whose declared stationary target differs from the one the checkpoint
    was written with.

    the target is not a field, so it leaves no trace in the file's data and a changed one is
    invisible: the resumed run integrates different equations — a different imbalance is
    subtracted at every stage, and a different state is held exactly — while every plot looks
    the way it should. the ordinary immutable-field merge cannot catch this because the target
    is a computed expression graph rather than a model field.

    the comparison is structural, on the parsed payloads, so whitespace or key ordering from
    two different json writers does not read as a physics change.
    """
    import json

    from .problem import ConfigError

    # SCHEME-CHANGE GUARD. `solver = hllc_lm` changed meaning on 2026-08-15: the clamped
    # variant was retired and the name now denotes the published Fleischmann ramp. a series
    # recorded under the old meaning must not be continued under the new one -- same string,
    # different numerics, and nothing downstream could ever tell. the discriminator is the
    # stored config itself: `wb_reconstruction` entered the model with the new scheme, so a
    # checkpoint whose config lacks the key is clamp-era by construction.
    recorded_solver = str(getattr(metadata, "solver", "") or "")
    if recorded_solver in ("hllc_lm", "hllc-lm"):
        stored_cfg = getattr(metadata, "config", None) or {}
        if hasattr(stored_cfg, "keys") and "wb_reconstruction" not in stored_cfg:
            from .problem import ConfigError

            raise ConfigError(
                "this checkpoint was written by the RETIRED clamped hllc_lm scheme (its "
                "stored config predates wb_reconstruction). the name now denotes the "
                "published Fleischmann ramp, so continuing the series would silently change "
                "the numerics mid-run. start a fresh series, or pin the pre-2026-08-15 "
                "binary to continue this one."
            )

    declared = getattr(problem, "equilibrium_expressions", {}) or {}
    recorded_raw = getattr(metadata, "equilibrium_target", "") or ""
    try:
        recorded = json.loads(recorded_raw) if recorded_raw else {}
    except json.JSONDecodeError:
        recorded = {"unparsable": recorded_raw}

    if declared == recorded:
        return

    if recorded and not declared:
        raise ConfigError(
            "this checkpoint was written with a declared stationary target and the config "
            "now declares none. the run it resumes subtracted that target's imbalance at "
            "every stage; continuing without it integrates different equations. restore the "
            "`equilibrium_expressions` the run started with, or start a fresh run."
        )
    if declared and not recorded:
        raise ConfigError(
            "the config declares a stationary target but this checkpoint was written "
            "without one. adding a target mid-run changes the equations being integrated. "
            "start a fresh run to introduce it."
        )
    raise ConfigError(
        "the declared stationary target differs from the one this checkpoint was written "
        "with. the target sets which state the scheme holds exactly and which imbalance it "
        "subtracts every stage, so resuming against a different one silently integrates "
        "different equations. restore the original target, or start a fresh run."
    )


def merge_with_checkpoint(
    problem: SimbiProblem,
    checkpoint_path: Path,
) -> SimbiProblem:
    """
    merge user problem config with checkpoint metadata.

    - checkpoint_safe=True fields: user value is kept
    - checkpoint_safe=False fields: checkpoint value is used

    raises ConfigError if the user EXPLICITLY passed a flag for an immutable
    field with a value that disagrees with the checkpoint (a class default that
    differs is not an override — the checkpoint wins silently).

    args:
        problem: user-provided problem configuration
        checkpoint_path: path to checkpoint file

    returns:
        new problem instance with merged configuration
    """
    metadata, mesh_shape = load_checkpoint_metadata(checkpoint_path)
    checkpoint_config = metadata_to_config_dict(metadata, mesh_shape)

    # get field classifications
    immutable_fields = problem.get_checkpoint_immutable_fields()
    safe_fields = problem.get_checkpoint_safe_fields()

    # build merged config
    merged_data: dict[str, Any] = {}

    def normalize_value(val):
        """normalize value for pydantic validation."""
        if isinstance(val, Path):
            return str(val)
        elif hasattr(val, "value"):  # enum
            return val
        elif hasattr(val, "dtype"):  # numpy scalar
            return val.item()
        return val

    def coerce_to_field_type(value, field_info):
        """
        coerce checkpoint value to match field's expected type.
        uses pydantic validation to try each union member.
        """
        from pydantic import TypeAdapter, ValidationError

        annotation = field_info.annotation
        args = get_args(annotation)

        # filter out NoneType from optional fields
        union_members = [a for a in args if a is not type(None)] if args else []

        if len(union_members) > 1:
            # union type - try each member with pydantic validation
            for member_type in union_members:
                try:
                    adapter = TypeAdapter(member_type)
                    return adapter.validate_python(value)
                except (ValidationError, TypeError):
                    continue

            # failed - if single-element sequence, try unwrapping
            if isinstance(value, (list, tuple)) and len(value) == 1:
                for member_type in union_members:
                    try:
                        adapter = TypeAdapter(member_type)
                        return adapter.validate_python(value[0])
                    except (ValidationError, TypeError):
                        continue

            return value

        # not a union - try direct validation
        try:
            adapter = TypeAdapter(annotation)
            return adapter.validate_python(value)
        except (ValidationError, TypeError):
            # a lower-dimensional run restores from the 3-padded checkpoint
            # resolution (nx, 1, 1): trailing 1s carry no information, so a
            # scalar field takes the leading entry.
            if (
                isinstance(value, (list, tuple))
                and len(value) > 1
                and all(v == 1 for v in value[1:])
            ):
                try:
                    adapter = TypeAdapter(annotation)
                    return adapter.validate_python(value[0])
                except (ValidationError, TypeError):
                    pass
            # validation failed - try unwrapping sequences
            if isinstance(value, (list, tuple)):
                # if all elements are identical, extract first
                if len(value) > 0 and all(v == value[0] for v in value):
                    try:
                        adapter = TypeAdapter(annotation)
                        return adapter.validate_python(value[0])
                    except (ValidationError, TypeError):
                        pass
                # single-element sequence
                elif len(value) == 1:
                    try:
                        adapter = TypeAdapter(annotation)
                        return adapter.validate_python(value[0])
                    except (ValidationError, TypeError):
                        pass
            return value

    # the flags the user EXPLICITLY chose: from_cli records the argv-provided
    # dests; a directly-constructed problem carries the same fact in
    # model_fields_set. a class default that merely differs from the checkpoint
    # is NOT an override — the checkpoint wins silently.
    cli_explicit = getattr(problem, "_cli_explicit", None)
    explicit = cli_explicit if cli_explicit is not None else problem.model_fields_set

    for field_name, field_info in type(problem).model_fields.items():
        user_value = normalize_value(getattr(problem, field_name))

        if field_name in checkpoint_config:
            checkpoint_value = checkpoint_config[field_name]

            if field_name in immutable_fields:
                # must use checkpoint value, but coerce to field's expected type
                coerced = coerce_to_field_type(checkpoint_value, field_info)
                # an EXPLICIT user demand for a different value on an immutable
                # field cannot be honored — refuse loudly; silently running the
                # checkpoint's setting under the user's flag would hide the conflict.
                if field_name in explicit and not _values_agree(user_value, coerced):
                    from .problem import ConfigError

                    raise ConfigError(
                        f"'{field_name}' cannot be changed on restart: the checkpoint "
                        f"was written with {coerced!r} but the command line asks for "
                        f"{user_value!r}. drop the flag to continue the run as "
                        f"recorded, or start a fresh run (no --checkpoint) to change it."
                    )
                merged_data[field_name] = coerced
            else:
                # user can override
                merged_data[field_name] = user_value
        else:
            # field not in checkpoint, use user value
            merged_data[field_name] = user_value

    # the CLOSURE is not a knob a restart may drift on: the state in the file was integrated
    # under one equation of state, and continuing it under another is a different physics
    # problem wearing the same file name. unlike every other immutable field this refuses even
    # when the config merely DEFAULTS to the other closure — a config EDITED between the
    # original run and the restart (gamma-law -> synge, say) carries no explicit flag to catch,
    # and the silent checkpoint-wins rule would resume the old equations under a source file
    # that reads as the new ones.
    recorded_closure = checkpoint_config.get("eos")
    if recorded_closure is not None and recorded_closure != problem.eos:
        from .problem import ConfigError

        # the repair is a CONFIG edit, not a flag: `--eos` alone flips the closure without
        # the adiabatic_index that has to appear (ideal) or disappear (synge) alongside it,
        # so the model validator refuses the problem before the merge is ever reached.
        if recorded_closure == Eos.SYNGE:
            repair = "set `eos = Eos.SYNGE` in the config and drop its adiabatic_index"
        else:
            gamma = checkpoint_config.get("adiabatic_index")
            repair = (
                f"set `eos = Eos.IDEAL` and `adiabatic_index = {gamma!r}` in the config"
            )
        raise ConfigError(
            f"the closure cannot change on restart: the checkpoint was integrated with "
            f"eos = '{recorded_closure.value}' but this config declares "
            f"'{problem.eos.value}'. resuming would continue the file's state under "
            f"different equations of state. to continue the run as recorded, {repair}; "
            f"to use the new closure, start a fresh run (no --checkpoint)."
        )

    # the synge (taub-mathews) closure declares NO adiabatic index: the model validator
    # REFUSES a user-supplied one and then writes the inert placeholder 5/3 itself, purely
    # so the plumbing has a float to carry. the backend records that placeholder as the
    # checkpoint's `gamma`, so the merge above hands it straight back to the constructor —
    # where it reads as a DECLARED index and the validator kills the restart. drop it and
    # let the closure re-supply its own placeholder. a gamma-law restart is untouched:
    # its checkpoint gamma is real physics and stays immutable.
    if merged_data.get("eos") == Eos.SYNGE:
        merged_data.pop("adiabatic_index", None)

    _assert_same_equilibrium_target(problem, metadata)

    # RESUME at the checkpoint's physical time. start_time is the
    # sim clock (sim.time): a restart must continue from where the checkpoint left off, regardless
    # of what the config (or its validators) set start_time to. the LOG-checkpoint anchor is a
    # SEPARATE field (checkpoint_log_anchor) so the cadence stays fixed across the restart.
    if "start_time" in checkpoint_config:
        merged_data["start_time"] = checkpoint_config["start_time"]

    # the checkpoint INDEX is resume state: the next dump must continue the
    # monotonic numbering from where the run stopped (chkpt.030 -> chkpt.031), so force it from the
    # checkpoint like start_time. it is checkpoint_safe (serializable, user-visible), so the generic
    # field merge would otherwise reset it to the config default 0 and re-number every restart from
    # zero — silently overwriting the earlier checkpoints on disk.
    if "checkpoint_index" in checkpoint_config:
        merged_data["checkpoint_index"] = checkpoint_config["checkpoint_index"]

    # special handling for end_time: allow extending
    if "end_time" in safe_fields:
        user_end = getattr(problem, "end_time")
        checkpoint_start = checkpoint_config.get("start_time", 0.0)
        merged_data["end_time"] = max(user_end, checkpoint_start)

    # normalize types for pydantic validation
    # convert all Path objects to strings
    for key, value in merged_data.items():
        if isinstance(value, Path):
            merged_data[key] = str(value)

    # convert numpy integers to python ints
    for key, value in merged_data.items():
        if hasattr(value, "dtype"):  # numpy scalar
            merged_data[key] = value.item()
        elif isinstance(value, (list, tuple)):
            # check if list contains numpy types
            if value and hasattr(value[0], "dtype"):
                merged_data[key] = [
                    v.item() if hasattr(v, "dtype") else v for v in value
                ]

    # create new instance with merged config
    return type(problem)(**merged_data)
