# =============================================================================
# param.py
#
# field factory for SimbiProblem parameters.
# wraps pydantic Field with cli and checkpoint metadata.
#
# usage:
#   resolution: Annotated[int, ProblemParam(1000, cli=True, description="grid resolution")]
#   cfl_number: Annotated[float, ProblemParam(0.4, cli=True, checkpoint_safe=True)]
# =============================================================================
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from pydantic import Field
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class ParamMetadata:
    """metadata for cli and checkpoint behavior."""

    cli: bool = False
    checkpoint_safe: bool = False
    cli_name: Optional[str] = None
    # optional section label for the live-dashboard "problem setup" panel; custom params sharing a
    # group render together. None -> the default "Parameters" group.
    group: Optional[str] = None
    # the admissible values, when a parameter takes one of a fixed set. carried as
    # metadata consumed by the cli and checkpoint layers; this type does not validate membership.
    choices: Optional[tuple[Any, ...]] = None


def ProblemParam(
    default: Any = ...,
    *,
    cli: bool = False,
    checkpoint_safe: bool = False,
    cli_name: Optional[str] = None,
    description: Optional[str] = None,
    group: Optional[str] = None,
    choices: Optional[Sequence[Any]] = None,
    **field_kwargs: Any,
) -> FieldInfo:
    """
    create a problem parameter field with cli and checkpoint metadata.

    args:
        default: default value (use ... for required fields)
        cli: if True, expose this field as a CLI argument
        checkpoint_safe: if True, user can override this value when restarting
                        from checkpoint. if False, value must match checkpoint.
        cli_name: custom cli argument name (defaults to kebab-case of field name)
        description: help text for cli and documentation
        **field_kwargs: additional pydantic Field arguments

    returns:
        pydantic FieldInfo with embedded ParamMetadata

    examples:
        # required field, checkpoint-locked, must match checkpoint
        bounds: Annotated[Sequence[Sequence[float]], ProblemParam(..., description="domain bounds")]

        # optional with default, cli-configurable, safe to override on restart
        cfl_number: Annotated[float, ProblemParam(0.4, cli=True, checkpoint_safe=True)]

        # optional with default, cli-configurable, must match checkpoint
        resolution: Annotated[int, ProblemParam(1000, cli=True, checkpoint_safe=False)]
    """
    # `choices` is ours, not pydantic's: forwarding it into `Field` as an extra keyword
    # is deprecated and becomes an error in pydantic v3, so it rides in the metadata the
    # rest of the parameter description already travels in.
    metadata = ParamMetadata(
        cli=cli,
        checkpoint_safe=checkpoint_safe,
        cli_name=cli_name,
        group=group,
        choices=tuple(choices) if choices is not None else None,
    )
    extra: dict[str, Any] = {"param_metadata": metadata}
    return Field(
        default=default,
        description=description,
        json_schema_extra=extra,
        **field_kwargs,
    )


def get_param_metadata(field_info: FieldInfo) -> ParamMetadata:
    """extract ParamMetadata from a field, returning defaults if not present."""
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        metadata = extra.get("param_metadata")
        if isinstance(metadata, ParamMetadata):
            return metadata
    return ParamMetadata()
