# =============================================================================
# param.py
#
# field factory for SimbiProblem parameters.
# wraps pydantic Field with cli and checkpoint metadata.
#
# usage:
#   resolution: int = ProblemParam(1000, cli=True, description="grid resolution")
#   cfl_number: float = ProblemParam(0.4, cli=True, checkpoint_safe=True)
# =============================================================================
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import Field
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class ParamMetadata:
    """metadata for cli and checkpoint behavior."""

    cli: bool = False
    checkpoint_safe: bool = False
    cli_name: Optional[str] = None


def ProblemParam(
    default: Any = ...,
    *,
    cli: bool = False,
    checkpoint_safe: bool = False,
    cli_name: Optional[str] = None,
    description: Optional[str] = None,
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
        # required field, not cli-configurable, must match checkpoint
        bounds: Sequence[Sequence[float]] = ProblemParam(..., description="domain bounds")

        # optional with default, cli-configurable, safe to override on restart
        cfl_number: float = ProblemParam(0.4, cli=True, checkpoint_safe=True)

        # optional with default, cli-configurable, must match checkpoint
        resolution: int = ProblemParam(1000, cli=True, checkpoint_safe=False)
    """
    metadata = ParamMetadata(
        cli=cli,
        checkpoint_safe=checkpoint_safe,
        cli_name=cli_name,
    )

    return Field(
        default=default,
        description=description,
        json_schema_extra={"param_metadata": metadata},
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
