from typing import Any, Optional

from pydantic import Field


def ProblemField(
    default: Any = ...,
    *,
    description: Optional[str] = None,
    cli_name: Optional[str] = None,
    help_text: Optional[str] = None,
    choices: Optional[list[Any]] = None,
    expose_cli: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create a field that is both a Pydantic field and a CLI parameter.

    Any field defined with ProblemField in a BaseProblemConfig subclass
    will automatically:
    1. Be validated by Pydantic
    2. Become a CLI argument (if expose_cli=True)
    3. Be documented in help text
    4. Support tab completion (if using argcomplete)

    Args:
        default: Default value (use ... for required fields)
        description: Field description for docs and help
        cli_name: Override CLI parameter name (default: field name with dashes)
        help_text: CLI help text (default: use description)
        choices: Valid choices for CLI parameter
        expose_cli: Whether to expose on CLI (default: True)
        **kwargs: Additional Pydantic Field arguments

    Example:
        >>> class MyProblem(BaseProblemConfig):
        ...     resolution: int = ProblemField(
        ...         1000,
        ...         description="Number of grid cells",
        ...         cli_name="res",
        ...     )

        This creates: --res argument with default 1000
    """
    # Store CLI information in json_schema_extra
    extra = kwargs.pop("json_schema_extra", {})

    # Set up CLI info
    extra.update(
        {
            "cli_info": {
                "cli_name": cli_name,  # Will be set properly during __set_name__
                "help_text": help_text or description,
                "choices": choices,
                "expose_cli": expose_cli,
            }
        }
    )

    # Pass description to both Field and CLI info
    return Field(
        default, description=description, json_schema_extra=extra, **kwargs
    )
