from pydantic import Field
from typing import Any, Optional


def SimbiField(
    default: Any = ...,
    *,
    description: Optional[str] = None,
    name: Optional[str] = None,
    help_text: Optional[str] = None,
    choices: Optional[list[Any]] = None,
    expose_cli: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create a field that is both a Pydantic field and a CLI parameter.

    Args:
        default: Default value
        description: Field description for documentation
        cli_name: Name to use for CLI parameter (defaults to field name)
        help_text: Help text for CLI (defaults to description)
        cli_choices: Valid choices for CLI parameter
        expose_cli: Whether to expose this field on CLI
        **kwargs: Additional arguments for Field

    Returns:
        A Pydantic Field with CLI metadata
    """

    # Store CLI information in json_schema_extra
    extra = kwargs.pop("json_schema_extra", {})

    # Set up CLI info
    extra.update(
        {
            "cli_info": {
                "cli_name": name,  # Will be set properly during __set_name__
                "help_text": help_text or description,
                "choices": choices,
                "expose_cli": expose_cli,
                # "defined_at": caller_info,
            }
        }
    )

    # Pass description to both Field and CLI info
    return Field(default, description=description, json_schema_extra=extra, **kwargs)
