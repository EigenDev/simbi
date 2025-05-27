"""
CLI parameter handling for simbi configurations.

This module provides the base model for CLI-configurable Pydantic models.
"""

from pydantic import BaseModel, ConfigDict, ValidationError
from typing import (
    ClassVar,
    Optional,
    get_origin,
    get_args,
    Union,
    Any,
    cast,
    TypedDict,
)
import argparse
from pathlib import Path
from typing_extensions import TypeAlias


# Define types for CLI info structure
class CLIInfo(TypedDict):
    cli_name: Optional[str]
    help_text: Optional[str]
    choices: Optional[list[Any]]
    expose_cli: bool


# Define JsonSchema types for better type checking
JsonValue: TypeAlias = Union[
    int, float, str, bool, list["JsonValue"], dict[str, "JsonValue"], None
]


class CLIConfigurableModel(BaseModel):
    """Base model for CLI-configurable Pydantic models.

    This class extends Pydantic's BaseModel with methods for integrating
    with command-line interfaces. It processes SimbiField metadata to
    automatically register CLI parameters.
    """

    # Allow arbitrary types for functions, callables, etc.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Store CLI parser for class access
    cli_parser: ClassVar[Optional[argparse.ArgumentParser]] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: dict[str, Any]) -> None:
        """Process field metadata when subclass is created."""
        super().__pydantic_init_subclass__(**kwargs)

        # Process field names for fields with CLI info
        for field_name, field_info in cls.model_fields.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Check if this field has CLI info
            extra = cast(Optional[dict[str, Any]], field_info.json_schema_extra)
            if extra and "cli_info" in extra:
                cli_info = cast(CLIInfo, extra["cli_info"])

                # If cli_name not set, convert field name to kebab-case
                if cli_info["cli_name"] is None:
                    cli_info["cli_name"] = field_name.replace("_", "-")

    @classmethod
    def register_cli_parameters(cls, parser: argparse.ArgumentParser) -> None:
        """Register CLI parameters based on field metadata.

        Args:
            parser: The argument parser to register parameters with
        """
        group_name = cls.__name__
        group = parser.add_argument_group(f"{group_name} parameters")

        for field_name, field_info in cls.model_fields.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Get CLI info if available
            cli_info: Optional[CLIInfo] = None
            extra = cast(Optional[dict[str, Any]], field_info.json_schema_extra)

            if extra and "cli_info" in extra:
                cli_info = cast(CLIInfo, extra["cli_info"])

            if cli_info and cli_info.get("expose_cli", True):
                # This field should be exposed on CLI
                cli_name = cli_info["cli_name"]
                kwargs: dict[str, Any] = {
                    "help": cli_info["help_text"]
                    or field_info.description
                    or f"Set {field_name}",
                    "dest": field_name,
                }

                # Add choices if specified
                if cli_info["choices"]:
                    kwargs["choices"] = cli_info["choices"]

                # Set default if available
                if field_info.default is not None and field_info.default is not ...:
                    kwargs["default"] = field_info.default

                # Handle different types
                cls._add_type_info(kwargs, field_info)

                # Add the argument to the parser
                group.add_argument(f"--{cli_name}", **kwargs)

    @classmethod
    def _add_type_info(cls, kwargs: dict[str, Any], field_info: Any) -> None:
        """Add type information to kwargs for argparse.

        Args:
            kwargs: Dictionary of argument parser kwargs to update
            field_info: Field information from Pydantic model
        """
        # Handle boolean fields
        if isinstance(field_info.annotation, bool):
            default = field_info.default
            if default is None or default is ...:
                default = False
            kwargs["action"] = "store_true" if not default else "store_false"
            return

        # Handle simple types directly
        simple_types = {str, int, float, Path}
        if field_info.annotation in simple_types:
            kwargs["type"] = field_info.annotation
            return

        if get_origin(field_info.annotation) in (tuple, list):
            # Custom converter for tuple/list types
            element_type = get_args(field_info.annotation)[0]  # Get the element type
            kwargs["type"] = lambda x: tuple(
                element_type(item) for item in x.split(",")
            )
            kwargs["help"] = f"{kwargs.get('help', '')} (comma-separated values)"
            return

        # Handle Optional[T] - extract the inner type
        origin = get_origin(field_info.annotation)
        if origin is Union:
            args = get_args(field_info.annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1 and non_none_args[0] in simple_types:
                kwargs["type"] = non_none_args[0]
                return

    @classmethod
    def setup_cli(cls, subparser: argparse.ArgumentParser) -> None:
        """Set up CLI for this configuration.

        Args:
            parser: Optional existing parser to extend

        Returns:
            The configured parser
        """
        cls.cli_parser = subparser
        cls.register_cli_parameters(subparser)

    @classmethod
    def from_cli(cls, main_parser: argparse.ArgumentParser) -> "CLIConfigurableModel":
        """Create configuration from command line arguments.

        Args:
            args: Optional list of command line arguments

        Returns:
            An instance of this class with values from CLI args
        """
        assert cls.cli_parser is not None
        namespace = main_parser.parse_args()
        return cls.from_namespace(namespace)

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "CLIConfigurableModel":
        """Create instance from parsed namespace.

        Args:
            namespace: The parsed argument namespace

        Returns:
            An instance of this class with values from the namespace
        """
        data = {}
        for field_name in cls.model_fields:
            if hasattr(namespace, field_name):
                value = getattr(namespace, field_name)
                if value is not None:
                    data[field_name] = value

        # Let Pydantic handle validation
        try:
            return cls(**data)
        except ValidationError as e:
            # Provide a clearer error message
            error_msgs = []
            for err in e.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                error_msgs.append(f"Error in {field}: {err['msg']}")

            error_summary = "\n".join(error_msgs)
            raise ValueError(f"Invalid configuration values:\n{error_summary}") from e
