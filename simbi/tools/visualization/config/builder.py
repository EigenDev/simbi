from typing import Any, Sequence, Union
import argparse
from .metadata import ArgumentMapping


class ConfigBuilder:
    """Builds structured configuration from various input sources"""

    @classmethod
    def from_args(
        cls, args: Union[argparse.Namespace, dict[str, Any]]
    ) -> dict[str, Any]:
        """Build configuration from argparse args or kwargs dictionary"""
        # Convert Namespace to dict if needed
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Initialize empty config structure
        config: dict[str, dict[str, Any]] = {
            "plot": {},
            "style": {},
            "multidim": {},
            "animation": {},
        }

        # Distribute arguments to their proper sections
        for arg_name, value in args.items():
            # Skip None values and private args (starting with _)
            if value is None or arg_name.startswith("_"):
                continue

            # Get section, possible transformation, and final name
            section = ArgumentMapping.get_section(arg_name)
            transform = ArgumentMapping.get_transform(arg_name)
            final_name = ArgumentMapping.get_name(arg_name)

            # Apply transformation if needed
            if transform is not None:
                value = transform(value)

            # Add to config
            config[section][final_name] = value

        return config

    @classmethod
    def merge_with_defaults(
        cls, config: dict[str, Any], defaults: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user config with defaults, preserving user values"""
        result = {**defaults}

        # Merge each section
        for section in result:
            if section in config:
                result[section] = {**result.get(section, {}), **config.get(section, {})}

        return result

    @classmethod
    def create_config(
        cls,
        plot_type: str,
        files: Sequence[str] = None,
        fields: Sequence[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a complete configuration from parameters and kwargs"""
        # Create basic config from explicit parameters
        config = {
            "plot": {
                "plot_type": plot_type,
                "files": files or [],
                "fields": fields or ["rho"],
            }
        }

        # Add kwargs to config
        kwargs_config = cls.from_args(kwargs)

        # Merge configs
        for section, values in kwargs_config.items():
            if section not in config:
                config[section] = {}
            config[section].update(values)

        return config
