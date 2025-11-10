from pathlib import Path
from typing import Iterator


class ConfigDiscovery:
    """Handles finding and validating simulation config files"""

    CONFIG_DIR = "simbi_configs"

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize config name to underscore format

        kelvin-helmholtz -> kelvin_helmholtz
        KELVINhelmholtz -> kelvin_helmholtz
        """
        return name.replace("-", "_").lower()

    @classmethod
    def find_configs_dir(cls) -> Path | None:
        """Find simbi_configs directory in current working directory"""
        cwd = Path.cwd()
        config_dir = cwd / cls.CONFIG_DIR
        return config_dir if config_dir.is_dir() else None

    @classmethod
    def available_configs(cls) -> Iterator[Path]:
        """Get all available config files"""
        config_dir = cls.find_configs_dir()
        if not config_dir:
            return iter([])
        return config_dir.glob("**/*.py")

    @classmethod
    def find_config(cls, name: str) -> Path:
        """Find config file from name or path

        Supports:
        - Full paths: /path/to/config.py
        - Names with dashes: kelvin-helmholtz
        - Names with underscores: kelvin_helmholtz
        - Just names: sod (will look for sod.py)
        """
        # Handle full paths
        if "/" in name or "\\" in name:
            path = Path(name)
            if not path.exists():
                raise ValueError(f"Config file not found: {path}")
            if path.suffix.lower() != ".py":
                raise ValueError(f"Config file must be a .py file: {path}")
            return path

        # Look in simbi_configs directory
        config_dir = cls.find_configs_dir()
        if not config_dir:
            raise ValueError(
                f"No '{cls.CONFIG_DIR}' directory found in current directory"
            )

        # Normalize name and look for match
        normalized = cls.normalize_name(name)
        for config in cls.available_configs():
            if cls.normalize_name(config.stem) == normalized:
                return config

        # No match found - show available configs
        available = sorted(
            [p.stem.replace("_", "-") for p in cls.available_configs()]
        )

        raise ValueError(
            f"No config named '{name}' found in {cls.CONFIG_DIR}/\n"
            "Available configs:\n"
            + "\n".join(f"- {name}" for name in available)
        )
