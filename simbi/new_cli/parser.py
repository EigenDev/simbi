import importlib.util
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from simbi.simulation.base import BaseProblemConfig
from simbi.simulation.discovery import ConfigDiscovery


@dataclass
class GlobalConfig:
    """Global simulation settings"""

    mode: Literal["cpu", "gpu", "omp"] = "cpu"
    threads: int | None = None
    checkpoint: Path | None = None
    gpu_block_dims: list[int] = field(default_factory=list)


def find_config_file(name: str) -> Path:
    """Find config file from name, supporting multiple formats

    Examples:
        kelvin-helmholtz -> configs/kelvin_helmholtz.py
        kelvin_helmholtz -> configs/kelvin_helmholtz.py
        /full/path/to/custom.py -> custom.py
    """
    name = name.replace("-", "_")

    # Full path provided
    if "/" in name or "\\" in name:
        path = Path(name)
        if path.suffix.lower() != ".py":
            raise ValueError(f"Config file must be a .py file: {name}")
        return path

    # Just name provided - search in configs dir
    configs_dir = Path(__file__).parent.parent / "configs"
    candidates = [p for p in configs_dir.glob("*.py") if p.stem == name]

    if not candidates:
        available = [p.stem.replace("_", "-") for p in configs_dir.glob("*.py")]
        raise ValueError(
            f"No config named '{name}'. Available configs:\n"
            + "\n".join(f"- {n}" for n in sorted(available))
        )

    return candidates[0]


def load_problem_config(path: Path) -> BaseProblemConfig:
    """Load problem config from path"""
    spec = importlib.util.spec_from_file_location("config", path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Config class must be named Config or ProblemConfig
    for _, obj in vars(module).items():
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseProblemConfig)
            and obj is not BaseProblemConfig
        ):  # Don't return the base class itself
            return obj()

    raise ValueError(f"No Config or ProblemConfig class found in {path}")


def parse_args() -> tuple[GlobalConfig, BaseProblemConfig]:
    """Parse CLI args into global and problem configs"""
    parser = ArgumentParser()

    # Global args
    parser.add_argument("config", help="Problem config file or name")
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu", "omp"],
        default="cpu",
        help="Compute mode",
    )
    parser.add_argument(
        "--threads", type=int, help="Thread count for CPU/OMP mode"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Checkpoint file to restart from"
    )
    parser.add_argument(
        "--gpu-block-dims", type=int, nargs="+", help="GPU block dimensions"
    )

    # First parse to get config file
    partial_args, remaining = parser.parse_known_args()

    # Load problem config
    try:
        config_path = ConfigDiscovery.find_config(partial_args.config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    problem_config = load_problem_config(config_path)

    # Add problem-specific args from ProblemFields
    for field_name, val in BaseProblemConfig.model_fields.items():
        schema = val.json_schema_extra
        if isinstance(schema, dict):
            if not isinstance(schema["cli_info"], dict):
                continue

            if not schema["cli_info"]["expose_cli"]:
                continue

            cli_info = schema["cli_info"]
            parser.add_argument(
                f"--{cli_info['cli_name'] or field_name.replace('_', '-')}",
                help=cli_info["help_text"],
                choices=cli_info["choices"],
                default=val.default,
            )

    # Parse all args
    args = parser.parse_args()

    # Create configs
    global_config = GlobalConfig(
        mode=args.mode,
        threads=args.threads,
        checkpoint=args.checkpoint,
        gpu_block_dims=args.gpu_block_dims or [],
    )

    # Update problem config from CLI args
    for field_name, val in BaseProblemConfig.model_fields.items():
        schema = val.json_schema_extra
        if isinstance(schema, dict):
            if not isinstance(schema["cli_info"], dict):
                continue

            if schema["cli_info"]["expose_cli"]:
                cli_name = schema["cli_info"]["cli_name"] or field_name
                if isinstance(cli_name, str):
                    cli_name = cli_name.replace("-", "_")
                    setattr(problem_config, field_name, getattr(args, cli_name))

    return global_config, problem_config


if __name__ == "__main__":
    global_cfg, problem_cfg = parse_args()
    print("Global Config:", global_cfg)
    print("Problem Config:", problem_cfg)
