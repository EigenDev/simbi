# =============================================================================
# run/parser.py
#
# simplified cli for running simbi simulations.
# discovers problem classes from config files and runs them.
#
# usage:
#   simbi run sod --end-time 0.2 --resolution 1000
#   simbi run sod --checkpoint data/checkpoint.h5 --end-time 1.0
# =============================================================================
from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from typing import Optional

from ...actions import (
    ComputeModeAction,
    RegisterGPUBlockDimensions,
    get_available_configs,
    print_available_configs,
)
from ...utils.formatter import HelpFormatter


def _validate_config_script(param: str) -> str:
    """validate and resolve config script path."""
    path = Path(param)
    base = path.stem
    ext = path.suffix

    # if it's already a .py file, return as-is
    if ext.lower() == ".py":
        return param

    # otherwise, search available configs
    available_configs = get_available_configs()
    for file in available_configs:
        file_stem = Path(file).stem
        # match exact name or kebab-case variant
        if base == file_stem or base.replace("-", "_") == file_stem:
            return str(file)

    # not found - show helpful error
    config_names = sorted(
        [Path(f).stem.replace("_", "-") for f in available_configs]
    )
    raise ArgumentTypeError(
        f"no configuration named '{base}'. available configs:\n"
        + "".join(f"  > {name}\n" for name in config_names)
    )


def setup_parser(subparsers) -> None:
    """setup run command parser."""
    run_parser = subparsers.add_parser(
        "run",
        help="run a simulation from a config file",
        formatter_class=HelpFormatter,
        usage="simbi run <config> [options]",
        # we handle -h/--help ourselves (below) so that `simbi run <config> --help`
        # can load the config and show ITS flags, not just the generic run options.
        add_help=False,
    )

    # main argument: optional so `simbi run --help` / `--configs` work with no config.
    run_parser.add_argument(
        "config_script",
        nargs="?",
        default=None,
        help="config file or registered config name (e.g., 'sod' or 'sod.py')",
        type=_validate_config_script,
    )

    # compute mode
    mode_group = run_parser.add_argument_group("compute mode")
    mode_group.add_argument(
        "--mode",
        help="execution mode",
        default="cpu",
        choices=["cpu", "omp", "gpu"],
        dest="compute_mode",
    )
    mode_group.add_argument("--cpu", action=ComputeModeAction, const="cpu")
    mode_group.add_argument("--gpu", action=ComputeModeAction, const="gpu")
    mode_group.add_argument("--omp", action=ComputeModeAction, const="omp")

    # checkpoint
    checkpoint_group = run_parser.add_argument_group("checkpoint")
    checkpoint_group.add_argument(
        "--checkpoint",
        help="checkpoint file to resume from",
        default=None,
        type=str,
    )

    # gpu options
    gpu_group = run_parser.add_argument_group("gpu options")
    gpu_group.add_argument(
        "--tile-block-dims",
        help="gpu block dimensions (x y z)",
        default=[],
        type=int,
        action=RegisterGPUBlockDimensions,
        nargs="+",
    )

    # monitoring
    monitor_group = run_parser.add_argument_group("monitoring")
    monitor_group.add_argument(
        "--live",
        dest="live_monitor",
        help="write a read-only snapshot each cadence so `simbi attach <data_dir>` "
        "can monitor a headless (batch/cluster) run",
        default=False,
        action="store_true",
    )

    # utilities
    util_group = run_parser.add_argument_group("utilities")
    # -h/--help/--peek/--info all show the config's flags: with a <config> given they
    # print THAT problem's parameters (--gamma-shock0, --has-mesh-motion, ...); with no
    # config they print the generic run help. one flag, all the aliases a user reaches for.
    util_group.add_argument(
        "-h",
        "--help",
        "--peek",
        "--info",
        dest="info",
        help="show the problem's configurable flags (or the generic run help) without running",
        default=False,
        action="store_true",
    )
    util_group.add_argument(
        "--configs",
        help="list available configs",
        action=print_available_configs,
    )

    run_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """execute run command."""
    from .executor import run_config

    run_config(args, argv)
