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
import difflib
import sys
from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from typing import Optional

from ...actions import (
    ComputeModeAction,
    RegisterGPUBlockDimensions,
    get_available_configs,
    PrintAvailableConfigsAction,
)
from ...utils.formatter import HelpFormatter


def _resolve_ambiguous(base: str, matches: list[Path]) -> str:
    """several discovered configs share the requested name. on a tty, ask the
    user which one to run; otherwise (scripts, pipes, ci) fail loudly with the
    full paths so the caller can pass one explicitly. a silent first-match pick
    would run whichever file the directory walk visited first."""
    if sys.stdin.isatty() and sys.stderr.isatty():
        print(
            f"found {len(matches)} configs named '{base}':", file=sys.stderr
        )
        for ii, mm in enumerate(matches, 1):
            print(f"  [{ii}] {mm}", file=sys.stderr)
        while True:
            choice = input(
                f"select [1-{len(matches)}], or q to abort: "
            ).strip()
            if choice.lower() in ("q", "quit", ""):
                raise ArgumentTypeError(f"'{base}' selection aborted")
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                return str(matches[int(choice) - 1])
            print(f"invalid choice '{choice}'", file=sys.stderr)
    raise ArgumentTypeError(
        f"'{base}' is ambiguous — {len(matches)} configs share the name:\n"
        + "".join(f"  > {mm}\n" for mm in matches)
        + "pass the config path explicitly."
    )


def _validate_config_script(param: str) -> str:
    """validate and resolve config script path."""
    path = Path(param)
    base = path.stem
    ext = path.suffix

    # an explicit .py path bypasses discovery, but a typo'd path must fail HERE
    # with a clear message, not later inside the loader with an import error.
    if ext.lower() == ".py":
        if not path.is_file():
            raise ArgumentTypeError(f"config script '{param}' does not exist")
        return param

    # otherwise, search available configs. collect EVERY match — the same stem
    # can exist in several config dirs (bundled examples vs a cwd-local tree).
    available_configs = get_available_configs()
    matches = [
        Path(file)
        for file in available_configs
        if base == Path(file).stem or base.replace("-", "_") == Path(file).stem
    ]
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        return _resolve_ambiguous(base, matches)

    # not found - show the closest names first, then everything
    config_names = sorted(
        {Path(f).stem.replace("_", "-") for f in available_configs}
    )
    close = difflib.get_close_matches(
        base.replace("_", "-"), config_names, n=3, cutoff=0.5
    )
    hint = (
        "did you mean: " + ", ".join(close) + "?\n" if close else ""
    )
    raise ArgumentTypeError(
        f"no configuration named '{base}'. {hint}available configs:\n"
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
        action=PrintAvailableConfigsAction,
    )

    run_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """execute run command."""
    from .executor import run_config

    run_config(args, argv)
