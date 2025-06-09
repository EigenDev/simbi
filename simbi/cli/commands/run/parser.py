from argparse import (
    ArgumentParser,
    Namespace,
    HelpFormatter,
    ArgumentTypeError,
    BooleanOptionalAction,
)
from typing import Optional
from ....detail import max_thread_count
from ...actions import ComputeModeAction, RegisterGPUBlockDimensions
from ....detail import bcolors
from pathlib import Path
from ...actions import print_available_configs, get_available_configs
from ...utils.formatter import HelpFormatter


def validate_simbi_script(param):
    path = Path(param)
    base = path.stem
    ext = path.suffix
    available_configs = get_available_configs()
    if ext.lower() != ".py":
        param = None
        for file in available_configs:
            if base == Path(file).stem:
                param = file
            elif base.replace("-", "_") == Path(file).stem:
                param = file

        if not param:
            available_configs = sorted([Path(conf).stem for conf in available_configs])
            raise ArgumentTypeError(
                "No configuration named {}{}{}. The valid configurations are:\n{}".format(
                    bcolors.OKCYAN,
                    base,
                    bcolors.ENDC,
                    "".join(
                        f"> {bcolors.BOLD}{conf.replace('_', '-')}{bcolors.ENDC}\n"
                        for conf in available_configs
                    ),
                )
            )
    return param


def _add_global_args(parser: ArgumentParser) -> None:
    """"""
    parser.add_argument(
        "--nthreads",
        "-p",
        help="number of omp threads to run with",
        type=max_thread_count,
        default=None,
    )
    parser.add_argument(
        "--info", help="print setup-script usage", default=False, action="store_true"
    )
    parser.add_argument(
        "--type-check",
        help="flag for static type checking configration files",
        default=True,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--gpu-block-dims",
        help="gpu dim3 thread block dimensions",
        default=[],
        type=int,
        action=RegisterGPUBlockDimensions,
        nargs="+",
    )
    parser.add_argument(
        "--configs",
        help="print the available config files",
        action=print_available_configs,
    )
    parser.add_argument("--cpu", action=ComputeModeAction, const="cpu")
    parser.add_argument("--gpu", action=ComputeModeAction, const="gpu")
    parser.add_argument(
        "--omp",
        action=ComputeModeAction,
        const="omp",
    )


def _add_onthefly_args(parser: ArgumentParser) -> None:
    """"""
    parser.add_argument(
        "--mode",
        help="execution mode for computation",
        default="cpu",
        choices=["cpu", "omp", "gpu"],
        dest="compute_mode",
    )
    parser.add_argument(
        "--checkpoint",
        help="checkpoint file to restart computation from",
        default=None,
        type=str,
    )


def setup_parser(subparsers) -> None:
    """Setup run command parser"""
    run_parser = subparsers.add_parser(
        "run",
        help="runs the setup script",
        formatter_class=HelpFormatter,
        usage="simbi run <setup_script> [options]",
    )

    # Add argument groups
    overridable = run_parser.add_argument_group("override")
    global_args = run_parser.add_argument_group("globals")
    onthefly = run_parser.add_argument_group("onthefly")

    # Add core arguments
    run_parser.add_argument(
        "setup_script",
        help="setup script for simulation run",
        type=validate_simbi_script,
    )

    # Add group arguments
    _add_global_args(global_args)
    _add_onthefly_args(onthefly)

    run_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """Execute run command"""
    from .executor import run_simulation
    from .config import configure_state

    states, state_docs = configure_state(args, argv)
    run_simulation(states, state_docs, args)
