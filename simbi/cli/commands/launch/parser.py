# =============================================================================
# simbi/cli/commands/launch/parser.py
#
# the `simbi launch` parser: the config script, the layout file, the checkpoint
# to resume from, and the worker-role flags the launcher passes to the
# processes it spawns. problem parameters pass through to the config's own
# parser exactly as `simbi run` forwards them.
#
# usage:
#   simbi launch kh.py --layout cluster.toml [--checkpoint file.h5] [problem flags]
# =============================================================================
from argparse import Namespace
from typing import Optional

from ...utils.formatter import HelpFormatter
from ..run.parser import _validate_config_script


def setup_parser(subparsers) -> None:
    """setup launch command parser."""
    parser = subparsers.add_parser(
        "launch",
        help="evolve a problem across worker processes over the fabric",
        formatter_class=HelpFormatter,
        usage="simbi launch <config> --layout <layout.toml> [options]",
        add_help=False,
    )
    parser.add_argument(
        "config_script",
        nargs="?",
        default=None,
        help="config file or registered config name",
        type=_validate_config_script,
    )
    layout = parser.add_argument_group("layout")
    layout.add_argument(
        "--layout",
        dest="layout",
        default=None,
        help="the execution layout: [execution] workers, [partition] shape or cuts, "
        "[checkpoint] directory and staging_mb",
    )
    checkpoint = parser.add_argument_group("checkpoint")
    checkpoint.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=None,
        help="checkpoint file to resume from",
    )
    bounded = parser.add_argument_group("utilities")
    bounded.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=0,
        help="stop after this many steps (0 = run to end_time); the final checkpoint is written either way",
    )
    bounded.add_argument(
        "--info",
        action="store_true",
        dest="info",
        default=False,
        help="show the problem's configurable flags without running",
    )
    # the worker role: set by the launcher on the processes it spawns
    worker = parser.add_argument_group("worker role (set by the launcher)")
    worker.add_argument("--worker", dest="worker", type=int, default=None, help="this worker's id")
    worker.add_argument("--workers", dest="workers", type=int, default=None, help="the worker count")
    worker.add_argument("--rendezvous", dest="rendezvous", default=None, help="the rendezvous file")
    worker.add_argument("--credential", dest="credential", type=int, default=None, help="the session credential")
    worker.add_argument("--owner", dest="owner", default=None, help="the tile owner map, comma separated")
    worker.add_argument("--cuts", dest="cuts", default=None, help="the per-axis cuts, semicolon separated axes")
    worker.add_argument("--staging-cells", dest="staging_cells", type=int, default=None, help="checkpoint block cells")
    parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """execute launch command."""
    from .executor import launch_config

    launch_config(args, argv)
