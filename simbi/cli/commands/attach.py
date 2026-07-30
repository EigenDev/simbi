# =============================================================================
# simbi/cli/commands/attach.py
#
# `simbi attach <data_dir>` — read-only live monitor for a headless run.
# a run started with `--live` writes a snapshot to <data_dir>/.simbi-live/ each
# diagnostic cadence; this polls that file and renders the same tabbed dashboard
# the interactive run shows. one-way: the client never writes back, so pausing /
# stepping / checkpointing are absent (they do not apply to a remote run).
# =============================================================================
import sys
from argparse import Namespace, _SubParsersAction
from typing import Optional

from ..utils.formatter import HelpFormatter


def setup_parser(subparsers: _SubParsersAction) -> None:
    """setup attach command parser."""
    attach_parser = subparsers.add_parser(
        "attach",
        help="monitor a running (--live) simulation's dashboard read-only",
        formatter_class=HelpFormatter,
        usage="simbi attach <data_dir> [--poll-ms N]",
    )
    attach_parser.add_argument(
        "data_dir",
        help="the run's data directory (holds .simbi-live/snapshot.bin)",
        type=str,
    )
    attach_parser.add_argument(
        "--poll-ms",
        dest="poll_ms",
        help="snapshot poll interval in milliseconds",
        default=250,
        type=int,
    )
    attach_parser.set_defaults(func=execute)


def execute(args: Namespace, _: Optional[list] = None) -> None:
    """execute attach command."""
    import importlib

    try:
        backend = importlib.import_module("simbi.libs.cpu_ext")
    except ImportError as exc:
        print(f"error: could not load backend: {exc}", file=sys.stderr)
        sys.exit(1)

    backend.attach_dashboard(args.data_dir, args.poll_ms)
