from argparse import Namespace
from typing import Optional
from ..utils.formatter import HelpFormatter


def setup_parser(subparsers) -> None:
    """Setup afterglow command parser"""
    afterglow_parser = subparsers.add_parser(
        "afterglow",
        help="compute the afterglow for given data",
        usage="simbi afterglow <files> [options]",
        formatter_class=HelpFormatter,
        # parents=[subparsers._action_groups[0]],
    )
    afterglow_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """Execute afterglow command"""
    from ...afterglow import radiation

    radiation.run(args, argv)
