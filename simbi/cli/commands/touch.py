from argparse import Namespace
from typing import Optional
from ..utils.formatter import HelpFormatter


def setup_parser(subparsers) -> None:
    """Setup touch command parser"""
    touch_parser = subparsers.add_parser(
        "touch", formatter_class=HelpFormatter, help="Generate a simbi configuration file"
    )
    touch_parser.add_argument(
        "--name",
        help="name of the generated setup script",
        default="some_skeleton.py",
        type=str,
        dest="skeleton_name",
    )
    touch_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """Execute touch command"""
    from ... import skeleton

    skeleton.generate(args.skeleton_name)
