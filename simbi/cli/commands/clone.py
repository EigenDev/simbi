from argparse import Namespace
from typing import Optional
from ..utils.formatter import HelpFormatter


def setup_parser(subparsers) -> None:
    """Setup clone command parser"""
    clone_parser = subparsers.add_parser(
        "clone", formatter_class=HelpFormatter, help="Clone a simbi configuration file"
    )
    clone_parser.add_argument(
        "--name",
        help="name of the generated setup script",
        default="some_clone.py",
        type=str,
        dest="clone_name",
    )
    clone_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """Execute clone command"""
    from ... import clone

    clone.generate(args.clone_name)
