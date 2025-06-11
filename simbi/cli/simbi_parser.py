from .commands import run, plot, afterglow, touch
from .actions import print_the_version
from .utils.formatter import HelpFormatter
from .base_parser import BaseParser
from typing import Any
from argparse import ArgumentParser


class SimbiParser(BaseParser):
    """Main parser for simbi CLI"""

    command: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            prog="simbi",
            usage="%(prog)s <command> <input> [options]",
            description="Relativistic magneto-gas dynamics module",
            formatter_class=HelpFormatter,
            add_help=False,
        )
        self.add_argument("--version", action=print_the_version)
        # Specify parser_class to prevent recursion
        self.subparsers = self.add_subparsers(
            dest="command",
            parser_class=ArgumentParser,
            title="commands",
            metavar="<command>",
            required=True,
        )
        self._add_subcommands()

    def _add_subcommands(self) -> None:
        plot.setup_parser(self.subparsers)
        run.setup_parser(self.subparsers)
        afterglow.setup_parser(self.subparsers)
        touch.setup_parser(self.subparsers)
