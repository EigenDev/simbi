# =============================================================================
# simbi_parser.py
#
# main cli parser for simbi.
# =============================================================================
from argparse import ArgumentParser
from typing import Any

from .actions import print_the_version
from .base_parser import BaseParser
from .commands import afterglow, plot, run
from .utils.formatter import HelpFormatter


class SimbiParser(BaseParser):
    """main parser for simbi cli."""

    command: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            prog="simbi",
            usage="%(prog)s <command> <input> [options]",
            description="relativistic magneto-gas dynamics simulation framework",
            formatter_class=HelpFormatter,
            add_help=False,
        )
        self.add_argument("--version", action=print_the_version)
        self.subparsers = self.add_subparsers(
            dest="command",
            parser_class=ArgumentParser,
            title="commands",
            metavar="<command>",
            required=True,
        )
        self._add_subcommands()

    def _add_subcommands(self) -> None:
        run.setup_parser(self.subparsers)
        plot.setup_parser(self.subparsers)
        afterglow.setup_parser(self.subparsers)
