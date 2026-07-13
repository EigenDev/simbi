# =============================================================================
# simbi/cli/base_parser.py
#
# base argument parser with subcommand tracking and clinical error handling:
# every parse error prints the REAL message (with a did-you-mean for a mistyped
# command), then the relevant help, then exits 2. error handling never re-enters
# parsing — the historical error() -> parse_args() -> error() loop hung the
# process on any unregistered single-argument invocation.
# =============================================================================
import difflib
import re
import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Optional, Sequence


class BaseParser(ArgumentParser):
    """argument parser with subparser tracking + fail-loud error reporting."""

    command: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._subparser_map: dict[str, ArgumentParser] = {}
        self._subparser_added = False

    def add_subparsers(self, **kwargs) -> Any:
        if self._subparser_added:
            raise ValueError("Cannot have multiple subparser arguments.")

        self._subparser_added = True
        subparsers = super().add_subparsers(**kwargs)

        original_add_parser = subparsers.add_parser

        def add_parser_wrapper(name: str, **parser_kwargs) -> ArgumentParser:
            parser = original_add_parser(name, **parser_kwargs)
            self._subparser_map[name] = parser
            return parser

        subparsers.add_parser = add_parser_wrapper
        return subparsers

    def error(self, message: str):
        """print the real error, suggest a close command for an invalid choice,
        show the relevant help, exit 2. NEVER re-enters parsing (re-entry both
        recursed on repeated errors and let --help's SystemExit(0) preempt the
        non-zero exit a failed parse owes the shell)."""
        sys.stderr.write(f"error: {message}\n")
        if "invalid choice" in message and self._subparser_map:
            m = re.search(r"invalid choice: '([^']+)'", message)
            if m:
                close = difflib.get_close_matches(
                    m.group(1), list(self._subparser_map), n=1
                )
                if close:
                    sys.stderr.write(f"did you mean '{close[0]}'?\n")
        active = self._subparser_map.get(self.command)
        (active or self).print_help(sys.stderr)
        self.exit(2)

    def parse_known_args(
        self,
        args: Optional[Sequence[Any]] = None,
        namespace: Optional[Namespace] = None,
    ) -> tuple[Namespace, list[str]]:
        if args is None:
            args = sys.argv[1:]

        # no try/except: a real parse error must surface through error() with its
        # own message, not be swallowed into a generic help screen.
        parsed_args, argv = super().parse_known_args(args, namespace)
        self.command = getattr(parsed_args, "command", "") or ""
        setattr(parsed_args, "main_parser", self)
        setattr(
            parsed_args,
            "active_parser",
            self._subparser_map.get(self.command, self),
        )
        return parsed_args, argv

    def parse_args(
        self, args: Optional[Any] = None, namespace: Optional[Namespace] = None
    ):
        args, argv = self.parse_known_args(args, namespace)
        self.command = args.command

        if argv:
            msg = "unrecognized arguments: {:s}"
            self.error(msg.format(" ".join(argv)))

        # store the command, but delete it from the namespace
        delattr(args, "command")
        return args
