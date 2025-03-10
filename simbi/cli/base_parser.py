import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Optional


class BaseParser(ArgumentParser):
    """Base parser interface"""

    command: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._subparser_map = {}
        self._subparser_added = False  # Track if subparser was added

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
        if not self.command:
            # No command provided - show main help
            if "unrecognized command" in message:
                sys.stderr.write(f"error: {message}\n")
            self.print_help()
        elif "too few arguments" in message:
            # Command provided but no args - show command help
            self.parse_args([self.command, "--help"])
        else:
            # Other errors - show specific error and command help
            sys.stderr.write(f"error: {message}\n")
            self.parse_args([self.command, "--help"])
        self.exit(2)

    def parse_known_args(
        self, args: Optional[Any] = None, namespace: Optional[Namespace] = None
    ) -> tuple[Namespace, list[str]]:
        """Override to handle subcommand help printing"""
        # Get actual args if none provided
        if args is None:
            args = sys.argv[1:]

        # Check for command without args
        if len(args) == 1 and not args[0].startswith("-"):
            if args[0] in ["run", "plot", "clone", "afterlow"]:
                self.command = args[0]
                self.parse_args([self.command, "--help"])
            else:
                self.error("unrecognized command: {:s}".format(args[0]))
                # self.parse_args(["--help"])

        # Normal parsing
        try:
            parsed_args, argv = super().parse_known_args(args, namespace)
            self.command = getattr(parsed_args, "command", None)
            setattr(parsed_args, "main_parser", self)
            setattr(parsed_args, "active_parser", self._subparser_map[self.command])
            return parsed_args, argv
        except Exception as e:
            if self.command:
                self.parse_args([self.command, "--help"])
            else:
                self.print_help()
            self.exit(2)

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
