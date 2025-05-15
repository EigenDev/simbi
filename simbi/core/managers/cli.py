from dataclasses import dataclass, field
from ..types.dynarg import DynamicArg
from typing import Any
import argparse


@dataclass
class CLIManager:
    """Manages command line interface for BaseConfigs"""

    main_parser: argparse.ArgumentParser
    run_parser: argparse.ArgumentParser
    dynamic_args: list[DynamicArg] = field(default_factory=list)
    property_overrides: dict[str, Any] = field(default_factory=dict)
    arg_group_set: bool = False

    @classmethod
    def from_parsers(
        cls, main_parser: argparse.ArgumentParser, run_parser: argparse.ArgumentParser
    ) -> "CLIManager":
        """Create a CLIManager from main and run parsers"""
        return cls(main_parser=main_parser, run_parser=run_parser)

    def register_dynamic_arg(self, arg: DynamicArg, name: str) -> None:
        """Register a dynamic argument"""
        if arg.name in [a.name for a in self.dynamic_args]:
            return
        self.dynamic_args.append(arg)
        # the problem args group should only be called once
        if not self.arg_group_set:
            problem_args = self.run_parser.add_argument_group(name)
            setattr(self.run_parser, "problem_args", problem_args)
            self.arg_group_set = True
        else:
            problem_args = getattr(self.run_parser, "problem_args")

        if isinstance(arg.value, bool):
            problem_args.add_argument(
                f"--{arg.name}", help=arg.help, action=arg.action, default=arg.value
            )
        else:
            try:
                problem_args.add_argument(
                    f"--{arg.name}",
                    help=arg.help,
                    type=arg.var_type,
                    choices=arg.choices,
                    default=arg.value,
                )
            except argparse.ArgumentError:
                # this will happen if the argument
                # is already registered
                pass

    def parse_args(self) -> dict[str, Any]:
        """Parse arguments and return values"""
        args = vars(self.main_parser.parse_args())
        return {
            arg.name.replace("-", "_"): args[arg.name.replace("-", "_")]
            for arg in self.dynamic_args
            if arg.name.replace("-", "_") in args
        }
