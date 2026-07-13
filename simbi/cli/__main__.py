# =============================================================================
# simbi/cli/__main__.py
#
# entry point for `python -m simbi.cli` or the `simbi` console script.
# parses command line arguments and dispatches to the appropriate subcommand.
# =============================================================================
import sys

from simbi.reader.computation import FieldComputationError
from simbi.simulation.problem import ConfigError

from .simbi_parser import SimbiParser


def main() -> None:
    try:
        from rich.traceback import install

        install()
    except ImportError:
        pass

    parser = SimbiParser()
    args, remaining = parser.parse_known_args()

    # `run` forwards leftover flags to the config's own parser (from_cli), which
    # rejects unknowns itself; every OTHER subcommand consumes nothing, so a
    # leftover flag there is a typo that must fail loudly, not silently drop.
    if args.command != "run" and remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))

    if hasattr(args, "func"):
        try:
            args.func(args, remaining)
        except (ConfigError, FieldComputationError, FileNotFoundError, OSError) as exc:
            # a user-facing error: print the formatted message and exit
            # non-zero WITHOUT the rich traceback (which buries the message).
            print(f"\nerror: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
