# =============================================================================
# simbi/cli/__main__.py
#
# entry point for `python -m simbi.cli` or the `simbi` console script.
# parses command line arguments and dispatches to the appropriate subcommand.
# =============================================================================
from .simbi_parser import SimbiParser


def main() -> None:
    try:
        from rich.traceback import install

        install()
    except ImportError:
        pass

    parser = SimbiParser()
    args, remaining = parser.parse_known_args()

    if args.command == "plot" and remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))

    if hasattr(args, "func"):
        args.func(args, remaining)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
