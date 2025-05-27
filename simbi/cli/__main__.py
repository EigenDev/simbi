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
