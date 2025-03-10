from .simbi_parser import SimbiParser


def main() -> None:
    try:
        from rich.traceback import install

        install()
    except ImportError:
        pass

    parser = SimbiParser()
    args, argv = parser.parse_known_args()

    if args.command == "plot" and argv:
        parser.error("unrecognized arguments: " + " ".join(argv))

    if hasattr(args, "func"):
        args.func(args, argv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
