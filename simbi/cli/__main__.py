from .simbi_parser import SimbiParser


def main() -> None:
    try:
        from rich.traceback import install

        install()
    except ImportError:
        pass

    parser = SimbiParser()
    args, argv = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args, argv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
