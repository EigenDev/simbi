from argparse import _SubParsersAction, Namespace
from typing import Optional

from simbi.tools.utility import get_dimensionality
from ..utils.formatter import HelpFormatter
from ...tools.visualization.cli import setup_parser as setup_viz_parser
import sys


def setup_parser(subparsers: _SubParsersAction) -> None:
    """Setup plot command parser"""
    plot_parser = subparsers.add_parser(
        "plot",
        help="plots the given simbi checkpoint file",
        formatter_class=HelpFormatter,
        usage="simbi plot <checkpoints> <setup_name> [options]",
    )
    setup_viz_parser(plot_parser)
    plot_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """Execute plot command using new component-based API"""
    from ...tools.visualization import api

    # Get basic parameters
    files = args.files
    setup = args.setup
    theme = args.theme

    # If files don't exist, show error
    if not files:
        print("Error: No files specified")
        sys.exit(1)

    var_set = {
        "files": files,
        "setup": setup,
        "plot_type": args.plot_type,
        "fields": args.fields,
        "save_as": args.save_as,
        "frame_rate": args.frame_rate,
        "theme": args.theme,
    }

    # get subset of args that are not
    # in var_set and turn this into a
    # kwargs dict
    kwargs = {k: v for k, v in vars(args).items() if k not in var_set}

    # Remove unnecessary args
    for arg in ["func", "files", "active_parser", "main_parser"]:
        if arg in kwargs:
            del kwargs[arg]

    # Handle animation flag
    is_animation = args.kind == "movie"

    ndim = get_dimensionality(files)
    kwargs["ndim"] = ndim
    if not args.plot_type:
        if ndim == 1 or args.slice_along:
            plot_type = "line"
        else:
            plot_type = "multidim"
    else:
        plot_type = args.plot_type

    # Call the appropriate API function
    if is_animation:
        api.animate(
            files=files,
            plot_type=plot_type,
            fields=args.fields,
            save_as=args.save_as,
            frame_rate=args.frame_rate,
            setup=setup,
            theme=theme,
            **kwargs,
        )
    else:
        if plot_type == "line":
            api.plot_line(
                files, args.fields, args.save_as, setup=setup, theme=theme, **kwargs
            )
        elif plot_type == "multidim":
            api.plot_multidim(
                files, args.fields, args.save_as, setup=setup, theme=theme, **kwargs
            )
        elif plot_type == "histogram":
            api.plot_histogram(
                files, args.fields, args.save_as, setup=setup, theme=theme, **kwargs
            )
        elif plot_type == "temporal":
            api.plot_temporal(
                files, args.fields, args.save_as, setup=setup, theme=theme, **kwargs
            )
