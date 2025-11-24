# =============================================================================
# simbi/cli/commands/plot.py
#
# plot command for visualizing simulation checkpoints.
# dispatches to appropriate visualization functions based on plot type.
# =============================================================================
import sys
from argparse import Namespace, _SubParsersAction
from typing import Optional

from simbi.viz import config_from_args, setup_viz_parser

from ..utils.formatter import HelpFormatter


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


def execute(args: Namespace, _: Optional[list] = None) -> None:
    """Execute plot command using new component-based API"""
    from simbi.viz import api

    if not args.files:
        print("Error: No files specified for 'plot' command.")
        sys.exit(1)

    config = config_from_args(args)
    is_animation = getattr(args, "animate", False) or args.kind == "movie"
    plot_type = config.plot.plot_type

    cli_args = vars(args).copy()
    processed_args = {
        "func",
        "active_parser",
        "main_parser",
        "files",
        "fields",
        "save_as",
        "setup",
        "theme",
        "plot_type",
        "frame_rate",
        "kind",
        "animate",
        "no_show",
        "vector_field",
        "scale",
        "rend",
        "rbeg",
    }
    pass_through_kwargs = {
        k: v for k, v in cli_args.items() if k not in processed_args
    }
    pass_through_kwargs["show"] = not getattr(args, "no_show", False)

    # 5. Handle Animation
    if is_animation:
        raise NotImplementedError(
            "Animation support is not yet implemented in the refactored API."
        )
        # api.animate(
        #     config=config,
        #     files=args.files,
        #     fields=args.fields,
        #     plot_type=plot_type,
        #     save_as=args.save_as,
        #     frame_rate=config.animation.frame_rate,  # Get from config
        #     setup=args.setup,
        #     theme=args.theme,
        #     **pass_through_kwargs,
        # )
    else:
        plot_dispatch = {
            "line": api.plot,
            "multidim": api.plot,
            "coordinate_bin": api.plot_coordinate_profile,
            "time_series": api.plot_time_series,
            # "histogram": api.plot_histogram,
        }

        plot_func = plot_dispatch.get(plot_type)
        if plot_func is None:
            print(f"Error: Unknown plot type '{plot_type}'")
            sys.exit(1)

        plot_func(
            config=config,
            files=args.files,
            fields=args.fields,
            save_as=args.save_as,
            setup=args.setup,
            theme=args.theme,
            **pass_through_kwargs,
        )
