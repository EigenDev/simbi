# =============================================================================
# simbi/cli/commands/plot.py
#
# plot command for visualizing simulation checkpoints.
# dispatches to appropriate visualization functions based on plot type.
# =============================================================================
import sys
from argparse import Namespace, _SubParsersAction
from typing import Optional

from simbi.viz import (
    config_from_args,
    handle_generate_config,
    load_props_from_args,
    setup_viz_parser,
)

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

    # handle --generate-config: print example and exit
    if handle_generate_config(args):
        return

    if not args.files:
        print("Error: No files specified for 'plot' command.")
        sys.exit(1)

    config = config_from_args(args)

    # load component props from --config file and/or --props overrides
    component_props = load_props_from_args(args)

    is_animation = getattr(args, "animate", False) or args.kind == "movie"
    is_overlay = getattr(args, "overlay", False)
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
        "overlay",
        "no_show",
        "vector_field",
        "scale",
        "rend",
        "rbeg",
        "config",
        "props",
        "generate_config",
    }
    pass_through_kwargs = {
        k: v for k, v in cli_args.items() if k not in processed_args
    }
    pass_through_kwargs["show"] = not getattr(args, "no_show", False)

    plot_dispatch = {
        "line": api.plot,
        "multidim": api.plot,
        "coordinate_bin": api.plot_coordinate_profile,
        "time_series": api.plot_time_series,
    }

    overlay_dispatch = {
        "line": api.plot_overlay,
        "coordinate_bin": api.plot_coordinate_profile_overlay,
    }

    if is_overlay and is_animation:
        print("Error: --overlay and --animate are mutually exclusive")
        sys.exit(1)

    if is_overlay:
        if plot_type not in overlay_dispatch:
            print(f"Error: overlay not supported for plot type '{plot_type}'")
            sys.exit(1)
        if len(args.files) < 2:
            print("Error: --overlay requires at least 2 files")
            sys.exit(1)

        overlay_func = overlay_dispatch[plot_type]
        overlay_func(
            config=config,
            files=args.files,
            fields=args.fields,
            save_as=args.save_as,
            component_props=component_props,
            **pass_through_kwargs,
        )
    elif is_animation:
        if plot_type == "coordinate_bin":
            api.animate_coordinate_profile(
                config=config,
                files=args.files,
                fields=args.fields,
                save_as=args.save_as,
                component_props=component_props,
                **pass_through_kwargs,
            )
        else:
            api.animate(
                config=config,
                files=args.files,
                fields=args.fields,
                save_as=args.save_as,
                component_props=component_props,
                **pass_through_kwargs,
            )
    else:
        plot_func = plot_dispatch.get(plot_type)
        if plot_func is None:
            print(f"Error: Unknown plot type '{plot_type}'")
            sys.exit(1)

        plot_func(
            config=config,
            files=args.files,
            fields=args.fields,
            save_as=args.save_as,
            component_props=component_props,
            **pass_through_kwargs,
        )
