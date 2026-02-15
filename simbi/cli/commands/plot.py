# =============================================================================
# simbi/cli/commands/plot.py
#
# plot command for visualizing simulation checkpoints.
# contains the argparse parser setup, argument helpers, and dispatch logic.
# =============================================================================
import argparse
import sys
import warnings
from typing import Optional

from simbi.reader.checkpoint_utils import (
    extract_timestep,
    glob_checkpoints,
)
from simbi.viz.config import _PLOT_TYPE_TO_REGISTRY

from ..utils.formatter import HelpFormatter
from .plot_config import (
    config_from_args,
    handle_generate_config,
    load_props_from_args,
)

VALID_PLOT_TYPES = sorted(_PLOT_TYPE_TO_REGISTRY.keys())

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cmasher  # noqa: F401
except ImportError:
    pass


# =========================================================================
# argparse helpers
# =========================================================================
class _glob_files(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        from pathlib import Path

        files: list[list[Path]] = []
        try:
            if values:
                for f in values:
                    if Path(f).is_dir():
                        files.append(glob_checkpoints(f, filter_invalid=True))
                    else:
                        files.append([Path(f)])

        except ValueError as ex:
            message = f"\nTraceback: {ex}"
            raise argparse.ArgumentError(self, str(message))

        flat_files = [x for y in files for x in y]
        setattr(namespace, self.dest, flat_files)


class _ParseKVAction(argparse.Action):
    """parse key=value pairs into a dict."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        try:
            the_dict = dict(map(lambda x: x.split("="), values))
        except ValueError as ex:
            message = f"\nTraceback: {ex}"
            message += (
                f"\nError on '{','.join(values)}' || expected 'key=value'"
            )
            raise argparse.ArgumentError(self, str(message))
        setattr(namespace, self.dest, the_dict)


def _nullable_string(val: str) -> Optional[str]:
    """return None for empty strings."""
    return val if val else None


def _timestep_converter(val: str) -> float:
    """parse timestep value with underscore separators."""
    return float(val.replace("_", ""))


_LINESTYLE_ALIASES = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot": "-.",
}


def _linestyle_converter(val: str) -> str:
    """convert word aliases to matplotlib linestyle strings."""
    return _LINESTYLE_ALIASES.get(val.lower(), val)


def _time_scale_converter(val: str) -> Optional[float]:
    """parse time scale value, supporting 'pi' and 'e' constants."""
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        try:
            import math
            import re

            parts = re.findall(r"[\d\.]+|[^\d\.]+", val)
            for ii, part in enumerate(parts):
                if part == "pi":
                    parts[ii] = math.pi
                elif part == "e":
                    parts[ii] = math.e
                else:
                    parts[ii] = float(part)
            return math.prod(parts)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"could not convert '{val}' to float: {e}"
            )


def filter_files(
    files: list,
    tmin: float | None = None,
    tmax: float | None = None,
    stride: int = 1,
) -> list:
    """
    filter file list by timestep range and stride.
    call this after parsing to apply --tmin, --tmax, --stride.
    """
    if not files:
        return files

    if tmin is not None or tmax is not None:
        filtered = []
        for f in files:
            ts = extract_timestep(f.name)
            if tmin is not None and ts < tmin:
                continue
            if tmax is not None and ts > tmax:
                continue
            filtered.append(f)
        files = filtered

    if stride > 1:
        files = files[::stride]

    return files


# =========================================================================
# parser setup
# =========================================================================
def _setup_plot_args(parser: argparse.ArgumentParser) -> None:
    """add all visualization arguments to the parser."""

    # required / positional
    parser.add_argument(
        "files",
        nargs="*",
        help="checkpoint file(s) or directory",
        action=_glob_files,
    )
    parser.add_argument(
        "--fields", nargs="+", default=["rho"], help="field(s) to visualize"
    )

    # file filtering
    parser.add_argument(
        "--tmin",
        type=_timestep_converter,
        default=None,
        help="minimum timestep to include (supports underscores: 1_000_000)",
    )
    parser.add_argument(
        "--tmax",
        type=_timestep_converter,
        default=None,
        help="maximum timestep to include (supports underscores: 1_000_000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="use every Nth file (default: 1 = all files)",
    )

    # plot type and dispatch
    parser.add_argument(
        "--plot-type",
        choices=VALID_PLOT_TYPES,
        default=None,
        help="type of plot (auto-detected if omitted)",
    )
    parser.add_argument(
        "--setup", default="Simulation", help="setup name for title"
    )

    # output control
    parser.add_argument("--save-as", help="save output to file")
    parser.add_argument(
        "--no-show", action="store_true", help="don't display the plot"
    )
    parser.add_argument(
        "--bbox-inches",
        type=_nullable_string,
        default="tight",
        help="bounding box for saving",
    )

    # animation
    parser.add_argument(
        "--animate", action="store_true", help="create animation"
    )
    parser.add_argument(
        "--kind",
        choices=["snapshot", "movie"],
        default="snapshot",
        help="visualization kind",
    )
    parser.add_argument(
        "--frame-rate", type=int, default=10, help="animation frame rate"
    )

    # overlay (multiple files on same axes)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="overlay multiple files on same axes (line/coordinate_bin/power_spectrum)",
    )
    parser.add_argument(
        "--normalizations",
        nargs="+",
        type=float,
        default=None,
        metavar="NORM",
        help="per-file y-axis normalizations for overlay plots",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="custom per-file legend labels for overlay plots",
    )
    parser.add_argument(
        "--linestyles",
        nargs="+",
        default=None,
        type=_linestyle_converter,
        metavar="LS",
        help="per-file line styles (solid, dashed, dotted, dashdot)",
    )
    parser.add_argument(
        "--x-normalizations",
        nargs="+",
        type=float,
        default=None,
        metavar="XNORM",
        help="per-file x-axis normalizations (overrides auto max-extent detection)",
    )

    # subplot grid (multi-panel comparison)
    parser.add_argument(
        "--subplot",
        action="store_true",
        help="enable grid mode (auto-layout from file count)",
    )
    parser.add_argument(
        "--layout",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        default=None,
        help="explicit grid dimensions (e.g., --layout 2 3)",
    )
    parser.add_argument(
        "--panel-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="custom per-panel titles",
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="derive panel titles from checkpoint metadata",
    )
    parser.add_argument(
        "--no-shared-colorbar",
        action="store_true",
        help="use per-panel colorbars instead of a single shared one",
    )
    parser.add_argument(
        "--annotate-inside",
        action="store_true",
        help="place panel labels inside the plot area instead of as titles",
    )
    parser.add_argument(
        "--wspace",
        type=float,
        default=None,
        help="horizontal spacing between subplots (0.0 = no gap)",
    )
    parser.add_argument(
        "--hspace",
        type=float,
        default=None,
        help="vertical spacing between subplots (0.0 = no gap)",
    )
    parser.add_argument(
        "--max-xticks",
        type=int,
        default=None,
        help="max number of x-axis ticks per panel (reduces label collisions)",
    )
    parser.add_argument(
        "--max-yticks",
        type=int,
        default=None,
        help="max number of y-axis ticks per panel (reduces label collisions)",
    )

    # field overlays (e.g., contour lines over 2d plots)
    parser.add_argument(
        "--field-overlay",
        nargs="+",
        action="append",
        metavar="SPEC",
        dest="field_overlays",
        help=(
            "add field overlay (e.g., --field-overlay mach:contour:1.0). "
            "format: FIELD:COMPONENT:LEVELS where LEVELS is comma-separated. "
            "can be specified multiple times for multiple overlays."
        ),
    )
    parser.add_argument(
        "--overlay-color",
        type=str,
        default="lightgrey",
        help="default color for field overlays",
    )
    parser.add_argument(
        "--overlay-linewidth",
        type=float,
        default=1.5,
        help="default linewidth for field overlays",
    )

    # data pipeline
    parser.add_argument(
        "--slice",
        nargs="+",
        action=_ParseKVAction,
        default=None,
        help="slice data (e.g., --slice x3=0.0 x2=0.1)",
    )
    parser.add_argument(
        "--active-levels",
        nargs="+",
        default=None,
        help="refinement levels to display (e.g., '0 1' or 'all')",
    )
    parser.add_argument(
        "--composite-view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use composite view for refined data",
    )
    parser.add_argument(
        "--render-mode",
        choices=["pcolormesh", "polygons"],
        default="pcolormesh",
        help="2d rendering mode",
    )

    # vector fields
    parser.add_argument(
        "--vector-fields",
        nargs="+",
        default=None,
        help="vector field components (e.g., v1 v2)",
    )
    parser.add_argument(
        "--vector-type",
        choices=["quiver", "stream"],
        default="quiver",
        help="vector visualization type",
    )

    # figure layout (FigureConfig)
    parser.add_argument(
        "--fig-size",
        nargs=2,
        type=float,
        help="figure dimensions (width height)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="output dpi")
    parser.add_argument(
        "--xlims",
        nargs=2,
        type=float,
        default=[None, None],
        help="x axis limits",
    )
    parser.add_argument(
        "--ylims",
        nargs=2,
        type=float,
        default=[None, None],
        help="y axis limits",
    )
    parser.add_argument("--xlabel", default=None, help="x axis label")
    parser.add_argument("--ylabel", default=None, help="y axis label")
    parser.add_argument(
        "--xscale",
        choices=["linear", "log", "symlog", "asinh"],
        default="linear",
        help="x axis scale",
    )
    parser.add_argument(
        "--yscale",
        choices=["linear", "log", "symlog", "asinh"],
        default="linear",
        help="y axis scale",
    )

    # theme and global style
    parser.add_argument(
        "--theme",
        choices=["default", "dark", "scientific"],
        default="default",
        help="visualization theme",
    )
    parser.add_argument(
        "--color-cycle",
        type=str,
        default=None,
        help="colormap for line color cycle (e.g., tab10, cmasher.rainforest)",
    )
    parser.add_argument(
        "--color-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="sampling range for the color cycle (default: 0.1 0.9)",
    )
    parser.add_argument(
        "--color-indices",
        nargs="+",
        type=int,
        default=None,
        help="pick specific indices from a discrete colormap (e.g., 0 5 1 for Pastel1's red, yellow, blue)",
    )
    parser.add_argument(
        "--transparent", action="store_true", help="transparent background"
    )
    parser.add_argument(
        "--use-tex", action="store_true", help="use latex for text"
    )

    # time display
    parser.add_argument(
        "--time-scale",
        type=_time_scale_converter,
        help="characteristic time scale (supports 'pi', 'e')",
    )
    parser.add_argument(
        "--time-units", type=str, default="", help="time units string"
    )

    # special features
    parser.add_argument(
        "--draw-bodies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="draw immersed bodies on hydro plots",
    )

    # coordinate binning options
    parser.add_argument(
        "--n-bins",
        type=int,
        default=64,
        help="number of bins for coordinate profiles",
    )

    # time series options
    parser.add_argument(
        "--weight", default=None, help="weight field for time series averaging"
    )

    # temporal spectrum options
    parser.add_argument(
        "--psd-method",
        choices=["standard", "welch"],
        default="standard",
        help="PSD estimation method (welch averages over segments for lower variance)",
    )
    parser.add_argument(
        "--psd-segments",
        type=int,
        default=8,
        help="number of segments for welch method (default: 8)",
    )
    parser.add_argument(
        "--psd-overlap",
        type=float,
        default=0.5,
        help="fractional overlap between welch segments (default: 0.5)",
    )
    parser.add_argument(
        "--normalize-psd",
        action="store_true",
        help="normalize PSD to integrate to 1",
    )

    # config file and props overrides
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="yaml/json config file for component props",
    )
    parser.add_argument(
        "--props",
        nargs="+",
        default=[],
        metavar="COMPONENT.FIELD=VALUE",
        help="override component props (e.g., polygon.cmap=inferno)",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="print example config and exit",
    )

    # tui mode
    parser.add_argument(
        "--tui",
        action="store_true",
        help="launch interactive TUI for plot configuration",
    )


# =========================================================================
# subcommand registration
# =========================================================================
def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    """register the plot subcommand."""
    plot_parser = subparsers.add_parser(
        "plot",
        help="plots the given simbi checkpoint file",
        formatter_class=HelpFormatter,
        usage="simbi plot <checkpoints> <setup_name> [options]",
    )
    _setup_plot_args(plot_parser)
    plot_parser.set_defaults(func=execute)


# =========================================================================
# execution
# =========================================================================
def execute(args: argparse.Namespace, _: Optional[list] = None) -> None:
    """execute plot command."""

    if getattr(args, "tui", False):
        from simbi.viz.tui import run_plot_tui

        run_plot_tui(initial_path=args.files[0] if args.files else None)
        return

    if handle_generate_config(args):
        return

    if not args.files:
        print("Error: No files specified for 'plot' command.")
        sys.exit(1)

    from simbi.viz import api

    config = config_from_args(args)
    component_props, per_file_overrides = load_props_from_args(args)

    # deprecation warnings for legacy per-file flags
    for flag in ("normalizations", "x_normalizations", "labels"):
        if getattr(args, flag, None):
            warnings.warn(
                f"--{flag.replace('_', '-')} is deprecated. "
                "use --props N:component.field=value instead.",
                DeprecationWarning,
                stacklevel=1,
            )

    is_grid = getattr(args, "subplot", False) or getattr(args, "layout", None)

    if is_grid and getattr(args, "config", None):
        from simbi.viz.config_loader import load_grid_config

        args._grid_config = load_grid_config(args.config)
    else:
        args._grid_config = None

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
        "normalizations",
        "labels",
        "linestyles",
        "x_normalizations",
        "no_show",
        "vector_field",
        "scale",
        "rend",
        "rbeg",
        "config",
        "props",
        "generate_config",
        "subplot",
        "layout",
        "panel_labels",
        "auto_label",
        "no_shared_colorbar",
        "annotate_inside",
        "wspace",
        "hspace",
        "_grid_config",
        "render_mode",
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
        "power_spectrum": api.plot_power_spectrum,
        "temporal_spectrum": api.plot_temporal_spectrum,
    }

    overlay_dispatch = {
        "line": api.plot_overlay,
        "coordinate_bin": api.plot_coordinate_profile_overlay,
        "power_spectrum": api.plot_power_spectrum_overlay,
    }

    if is_grid:
        if len(args.files) < 2:
            print("Error: grid mode requires at least 2 files")
            sys.exit(1)

        layout = tuple(args.layout) if getattr(args, "layout", None) else None
        panel_labels = getattr(args, "panel_labels", None)
        auto_label = getattr(args, "auto_label", False)
        shared_colorbar = not getattr(args, "no_shared_colorbar", False)
        annotate_inside = getattr(args, "annotate_inside", False)
        wspace = getattr(args, "wspace", None)
        hspace = getattr(args, "hspace", None)

        panel_overrides = dict(per_file_overrides) if per_file_overrides else {}
        grid_config = getattr(args, "_grid_config", None)
        if grid_config and "panels" in grid_config:
            # yaml config provides the base; cli per-file overrides win
            for k, v in grid_config["panels"].items():
                idx = int(k)
                if idx in panel_overrides:
                    merged = dict(v)
                    for comp, fields in panel_overrides[idx].items():
                        merged.setdefault(comp, {}).update(fields)
                    panel_overrides[idx] = merged
                else:
                    panel_overrides[idx] = v
        panel_overrides = panel_overrides or None

        api.plot_grid(
            config=config,
            files=args.files,
            fields=args.fields,
            layout=layout,
            panel_labels=panel_labels,
            auto_label=auto_label,
            shared_colorbar=shared_colorbar,
            annotate_inside=annotate_inside,
            wspace=wspace,
            hspace=hspace,
            save_as=args.save_as,
            component_props=component_props,
            panel_overrides=panel_overrides,
            **pass_through_kwargs,
        )
        return

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
            per_file_overrides=per_file_overrides or None,
            normalizations=getattr(args, "normalizations", None),
            labels=getattr(args, "labels", None),
            linestyles=getattr(args, "linestyles", None),
            x_normalizations=getattr(args, "x_normalizations", None),
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
