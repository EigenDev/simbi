# =============================================================================
# cli.py
#
# visualization cli argument parser.
# orchestration and figure-level args only.
# component-specific styling handled entirely by --config and --props.
# =============================================================================
import argparse
import warnings
from typing import Optional

from simbi.viz.checkpoint.checkpoint_utils import (
    extract_timestep,
    glob_checkpoints,
)

VALID_PLOT_TYPES = ["line", "multidim", "coordinate_bin", "time_series"]

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cmasher  # noqa: F401
except ImportError:
    pass


class glob_files(argparse.Action):
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


class ParseKVAction(argparse.Action):
    """Parse key=value pairs into a dict."""

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


def nullable_string(val: str) -> Optional[str]:
    """Return None for empty strings."""
    return val if val else None


def timestep_converter(val: str) -> float:
    """
    parse timestep value with underscore separators.
    e.g., '1_000_000' -> 1000000.0
    """
    return float(val.replace("_", ""))


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

    # filter by timestep range
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

    # apply stride
    if stride > 1:
        files = files[::stride]

    return files


def time_scale_converter(val: str) -> Optional[float]:
    """
    Parse time scale value, supporting 'pi' and 'e' constants.
    e.g., '4pi' -> 4 * 3.14159...
    """
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        try:
            import math
            import re

            parts = re.findall(r"[\d\.]+|[^\d\.]+", val)
            for i, part in enumerate(parts):
                if part == "pi":
                    parts[i] = math.pi
                elif part == "e":
                    parts[i] = math.e
                else:
                    parts[i] = float(part)
            return math.prod(parts)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"could not convert '{val}' to float: {e}"
            )


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Setup visualization parser."""

    # =========================================================================
    # required / positional
    # =========================================================================
    parser.add_argument(
        "files",
        nargs="+",
        help="checkpoint file(s) or directory for body diagnostics",
        action=glob_files,
    )
    parser.add_argument(
        "--fields", nargs="+", default=["rho"], help="field(s) to visualize"
    )
    parser.add_argument(
        "--norm",
        default=None,
        metavar="VALUE|max|min",
        help="normalize each field by: a number (e.g., 5e-7), or 'max'/'min' (the field's own "
        "extremum). for M&DZ-style 'relative to max initial value', pass that number.",
    )

    # =========================================================================
    # file filtering
    # =========================================================================
    parser.add_argument(
        "--tmin",
        type=timestep_converter,
        default=None,
        help="minimum timestep to include (supports underscores: 1_000_000)",
    )
    parser.add_argument(
        "--tmax",
        type=timestep_converter,
        default=None,
        help="maximum timestep to include (supports underscores: 1_000_000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="use every Nth file (default: 1 = all files)",
    )

    # =========================================================================
    # plot type and dispatch
    # =========================================================================
    parser.add_argument(
        "--plot-type",
        choices=VALID_PLOT_TYPES,
        default=None,
        help="type of plot (auto-detected if omitted)",
    )
    parser.add_argument(
        "--setup", default="Simulation", help="setup name for title"
    )

    # =========================================================================
    # output control
    # =========================================================================
    parser.add_argument("--save-as", help="save output to file")
    parser.add_argument(
        "--print",
        action="store_true",
        help="print (publication) rendering: the title carries no time stamp",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="don't display the plot"
    )
    parser.add_argument(
        "--bbox-inches",
        type=nullable_string,
        default="tight",
        help="bounding box for saving",
    )

    # =========================================================================
    # animation
    # =========================================================================
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

    # =========================================================================
    # overlay (multiple files on same axes)
    # =========================================================================
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="overlay multiple files on same axes (line/coordinate_bin only)",
    )

    # =========================================================================
    # data pipeline
    # =========================================================================
    parser.add_argument(
        "--slice",
        nargs="+",
        action=ParseKVAction,
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

    # =========================================================================
    # vector fields
    # =========================================================================
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

    # =========================================================================
    # figure layout (FigureConfig)
    # =========================================================================
    parser.add_argument(
        "--fig-size",
        nargs=2,
        type=float,
        help="figure dimensions (width height)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="output dpi")
    parser.add_argument(
        "--draw-bodies",
        action="store_true",
        help="overlay each immersed body's silhouette (at its evolved pose) on the "
        "field plot; follows --slice (cartesian only)",
    )
    parser.add_argument(
        "--draw-tracers",
        action="store_true",
        help="scatter mass-transport tracers on the field plot; follows --slice",
    )
    parser.add_argument(
        "--tracers-only",
        action="store_true",
        help="render only the mass-transport tracer cloud without an eulerian field",
    )
    parser.add_argument(
        "--tracer-render",
        choices=["concentration", "cohort-ratio", "scatter"],
        default="concentration",
        help="tracer-only rendering mode",
    )
    parser.add_argument(
        "--tracer-smoothing",
        type=float,
        default=None,
        help="concentration smoothing width in display cells; default is adaptive, 0 is exact",
    )
    parser.add_argument(
        "--tracer-color-by",
        choices=["flag", "cohort", "reservoir", "id", "none"],
        default="flag",
        help="scatter-mode tracer coloring",
    )
    parser.add_argument(
        "--tracer-cohort",
        type=int,
        default=None,
        help="render only this initial-material cohort in concentration mode",
    )
    parser.add_argument(
        "--draw-horizon",
        action="store_true",
        help="overlay the black-hole event horizon (r_+ = M + sqrt(M^2 - a^2)) of a "
        "curved-spacetime run, read from the checkpoint metadata; flat runs draw nothing",
    )
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

    # =========================================================================
    # theme and global style
    # =========================================================================
    parser.add_argument(
        "--theme",
        choices=["default", "dark", "scientific"],
        default="default",
        help="visualization theme",
    )
    parser.add_argument(
        "--transparent", action="store_true", help="transparent background"
    )
    parser.add_argument(
        "--use-tex", action="store_true", help="use latex for text"
    )

    # =========================================================================
    # time display
    # =========================================================================
    parser.add_argument(
        "--time-scale",
        type=time_scale_converter,
        help="characteristic time scale (supports 'pi', 'e')",
    )
    parser.add_argument(
        "--time-units", type=str, default="", help="time units string"
    )

    # =========================================================================
    # coordinate binning options
    # =========================================================================
    parser.add_argument(
        "--n-bins",
        type=int,
        default=64,
        help="number of bins for coordinate profiles",
    )

    # =========================================================================
    # time series options
    # =========================================================================
    parser.add_argument(
        "--weight", default=None, help="weight field for time series averaging"
    )

    # =========================================================================
    # config file and props overrides (ALL component styling goes here)
    # =========================================================================
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
        help="override component props (e.g., polygon.cmap=inferno). qualify the "
        "component with a data field to style just the panel drawing it, when "
        "several fields share a chart (e.g., quad:u.log_scale=false)",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="print example config and exit",
    )
