import argparse
from itertools import cycle
from typing import Any, Optional

from . import api
from .config.builder import ConfigBuilder

try:
    import cmasher
except ImportError:
    pass


class ParseKVAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        try:
            the_dict = dict(map(lambda x: x.split("="), values))
        except ValueError as ex:
            message = "\nTraceback: {}".format(ex)
            message += "\nError on '{}' || It should be 'key=value'".format(each)
            raise argparse.ArgumentError(self, str(message))
        setattr(namespace, self.dest, the_dict)


class ParseKVActionToList(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                if "," in value:
                    value = value.split(",")
                if isinstance(value, str):
                    value = [value]
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


def tuple_arg(param: str) -> tuple[int, ...]:
    """Parse a tuple of ints from the command line"""
    try:
        return tuple(int(arg) for arg in param.split(","))
    except BaseException:
        raise argparse.ArgumentTypeError("argument must be tuple of ints")


def colorbar_limits(c):
    """Parse the colorbar limits from the command line"""
    try:
        vmin, vmax = map(float, c.split(","))
        if vmin > vmax:
            return vmax, vmin
        return vmin, vmax
    except BaseException:
        raise argparse.ArgumentTypeError(
            "Colorbar limits must be in the format: vmin,vmax"
        )


def nullable_string(val: str) -> Optional[str]:
    """If a user passes an empty string to this argument, return None"""
    if not val:
        return None
    return val


class PlotStyleAction(argparse.Action):
    """Custom action to set plot style from flag or direct argument"""

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        nargs: Optional[int] = None,
        **kwargs: Any,
    ):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        if not option_string:
            raise ValueError("No option string provided")

        if option_string == "--plot-type":
            if values not in VALID_PLOT_TYPES:
                raise ValueError(f"Invalid plot style: {values}")
            setattr(namespace, self.dest, values)
        elif option_string.startswith("--"):
            # Convert flag to style name (e.g. --line -> "line")
            style = option_string.replace("--", "")
            setattr(namespace, self.dest, style)


class CycleAction(argparse.Action):
    """Custom action to turn list of items into cycle"""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        setattr(namespace, self.dest, cycle(values))


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Setup visualization parser with all plot arguments"""
    # Common arguments
    parser.add_argument("files", nargs="+", help="Data files to visualize")
    parser.add_argument(
        "--fields", nargs="+", default=["rho"], help="Fields to visualize"
    )
    parser.add_argument("--setup", default="Simulation", help="Setup name")
    parser.add_argument(
        "--plot-type",
        choices=["line", "multidim", "histogram", "temporal"],
        default=None,
        help="Type of plot to create",
    )
    parser.add_argument(
        "--theme",
        default="default",
        help="Theme to use for visualization",
        choices=["default", "dark", "scientific"],
    )
    parser.add_argument("--save-as", help="Save output to file")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    parser.add_argument("--ndim", type=int, help="Number of dimensions")
    parser.add_argument("--log", action="store_true", help="Use log scale")
    parser.add_argument("--semilogx", action="store_true", help="Log scale for x-axis")
    parser.add_argument("--semilogy", action="store_true", help="Log scale for y-axis")
    parser.add_argument(
        "--cmap",
        nargs="+",
        help="Colormap(s) to use",
        default=cycle(["viridis"]),
        action=CycleAction,
    )
    parser.add_argument(
        "--fig-size", nargs=2, type=float, help="Figure dimensions (width height)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument(
        "--no-legend", dest="legend", action="store_false", help="Hide legend"
    )
    parser.add_argument("--legend-loc", help="Location of legend")
    parser.add_argument("--labels", nargs="+", help="Labels for plots", default=[])
    parser.add_argument(
        "--xlims", nargs=2, type=float, default=[None, None], help="X axis limits"
    )
    parser.add_argument(
        "--ylims", nargs=2, type=float, default=[None, None], help="Y axis limits"
    )
    parser.add_argument("--xlabel", default="x", help="X axis label")
    parser.add_argument("--ylabel", default="y", help="Y axis label")
    parser.add_argument("--nplots", type=int, default=1, help="Number of subplots")
    parser.add_argument(
        "--kind",
        choices=["snapshot", "movie"],
        default="snapshot",
        help="Kind of visualization",
    )

    # Animation
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument(
        "--frame-rate", type=int, default=10, help="Animation frame rate"
    )
    parser.add_argument(
        "--pan-speed", type=float, help="Speed of camera pan for animations"
    )
    parser.add_argument(
        "--extension", type=float, help="Maximum extent for animation span"
    )

    # Line plot options
    parser.add_argument(
        "--slice-along", choices=["x1", "x2", "x3"], help="Axis to slice along"
    )
    parser.add_argument(
        "--coords",
        nargs="+",
        help="Slice coordinates (key=value format)",
        action=ParseKVActionToList,
        default={"xj": [0.0], "xk": [0.0]},
    )

    # Multidim options
    parser.add_argument(
        "--projection",
        type=tuple_arg,
        default=(1, 2, 3),
        help="Projection axes (x y z)",
        choices=[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)],
    )
    parser.add_argument(
        "--box-depth", type=float, help="Depth for 3D projection", default=0.0
    )
    parser.add_argument("--bipolar", action="store_true", help="Use bipolar plotting")
    parser.add_argument(
        "--bbox-inches",
        type=nullable_string,
        default="tight",
        help="Bounding box in inches",
    )
    parser.add_argument(
        "--draw-bodies", action="store_true", help="Draw immersed bodies"
    )
    parser.add_argument(
        "--color-range",
        nargs="+",
        help="Color range(s) (min,max format)",
        type=colorbar_limits,
        default=cycle([(None, None)]),
        action=CycleAction,
    )

    # Style options
    parser.add_argument(
        "--transparent", action="store_true", help="Transparent background"
    )
    parser.add_argument("--dbg", action="store_true", help="Dark background")
    parser.add_argument("--use-tex", action="store_true", help="Use LaTeX for text")
    parser.add_argument(
        "--print", action="store_true", help="Publication-quality style"
    )
    parser.add_argument("--pictorial", action="store_true", help="Pictorial style")
    parser.add_argument("--scale-downs", nargs="+", type=float, help="Scale factors")
    parser.add_argument("--time-modulus", type=float, help="Time modulus value")
    parser.add_argument("--norm", action="store_true", help="Normalize plot axes")
    parser.add_argument("--font-color", default="black", help="Font color")
    parser.add_argument(
        "--cbar", action="store_true", default=True, help="Show colorbar"
    )
    parser.add_argument(
        "--rev-cmap", dest="rcmap", action="store_true", help="Reverse colormap"
    )
    parser.add_argument(
        "--cbar-orient",
        dest="colorbar_orientation",
        choices=["horizontal", "vertical"],
        default="vertical",
        help="Colorbar orientation",
    )
    parser.add_argument(
        "--annot-loc",
        choices=[
            "lower left",
            "lower right",
            "upper left",
            "upper right",
            "upper center",
            "lower center",
            "center",
            "center right",
            "center left",
        ],
        help="Annotation location",
    )
    parser.add_argument("--annot-text", nargs="+", help="Annotation text")
    parser.add_argument("--power", type=float, default=1.0, help="Power norm exponent")
    parser.add_argument(
        "--nlinestyles",
        type=int,
        help="Number of line styles to use for line plots",
    )
    parser.add_argument(
        "--bbox-kind",
        choices=["tight", "standard", "full"],
        default="tight",
        help="Bounding box kind for saving figures",
    )
    parser.add_argument(
        "--annotation-anchor",
        nargs=2,
        type=float,
        default=(1.0, 1.0),
        help="Annotation anchor point (x, y)",
    )
    parser.add_argument(
        "--annotation-text",
        type=str,
        default="",
        help="Text to display in the annotation",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        help="Maximum x value for plots (overrides xlims)",
    )

    # Histogram options
    parser.add_argument(
        "--hist-type",
        choices=["kinetic", "enthalpy", "mass", "energy"],
        help="Histogram type",
    )
    parser.add_argument("--powerfit", action="store_true", help="Fit power law")

    # Temporal options
    parser.add_argument("--weight", help="Weight field for temporal average")
    parser.add_argument(
        "--orbital-params",
        nargs="+",
        help="Orbital parameters (key=value format)",
        action=ParseKVAction,
    )
    parser.add_argument(
        "--inset",
        action=ParseKVAction,
        help="Inset plot parameters (key=value format)",
        default=None,
        metavar="KEY=VALUE",
    )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Simbi Visualization Tool")
    setup_parser(parser)
    args = parser.parse_args()

    # Build structured configuration
    config = ConfigBuilder.from_args(args)

    # Determine if animation
    is_animation = args.animate or args.kind == "movie"

    # Create visualization
    if is_animation:
        api.animate(
            files=args.files,
            plot_type=args.plot_type,
            fields=args.fields,
            save_as=args.save_as,
            show=not args.no_show,
            frame_rate=args.frame_rate,
            config=config,
        )
    else:
        plot_type = args.plot_type
        # Auto-detect plot type if not specified
        if not plot_type and args.files:
            from ...tools.utility import get_dimensionality

            ndim = get_dimensionality(args.files[0])
            plot_type = "line" if ndim == 1 else "multidim"

        if plot_type == "line":
            api.plot_line(
                args.files, args.fields, args.save_as, not args.no_show, config=config
            )
        elif plot_type == "multidim":
            api.plot_multidim(
                args.files, args.fields, args.save_as, not args.no_show, config=config
            )
        elif plot_type == "histogram":
            api.plot_histogram(
                args.files, args.fields, args.save_as, not args.no_show, config=config
            )
        elif plot_type == "temporal":
            api.plot_temporal(
                args.files, args.fields, args.save_as, not args.no_show, config=config
            )


if __name__ == "__main__":
    main()
