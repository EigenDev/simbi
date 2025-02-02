import argparse
import sys
import matplotlib.pyplot as plt
from . import old_visual
from ..tools import visual
from typing import Optional, Any
from ..tools.utility import (
    BIGGER_SIZE,
    DEFAULT_SIZE,
    get_dimensionality,
    get_file_list,
)
from ..detail import get_subparser, ParseKVAction
from pathlib import Path

tool_src = Path(__file__).resolve().parent / "tools"
VALID_PLOT_TYPES= ["line", "multidim", "temporal", "histogram"]


def colorbar_limits(c):
    """Parse the colorbar limits from the command line"""
    try:
        vmin, vmax = map(float, c.split(","))
        if vmin > vmax:
            return vmax, vmin
        return vmin, vmax
    except:
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
        option_string: Optional[str] = None
    ) -> None:
        if option_string == '--plot-type':
            if values not in VALID_PLOT_TYPES:
                raise ValueError(f'Invalid plot style: {values}')
            setattr(namespace, self.dest, values)
        else:
            # Convert flag to style name (e.g. --line -> "line")
            style = option_string.replace('--', '')
            setattr(namespace, self.dest, style)


def parse_plotting_arguments(
    parser: argparse.ArgumentParser,
) -> tuple[argparse.ArgumentParser, tuple[argparse.Namespace, list[str]]]:
    plot_parser = get_subparser(parser, 1)
    plot_parser.add_argument("files", nargs="+", help="checkpoints files to plot")
    plot_parser.add_argument(
        "setup", type=str, help="The name of the setup being plotted"
    )
    plot_parser.add_argument(
        "--fields",
        default=["rho"],
        nargs="+",
        help="the name of the field variable",
        choices=old_visual.field_choices,
    )
    plot_parser.add_argument(
        "--ndim",
        default=1,
        type=int,
        help="the dimensionality of the data",
    )
    plot_parser.add_argument(
        "--cartesian",
        default=False,
        action="store_true",
        help="flag for cartesian plotting",
    )
    plot_parser.add_argument(
        "--xmax",
        default=None,
        help="the domain range",
        type=float,
    )
    plot_parser.add_argument(
        "--log",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="logarithmic plotting scale",
    )
    plot_parser.add_argument(
        "--semilogx",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="logarithmic plotting scale for x-axis",
    )
    plot_parser.add_argument(
        "--semilogy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="logarithmic plotting scale for y-axis",
    )
    plot_parser.add_argument(
        "--plot-type",
        default=None,
        type=str,
        help="plot type",
        choices=["line", "multidim", "temmporal", "histogram"],
    )
    plot_parser.add_argument(
        "--ax-anchor",
        default=None,
        type=str,
        nargs="+",
        help="anchor annotation text for each plot",
    )
    plot_parser.add_argument(
        "--norm", default=False, action="store_true", help="flag to normalize plot axes"
    )
    plot_parser.add_argument(
        "--labels", default=None, nargs="+", help="list of legend labels"
    )
    plot_parser.add_argument(
        "--xlims", default=[None, None], type=float, nargs=2, help="limits of x axis"
    )
    plot_parser.add_argument(
        "--ylims", default=[None, None], type=float, nargs=2, help="limits of y axis"
    )
    plot_parser.add_argument(
        "--units",
        default=False,
        action="store_true",
        help="flag for dimensionful units",
    )
    plot_parser.add_argument(
        "--power", default=1.0, type=float, help="exponent of power-law norm"
    )
    plot_parser.add_argument(
        "--scale-downs",
        default=[1],
        type=float,
        nargs="+",
        help="list of values to scale plotted variables down by",
    )
    plot_parser.add_argument(
        "--dbg",
        default=False,
        action="store_true",
        help="flag for dark background style",
    )
    plot_parser.add_argument(
        "--tex", default=False, action="store_true", help="flag for latex typesetting"
    )
    plot_parser.add_argument(
        "--print",
        default=False,
        action="store_true",
        help="flag for publications plot formatting",
    )
    plot_parser.add_argument(
        "--pictorial",
        default=False,
        action="store_true",
        help="flag for creating figs without data",
    )
    plot_parser.add_argument(
        "--annot-loc",
        default=None,
        type=str,
        help="location of annotations",
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
    )
    plot_parser.add_argument(
        "--legend-loc",
        default=None,
        type=str,
        help="location of legend",
        choices=[
            "lower left",
            "lower right",
            "upper left",
            "upper right",
            "upper center",
            "lower center",
            "center" "center left",
            "center right",
        ],
    )
    plot_parser.add_argument(
        "--annot-text", default=None, nargs="+", type=str, help="text in annotations"
    )
    plot_parser.add_argument(
        "--inset",
        default=None,
        action=ParseKVAction,
        metavar="KEY=VALUE",
        nargs="+",
        help="flag for inset plot. Takes KEY=VALUE for inset x-ylims",
    )
    plot_parser.add_argument(
        "--png",
        default=False,
        action="store_true",
        help="flag for saving figure as png",
    )
    plot_parser.add_argument(
        "--fig-dims", default=[4, 4], type=float, nargs=2, help="figure dimensions"
    )
    plot_parser.add_argument(
        "--legend",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="flag for legend output",
    )
    plot_parser.add_argument(
        "--extra-files",
        default=None,
        nargs="+",
        help="extra 1D files to plot alongside 2D plots",
    )
    plot_parser.add_argument("--save", default=None, help="Save the fig with some name")
    plot_parser.add_argument(
        "--kind",
        default="snapshot",
        type=str,
        choices=["snapsoht", "movie"],
        help="kind of visual to output",
    )
    plot_parser.add_argument(
        "--cmap", default=["viridis"], type=str, nargs="+", help="matplotlib color map"
    )
    plot_parser.add_argument("--nplots", default=1, type=int, help="number of subplots")
    plot_parser.add_argument(
        "--cbar-range",
        default=[(None, None)],
        nargs="+",
        type=colorbar_limits,
        help="The colorbar range",
    )
    plot_parser.add_argument(
        "--weight",
        help="plot weighted avg of desired var as function of time",
        default=None,
        choices=old_visual.field_choices + old_visual.derived,
        type=str,
    )
    plot_parser.add_argument(
        "--powerfit",
        help="plot power-law fit on top of histogram",
        default=False,
        action="store_true",
    )
    plot_parser.add_argument(
        "--break-time",
        help="break time of relativistic blast wave",
        type=float,
        default=None,
    )
    plot_parser.add_argument(
        "--sort",
        help="flag to sort file list",
        type=argparse.BooleanOptionalAction,
        default=False,
    )
    plot_parser.add_argument(
        "--cutoffs",
        default=[0.0],
        type=float,
        nargs="+",
        help="The 4-velocity cutoff value for the dE/dOmega plot",
    )
    plot_parser.add_argument(
        "--bbox-kind",
        default="tight",
        type=nullable_string,
        help="tset bbox type during figure save",
    )
    plot_parser.add_argument(
        "--transparent",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="flag for transparent plot background on save",
    )
    plot_parser.add_argument(
        "--extra-args",
        nargs="+",
        action=ParseKVAction,
        help="accepts dict style args KEY=VALUE",
        metavar="KEY=VALUE",
    )
    plot_parser.add_argument(
        "--frame-rate",
        type=int,
        default=10,
        help="frame rate in ms",
    )
    plot_parser.add_argument(
        "--font-color",
        type=str,
        default="black",
        help="font color for plot",
    )
    fillgroup = plot_parser.add_mutually_exclusive_group()
    fillgroup.add_argument(
        "--xfill-scale",
        type=float,
        default=None,
        help="Set the x-scale to start plt.fill_between",
    )
    fillgroup.add_argument(
        "--yfill-scale",
        type=float,
        default=None,
        help="Set the y-scale to start plt.fill_between",
    )

    for style in VALID_PLOT_TYPES:
        plot_parser.add_argument(
            f"--{style}",
            action=PlotStyleAction,
            dest="plot_type",
            const=style,
            help=f"Set plot style to {style}",
        )
        
    return parser, parser.parse_known_args(
        args=None if sys.argv[2:] else ["plot", "--help"]
    )


def main(parser: argparse.ArgumentParser, args: argparse.Namespace, *_) -> None:
    parser, (args, _) = parse_plotting_arguments(parser)

    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    if args.tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": BIGGER_SIZE,
                "text.color": args.font_color,
                "axes.labelcolor": args.font_color,
                "xtick.color": args.font_color,
                "ytick.color": args.font_color,
                "axes.edgecolor": args.font_color,
            }
        )

        if args.print:
            plt.rcParams.update(
                {
                    "legend.fontsize": DEFAULT_SIZE,
                }
            )

    sys.path.insert(1, f"{tool_src}")
    file_list, _ = get_file_list(args.files)
    ndim = get_dimensionality(file_list)

    visual.visualize(parser, ndim)
