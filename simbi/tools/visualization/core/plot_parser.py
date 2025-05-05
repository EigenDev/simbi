import argparse
from itertools import cycle
from typing import Any, Optional, Dict
from ..config.config import PlotGroup, StyleGroup, AnimationGroup, MultidimGroup
from .constants import FIELD_ALIASES, VALID_PLOT_TYPES, FIELD_CHOICES, LEGEND_LOCATIONS
from ....cli.base_parser import BaseParser


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


class ParseKVAction(argparse.Action):
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
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


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


class PlottingArgumentBuilder:
    """Handles parsing of plotting arguments"""

    def __init__(self) -> None:
        self.plot_group = []
        self.style_group = []
        self.animation_group = []
        self._build_argument_groups()

    def _build_argument_groups(self) -> None:
        """Build argument groups"""
        self._build_plot_arguments()
        self._build_style_arguments()
        self._build_animation_arguments()

    def _build_plot_arguments(self) -> None:
        """Add basic plot arguments"""
        self.plot_group = [
            (["files"], {"nargs": "+", "help": "checkpoints files to plot"}),
            (["setup"], {"type": str, "help": "The name of the setup being plotted"}),
            (
                ["--plot-type"],
                {"type": str, "choices": VALID_PLOT_TYPES, "help": "plot type"},
            ),
            (
                ["--fields"],
                {
                    "default": ["rho"],
                    "nargs": "+",
                    "help": "the name of the field variable",
                    "choices": FIELD_CHOICES + list(FIELD_ALIASES.keys()),
                },
            ),
            (
                ["--ndim"],
                {"default": 1, "type": int, "help": "the dimensionality of the data"},
            ),
            (
                ["--cartesian"],
                {
                    "default": False,
                    "action": argparse.BooleanOptionalAction,
                    "help": "flag for cartesian plotting",
                },
            ),
            (["--xmax"], {"default": None, "help": "the domain range", "type": float}),
            (["--save-as"], {"default": None, "help": "Save the fig with some name"}),
            (["--dpi"], {"default": 300, "help": "dpi of the saved fig", "type": int}),
            (
                ["--kind"],
                {
                    "default": "snapshot",
                    "type": str,
                    "choices": ["snapsoht", "movie"],
                    "help": "kind of visual to output",
                },
            ),
            (
                ["--slice-along"],
                {
                    "help": "free coordinate for one-d projection",
                    "default": None,
                    "choices": ["x1", "x2", "x3"],
                    "type": str,
                },
            ),
            (
                ["--orbital-params"],
                {
                    "help": "orbital parameters for plotting",
                    "default": None,
                    "action": ParseKVAction,
                    "nargs": "+",
                    "type": str,
                },
            ),
            (
                ["--nlines"],
                {
                    "help": "number of linestyles to plot",
                    "default": None,
                    "type": int,
                },
            ),
            (
                ["--extend"],
                {
                    "default": None,
                    "nargs": "+",
                    "help": "path(s) to Python script(s) with custom plotting functions",
                    "type": str,
                },
            ),
        ]

    def _build_style_arguments(self) -> None:
        """Add style arguments"""
        self.style_group = [
            (
                ["--cmap"],
                {
                    "default": cycle(["viridis"]),
                    "type": str,
                    "nargs": "+",
                    "action": CycleAction,
                    "help": "matplotlib color map",
                },
            ),
            (
                ["--log"],
                {
                    "default": False,
                    "action": argparse.BooleanOptionalAction,
                    "help": "logarithmic plotting scale",
                },
            ),
            (
                ["--semilogx"],
                {
                    "default": False,
                    "action": argparse.BooleanOptionalAction,
                    "help": "logarithmic plotting scale for x-axis",
                },
            ),
            (
                ["--semilogy"],
                {
                    "default": False,
                    "action": argparse.BooleanOptionalAction,
                    "help": "logarithmic plotting scale for y-axis",
                },
            ),
            (
                ["--ax-anchor"],
                {
                    "default": None,
                    "type": str,
                    "nargs": "+",
                    "help": "anchor annotation text for each plot",
                },
            ),
            (
                ["--norm"],
                {
                    "default": False,
                    "action": "store_true",
                    "help": "flag to normalize plot axes",
                },
            ),
            (
                ["--labels"],
                {"default": None, "nargs": "+", "help": "list of legend labels"},
            ),
            (
                ["--xlims"],
                {
                    "default": [None, None],
                    "type": float,
                    "nargs": 2,
                    "help": "limits of x axis",
                },
            ),
            (
                ["--ylims"],
                {
                    "default": [None, None],
                    "type": float,
                    "nargs": 2,
                    "help": "limits of y axis",
                },
            ),
            (
                ["--units"],
                {
                    "default": False,
                    "action": "store_true",
                    "help": "flag for dimensionful units",
                },
            ),
            (
                ["--power"],
                {"default": 1.0, "type": float, "help": "exponent of power-law norm"},
            ),
            (
                ["--scale-downs"],
                {
                    "default": [1],
                    "type": float,
                    "nargs": "+",
                    "help": "list of values to scale plotted variables down by",
                },
            ),
            (
                ["--time-modulus"],
                {
                    "default": 1,
                    "type": float,
                    "nargs": 1,
                },
            ),
            (
                ["--dbg"],
                {
                    "default": False,
                    "action": "store_true",
                    "help": "flag for dark background style",
                },
            ),
            (
                ["--use-tex"],
                {
                    "default": False,
                    "action": "store_true",
                    "help": "flag for latex typesetting",
                },
            ),
            (
                ["--print"],
                {
                    "default": False,
                    "action": "store_true",
                    "help": "flag for publications plot formatting",
                },
            ),
            (
                ["--pictorial"],
                {
                    "default": False,
                    "action": "store_true",
                    "help": "flag for creating figs without data",
                },
            ),
            (
                ["--annot-loc"],
                {
                    "default": None,
                    "type": str,
                    "help": "location of annotations",
                    "choices": [
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
                },
            ),
            (
                ["--legend-loc"],
                {
                    "default": None,
                    "type": str,
                    "help": "location of legend",
                    "choices": [
                        "lower left",
                        "lower right",
                        "upper left",
                        "upper right",
                        "upper center",
                        "lower center",
                        "centercenter left",
                        "center right",
                    ],
                },
            ),
            (
                ["--annot-text"],
                {
                    "default": None,
                    "nargs": "+",
                    "type": str,
                    "help": "text in annotations",
                },
            ),
            (
                ["--inset"],
                {
                    "default": None,
                    "action": ParseKVAction,
                    "metavar": "KEY=VALUE",
                    "nargs": "+",
                    "help": "flag for inset plot. Takes KEY=VALUE for inset x-ylims",
                },
            ),
            (
                ["--png"],
                {
                    "default": False,
                    "action": "store_true",
                    "help": "flag for saving figure as png",
                },
            ),
            (
                ["--fig-dims"],
                {
                    "default": [4, 4],
                    "type": float,
                    "nargs": 2,
                    "help": "figure dimensions",
                },
            ),
            (
                ["--legend"],
                {
                    "default": True,
                    "action": argparse.BooleanOptionalAction,
                    "help": "flag for legend output",
                },
            ),
            (["--nplots"], {"default": 1, "type": int, "help": "number of subplots"}),
            (
                ["--cbar-range"],
                {
                    "default": cycle([(None, None)]),
                    "nargs": "+",
                    "type": colorbar_limits,
                    "action": CycleAction,
                    "help": "The colorbar range(s)",
                },
            ),
            (
                ["--weight"],
                {
                    "help": "plot weighted avg of desired var as function of time",
                    "default": None,
                    "choices": FIELD_CHOICES,
                    "type": str,
                },
            ),
            (
                ["--powerfit"],
                {
                    "help": "plot power-law fit on top of histogram",
                    "default": False,
                    "action": "store_true",
                },
            ),
            (
                ["--cutoffs"],
                {
                    "default": [0.0],
                    "type": float,
                    "nargs": "+",
                    "help": "The 4-velocity cutoff value for the dE/dOmega plot",
                },
            ),
            (
                ["--bbox-kind"],
                {
                    "default": "tight",
                    "type": nullable_string,
                    "help": "tset bbox type during figure save",
                },
            ),
            (
                ["--transparent"],
                {
                    "default": False,
                    "action": argparse.BooleanOptionalAction,
                    "help": "flag for transparent plot background on save",
                },
            ),
            (
                ["--extra-args"],
                {
                    "nargs": "+",
                    "action": ParseKVAction,
                    "help": "accepts dict style args KEY=VALUE",
                    "metavar": "KEY=VALUE",
                },
            ),
            (
                ["--font-color"],
                {"type": str, "default": "black", "help": "font color for plot"},
            ),
            (
                ["--cbar"],
                {
                    "action": argparse.BooleanOptionalAction,
                    "default": True,
                    "help": "colobar visible switch",
                },
            ),
            (
                ["--rev-cmap"],
                {
                    "dest": "rcmap",
                    "action": "store_true",
                    "default": False,
                    "help": "True if you want the colormap to be reversed",
                },
            ),
            (
                ["--cbar-orient"],
                {
                    "dest": "cbar_orient",
                    "default": "vertical",
                    "type": str,
                    "help": "Colorbar orientation",
                    "choices": ["horizontal", "vertical"],
                },
            ),
            (
                ["--bipolar"],
                {"dest": "bipolar", "default": False, "action": "store_true"},
            ),
            (
                ["--subplots"],
                {"dest": "subplots", "default": None, "type": int},
            ),
            (
                ["--sub_split"],
                {"dest": "sub_split", "default": None, "nargs": "+", "type": int},
            ),
            (
                ["--coords"],
                {
                    "help": "coordinates of fixed vars for (n-m)d projection",
                    "action": ParseKVAction,
                    "nargs": "+",
                    "default": {"xj": "0.0", "xk": "0.0"},
                },
            ),
            (
                ["--projection"],
                {
                    "help": "axes to project multidim solution onto",
                    "default": [1, 2, 3],
                    "type": tuple_arg,
                    "choices": [
                        (1, 2, 3),
                        (1, 3, 2),
                        (2, 3, 1),
                        (2, 1, 3),
                        (3, 1, 2),
                        (3, 2, 1),
                    ],
                },
            ),
            (
                ["--box-depth"],
                {
                    "help": "index depth for projecting 3D data onto 2D plane",
                    "type": float,
                    "default": 0,
                },
            ),
            (
                ["--xlabel"],
                {"nargs": 1, "default": "x", "help": "X label name"},
            ),
            (["--ylabel"], {"nargs": 1, "default": "y", "help": "Y label name"}),
        ]
        for style in VALID_PLOT_TYPES:
            self.style_group.append(
                (
                    [f"--{style}"],
                    {
                        "action": PlotStyleAction,
                        "help": f"set plot style to {style}",
                        "dest": "plot_type",
                    },
                )
            )

    def _build_animation_arguments(self) -> None:
        """Add animation arguments"""
        self.animation_group = [
            (
                ["--pan-speed"],
                {
                    "default": 0.1,
                    "type": float,
                    "help": "speed of camera pan for animations",
                },
            ),
            (
                ["--extension"],
                {
                    "default": 1.0,
                    "type": float,
                    "help": "max extent for end of camera span",
                },
            ),
            (
                ["--frame-rate"],
                {"default": 10, "type": int, "help": "frame rate in ms"},
            ),
        ]

    @staticmethod
    def _build_config(args: Dict[str, Any]) -> Dict[str, Any]:
        config = {
            "plot": PlotGroup(
                setup=args["setup"],
                files=args["files"],
                plot_type=args["plot_type"],
                fields=args["fields"],
                ndim=args["ndim"],
                cartesian=args["cartesian"],
                nplots=args["nplots"],
                kind=args["kind"],
                powerfit=args["powerfit"],
                weight=args["weight"],
                save_as=args["save_as"],
                extension=args["extension"],
            ),
            "style": StyleGroup(
                color_maps=args["cmap"],
                log=args["log"],
                semilogx=args["semilogx"],
                legend_loc=args["legend_loc"],
                fig_dims=args["fig_dims"],
                use_tex=args["use_tex"],
                print=args["print"],
                pictorial=args["pictorial"],
                annotation_loc=args["annot_loc"],
                annotation_text=args["annot_text"],
                annotation_anchor=args["ax_anchor"],
                labels=args["labels"],
                xlims=args["xlims"],
                ylims=args["ylims"],
                power=args["power"],
                scale_downs=args["scale_downs"],
                time_modulus=args["time_modulus"],
                black_background=args["dbg"],
                units=args["units"],
                normalize=args["norm"],
                semilogy=args["semilogy"],
                bbox_kind=args["bbox_kind"],
                transparent=args["transparent"],
                dpi=args["dpi"],
                # extra_args=args.get("extra_args"),
                font_color=args["font_color"],
                show_colorbar=args["cbar"],
                reverse_colormap=args["rcmap"],
                colorbar_orientation=args["cbar_orient"],
                split_into_subplots=args["subplots"],
                # sub_split=args["sub_split"],
                bipolar=args["bipolar"],
                color_range=args["cbar_range"],
                xmax=args["xmax"],
                orbital_params=args["orbital_params"],
                nlinestyles=args["nlines"],
            ),
            "animation": AnimationGroup(
                frame_rate=args["frame_rate"],
                pan_speed=args.get("pan_speed"),
                extent=args.get("extent"),
            ),
            "multidim": MultidimGroup(
                projection=args["projection"],
                box_depth=args["box_depth"],
                bipolar=args["bipolar"],
                slice_along=args["slice_along"],
            ),
        }

        # Validate configuration
        PlottingArgumentBuilder()._validate_config(config)

        return config

    @staticmethod
    def get_config(args: argparse.Namespace) -> Dict[str, Any]:
        """Parse arguments into configuration groups"""
        # get all args except the first one which is the command name
        args_dict = vars(args)
        return PlottingArgumentBuilder()._build_config(args_dict)

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration groups"""
        # Validate plot settings
        if config["plot"].ndim < 1:
            raise ValueError("ndim must be >= 1")

        if config["plot"].plot_type == "multidim" and config["plot"].ndim < 2:
            raise ValueError("multidim plots require ndim >= 2")

        # Validate style settings
        if (
            config["style"].legend_loc
            and config["style"].legend_loc not in LEGEND_LOCATIONS
        ):
            raise ValueError(f"Invalid legend location: {config['style'].legend_loc}")

        # Validate animation settings
        if config["animation"].frame_rate <= 0:
            raise ValueError("frame_rate must be positive")

    def add_to_subparser(self, parser: BaseParser) -> None:
        """Add argument groups to subparser"""
        plot_group = parser.add_argument_group("plot")
        style_group = parser.add_argument_group("style")
        anim_group = parser.add_argument_group("animation")

        # Add arguments to groups
        for args, kwargs in self.plot_group:
            plot_group.add_argument(*args, **kwargs)

        for args, kwargs in self.style_group:
            style_group.add_argument(*args, **kwargs)

        for args, kwargs in self.animation_group:
            anim_group.add_argument(*args, **kwargs)
