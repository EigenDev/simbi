import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import matplotlib.ticker as tkr
from itertools import cycle
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from typing import Iterable
from . import utility as util
from ..detail import ParseKVAction
from ..detail.slogger import logger
from ..detail.helpers import (
    get_iterable,
    calc_cell_volume1D,
    calc_cell_volume2D,
    calc_cell_volume3D,
    calc_domega,
    find_nearest,
    calc_any_mean,
)
from ..detail import get_subparser

try:
    import cmasher
except ImportError:
    pass


derived = [
    "D",
    "momentum",
    "energy",
    "energy_rst",
    "enthalpy",
    "temperature",
    "T_eV",
    "mass",
    "chi_dens",
    "mach",
    "u1",
    "u2",
    "u3",
    "u",
    "tau-s",
    "ptot",
    "pmag",
    "sigma",
]
field_choices = [
    "rho",
    "v1",
    "v2",
    "v3",
    "v",
    "p",
    "gamma_beta",
    "chi",
    "b1",
    "b2",
    "b3",
] + derived
lin_fields = [
    "chi",
    "gamma_beta",
    "u1",
    "u2",
    "u3",
    "u",
    "tau-s",
    "v",
    "v1",
    "v2",
    "v3",
]


def tuple_arg(param: str) -> tuple[int]:
    try:
        return tuple(int(arg) for arg in param.split(","))
    except BaseException:
        raise argparse.ArgumentTypeError("argument must be tuple of ints")


class Visualizer:
    def __init__(
        self,
        parser,
        ndim=1,
        files=None,
        sort=False,
        dbg=False,
        hist=False,
        weight=False,
        dx_domega=False,
    ):
        self.current_frame = slice(None)
        self.break_time = None
        self.plotted_references = False
        self.ndim = ndim
        self.refs = []
        self.oned_slice = False
        self.coords = {"x2v": "0.0", "x3v": "0.0"}
        self.dbg = dbg
        self.hist = hist
        self.weight = weight
        self.dx_domega = dx_domega
        self.files = files
        self.sort = sort

        if self.ndim != 1:
            self.setup_plot_parser(parser)

        self.update_vars(parser)
        self.setup_plot_style()
        self.color_map = []
        self.cartesian = True
        self.flist, self.frame_count = util.get_file_list(self.files, self.sort)

        if self.ndim != 1:
            self.setup_color_map()
            self.check_cartesian()

        self.color_map = cycle(self.color_map)
        self.setup_vrange()
        self.square_plot = self.check_square_plot()
        self.setup_coords()
        self.create_figure()

    def setup_plot_parser(self, parser):
        plot_parser = get_subparser(parser, 1)
        plot_parser.add_argument(
            "--cbar_sub",
            dest="cbar2",
            metavar="Range of Color Bar for secondary plot",
            nargs="+",
            type=float,
            default=[None, None],
            help="The colorbar range you'd like to plot",
        )
        plot_parser.add_argument(
            "--cbar",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="colobar visible switch",
        )
        plot_parser.add_argument(
            "--cmap2",
            dest="cmap2",
            metavar="Color Bar Colarmap 2",
            default="magma",
            help="The secondary colorbar cmap you'd like to plot",
        )
        plot_parser.add_argument(
            "--rev-cmap",
            dest="rcmap",
            action="store_true",
            default=False,
            help="True if you want the colormap to be reversed",
        )
        plot_parser.add_argument(
            "--x",
            nargs="+",
            default=None,
            type=float,
            help="List of x values to plot field max against",
        )
        plot_parser.add_argument("--xlabel", nargs=1, default="X", help="X label name")
        plot_parser.add_argument(
            "--dx-domega",
            action="store_true",
            default=False,
            help="Plot the d(var)/dOmega plot",
        )
        plot_parser.add_argument(
            "--dec-rad",
            dest="dec_rad",
            default=False,
            action="store_true",
            help="Compute dr as function of angle",
        )
        plot_parser.add_argument(
            "--nwedge", dest="nwedge", default=0, type=int, help="Number of wedges"
        )
        plot_parser.add_argument(
            "--cbar-orient",
            dest="cbar_orient",
            default="vertical",
            type=str,
            help="Colorbar orientation",
            choices=["horizontal", "vertical"],
        )
        plot_parser.add_argument(
            "--wedge-lims",
            dest="wedge_lims",
            default=[0.4, 1.4, 70, 110],
            type=float,
            nargs=4,
            help="wedge limits",
        )
        plot_parser.add_argument(
            "--bipolar", dest="bipolar", default=False, action="store_true"
        )
        plot_parser.add_argument("--subplots", dest="subplots", default=None, type=int)
        plot_parser.add_argument(
            "--sub_split", dest="sub_split", default=None, nargs="+", type=int
        )
        plot_parser.add_argument(
            "--viewing",
            help="viewing angle of simulation in [deg]",
            type=float,
            default=None,
            nargs="+",
        )
        plot_parser.add_argument(
            "--oned-slice",
            help="free coordinate for one-d projection",
            default=None,
            choices=["x1", "x2", "x3"],
            type=str,
        )
        plot_parser.add_argument(
            "--coords",
            help="coordinates of fixed vars for (n-m)d projection",
            action=ParseKVAction,
            nargs="+",
            default={"xj": "0.0", "xk": "0.0"},
        )
        plot_parser.add_argument(
            "--projection",
            help="axes to project multidim solution onto",
            default=[1, 2, 3],
            type=tuple_arg,
            choices=[(1, 2, 3), (1, 3, 2), (2, 3, 1), (2, 1, 3), (3, 1, 2), (3, 2, 1)],
        )
        plot_parser.add_argument(
            "--box-depth",
            help="index depth for projecting 3D data onto 2D plane",
            type=float,
            default=0,
        )
        plot_parser.add_argument(
            "--pan-speed",
            help="speed of camera pan for animations",
            type=float,
            default=None,
        )
        plot_parser.add_argument(
            "--extent",
            help="max extent for end of camera span",
            type=float,
            default=None,
        )

    def update_vars(self, parser):
        vars(self).update(**vars(parser.parse_args()))

    def setup_plot_style(self):
        if self.cmap == "grayscale":
            plt.style.use("grayscale")
        else:
            plt.style.use("seaborn-v0_8-colorblind")

        if self.dbg:
            plt.style.use("dark_background")

    def setup_color_map(self):
        self.color_map = [
            plt.get_cmap(cmap).reversed() if self.rcmap else plt.get_cmap(cmap)
            for cmap in self.cmap
        ]

    def check_cartesian(self):
        if isinstance(self.flist, dict):
            self.cartesian = util.read_file(self.flist[0][0])[1]["is_cartesian"]
        else:
            self.cartesian = util.read_file(self.flist[0])[1]["is_cartesian"]

    def setup_vrange(self):
        self.vrange = self.cbar_range
        if len(self.vrange) != len(self.fields):
            self.vrange += [(None, None)] * (abs(len(self.fields) - len(self.vrange)))
        self.vrange = cycle(self.vrange)

    def check_square_plot(self):
        return (
            self.cartesian
            or self.ndim == 1
            or self.hist
            or self.weight
            or self.dx_domega
            or self.oned_slice
        )

    def setup_coords(self):
        self.coords.setdefault("xj", "0.0")
        self.coords.setdefault("xk", "0.0")

    def place_annotation(self, ax: plt.Axes, anchor_text: str) -> None:
        at = AnchoredText(
            rf"{anchor_text}",
            frameon=False,
            loc=self.annot_loc,
        )
        ax.add_artist(at)

    def plot_line(self):
        field_str = util.get_field_str(self)
        scale_cycle = cycle(self.scale_downs)
        refcount = 0

        for ax in get_iterable(self.axs, func=list if self.nplots == 1 else iter):
            for fidx, file in enumerate(get_iterable(self.flist[self.current_frame])):
                fields, setup, mesh = util.read_file(file)
                for idx, field in enumerate(get_iterable(self.fields)):
                    var = (
                        util.prims2var(fields, field)
                        if field in derived
                        else fields[field if field != "v" else "v1"]
                    )

                    if self.units:
                        if field in ["p", "energy", "energy_rst"]:
                            var *= util.edens_scale.value
                        elif field in ["rho", "D"]:
                            var *= util.rho_scale.value

                    label = (
                        self.labels[fidx]
                        if self.labels
                        else (field_str[idx] if len(self.fields) > 1 else field_str)
                    )
                    scale = next(scale_cycle)
                    if scale != 1 and label:
                        label += f"/{int(scale)}"

                    if self.oned_slice:
                        x = calc_any_mean(
                            mesh[f"{self.oned_slice}v"],
                            setup[f"{self.oned_slice}_cell_spacing"],
                        )
                        for xkcoord in map(float, self.coords["xk"].split(",")):
                            for xjcoord in map(float, self.coords["xj"].split(",")):
                                coord_label = (
                                    f"{label or ''}, $x_j={xjcoord:.1f}$".strip(", ")
                                )
                                if self.ndim == 3:
                                    coord_label += f", $x_k={xkcoord:.1f}$"
                                if not self.cartesian:
                                    xjcoord = np.deg2rad(xjcoord)
                                    xkcoord = np.deg2rad(xkcoord)
                                if self.oned_slice == "x1":
                                    xj, xk = "x2v", "x3v"
                                elif self.oned_slice == "x2":
                                    xj, xk = "x3v", "x1v"
                                else:
                                    xj, xk = "x1v", "x2v"
                                jidx = find_nearest(mesh.get(xj, np.linspace(0, 1)), xjcoord)[0]
                                kidx = find_nearest(mesh.get(xk, np.linspace(0, 1)), xkcoord)[0]
                                if self.ndim == 2:
                                    yvar = var[jidx]
                                else:
                                    if self.oned_slice == "x1":
                                        slice = np.s_[kidx, jidx, :]
                                    elif self.oned_slice == "x2":
                                        slice = np.s_[jidx, :, kidx]
                                    else:
                                        slice = np.s_[:, kidx, jidx]
                                    yvar = var[slice]
                                if self.shock_coord:
                                    xvar = x / x[np.argmax(yvar)]
                                else:
                                    xvar = x
                                scale = (
                                    yvar[np.argmax(yvar)] if self.shock_coord else scale
                                )
                                (line,) = ax.plot(xvar, yvar / scale, label=coord_label)
                                if self.nplots != 1:
                                    ax.set_ylabel(coord_label)
                    else:
                        x = calc_any_mean(mesh["x1v"], setup["x1_cell_spacing"])
                        (line,) = ax.plot(x, var / scale, label=label)

                    self.frames.append(line)

                    if self.pictorial and refcount == 0:
                        x = np.linspace(mesh["x1v"][var.argmax()], 1, 1000)
                        (ref,) = ax.plot(
                            x,
                            var.max() * (x / x[0]) ** (-3 / 2),
                            linestyle="--",
                            color="grey",
                            alpha=0.4,
                        )
                        self.refx, self.refy = x[0], var.max()
                        self.refs.append(ref)
                        refcount += 1

        if self.setup:
            ax.set_title(f'{self.setup} t = {setup["time"]:.1f}')
        if self.log or setup["x1_cell_spacing"] == "log" or self.semilogx:
            ax.set_xscale("log")
        if self.log or self.semilogy:
            ax.set_yscale("log")

        if len(self.fields) == 1:
            ax.set_ylabel(field_str)
        if label and self.legend:
            if self.nplots == 1:
                ax.legend(loc=self.legend_loc)
            else:
                self.axs[0].legend(loc=self.legend_loc)

        ax.set_xlim(
            *self.xlims if any(self.xlims) else (mesh["x1v"][0], mesh["x1v"][-1])
        )
        ax.set_ylim(*self.ylims if any(self.ylims) else (None, None))
        if self.oned_slice:
            ax.set_xlabel(f"$x_{self.oned_slice[1]}$")
        else:
            ax.set_xlabel(r"$x$" if self.cartesian else r"$r$")

    def plot_multidim(self) -> None:
        def theta_sign(quadrant: int) -> int:
            return 1 if quadrant in [0, 3] else -1

        field_str = get_iterable(util.get_field_str(self))
        cbar_orientation = "vertical"
        patches = len(self.fields) + (len(self.fields) == 1)
        if self.bipolar and patches == 2:
            patches *= 2

        theta_cycle = cycle([0, -np.pi * 0.5, -np.pi, np.pi * 0.5])
        the_fields = cycle(self.fields)
        if any(self.xlims) and not self.cartesian:
            edge = self.xlims[1] + (len(self.fields) - 1) * (
                self.xlims[1] - self.xlims[0]
            )
            xextent = np.deg2rad([self.xlims[0], edge])

        for ax in get_iterable(self.axs):
            for file in get_iterable(self.flist[self.current_frame]):
                fields, setup, mesh = util.read_file(file)
                for idx in range(patches):
                    field = next(the_fields)
                    var = (
                        util.prims2var(fields, field)
                        if field in derived
                        else fields.get(
                            field, fields["v1"] if field == "v" else fields[field]
                        )
                    )

                    if self.units:
                        if field in ["p", "energy", "energy_rst"]:
                            var *= util.edens_scale.value
                        elif field in ["rho", "D"]:
                            var *= util.rho_scale.value

                    xx = (
                        mesh["x1v"]
                        if self.ndim == 2
                        else mesh[f"x{self.projection[0]}v"]
                    )
                    yy = (
                        mesh["x2v"]
                        if self.ndim == 2
                        else mesh[f"x{self.projection[1]}v"]
                    )

                    if setup["coord_system"] == "axis_cylindrical" and idx == 1:
                        xx *= -1

                    if self.ndim == 3:
                        box_depth = (
                            np.deg2rad(self.box_depth)
                            if not self.cartesian
                            else self.box_depth
                        )
                        coord_idx = find_nearest(
                            mesh[f"x{self.projection[2]}v"], box_depth
                        )[0]
                        var = (
                            var[coord_idx]
                            if self.projection[2] == 3
                            else (
                                var[:, coord_idx, :]
                                if self.projection[2] == 2
                                else var[:, :, coord_idx]
                            )
                        )

                    if not self.cartesian:
                        xx, yy = np.meshgrid(xx, yy)[::-1]
                        max_angle = np.abs(xx.max())
                        if any(self.xlims):
                            dtheta = self.xlims[1] - self.xlims[0]
                            ax.set_thetamin(
                                self.xlims[0] - (patches - 1) * dtheta * self.bipolar
                            )
                            ax.set_thetamax(
                                self.xlims[1]
                                + (patches - 1) * dtheta * (not self.bipolar)
                            )
                            low_wing, hi_wing = map(
                                lambda lim: util.find_nearest(
                                    mesh["x2v"], np.deg2rad(lim)
                                )[0],
                                self.xlims,
                            )
                            xx, yy, var = (
                                xx[low_wing:hi_wing] + idx * np.deg2rad(dtheta),
                                yy[low_wing:hi_wing],
                                var[low_wing:hi_wing],
                            )
                            if idx == 1:
                                xx = (
                                    -xx[::+1] + np.deg2rad(dtheta)
                                    if self.bipolar
                                    else xx[::-1]
                                )
                        elif max_angle == 0.5 * np.pi:
                            cbar_orientation = "horizontal"
                            self.axs.set_thetamin(-90)
                            self.axs.set_thetamax(+90)
                            xx = xx[:: theta_sign(idx)] + next(theta_cycle)
                            if idx in [2, 3]:
                                xx = xx[::-1]
                        elif (
                            max_angle > 0.5 * np.pi
                            and max_angle < 2.0 * np.pi
                            and patches > 1
                        ):
                            hemisphere = (
                                np.s_[: xx.shape[0]]
                                if patches == 2
                                or (patches == 3 and idx == 0)
                                or idx in [0, 1]
                                else np.s_[xx.shape[0] // 2 :]
                            )
                            xx, yy, var = (
                                theta_sign(idx) * xx[hemisphere],
                                yy[hemisphere],
                                var[hemisphere],
                            )
                        elif max_angle == 2.0 * np.pi and patches == 2:
                            hemisphere = (
                                slice(None, xx.shape[0] // 2)
                                if idx == 0
                                else slice(xx.shape[0] // 2, None)
                            )
                            xx, yy, var = (
                                xx[hemisphere],
                                yy[hemisphere],
                                var[hemisphere],
                            )

                    color_range = next(self.vrange)
                    norm = (
                        mcolors.LogNorm(vmin=color_range[0], vmax=color_range[1])
                        if self.log and field not in lin_fields
                        else mcolors.PowerNorm(
                            gamma=self.power, vmin=color_range[0], vmax=color_range[1]
                        )
                    )

                    self.frames.append(
                        ax.pcolormesh(
                            xx,
                            yy,
                            var,
                            cmap=next(self.color_map),
                            shading="auto",
                            norm=norm,
                        )
                    )

                    if self.cbar and idx < len(self.fields):
                        if self.cartesian:
                            divider = make_axes_locatable(ax)
                            side = (
                                "right"
                                if idx == 0
                                else (
                                    "left"
                                    if setup["coord_system"] == "axis_cylindrical"
                                    else "right"
                                )
                            )
                            cbaxes = divider.append_axes(side, size="5%", pad=0.05)
                        else:
                            single_width, height = 0.8, 0.05
                            width = single_width / len(self.fields)
                            x = (
                                0.1
                                if width == single_width
                                else (0.1 - 4e-2) + (1 - idx) * (width + 8e-2)
                            )
                            cbaxes = (
                                self.fig.add_axes([x, 0.2, width, height])
                                if cbar_orientation == "horizontal"
                                else self.fig.add_axes(
                                    [
                                        0.85 if idx == 0 else 0.10,
                                        0.1,
                                        0.03,
                                        0.8
                                        / (
                                            2
                                            if any(self.xlims) and not self.cartesian
                                            else 1
                                        ),
                                    ]
                                )
                            )

                        cbarfmt = (
                            tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
                            if self.log and field not in lin_fields
                            else None
                        )
                        cbar = self.fig.colorbar(
                            self.frames[idx],
                            orientation=cbar_orientation,
                            cax=cbaxes,
                            format=cbarfmt,
                        )
                        set_cbar_label = (
                            cbar.ax.set_xlabel
                            if cbar_orientation == "horizontal"
                            else cbar.ax.set_ylabel
                        )
                        labelpad = (
                            -50
                            if cbar_orientation == "vertical"
                            and idx in [1, 2]
                            and not (any(self.xlims) and not self.cartesian)
                            else None
                        )
                        if idx in [1, 2] and cbar_orientation == "vertical":
                            cbaxes.yaxis.set_ticks_position("left")
                        set_cbar_label(
                            (
                                r"$\log~${}".format(field_str[idx])
                                if self.log and field not in lin_fields
                                else r"{}".format(field_str[idx])
                            ),
                            labelpad=labelpad,
                        )

            if self.extra_args:
                angs = (
                    np.linspace(xextent[0], xextent[1], 1000)
                    if any(self.xlims)
                    else np.linspace(mesh["x2v"][0], mesh["x2v"][-1], mesh["x2v"].size)
                )
                if "eps" in self.extra_args:
                    eps = float(self.extra_args["eps"])
                    a, b = self.extra_args["radius"] * (1 - eps) ** (
                        -1 / 3
                    ), self.extra_args["radius"] * (1 - eps) ** (2 / 3)

                    def radius(theta):
                        return (
                            a
                            * b
                            / ((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
                            ** 0.5
                        )

                    r_theta = radius(angs)
                else:
                    from .extras.helpers import equipotential_surfaces

                    r_theta = equipotential_surfaces(**self.extra_args)

                ax.plot(angs, r_theta, linewidth=1, linestyle="--", color="grey")
                ax.plot(-angs, r_theta, linewidth=1, linestyle="--", color="grey")

            time = setup["time"] * (util.time_scale if self.units else 1.0)
            precision = 1 if time < 1 and self.print else 2 if self.print else 0
            if self.setup:
                title = f"{self.setup} t = {time:.{precision}f}"
                if self.cartesian:
                    ax.set_title(title)
                else:
                    kwargs = {"y": 0.8 if mesh["x2v"].max() == 0.5 * np.pi else 1.03}
                    self.fig.suptitle(title, **kwargs)

            if not self.cartesian:
                ax.set_rmin(self.ylims[0] or yy[0, 0])
                ax.set_rmax(self.ylims[1] or yy[0, -1])
            else:
                ax.set_ylim(*self.ylims)
                ax.set_aspect("equal")
                xlabel, ylabel = (
                    (f"$x{self.projection[0]}$", f"$x{self.projection[1]}$")
                    if self.ndim == 3
                    else ("$x$", "$y$")
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            if self.xmax:
                ax.set_rmax(self.xmax)

            if self.cbar:
                self.cbaxes, self.cbarfmt, self.cbar_orientation = (
                    cbaxes,
                    cbarfmt,
                    cbar_orientation,
                )

    def plot_histogram(self) -> None:
        colormap = plt.get_cmap(self.cmap[0])
        set_labels = cycle([None]) if not self.labels else cycle(self.labels)
        annotation_placed = False
        dims = [1, 2, 2]
        for axidx, ax in enumerate(
            ax_iter := get_iterable(self.axs, func=list if self.nplots == 1 else iter)
        ):
            for idx, file in enumerate(get_iterable(self.flist[self.current_frame])):
                if self.nplots > 1:
                    if idx == len(self.flist) // 2:
                        ax = next(ax_iter)
                        axidx += 1
                        annotation_placed = False

                fields, setup, mesh = util.read_file(file)
                time = setup["time"] * util.time_scale
                if fields["rho"].ndim == 1:
                    dV = calc_cell_volume1D(x1=mesh["x1v"])
                elif fields["rho"].ndim == 2:
                    dV = calc_cell_volume2D(x1=mesh["x1v"], x2=mesh["x2v"])
                    if mesh["x2v"][-1] == 0.5 * np.pi:
                        dV *= 2.0
                else:
                    dV = calc_cell_volume3D(
                        x1=mesh["x1v"], x2=mesh["x2v"], x3=mesh["x3v"]
                    )

                if self.kinetic:
                    mass = dV * fields["W"] * fields["rho"]
                    var = (fields["W"] - 1.0) * mass * util.e_scale.value
                elif self.enthalpy:
                    enthalpy = 1.0 + fields["ad_gamma"] * fields["p"] / (
                        fields["rho"] * (fields["ad_gamma"] - 1.0)
                    )
                    var = (enthalpy - 1.0) * dV * util.e_scale.value
                elif self.mass:
                    var = dV * fields["W"] * fields["rho"] * util.mass_scale.value
                else:
                    edens_total = util.prims2var(fields, "energy")
                    var = edens_total * dV * util.e_scale.value

                u = fields["gamma_beta"]
                for cutoff in self.cutoffs:
                    if cutoff > 0:
                        mean_gb = np.sum(u[u > cutoff] * var[u > cutoff]) / np.sum(
                            var[u > cutoff]
                        )
                        print(f"Mean gb > {cutoff}: {mean_gb}")

                gbs = np.geomspace(1e-5, u.max(), 128)
                var = np.asanyarray([var[u > gb].sum() for gb in gbs])
                if self.powerfit:
                    E_seg_rat = var[1:] / var[:-1]
                    gb_seg_rat = gbs[1:] / gbs[:-1]
                    E_seg_rat[E_seg_rat == 0] = 1

                    slope = (var[1:] - var[:-1]) / (gbs[1:] - gbs[:-1])
                    power_law_region = np.argmin(slope)
                    up_min = find_nearest(gbs, 2 * gbs[power_law_region:][0])[0]
                    upower = gbs[up_min:]

                    # Fix the power law segment, ignoring the sharp dip at the
                    # tail of the CDF
                    epower_law_seg = E_seg_rat[up_min : np.argmin(E_seg_rat > 0.8)]
                    gbpower_law_seg = gb_seg_rat[up_min : np.argmin(E_seg_rat > 0.8)]
                    segments = np.log10(epower_law_seg) / np.log10(gbpower_law_seg)
                    alpha = 1.0 - np.mean(segments)
                    E_0 = var[up_min] * upower[0] ** (alpha - 1)
                    print("Avg power law index: {:.2f}".format(alpha))
                    ax.plot(upower, E_0 * upower ** (-(alpha - 1)), "--")

                label = next(set_labels)

                if self.xfill_scale:
                    util.fill_below_intersec(gbs, var, self.xfill_scale, axis="x")
                elif self.yfill_scale:
                    util.fill_below_intersec(
                        gbs, var, self.yfill_scale * var.max(), axis="y"
                    )

                self.frames += [
                    ax.hist(
                        gbs,
                        bins=gbs,
                        weights=var,
                        label=label,
                        histtype="step",
                        rwidth=1.0,
                        linewidth=3.0,
                    )
                ]
                print(f"Computed histogram for {file}")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
                ax.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
                if any(self.xlims):
                    ax.set_xlim(*self.xlims)

                if self.nplots == 1 or idx == len(self.flist) - 1:
                    if any(self.ylims):
                        ax.set_ylim(*self.ylims)
                    ax.set_xlabel(r"$\Gamma\beta $")

                if self.kinetic:
                    ax.set_ylabel(r"$E_{\rm k}( > \Gamma \beta) \ [\rm{erg}]$")
                elif self.enthalpy:
                    ax.set_ylabel(r"$H ( > \Gamma \beta) \ [\rm{erg}]$")
                elif self.mass:
                    ax.set_ylabel(r"$M ( > \Gamma \beta) \ [\rm{g}]$")
                else:
                    ax.set_ylabel(r"$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$")

                if self.annot_text:
                    if not annotation_placed:
                        try:
                            annotation = self.annot_text[axidx]
                        except IndexError:
                            annotation = ""
                        self.place_annotation(ax, annotation)
                        annotation_placed = True

            if self.setup:
                if len(self.frames) == 1:
                    setup = self.setup + f",$~t={time:.1f}$"
                else:
                    setup = self.setup
                ax.set_title(f"{setup}")

            if self.labels:
                if self.nplots > 1:
                    self.axs[0].legend(loc=self.legend_loc)
                else:
                    alpha = 0.0 if self.transparent else 1.0
                    ax.legend(loc=self.legend_loc, framealpha=alpha)

    def plot_mean_vs_time(self) -> None:
        weighted_vars = []
        times = []
        label = self.labels[0] if self.labels else None

        self.axs.set_title(f"{self.setup}")
        self.axs.set_xlabel("$t$")
        if not isinstance(self.flist, dict):
            self.flist = {0: self.flist}

        for key in self.flist.keys():
            weighted_vars = []
            times = []
            label = self.labels[key] if self.labels else None
            for idx, file in enumerate(self.flist[key]):
                fields, setup, mesh = util.read_file(file)
                if self.fields[0] in derived:
                    var = util.prims2var(fields, self.fields[0])
                else:
                    var = fields[self.fields[0]]

                if self.weight != self.fields[0]:
                    weights = util.prims2var(fields, self.weight)

                    if self.ndim == 1:
                        dV = calc_cell_volume1D(x1=mesh["x1v"])
                    elif self.ndim == 2:
                        dV = calc_cell_volume2D(x1=mesh["x1v"], x2=mesh["x2v"])
                    else:
                        dV = calc_cell_volume3D(
                            x1=mesh["x1v"], x2=mesh["x2v"], x3=mesh["x3v"]
                        )
                    weighted = np.sum(weights * var * dV) / np.sum(weights * dV)
                else:
                    weighted = np.max(var)

                weighted_vars += [weighted]
                times += [setup["time"]]
                print(f"processed file {file}...", flush=True, end="\n")

            times = np.asanyarray(times)
            data = np.asanyarray(weighted_vars)
            self.frames += [self.axs.plot(times, data, label=label, alpha=1.0)]

            if self.pictorial:
                at_the_end = key == len(self.flist.keys()) - 1
                if self.fields[0] in ["gamma_beta", "u1", "u"] and at_the_end:
                    self.axs.plot(
                        times,
                        data[0] * (times / times[0]) ** (-3 / 2),
                        label=r"$\propto t^{-3/2}$",
                        color="grey",
                        linestyle=":",
                    )
                    if self.break_time:
                        tb_index = int(np.argmin(np.abs(times - self.break_time)))
                        tref = times[tb_index:]
                        exp_curve = np.exp(1 - tref / tref[0])
                        self.axs.plot(
                            tref,
                            data[tb_index] * exp_curve,
                            label=r"$\propto \exp(-t)$",
                            color="grey",
                            linestyle="-.",
                        )
                        self.axs.plot(
                            tref,
                            data[tb_index] * (tref / tref[0]) ** (-3),
                            label=r"$\propto t^{-3}$",
                            color="grey",
                            linestyle="--",
                        )

        if self.log:
            self.axs.set_xscale("log")
            if (
                self.fields[0] in ["gamma_beta", "u", "u1"]
                or self.fields[0] not in lin_fields
            ):
                self.axs.set(yscale="log")

        ylabel = util.get_field_str(self)
        self.axs.set_xlabel(r"$t$")
        if self.weight == self.fields[0]:
            self.axs.set_ylabel(rf"$($ {ylabel} $)_{{\rm max}}$")
        else:
            self.axs.set_ylabel(rf"$\langle$ {ylabel} $\rangle$")

        if self.legend:
            self.axs.legend(loc=self.legend_loc)

    def plot_dx_domega(self) -> None:
        annotation_placed = False
        for axidx, ax in enumerate(
            ax_iter := get_iterable(self.axs, func=list if self.nplots == 1 else iter)
        ):
            for idx, file in enumerate(get_iterable(self.flist[self.current_frame])):
                fields, setup, mesh = util.read_file(file)
                gb = fields["gamma_beta"]
                time = setup["time"] * util.time_scale

                if self.nplots > 1:
                    if idx == len(self.flist) // 2:
                        ax = next(ax_iter)
                        axidx += 1
                        annotation_placed = False

                if self.ndim == 2:
                    dV = calc_cell_volume2D(x1=mesh["x1v"], x2=mesh["x2v"])
                    domega = calc_domega(x2=mesh["x2v"])
                else:
                    dV = calc_cell_volume3D(
                        x1=mesh["x1v"], x2=mesh["x2v"], x3=mesh["x3v"]
                    )
                    domega = calc_domega(x2=mesh["x2v"], x3=mesh["x3v"])

                if self.kinetic:
                    mass = dV * fields["W"] * fields["rho"]
                    var = (fields["W"] - 1.0) * mass * util.e_scale.value
                elif self.enthalpy:
                    enthalpy = 1.0 + fields["ad_gamma"] * fields["p"] / (
                        fields["rho"] * (fields["ad_gamma"] - 1.0)
                    )
                    var = (enthalpy - 1.0) * dV * util.e_scale.value
                elif self.mass:
                    var = dV * fields["W"] * fields["rho"] * util.mass_scale.value
                elif self.momentum:
                    mass = dV * fields["W"] * fields["rho"] * util.mass_scale.value
                    var = mass * (1 - 1 / fields["W"] ** 2) ** (0.5) * util.c.value
                else:
                    edens_total = util.prims2var(fields, "energy")
                    var = edens_total * dV * util.e_scale.value

                theta = np.rad2deg(mesh["x2v"])
                for cidx, cutoff in enumerate(self.cutoffs):
                    deg_per_bin = 0.0001  # degrees in bin
                    num_bins = int((mesh["x2v"][-1] - mesh["x2v"][0]) / deg_per_bin)
                    if num_bins > theta.size:
                        num_bins = theta.size
                    tbins = np.linspace(mesh["x2v"][0], mesh["x2v"][-1], num_bins)
                    tbin_edges = np.linspace(
                        mesh["x2v"][0], mesh["x2v"][-1], num_bins + 1
                    )
                    domega_bins = (
                        2.0
                        * np.pi
                        * np.array(
                            [
                                np.cos(tl) - np.cos(tr)
                                for tl, tr in zip(tbin_edges, tbin_edges[1:])
                            ]
                        )
                    )

                    # ===================
                    # Manual Way
                    # ===================
                    # dvar        = var / domega[:,np.newaxis]
                    # cdf         = np.array([x[gb[idx] > cutoff].sum() for idx, x in enumerate(dvar)])
                    # bin_step    = int(theta.size / num_bins)
                    # domega_bins = 2.0 * np.pi * np.array([np.cos(tl) - np.cos(tr) for tl, tr in zip(tbin_edges, tbin_edges[1:])])
                    # dx_domega   = np.array([cdf[i:i+bin_step].sum() for i in range(0, bin_step * num_bins, bin_step)])
                    # iso_var     = 4.0 * np.pi * dx_domega

                    # ==================
                    # Numpy Hist way
                    # ==================
                    cdf = np.array(
                        [x[gb[idx] > cutoff].sum() for idx, x in enumerate(var)]
                    )
                    dx, _ = np.histogram(mesh["x2v"], weights=cdf, bins=tbin_edges)
                    dw, _ = np.histogram(mesh["x2v"], weights=domega, bins=tbin_edges)
                    dx_domega = dx / dw
                    iso_var = 4.0 * np.pi * dx_domega

                    # if the maximum is near the pole,
                    # it's a jet otherwise it's a ring
                    if np.rad2deg(tbins[np.argmax(iso_var)]) <= 45:
                        x = 2
                    else:
                        x = 1

                    viso = (
                        4.0
                        * np.pi
                        * (dx_domega * dx_domega * domega_bins).sum()
                        / (dx_domega * domega_bins).sum()
                    )
                    vtot = (domega_bins * dx_domega).sum()
                    thetax = x * np.arcsin((vtot / viso) ** (1 / x))
                    print(f"{'gamma_beta':.<50}: ", cutoff)
                    print(f"{'X_iso':.<50}: ", viso)
                    print(f"{'X_available':.<50}: ", vtot)
                    print(f"{'opening angle[deg]':.<50}: ", np.rad2deg(thetax))
                    print("")

                    tbins = np.rad2deg(tbins)
                    if cutoff.is_integer():
                        cprecision = 0
                    else:
                        cprecision = 1

                    if idx == 0:
                        label = rf"$\Gamma \beta > {cutoff:.{cprecision}f}$"
                    else:
                        label = None
                    color_idx = idx if len(self.fields) > 1 else cidx
                    if self.norm:
                        # iso_var *= dw / (4.0 * np.pi)
                        iso_var /= 4.0 * np.pi

                    if self.xlims == [-90, 90]:
                        tbins -= 90
                    # ax.plot(np.rad2deg(mesh['x2']), cdf, label=label)
                    ax.step(tbins, iso_var, label=label)
                    if self.log:
                        ax.set_yscale("log")

                    if self.broken_ax:
                        d = 0.5  # proportion of vertical to horizontal extent of the slanted line
                        kwargs = dict(
                            marker=[(-1, -d), (1, d)],
                            markersize=12,
                            linestyle="none",
                            color="k",
                            mec="k",
                            mew=1,
                            clip_on=False,
                        )

                        # hide the spines between ax and ax2
                        if axidx == 0:
                            ax.spines["bottom"].set_visible(False)
                            ax.set_ylim(1e49, 1e50)
                            ax.plot(0, 0, transform=self.axs[0].transAxes, **kwargs)
                            ax.set_xticks([])
                            # mf = tkr.ScalarFormatter(useMathText=True)
                            # mf.set_powerlimits((-1,1))
                            # ax.yaxis.set_major_formatter(mf)
                            # ax.set_yticklabels([r'$10^{50}$', r'$10^{51}$'])
                            # mf = tkr.ScalarFormatter(useMathText=True)
                            # mf.set_powerlimits((-2,2))
                            # plt.gca().yaxis.set_major_formatter(mf)
                            # ax.ticklabel_format(axis='y', scilimits=[-3, 3])
                        else:
                            ax.plot(0, 1, transform=self.axs[1].transAxes, **kwargs)
                            ax.set_ylim(1e44, 5e45)

                    # inset axes....
                    if self.inset is not None:
                        import ast

                        if cidx == 0:
                            if self.broken_ax:
                                axins = self.axs[1].inset_axes([0.2, 0.15, 0.47, 0.87])
                            else:
                                axins = ax.inset_axes([0.12, 0.10, 0.47, 0.47])
                        axins.step(tbins, iso_var)
                        # subregion of the original image
                        axins.set_xlim(*ast.literal_eval(self.inset["xlims"]))
                        axins.set_ylim(*ast.literal_eval(self.inset["ylims"]))
                        axins.set_yscale("log")
                        axins.tick_params(labelsize=10)

                    esn = np.sum(var[gb < 0.1])
                    print(f"Energy left for supernova: {esn:.2e}")

            if axidx == len(get_iterable(self.axs)) - 1:
                ax.set_xlabel(r"$\theta~\rm[deg]$")

            if self.norm:
                # ylabel = r'$E_{k,\phi}(> \Gamma\beta) [\rm erg]$'
                ylabel = r"$dE_{k}/d\Omega(> \Gamma\beta) [\rm erg]$"
            elif self.kinetic:
                ylabel = r"$E_{k,\rm iso}(> \Gamma\beta) [\rm erg]$"
            elif self.mass:
                ylabel = r"$M_{\rm iso}(> \Gamma\beta) [\rm g]$"
            elif self.momentum:
                ylabel = r"$dP/d\Omega(> \Gamma\beta) [\rm g~cm~s^{-1}]$"
            else:
                ylabel = r"$E_{\rm iso}(> \Gamma\beta) [\rm g]$"

            if len(get_iterable(self.axs)) > 1:
                self.fig.supylabel(ylabel)
            else:
                ax.set_ylabel(ylabel)

            if any(self.xlims):
                ax.set_xlim(*self.xlims)
            else:
                ax.set_xlim(theta[0], theta[-1])

            if self.ylims:
                ax.set_ylim(*self.ylims)

            if self.setup:
                ax.set_title(f"{self.setup}")

            if self.annot_text:
                if not annotation_placed:
                    try:
                        annotation = self.annot_text[axidx]
                    except IndexError:
                        annotation = ""
                    self.place_annotation(ax, annotation)
                    annotation_placed = True

        if self.nplots > 1 or self.broken_ax:
            self.axs[0].legend(loc=self.legend_loc)
        else:
            ax.legend(loc=self.legend_loc)

    def plot(self) -> None:
        self.frames = []
        if self.hist:
            self.plot_histogram()
        elif self.weight:
            self.plot_mean_vs_time()
        elif self.ndim > 1 and self.dx_domega:
            self.plot_dx_domega()
        else:
            if self.ndim == 1 or self.oned_slice:
                self.plot_line()
            else:
                self.plot_multidim()

    def show(self) -> None:
        plt.show()

    def save_fig(self) -> None:
        if self.kind == "snapshot":
            ext = "png" if self.png else "pdf"
            fig_name = f"{self.save}.{ext}".replace("-", "_")
            logger.debug(f"Saving figure as {fig_name}")
            self.fig.savefig(
                fig_name,
                dpi=600,
                transparent=self.transparent,
                bbox_inches=self.bbox_kind,
            )
        else:
            ext = "mp4"
            fig_name = f"{self.save}.{ext}".replace("-", "_")
            logger.debug(f"Saving movie as {fig_name}")
            self.animation.save(
                fig_name,
                dpi=600,
                # bbox_inches=self.bbox_kind,
                progress_callback=lambda i, n: print(
                    f"Saving frame {i} of {n}", end="\r", flush=True
                ),
            )

    def create_figure(self) -> None:
        colormap = plt.get_cmap(self.cmap[0])
        if self.nplots == 1:
            nind_curves = max(
                len(self.fields),
                len(self.files),
                len(self.cutoffs),
                len(self.coords["xj"].split(",")) * len(self.coords["xk"].split(",")),
            )
        else:
            nind_curves = self.nplots // len(self.files)

        colors = np.array([colormap(k) for k in np.linspace(0.1, 0.9, nind_curves)])
        linestyles = [
            x[0] for x in zip(cycle(["-", "--", ":", "-."]), range(len(self.fields)))
        ]
        default_cycler = cycler(color=colors) * (cycler(linestyle=linestyles))

        plt.rc("axes", prop_cycle=default_cycler)
        if self.nplots == 1:
            if self.square_plot:
                nplots = self.broken_ax + 1
                self.fig, self.axs = plt.subplots(
                    nplots, 1, figsize=self.fig_dims, sharex=False
                )
                # self.fig.subplots_adjust(hspace=0.05)
                for ax in get_iterable(self.axs):
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
            else:
                self.fig, self.axs = plt.subplots(
                    1,
                    1,
                    subplot_kw={"projection": "polar"},
                    figsize=self.fig_dims,
                    constrained_layout=True,
                )
                self.axs.grid(False)
                self.axs.set_theta_zero_location("N")
                self.axs.set_theta_direction(-1)
                self.axs.set_xticklabels([])
                self.axs.set_yticklabels([])
        else:
            if not self.square_plot:
                raise NotImplementedError()

            self.fig, self.axs = plt.subplots(
                self.nplots, 1, figsize=self.fig_dims, sharex=True
            )
            for ax in self.axs:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

    def update_frame(self, frame: int):
        self.current_frame = frame
        fields, setups, mesh = util.read_file(self.flist[frame])
        time = setups["time"] * (util.time_scale if self.units else 1.0)
        if self.setup:
            title = rf"{self.setup} at t = {time:.1f}"
            if self.cartesian or self.oned_slice:
                self.axs.set_title(title)
            else:
                # speciifc to publication figure
                kwargs = {
                    "y": 0.95 if mesh["x2v"].max() == np.pi else 0.8,
                    # -------------------- Text for ring wedges
                    # 'y': 0.30,
                    # 'x': 0.80,
                    # 'color': 'white'
                    # ------------------- Text for jet wedges
                    # 'y': 0.9,
                    # 'x': 0.32,
                    # 'color': 'white',
                }
                self.fig.suptitle(title, **kwargs)

        scale_cycle = cycle(self.scale_downs)
        for idx, field in enumerate(self.fields):
            if field in derived:
                var = util.prims2var(fields, field)
            else:
                if field == "v":
                    field = "v1"
                var = fields[field]

            if self.units:
                if field in ["p", "energy", "energy_rst"]:
                    var *= util.edens_scale.value
                elif field in ["rho", "D"]:
                    var *= util.rho_scale.value

            if self.ndim == 1 or self.oned_slice:
                yvar = var
                self.axs.set_xlim(mesh["x1v"][0], mesh["x1v"][-1])
                x = calc_any_mean(mesh["x1v"], setups["x1_cell_spacing"])
                if self.oned_slice:
                    x = calc_any_mean(
                        mesh[f"{self.oned_slice}v"],
                        setups[f"{self.oned_slice}_cell_spacing"],
                    )
                    for xkcoord in map(float, self.coords["xk"].split(",")):
                        for xjcoord in map(float, self.coords["xj"].split(",")):
                            # coord_label =label + f", $x_2={xjcoord:.1f}$"
                            if not self.cartesian:
                                xjcoord = np.deg2rad(xjcoord)
                            yidx = find_nearest(mesh["x2v"], xjcoord)[0]
                            if self.ndim == 2:
                                yvar = var[yidx]
                            else:
                                # coord_label += f', $x_3={xkcoord:.1f}$'
                                if not self.cartesian:
                                    xkcoord = np.deg2rad(xkcoord)
                                zidx = find_nearest(mesh["x3v"], xkcoord)[0]
                                yvar = var[zidx, yidx]
                self.frames[idx].set_data(x, yvar / next(scale_cycle))
                # if self.refs:
                # x = mesh['x1']
                # self.refs[idx].set_data(x, self.refy * (x / self.refx) ** (-3/2))
            else:
                if len(self.fields) > 1:
                    self.frames[idx].set_array(var.ravel())
                else:
                    # affect the generator w/o using output
                    if len(self.fields) > 1:
                        any(
                            drawing.set_array(var.ravel())
                            for drawing in self.frames[idx]
                        )
                    else:
                        any(drawing.set_array(var.ravel()) for drawing in self.frames)

                if not self.square_plot:
                    if not self.xmax and not any(self.ylims):
                        self.axs.set_ylim(mesh["x1v"][0], mesh["x1v"][-1])
                    elif self.pan_speed:
                        max_extent = self.extent or mesh["x1v"][-1]
                        min_extent = self.xmax or 1.5 * mesh["x1v"][0]
                        self.axs.set_rmax(
                            min_extent
                            + max_extent * self.pan_speed * (frame / len(self.flist))
                        )

        return (self.frames,)

    def animate(self) -> None:
        from matplotlib.animation import FuncAnimation

        self.current_frame = 0
        self.plot()
        # self.frames.pop(0)
        self.animation = FuncAnimation(
            # Your Matplotlib Figure object
            self.fig,
            # The function that does the updating of the Figure
            self.update_frame,
            # Frame information (here just frame number)
            np.arange(self.frame_count),
            # blit = True,
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000 / self.frame_rate,
            repeat=True,
        )


def visualize(parser: argparse.ArgumentParser, ndim: int) -> None:
    viz = Visualizer(parser, ndim)
    if viz.kind == "movie":
        if viz.ndim == 3 and not viz.oned_slice:
            ...
            # raise NotImplementedError("3D movies not yet implemented")
        viz.animate()
    else:
        viz.plot()

    if viz.save:
        viz.save_fig()
    else:
        viz.show()
