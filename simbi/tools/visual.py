import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import matplotlib.ticker as tkr
from itertools import cycle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import utility as util
from ..detail.slogger import logger
from ..detail.helpers import (
    get_iterable, 
    calc_cell_volume1D,
    calc_cell_volume2D,
    calc_cell_volume3D
)
from ..detail import get_subparser

try:
    import cmasher as cmr
except ImportError:
    pass

derived = [
    'D',
    'momentum',
    'energy',
    'energy_rst',
    'enthalpy',
    'temperature',
    'T_eV',
    'mass',
    'chi_dens',
    'mach',
    'u1',
    'u2',
    'u3',
    'u']
field_choices = [
    'rho',
    'v1',
    'v2',
    'v3',
    'v',
    'p',
    'gamma_beta',
    'chi'] + derived
lin_fields = ['chi', 'gamma_beta', 'u1', 'u2', 'u3', 'u']


def tuple_arg(param: str) -> tuple[int]:
    try:
        return tuple(int(arg) for arg in param.split(','))
    except:
        raise argparse.ArgumentTypeError("argument must be tuple of ints")

class Visualizer:
    def __init__(self, parser: argparse.ArgumentParser, ndim: int) -> None:
        self.current_frame = slice(None)
        self.plotted_references = False
        self.ndim = ndim
        if self.ndim != 1:
            plot_parser = get_subparser(parser, 1)
            plot_parser.add_argument(
                '--cbar_sub',
                dest='cbar2',
                metavar='Range of Color Bar for secondary plot',
                nargs='+',
                type=float,
                default=[
                    None,
                    None],
                help='The colorbar range you\'d like to plot')
            plot_parser.add_argument(
                '--no_cbar',
                dest='no_cbar',
                action='store_true',
                default=False,
                help='colobar visible siwtch')
            plot_parser.add_argument(
                '--cmap2',
                dest='cmap2',
                metavar='Color Bar Colarmap 2',
                default='magma',
                help='The secondary colorbar cmap you\'d like to plot')
            plot_parser.add_argument(
                '--rev_cmap',
                dest='rcmap',
                action='store_true',
                default=False,
                help='True if you want the colormap to be reversed')
            plot_parser.add_argument(
                '--x',
                dest='x',
                nargs='+',
                default=None,
                type=float,
                help='List of x values to plot field max against')
            plot_parser.add_argument(
                '--xlabel',
                dest='xlabel',
                nargs=1,
                default='X',
                help='X label name')
            plot_parser.add_argument(
                '--de_domega',
                dest='de_domega',
                action='store_true',
                default=False,
                help='Plot the dE/dOmega plot')
            plot_parser.add_argument(
                '--dm_domega',
                dest='dm_domega',
                action='store_true',
                default=False,
                help='Plot the dM/dOmega plot')
            plot_parser.add_argument(
                '--dec_rad',
                dest='dec_rad',
                default=False,
                action='store_true',
                help='Compute dr as function of angle')
            plot_parser.add_argument(
                '--cutoffs',
                dest='cutoffs',
                default=[0.0],
                type=float,
                nargs='+',
                help='The 4-velocity cutoff value for the dE/dOmega plot')
            plot_parser.add_argument(
                '--nwedge',
                dest='nwedge',
                default=0,
                type=int,
                help='Number of wedges')
            plot_parser.add_argument(
                '--cbar_orient',
                dest='cbar_orient',
                default='vertical',
                type=str,
                help='Colorbar orientation',
                choices=[
                    'horizontal',
                    'vertical'])
            plot_parser.add_argument(
                '--wedge_lims',
                dest='wedge_lims',
                default=[0.4, 1.4, 70, 110],
                type=float,
                nargs=4,
                help="wedge limits")
            plot_parser.add_argument(
                '--bipolar',
                dest='bipolar',
                default=False,
                action='store_true')
            plot_parser.add_argument(
                '--subplots',
                dest='subplots',
                default=None,
                type=int)
            plot_parser.add_argument(
                '--sub_split',
                dest='sub_split',
                default=None,
                nargs='+',
                type=int)
            plot_parser.add_argument(
                '--tau_s',
                dest='tau_s',
                action='store_true',
                default=False,
                help='The shock optical depth')
            plot_parser.add_argument(
                '--viewing',
                help='viewing angle of simulation in [deg]',
                type=float,
                default=None,
                nargs='+')
            plot_parser.add_argument(
                '--oned_slice',
                help='index of x1 array for one-d projection',
                default=None,
                type=int)
            plot_parser.add_argument(
                '--projection',
                help='axes to project multidim solution onto',
                default=[1, 2, 3],
                type=tuple_arg,
                choices=[
                    (1, 2, 3), 
                    (1, 3, 2), 
                    (2, 3, 1),
                    (2, 1, 3),
                    (3, 1, 2),
                    (3, 2, 1)]
            )
            plot_parser.add_argument(
                '--box-depth',
                help='index depth for projecting 3D data onto 2D plane',
                type=int,
                default=0,
            )

        vars(self).update(**vars(parser.parse_args()))

        self.color_map = []
        self.cartesian = True
        self.flist, self.frame_count = util.get_file_list(self.files)
        if self.ndim != 1:
            for cmap in self.cmap:
                if self.rcmap:
                    self.color_map += [(plt.get_cmap(cmap)).reversed()]
                else:
                    self.color_map += [plt.get_cmap(cmap)]

            if isinstance(self.flist, dict):
                self.cartesian = util.read_file(
                    self, self.flist[0][0], self.ndim)[1]['is_cartesian']
            else:
                self.cartesian = util.read_file(
                    self, self.flist[0], self.ndim)[1]['is_cartesian']

        self.color_map = cycle(self.color_map)
        self.vrange = self.cbar
        if len(self.vrange) != len(self.fields):
            self.vrange += [(None, None)] * \
                (abs(len(self.fields) - len(self.vrange)))
        self.vrange = cycle(self.vrange)

        self.square_plot = False
        if self.cartesian or self.ndim == 1 or self.hist or self.weight:
            self.square_plot = True
        self.create_figure()

    def plot_1d(self):
        field_str   = util.get_field_str(self)
        scale_cycle = cycle(self.scale_downs)
        for ax in get_iterable(self.axs):
            for file in get_iterable(self.flist[self.current_frame]):
                fields, setup, mesh = util.read_file(
                    self, file, ndim=self.ndim)
                for idx, field in enumerate(self.fields):
                    if field in derived:
                        var = util.prims2var(fields, field)
                    else:
                        if field == 'v':
                            field = 'v1'
                        var = fields[field]

                    ax.set_title(f'{self.setup} at t = {setup["time"]:.2f}')
                    ax.set_xlim(mesh['x1'][0], mesh['x1'][-1])
                    if self.ylims:
                        ax.set_ylim(*self.ylims)
                    label = field_str[idx]
                    scale = next(scale_cycle)
                    if scale != 1:
                        label = label + f'/{int(scale)}'
                    line, = ax.plot(mesh['x1'], var / scale, label=label)
                    self.frames += [line]

        if self.log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        if len(self.fields) == 1:
            ax.set_ylabel(field_str)
        elif self.legend:
            ax.legend(loc=self.legend_loc)

        if self.cartesian:
            ax.set_xlabel('$x$')
        else:
            ax.set_xlabel('$r$')

    def plot_multidim(self) -> None:
        def theta_sign(quadrant: int) -> np.ndarray:
            if quadrant in [0, 3]:
                return 1
            else:
                return -1

        field_str = get_iterable(util.get_field_str(self))
        cbar_orientation = 'vertical'
        patches = len(self.fields)

        theta_cycle = cycle([0, -np.pi * 0.5, -np.pi, np.pi * 0.5])
        if patches == 1:
            patches += 1

        the_fields = cycle(self.fields)
        for ax in get_iterable(self.axs):
            for file in get_iterable(self.flist[self.current_frame]):
                fields, setup, mesh = util.read_file(self, file, ndim=self.ndim)
                ax.set_title(f'{self.setup} at t = {setup["time"]:.2f}')
                for idx in range(patches):
                    field = next(the_fields)
                    if field in derived:
                        var = util.prims2var(fields, field)
                    else:
                        if field == 'v':
                            field = 'v1'
                        var = fields[field]

                    xx = mesh['x1'] if self.ndim == 2 else mesh[f'x{self.projection[0]}']
                    yy = mesh['x2'] if self.ndim == 2 else mesh[f'x{self.projection[1]}']
                    if self.ndim == 3:
                        if self.projection[2] == 3:
                            var = var[self.box_depth]
                        elif self.projection[2] == 2:
                            var = var[:, self.box_depth, :]
                        else:
                            var = var[:, :, self.box_depth]
                        
                    if not self.cartesian:
                        # turn in mesh grid and then reverse
                        xx, yy = np.meshgrid(xx,yy)[::-1]
                        max_theta = np.abs(xx.max())
                        if max_theta < np.pi:
                            if patches <= 2:
                                cbar_orientation = 'horizontal'
                                self.axs.set_thetamin(-90)
                                self.axs.set_thetamax(+90)
                            else:
                                self.axs.set_thetamin(-180)
                                self.axs.set_thetamax(+180)
                            xx = xx[::theta_sign(idx)] + next(theta_cycle)
                        elif max_theta > 0.5 * np.pi and patches > 1:
                            if patches == 2:
                                hemisphere = np.s_[:]
                            elif patches == 3 and idx == 0:
                                hemisphere = np.s_[:]
                            elif idx in [0, 1]:
                                hemisphere = np.s_[: xx.shape[0] // 2]
                            else:
                                hemisphere = np.s_[xx.shape[0] // 2:]

                            xx = theta_sign(idx) * xx[hemisphere]
                            yy = yy[hemisphere]
                            var = var[hemisphere]

                    color_range = next(self.vrange)
                    if self.log and field not in lin_fields:
                        kwargs = {
                            'norm': mcolors.LogNorm(
                                vmin=color_range[0],
                                vmax=color_range[1])}
                    else:
                        kwargs = {
                            'norm': mcolors.PowerNorm(
                                gamma=self.power,
                                vmin=color_range[0],
                                vmax=color_range[1])}

                    self.frames += [ax.pcolormesh(
                        xx,
                        yy,
                        var,
                        cmap=next(self.color_map),
                        shading='auto',
                        **kwargs
                    )]

                    if not self.no_cbar:
                        if idx < len(self.fields):
                            if self.cartesian:
                                divider = make_axes_locatable(ax)
                                cbaxes = divider.append_axes(
                                    'right', size='5%', pad=0.05)
                            else:
                                if cbar_orientation == 'horizontal':
                                    single_width = 0.8
                                    width = single_width / len(self.fields)
                                    x = 0.1 + (1 - idx) * width * \
                                        (width != single_width)
                                    cbaxes = self.fig.add_axes(
                                        [x, 0.05, width, 0.05])
                                else:
                                    single_height = 0.8
                                    height = (
                                        single_height if len(self.fields) == 3 and idx == 0
                                        else 
                                        single_height / (len(self.fields) // 2)
                                    )

                                    x = [0.95, 0.03, 0.08, 0.9]
                                    y = [0.1, 0.1, 0.1, 0.1]
                                    cbaxes = self.fig.add_axes(
                                        [x[idx], y[idx], 0.03, height])

                            if self.log and field not in lin_fields:
                                logfmt = tkr.LogFormatterExponent(
                                    base=10.0, labelOnlyBase=True)
                                cbar = self.fig.colorbar(
                                    self.frames[idx], orientation=cbar_orientation, cax=cbaxes, format=logfmt)
                            else:
                                cbar = self.fig.colorbar(
                                    self.frames[idx], orientation=cbar_orientation, cax=cbaxes)

                            # Change the format of the field
                            set_cbar_label = cbar.ax.set_xlabel if cbar_orientation == 'horizontal' else cbar.ax.set_ylabel
                            labelpad = None
                            if cbar_orientation == 'vertical' and idx in [
                                    1, 2]:
                                labelpad = -60
                            if idx in [
                                    1, 2] and cbar_orientation == 'vertical':
                                cbaxes.yaxis.set_ticks_position('left')
                            if self.log and field not in lin_fields:
                                set_cbar_label(
                                    r'$\log$[{}]'.format(
                                        field_str[idx]), labelpad=labelpad)
                            else:
                                set_cbar_label(
                                    r'{}'.format(
                                        field_str[idx]),
                                    labelpad=labelpad)

                            if self.xmax:
                                self.axs.set_rmax(self.xmax)

    def plot_histogram(self) -> None:
        colors = plt.cm.twilight_shifted(
            np.linspace(0.25, 0.75, len(self.flist)))
        # ax.set_title(r'setup: {}'.format(args.setup))
        for ax in get_iterable(self.axs):
            for idx, file in enumerate(
                    get_iterable(self.flist[self.current_frame])):
                
                fields, setup, mesh = util.read_file(self, file, self.ndim)
                time = setup['time']
                if self.ndim == 1:
                    dV = calc_cell_volume1D(x1=mesh['x1'])
                elif self.ndim == 2:
                    dV = calc_cell_volume2D(x1=mesh['x1'], x2=mesh['x2'])
                else:
                    dV = calc_cell_volume3D(
                        x1=mesh['x1'], x2=mesh['x2'], x3=mesh['x3'])

                if self.kinetic:
                    mass = dV * fields['W'] * fields['rho']
                    var = (fields['W'] - 1.0) * mass * util.e_scale.value
                elif self.enthalpy:
                    enthalpy = 1.0 + \
                        fields['ad_gamma'] * fields['p'] / \
                        (fields['rho'] * (fields['ad_gamma'] - 1.0))
                    var = (enthalpy - 1.0) * dV * util.e_scale.value
                else:
                    edens_total = util.prims2var(fields, 'energy')
                    var = edens_total * dV * util.e_scale.value

                u = fields['gamma_beta']
                gbs = np.geomspace(1e-4, u.max(), 128)
                var = np.asanyarray([var[u > gb].sum() for gb in gbs])
                if self.powerfit:
                    E_seg_rat = energy[1:] / energy[:-1]
                    gb_seg_rat = gbs[1:] / gbs[:-1]
                    E_seg_rat[E_seg_rat == 0] = 1

                    slope = (energy[1:] - energy[:-1]) / (gbs[1:] - gbs[:-1])
                    power_law_region = np.argmin(slope)
                    up_min = find_nearest(
                        gbs, 2 * gbs[power_law_region:][0])[0]
                    upower = gbs[up_min:]

                    # Fix the power law segment, ignoring the sharp dip at the
                    # tail of the CDF
                    epower_law_seg = E_seg_rat[up_min: np.argmin(
                        E_seg_rat > 0.8)]
                    gbpower_law_seg = gb_seg_rat[up_min: np.argmin(
                        E_seg_rat > 0.8)]
                    segments = np.log10(epower_law_seg) / \
                        np.log10(gbpower_law_seg)
                    alpha = 1.0 - np.mean(segments)
                    E_0 = energy[up_min] * upower[0] ** (alpha - 1)
                    print('Avg power law index: {:.2f}'.format(alpha))
                    ax.plot(upower, E_0 * upower**(-(alpha - 1)), '--',
                            label=r'${}$ fit'.format(args.labels[case]))

                label = r'$E_T$' if not self.labels else f'{self.labels[idx]}, t={time:.2f}'
                self.frames += [ax.hist(gbs,
                                        bins=gbs,
                                        weights=var,
                                        label=label,
                                        color=colors[idx],
                                        histtype='step',
                                        rwidth=1.0,
                                        linewidth=3.0)]

            if self.nplots == 1:
                ax.set_xscale('log')
                ax.set_yscale('log')
                if self.xlims:
                    ax.set_xlim(*self.xlims)
                ax.set_xlabel(r'$\Gamma\beta $')
                if self.kinetic:
                    ax.set_ylabel(
                        r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$')
                elif self.enthalpy:
                    ax.set_ylabel(r'$H ( > \Gamma \beta) \ [\rm{erg}]$')
                else:
                    ax.set_ylabel(
                        r'$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$')

                if self.fill_scale is not None:
                    util.fill_below_intersec(
                        gbs, var, self.fill_scale * var.max(), colors[idx])

    def plot_mean_vs_time(self) -> None:
        weighted_vars = []
        times  = []
        colors = ['red', 'black']
        label  = self.labels[0] if self.labels else None

        self.axs.set_title(f'{self.setup}')
        self.axs.set_xlabel('$t$')
        if not isinstance(self.flist, dict):
            self.flist = {0: self.flist}

        for key in self.flist.keys():
            weighted_vars = []
            times = []
            label = self.labels[key] if self.labels else None
            for idx, file in enumerate(self.flist[key]):
                print(f'processing file {file}...', flush=True, end='\n')
                fields, setup, mesh = util.read_file(self, file, self.ndim)
                if self.fields[0] in derived:
                    var = util.prims2var(fields, self.fields[0])
                else:
                    var = fields[self.fields[0]]

                weights = util.prims2var(fields, self.weight)

                if self.ndim == 1:
                    dV = calc_cell_volume1D(x1=mesh['x1'])
                elif self.ndim == 2:
                    dV = calc_cell_volume2D(x1=mesh['x1'], x2=mesh['x2'])
                else:
                    dV = calc_cell_volume3D(
                        x1=mesh['x1'], x2=mesh['x2'], x3=mesh['x3'])
                weighted = np.sum(weights * var * dV) / np.sum(weights * dV)
                weighted_vars += [weighted]
                times += [setup['time']]

            times = np.asanyarray(times)
            data = np.asanyarray(weighted_vars)
            self.frames += [self.axs.plot(times,
                                          data,
                                          label=label,
                                          color=colors[key],
                                          alpha=1.0)]

            if self.fields[0] in ['gamma_beta', 'u1'] and not self.plotted_references:
                self.plotted_references = True
                exp_curve = np.exp(times[0] - times)
                if key == 0:
                    self.axs.plot(
                        times,
                        data[0] *
                        exp_curve,
                        label=r'$\propto \exp(-t)$',
                        color='grey',
                        linestyle='-.')
                    self.axs.plot(times,
                                  data[0] * (times / times[0]) ** (-3 / 2),
                                  label=r'$\propto t^{-3/2}$',
                                  color='grey',
                                  linestyle=':')
                    self.axs.plot(times,
                                  data[0] * (times / times[0]) ** (-3),
                                  label=r'$\propto t^{-3}$',
                                  color='grey',
                                  linestyle='--')

        if self.log:
            self.axs.set_xscale('log')
            if self.fields[0] == 'gamma_beta' or self.fields[0] not in lin_fields:
                self.axs.set(yscale='log')

        ylabel = util.get_field_str(self)
        self.axs.set_xlabel(r'$t$')
        self.axs.set_ylabel(rf"$\langle$ {ylabel} $\rangle$")

        if self.legend:
            self.axs.legend(loc=self.legend_loc)

    def plot(self) -> None:
        self.frames = []
        if self.hist:
            self.plot_histogram()
        elif self.weight:
            self.plot_mean_vs_time()
        else:
            if self.ndim == 1:
                self.plot_1d()
            else:
                self.plot_multidim()

    def show(self) -> None:
        plt.show()

    def save_fig(self) -> None:
        if self.kind == 'snapshot':
            ext = 'png' if self.png else 'pdf'
            fig_name = f'{self.save}.{ext}'.replace('-', '_')
            logger.info(f'Saving figure as {fig_name}')
            self.fig.savefig(fig_name, dpi=600, bbox_inches='tight')
        else:
            self.animation.save("{}.mp4".format(
                self.save.replace(" ", "_")),
                progress_callback=lambda i, n: print(
                f'Saving frame {i} of {n}', end='\r', flush=True)
            )

    def create_figure(self) -> None:
        if self.nplots == 1:
            if self.square_plot:
                self.fig, self.axs = plt.subplots(1, 1, figsize=self.fig_dims)
                self.axs.spines['top'].set_visible(False)
                self.axs.spines['right'].set_visible(False)
            else:
                if not self.hist:
                    self.fig, self.axs = plt.subplots(
                        1, 1, subplot_kw={
                            'projection': 'polar'}, figsize=self.fig_dims, constrained_layout=True)
                    self.axs.grid(False)
                    self.axs.set_theta_zero_location('N')
                    self.axs.set_theta_direction(-1)
                    self.axs.set_xticklabels([])
                    self.axs.set_yticklabels([])      
        else:
            raise NotImplementedError()

    def update_frame(self, frame: int):
        self.current_frame = frame
        fields, setups, mesh = util.read_file(
            self, self.flist[frame], ndim=self.ndim)
        time = setups['time'] * (util.time_scale if self.units else 1.0)
        if self.cartesian:
            self.axs.set_title('{} at t = {:.2f}'.format(self.setup, time))
        else:
            self.fig.suptitle(
                '{} at t = {:.2f}'.format(
                    self.setup, setups['time']), y=1.0)
        
        scale_cycle = cycle(self.scale_downs)
        for idx, field in enumerate(self.fields):
            if field in derived:
                var = util.prims2var(fields, field)
            else:
                if field == 'v':
                    field = 'v1'
                var = fields[field]

            if self.ndim == 1:
                self.axs.set_xlim(mesh['x1'][0], mesh['x1'][-1])
                self.frames[idx].set_data(mesh['x1'], var / next(scale_cycle))
            else:
                # affect the generator w/o using output
                any(drawing.set_array(var.ravel()) for drawing in self.frames)

        return self.frames,

    def animate(self) -> None:
        from matplotlib.animation import FuncAnimation
        self.current_frame = 0
        self.plot()
        self.animation = FuncAnimation(
            # Your Matplotlib Figure object
            self.fig,
            # The function that does the updating of the Figure
            self.update_frame,
            # Frame information (here just frame number)
            np.arange(self.frame_count),
            # blit = True,
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000 / 10,
            repeat=True
        )


def visualize(parser: argparse.ArgumentParser, ndim: int) -> None:
    viz = Visualizer(parser, ndim)
    if viz.kind == 'movie':
        viz.animate()
    else:
        viz.plot()

    if viz.save:
        viz.save_fig()
    else:
        viz.show()
