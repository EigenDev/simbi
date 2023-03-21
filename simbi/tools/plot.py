import argparse
import sys
import matplotlib.pyplot as plt
import importlib
from . import visual
from typing import Optional
from .utility import DEFAULT_SIZE, SMALL_SIZE, get_dimensionality, get_file_list
from ..detail import get_subparser
from pathlib import Path

tool_src = Path(__file__).resolve().parent / 'tools'

def colorbar_limits(c):
    try:
        vmin, vmax = map(float, c.split(','))
        return vmin, vmax
    except:
        raise argparse.ArgumentTypeError("Colorbar limits must be in the format: vmin,vmax")
    
def parse_plotting_arguments(
        parser: argparse.ArgumentParser) -> tuple[argparse.ArgumentParser, 
                                                  tuple[argparse.Namespace, list[str]]]:
    plot_parser = get_subparser(parser, 1)
    plot_parser.add_argument(
        'files',
        nargs='+',
        help='checkpoints files to plot')
    plot_parser.add_argument(
        'setup',
        type=str,
        help='The name of the setup being plotted')
    plot_parser.add_argument(
        '--fields',
        default=['rho'],
        nargs='+',
        help='the name of the field variable',
        choices=visual.field_choices)
    plot_parser.add_argument('--xmax', default=0.0, help='the domain range')
    plot_parser.add_argument(
        '--log',
        default=False,
        action='store_true',
        help='logarithmic plotting scale')
    plot_parser.add_argument(
        '--kinetic',
        default=False,
        action='store_true',
        help='plot the kinetic energy on the histogram')
    plot_parser.add_argument(
        '--enthalpy',
        default=False,
        action='store_true',
        help='plot the enthalpy on the histogram')
    plot_parser.add_argument(
        '--hist',
        default=False,
        action='store_true',
        help='convert plot to histogram')
    plot_parser.add_argument(
        '--mass',
        default=False,
        action='store_true',
        help='compute mass histogram')
    plot_parser.add_argument(
        '--dm-du',
        default=False,
        action='store_true',
        help='compute dM/dU over whole domain')
    plot_parser.add_argument(
        '--ax-anchor',
        default=None,
        type=str,
        nargs='+',
        help='anchor annotation text for each plot')
    plot_parser.add_argument(
        '--norm',
        default=False,
        action='store_true',
        help='flag to normalize plot axes')
    plot_parser.add_argument(
        '--labels',
        default=None,
        nargs='+',
        help='list of legend labels')
    plot_parser.add_argument(
        '--xlims',
        default=[None, None],
        type=float,
        nargs=2,
        help='limits of x axis')
    plot_parser.add_argument(
        '--ylims',
        default=[None, None],
        type=float,
        nargs=2,
        help='limits of y axis')
    plot_parser.add_argument(
        '--units',
        default=False,
        action='store_true',
        help='flag for dimensionful units')
    plot_parser.add_argument(
        '--power',
        default=1.0,
        type=float,
        help='exponent of power-law norm')
    plot_parser.add_argument(
        '--scale-downs',
        default = [1],
        type=float,
        nargs = '+',
        help='list of values to scale plotted variables down by',
    )
    plot_parser.add_argument(
        '--dbg',
        default=False,
        action='store_true',
        help='flag for dark background style')
    plot_parser.add_argument(
        '--tex',
        default=False,
        action='store_true',
        help='flag for latex typesetting')
    plot_parser.add_argument(
        '--print',
        default=False,
        action='store_true',
        help='flag for publications plot formatting')
    plot_parser.add_argument(
        '--pictorial',
        default=False,
        action='store_true',
        help='flag for creating figs without data')
    plot_parser.add_argument(
        '--anot-loc',
        default=None,
        type=str,
        help='location of annotations',
        choices=[
            'lower_left',
            'lower_right',
            'upper_left',
            'upper_right',
            'upper_center',
            'lower_center',
            'center'])
    plot_parser.add_argument(
        '--legend_loc',
        default=None,
        type=str,
        help='location of legend',
        choices=[
            'lower left',
            'lower right',
            'upper left',
            'upper right',
            'upper center',
            'lower center',
            'center'])
    plot_parser.add_argument(
        '--anot-text',
        default=None,
        type=str,
        help='text in annotations')
    plot_parser.add_argument(
        '--inset',
        default=False,
        action='store_true',
        help='flag for inset plot')
    plot_parser.add_argument(
        '--png',
        default=False,
        action='store_true',
        help='flag for saving figure as png')
    plot_parser.add_argument(
        '--fig-dims',
        default=[
            4,
            4],
        type=float,
        nargs=2,
        help='figure dimensions')
    plot_parser.add_argument(
        '--legend',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='flag for legend output')
    plot_parser.add_argument(
        '--extra-files',
        default=None,
        nargs='+',
        help='extra 1D files to plot alongside 2D plots')
    plot_parser.add_argument(
        '--save',
        default=None,
        help='Save the fig with some name')
    plot_parser.add_argument(
        '--kind',
        default='snapshot',
        type=str,
        choices=[
            'snapsoht',
            'movie'],
        help='kind of visual to output')
    plot_parser.add_argument(
        '--cmap',
        default=['viridis'],
        type=str,
        nargs = '+',
        help='matplotlib color map')
    plot_parser.add_argument(
        '--nplots',
        default=1,
        type=int,
        help='number of subplots')
    plot_parser.add_argument(
        '--cbar-range',
        default=[
            (None,
            None)],
        dest='cbar',
        nargs='+',
        type=colorbar_limits,
        help='The colorbar range')
    plot_parser.add_argument(
        '--fill-scale',
        type=float,
        default=None,
        help='Set the y-scale to start plt.fill_between')
    plot_parser.add_argument(
        '--weight', 
        help='plot weighted avg of desired var as function of time', 
        default=None, 
        choices = visual.field_choices + visual.derived,
        type=str
    )
    plot_parser.add_argument(
        '--powerfit',
        help='plot power-law fit on top of histogram',
        default=False,
        action='store_true'
    )
    
    return parser, parser.parse_known_args(
        args=None if sys.argv[2:] else ['plot', '--help'])


def main(
        parser: argparse.ArgumentParser,
        args:   argparse.Namespace,
        *_) -> None:
    parser, (args, _) = parse_plotting_arguments(parser)

    plt.rc('font', size=DEFAULT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=DEFAULT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=DEFAULT_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
    plt.rc('ytick', labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
    plt.rc('legend', fontsize=DEFAULT_SIZE)      # legend fontsize
    plt.rc('figure', titlesize=DEFAULT_SIZE)     # fontsize of the figure title
    if args.tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": DEFAULT_SIZE
            }
        )

    sys.path.insert(1, f'{tool_src}')
    file_list, _ = get_file_list(args.files)
    ndim = get_dimensionality(file_list)

    visual.visualize(parser, ndim)
