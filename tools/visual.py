#! /usr/bin/env python
import argparse 
import sys
import matplotlib.pyplot as plt 
import importlib
from utility import DEFAULT_SIZE, SMALL_SIZE, get_dimensionality, get_file_list

derived = ['D', 'momentum', 'energy', 'energy_rst', 'enthalpy', 'temperature', 'T_eV', 'mass', 'chi_dens',
          'mach', 'u1', 'u2']
field_choices = ['rho', 'v1', 'v2', 'v3', 'v', 'p', 'gamma_beta', 'chi'] + derived
lin_fields    = ['chi', 'gamma_beta', 'u1', 'u2', 'u3']

i = 0
def parse_arguments(cli_args: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visual simulations from SIMBI run')
    parser.add_argument('files',  nargs='+', help='A Data Source to Be Plotted')
    parser.add_argument('setup',  type=str, help='The name of the setup you are plotting (e.g., Blandford McKee)')
    parser.add_argument('--fields',     default=['rho'], nargs='+', help='The name of the field variable you\'d like to plot',choices=field_choices)
    parser.add_argument('--xmax',       default = 0.0, help='The domain range')
    parser.add_argument('--log',        default=False, action='store_true',  help='Logarithmic Radial Grid Option')
    parser.add_argument('--kinetic',    default=False, action='store_true',  help='Plot the kinetic energy on the histogram')
    parser.add_argument('--enthalpy',   default=False, action='store_true',  help='Plot the enthalpy on the histogram')
    parser.add_argument('--hist',       default=False, action='store_true',  help='Convert plot to histogram')
    parser.add_argument('--mass',       default=False, action='store_true',  help='Compute mass histogram')
    parser.add_argument('--dm_du',      default = False, action='store_true', help='Compute dM/dU over whole domain')
    parser.add_argument('--ax_anchor',  default=None, type=str, nargs='+', help='Anchor annotation text for each plot')
    parser.add_argument('--norm',       default=False, action='store_true', help='True if you want the plot normalized to max value')
    parser.add_argument('--labels',     default = None, nargs='+', help='Optionally give a list of labels for multi-file plotting')
    parser.add_argument('--xlims',      default = None, type=float, nargs=2)
    parser.add_argument('--ylims',      default = None, type=float, nargs=2)
    parser.add_argument('--units',      default = False, action='store_true')
    parser.add_argument('--power',      default = 1.0, type=float, help='exponent of power-law norm')
    parser.add_argument('--dbg',        default = False, action='store_true')
    parser.add_argument('--tex',        default = False, action='store_true')
    parser.add_argument('--print',      default = False, action='store_true')
    parser.add_argument('--pictorial',  default = False, action='store_true')
    parser.add_argument('--anot_loc',   default = None, type=str)
    parser.add_argument('--legend_loc', default = None, type=str)
    parser.add_argument('--anot_text',  default = None, type=str)
    parser.add_argument('--inset',      default=False, action= 'store_true')
    parser.add_argument('--png',        default=False, action= 'store_true')
    parser.add_argument('--fig_dims',   default = [4, 4], type=float, nargs=2)
    parser.add_argument('--legend',     default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--extra_files',default=None, nargs='+', help='extra 1D files to plot alongside 2D plots')
    parser.add_argument('--save',       default=None, help='Save the fig with some name')
    parser.add_argument('--kind',       default='snapshot', type=str, choices=['snapsoht', 'movie'])
    parser.add_argument('--cmap',       default='viridis', type=str, help='matplotlib color map')
    parser.add_argument('--nplots',     default=1, type=int, help='number of subplots')
    parser.add_argument('--cbar_range', default = [None, None], dest = 'cbar', nargs=2, help='The colorbar range')
    parser.add_argument('--fill_scale', type=float, default = None, help='Set the y-scale to start plt.fill_between')
    return parser, parser.parse_known_args(args=None if sys.argv[1:] else ['--help'])

def main():
    parser, (args, argv) = parse_arguments()
    if args.tex:
        plt.rc('font',   size=DEFAULT_SIZE)          # controls default text sizes
        plt.rc('axes',   titlesize=DEFAULT_SIZE)     # fontsize of the axes title
        plt.rc('axes',   labelsize=DEFAULT_SIZE)     # fontsize of the x and y labels
        plt.rc('xtick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('ytick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('legend', fontsize=DEFAULT_SIZE)      # legend fontsize
        plt.rc('figure', titlesize=DEFAULT_SIZE)     # fontsize of the figure title
        
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": DEFAULT_SIZE
            }
        )
    file_list, _ = get_file_list(args.files)
    ndim = get_dimensionality(file_list)
    visual_module = getattr(importlib.import_module(f'{args.kind}{ndim}d'), f'{args.kind}')
    visual_module(parser)
        


if __name__ == '__main__':
    sys.exit(main())