#! /usr/bin/env python

import numpy as np
from astropy import units, constants
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import argparse
import os
import sys
import h5py
import time as pytime
from numpy.typing import NDArray
from itertools import cycle
from .helpers import (
    get_tbin_edges,
    get_dL,
    generate_pseudo_mesh,
    read_afterglow_library_data,
    read_simbi_afterglow
)
from simbi import py_calc_fnu, py_log_events, find_nearest
from .scales import get_scale_model
from ..tools import utility as util
from ..detail import get_subparser

try:
    import cmasher as cmr
except ImportError:
    print("cannot find cmasher module, using basic matplotlib colors instead")


def day_type(param: str) -> units.Quantity:
    try:
        param = float(param) * units.day
    except BaseException:
        raise argparse.ArgumentTypeError(
            "time bin edges must be numeric types")

    return param

def deg_type(param: str):
    try:
        param = np.deg2rad(float(param))
    except BaseException:
        raise argparse.ArgumentTypeError(
            "viewing angle most be numerica type")

    return param

def parse_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace
):
    afterglow_parser = get_subparser(parser, 2)
    afterglow_parser.add_argument(
        '--files',
        nargs='+',
        help='Explicit filenames or directory')
    afterglow_parser.add_argument(
        '--theta-obs',
        help='observation angle in degrees',
        type=deg_type,
        default=0.0)
    afterglow_parser.add_argument(
        '--nu',
        help='Observed frequency',
        default=[1e9],
        type=float,
        nargs='+')
    afterglow_parser.add_argument(
        '--save',
        help='file name to save fig',
        default=None,
        type=str)
    afterglow_parser.add_argument(
        '--tex',
        help='true if want latex rendering on figs',
        default=False,
        action='store_true')
    afterglow_parser.add_argument(
        '--ntbins',
        type=int,
        help='number of time bins',
        default=50)
    afterglow_parser.add_argument(
        '--theta-samples',
        type=int,
        help='number of theta_samples',
        default=None)
    afterglow_parser.add_argument(
        '--phi-samples',
        type=int,
        help='number of phi',
        default=10)
    afterglow_parser.add_argument(
        '--example-data',
        type=str,
        help='data file(s) from other afterglow library',
        nargs='+',
        default=[])
    afterglow_parser.add_argument(
        '--load',
        type=str,
        help='data file from simbi-computed light curves',
        default=[],
        nargs='+')
    afterglow_parser.add_argument(
        '--cmap',
        help='colormap scheme for light curves',
        default=None,
        type=str)
    afterglow_parser.add_argument(
        '--clims',
        help='color value limits',
        nargs='+',
        type=float,
        default=[0.25, 0.75])
    afterglow_parser.add_argument(
        '--output',
        help='name of file to be saved as',
        type=str,
        default='some_lc.h5')
    afterglow_parser.add_argument(
        '--example-labels',
        help='label(s) of the example curve\'s markers',
        type=str,
        default=[],
        nargs='+')
    afterglow_parser.add_argument(
        '--xlims',
        help='x limits in plot',
        default=None,
        type=float,
        nargs='+')
    afterglow_parser.add_argument(
        '--ylims',
        help='y limits in plot',
        default=None,
        type=float,
        nargs='+')
    afterglow_parser.add_argument(
        '--fig-dims',
        help='figure dimensions',
        default=(5, 4),
        type=float,
        nargs='+')
    afterglow_parser.add_argument(
        '--title',
        help='title of plot',
        default=None)
    afterglow_parser.add_argument(
        '--spectra',
        help='set if want to plot spectra instead of light curve',
        default=False,
        action='store_true')
    afterglow_parser.add_argument(
        '--times',
        help='discrtete times for spectra calculation',
        default=[1],
        nargs='+',
        type=float)
    afterglow_parser.add_argument(
        '--z',
        help='redshift',
        type=float,
        default=0)
    afterglow_parser.add_argument(
        '--dL',
        help='luminosity distance in [cm]',
        default=1e28,
        type=float)
    afterglow_parser.add_argument(
        '--eps_e',
        help='energy density fraction in electrons',
        default=0.1,
        type=float)
    afterglow_parser.add_argument(
        '--eps_b',
        help='energy density fraction in magnetic field',
        default=0.1,
        type=float)
    afterglow_parser.add_argument(
        '--p',
        help='electron distribution index',
        default=2.5,
        type=float)
    afterglow_parser.add_argument(
        '--mode',
        help='mode to output radiation',
        default='fnu',
        choices=['fnu', 'events'],
        type=str)
    afterglow_parser.add_argument(
        '--scale',
        help='scale string for units',
        default='solar',
        type=str)
    afterglow_parser.add_argument(
        '--no-cut',
        help='flag for not cutting off grid edges',
        default=False,
        action='store_true'
    )
    afterglow_parser.add_argument(
        '--labels',
        help='list of label names for simbi checkpoints',
        default=[],
        type=str,
        nargs='+'
    )
    afterglow_parser.add_argument(
        '--legend-loc',
        help='location of legend',
        default=None,
        type=str,
        choices=[
            'lower left',
            'lower right',
            'lower center',
            'center',
            'upper right',
            'upper left',
            'upper center'
        ]
    )
    afterglow_parser.add_argument(
        '--vline',
        help='location of vertical line in plot',
        default=None,
        type=float
    )
    afterglow_parser.add_argument(
        '--tbins',
        help='time bin edges in units of days',
        default=None,
        nargs=2,
        type=day_type
    )

    return parser, parser.parse_args(
        args=None if sys.argv[2:] else ['afterglow', '--help'])


def run(parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        *_):

    parser, args = parse_args(parser, args)
    args.times.sort()
    args.nu.sort()

    scales = get_scale_model(args.scale)
    if args.tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman"
        })

    
    files = None if not args.files else util.get_file_list(args.files)[0]
    fig, ax = plt.subplots(figsize=args.fig_dims)
    freqs = np.asarray(args.nu) * units.Hz

    if args.cmap is not None:
        vmin, vmax = args.clims
        cinterval = np.linspace(vmin, vmax, len(args.nu))
        cmap = plt.cm.get_cmap(args.cmap)
        colors = util.get_colors(cinterval, cmap, vmin, vmax)
    else:
        cmap = plt.cm.get_cmap('viridis')
        colors = util.get_colors(np.linspace(0, 1, len(args.nu)), cmap)

    # ---------------------------------------------------------
    # Calculations
    # ---------------------------------------------------------
    at_pole = abs(np.cos(args.theta_obs)) == 1
    if args.files:
        dim = util.get_dimensionality(files)
        nbins = args.ntbins
        nbin_edges = nbins + 1
        if args.mode == 'fnu':
            tbin_edge = args.tbins or get_tbin_edges(
                args, files, scales.time_scale)
            tbin_edges = np.geomspace(
                tbin_edge[0] * 0.9,
                tbin_edge[1] * 1.1,
                nbin_edges)
            time_bins = np.sqrt(tbin_edges[1:] * tbin_edges[:-1])
            fnu = {i: np.zeros(nbins) * units.mJy for i in args.nu}
            fnu_contig = np.array(
                [fnu[key].value.flatten() for key in fnu.keys()], dtype=float
            ).flatten()

        scales_dict = {
            'time_scale': scales.time_scale.value,
            'length_scale': scales.length_scale.value,
            'rho_scale': scales.rho_scale.value,
            'pre_scale': scales.pre_scale.value,
            'v_scale': 1.0
        }

        sim_info = {
            'theta_obs': args.theta_obs,
            'nus': freqs.value,
            'z': args.z,
            'd_L': get_dL(args.z).value,
            'eps_e': args.eps_e,
            'eps_b': args.eps_b,
            'p': args.p,
        }

        for idx, file in enumerate(files):
            fields, setup, mesh = util.read_file(args, file, dim)
            # Generate a pseudo mesh if computing off-axis afterglows
            generate_pseudo_mesh(
                args,
                mesh,
                full_sphere=True,
                full_threed=False,
            )
            sim_info['dt'] = setup['dt']
            sim_info['adiabatic_gamma'] = setup['ad_gamma']
            sim_info['current_time'] = setup['time']
            t1 = pytime.time()
            if args.mode == 'fnu':
                py_calc_fnu(
                    fields=fields,
                    tbin_edges=tbin_edges.value,
                    flux_array=fnu_contig,
                    mesh=mesh,
                    qscales=scales_dict,
                    sim_info=sim_info,
                    chkpt_idx=idx,
                    data_dim=dim
                )
            else:
                py_log_events(
                    fields=fields,
                    photon_distro=photon_distro,
                    x_mu=x_mu,
                    mesh=mesh,
                    qscales=scales_dict,
                    sim_info=sim_info,
                    data_dim=dim,
                )
            print(
                f"Processed file {file} in {pytime.time() - t1:.2f} s",
                flush=True)

        fnu_contig = fnu_contig.reshape(len(args.nu), nbins) * (1.0 + args.z)
        for idx, key in enumerate(fnu.keys()):
            fnu[key] = fnu_contig[idx] * units.mJy

        # Save the data
        file_name = args.output
        if not file_name.endswith('.h5'):
            file_name += '.h5'

        isFile = os.path.isfile(file_name)
        dirname = os.path.dirname(file_name)
        if os.path.exists(dirname) == False and dirname != '':
            if not isFile:
                # Create a new directory because it does not exist
                os.makedirs(dirname)
                print(80 * '=')
                print(f"creating new directory named {dirname}...")

        print(80 * "=")
        print(f"Saving file as {file_name}...")
        print(80 * '=')
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('nu', data=[nu for nu in args.nu])
            hf.create_dataset('fnu', data=fnu_contig)
            hf.create_dataset('tbins', data=time_bins)

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    lines = ["--", "-.", ":", "-"]
    linecycler = cycle(lines)
    color_cycle = cycle(colors)
    sim_lines = []
    relevant_var = args.times if args.spectra else args.nu
    legend_labels = []
    for idx, val in enumerate(relevant_var):
        color = next(color_cycle)
        power_of_ten = int(np.floor(np.log10(val)))
        front_part = val / 10**power_of_ten
        var_label = r'$t=' if args.spectra else r'$\nu='
        if front_part == 1.0:
            label = f'{var_label}' + \
                r'10^{%d}$' % (power_of_ten) + \
                (' day' if args.spectra else ' Hz')
        else:
            label = f'{var_label}' + r'%d \times 10^{%d}$' % (
                int(front_part), power_of_ten) + (' day' if args.spectra else ' Hz')

        if files:
            if args.spectra:
                nearest_dat = find_nearest(time_bins.value, val)[0]
                relevant_flux = np.asanyarray(
                    [fnu[key][nearest_dat].value for key in fnu.keys()])
                xarr = args.nu
            else:
                relevant_flux = fnu[val]
                xarr = time_bins

            ax.plot(xarr, relevant_flux, color=color, label=label)

        for dat in args.example_data:
            example_data = read_afterglow_library_data(dat)
            if args.spectra:
                key = find_nearest(example_data['tday'].value, val)[
                    1] * units.day
                x = example_data['freq'] * units.Hz
                y = example_data['spectra'][key]
                m = 5.0
            else:
                key = val * units.Hz
                x = example_data['tday']
                y = example_data['fnu'][key]
                m = 0.5
            ax.plot(x, y, 'o', color=color, markersize=m)

        for dfile in args.load:
            linestyle = next(linecycler)
            dat = read_simbi_afterglow(dfile)
            if args.spectra:
                nearest_day = find_nearest(dat['tday'].value, val)[0]
                x = sorted(dat['freq'].value)
                y = np.array(
                    [dat['fnu'][key][nearest_day].value for key in sorted(dat['fnu'].keys())])
            else:
                x = dat['tday']
                y = dat['fnu'][val * units.Hz]

            ax.plot(x, y, color=color, label=label, linestyle=linestyle)

        if label not in legend_labels:
            # create dummy axes for color-coded legend labels
            line, = ax.plot([], [], color=color, label=label)
            sim_lines += [line]
            legend_labels += [label]

        linecycler = cycle(lines)

    if args.title:
        ax.set_title(rf'{args.title}')

    ylims = args.ylims if args.ylims else (1e-11, 1e4)
    ax.set_ylim(*ylims)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'Flux  Density [mJy]')
    if args.xlims:
        ax.set_xlim(*args.xlims)

    if args.spectra:
        ax.set_xlabel(r'$\nu_{\rm obs} [\rm Hz]$')
    else:
        ax.set_xlabel(r'$t_{\rm obs} [\rm day]$')

    if args.vline:
        ax.axvline(args.vline, color='black', linestyle=':', alpha=0.8)

    ex_lines = []
    marks = cycle(['o', 's'])
    for label in args.example_labels:
        ex_lines += [
            mlines.Line2D([0], [0],
                          marker=next(marks),
                          color='w',
                          label=label,
                          markerfacecolor='grey',
                          markersize=5)
        ]

    for label in args.labels:
        ex_lines += [
            mlines.Line2D([0, 1], [0, 1],
                          linestyle=next(linecycler),
                          label=label,
                          color='grey')
        ]

    ax.legend(handles=[*sim_lines, *ex_lines], loc=args.legend_loc)
    if args.save:
        file_str = f"{args.save}".replace(' ', '_')
        print(f'saving as {file_str}.pdf')
        fig.savefig(f'{file_str}.pdf')
        plt.show()
    else:
        plt.show()
