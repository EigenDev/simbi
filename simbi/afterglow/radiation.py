#! /usr/bin/env python

import numpy as np
import astropy.constants as const
import astropy.units as units
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import argparse
import os
import sys
import h5py
import time as pytime
from itertools import cycle
from .helpers import get_tbin_edges, get_dL, Scale, generate_pseudo_mesh, read_afterglow_library_data, read_simbi_afterglow
from simbi import py_calc_fnu, py_log_events
from simbi.tools import utility as util
from simbi._detail import get_subparser

try:
    import cmasher as cmr
except ImportError:
    print("cannot find cmasher module, using basic matplotlib colors insteads")


def parse_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace
):
    afterglow_parser = get_subparser(parser, 2)
    afterglow_parser.add_argument(
        'files',
        nargs='+',
        help='Explicit filenames or directory')
    afterglow_parser.add_argument(
        '--theta-obs',
        help='observation angle in degrees',
        type=float,
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
        default=None)
    afterglow_parser.add_argument(
        '--extra-files',
        type=str,
        help='data file from self computed light curves',
        default=None,
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
        '--dfile_out',
        dest='dfile_out',
        help='name of file to be saved as',
        type=str,
        default='some_lc.h5')
    afterglow_parser.add_argument(
        '--example-labels',
        help='label(s) of the example curve\'s markers',
        type=str,
        default=['example'],
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
        choices=['fnu', 'events', 'checkpoint'],
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

    return parser, parser.parse_args(
        args=None if sys.argv[2:] else ['afterglow', '--help'])


def run(parser: argparse.ArgumentParser = None,
        args: argparse.Namespace = None,
        *_):
    parser, args = parse_args(parser, args)

    scales = Scale(args.scale)
    if args.tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman"
        })

    files, _ = util.get_file_list(args.files)
    fig, ax = plt.subplots(figsize=args.fig_dims)
    freqs = np.array(args.nu) * units.Hz

    if args.cmap is not None:
        vmin, vmax = args.clims
        cinterval = np.linspace(vmin, vmax, len(args.nu))
        cmap = plt.cm.get_cmap(args.cmap)
        colors = util.get_colors(cinterval, cmap, vmin, vmax)
    else:
        colors = ['c', 'y', 'm', 'k']  # list of basic colors

    linestyles = ['-', '--', '-.', ':']  # list of basic linestyles
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    dim = util.get_dimensionality(files)
    if args.mode != 'checkpoint':
        nbins = args.ntbins
        nbin_edges = nbins + 1
        tbin_edge = get_tbin_edges(args, files, scales.time_scale)
        tbin_edges = np.geomspace(
            tbin_edge[0] * 0.9,
            tbin_edge[1] * 1.1,
            nbin_edges)
        time_bins = np.sqrt(tbin_edges[1:] * tbin_edges[:-1])
        fnu = {i: np.zeros(nbins) * units.mJy for i in args.nu}
        events_list = np.zeros(shape=(len(files), 2))
        storage = {}
        scales_dict = {
            'time_scale': scales.time_scale.value,
            'length_scale': scales.length_scale.value,
            'rho_scale': scales.rho_scale.value,
            'pre_scale': scales.pre_scale.value,
            'v_scale': 1.0
        }
        theta_obs = np.deg2rad(args.theta_obs)
        dL = get_dL(args.z)
        sim_info = {
            'theta_obs': theta_obs,
            'nus': freqs.value,
            'z': args.z,
            'd_L': dL.value,
            'eps_e': args.eps_e,
            'eps_b': args.eps_b,
            'p': args.p,
        }

        if args.mode == 'fnu':
            for idx, file in enumerate(files):
                fields, setup, mesh = util.read_file(args, file, dim)
                # Generate a pseudo mesh if computing off-axis afterglows
                generate_pseudo_mesh(
                    args,
                    mesh,
                    full_sphere=True,
                    full_threed=not args.theta_obs == 0)
                sim_info['dt'] = setup['dt']
                sim_info['adiabatic_gamma'] = setup['ad_gamma']
                sim_info['current_time'] = setup['time']
                t1 = pytime.time()
                py_calc_fnu(
                    fields=fields,
                    tbin_edges=tbin_edges.value,
                    flux_array=fnu,
                    mesh=mesh,
                    qscales=scales_dict,
                    sim_info=sim_info,
                    chkpt_idx=idx,
                    data_dim=dim
                )
                print(
                    f"Processed file {file} in {pytime.time() - t1:.2f} s",
                    flush=True)
        elif args.mode == 'events':
            for idx, file in enumerate(files):
                fields, setup, mesh = util.read_file(args, file, dim)
                # Generate a pseudo mesh if computing off-axis afterglows
                generate_pseudo_mesh(
                    args,
                    mesh,
                    full_sphere=True,
                    full_threed=not args.theta_obs == 0)
                photon_distro = np.zeros(shape=(mesh['phi']))
                sim_info['dt'] = setup['dt']
                sim_info['adiabatic_gamma'] = setup['ad_gamma']
                sim_info['current_time'] = setup['time']
                t1 = pytime.time()
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
                print(x_mu)
                zzz = input('')

        for key in fnu.keys():
            fnu[key] *= (1 + args.z)

        # Save the data
        if args.mode != 'checkpoint':
            file_name = args.dfile_out
            if os.path.splitext(file_name)[1] != '.h5':
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
                fnu_save = np.array([fnu[key] for key in fnu.keys()])
                dset = hf.create_dataset('sogbo_data', dtype='f')
                hf.create_dataset('nu', data=[nu for nu in args.nu])
                hf.create_dataset('fnu', data=fnu_save)
                hf.create_dataset('tbins', data=time_bins)

    color_cycle = cycle(colors)
    if args.spectra:
        sim_lines = [0] * len(args.times)
        for tidx, time in enumerate(args.times):
            see_day_idx = util.find_nearest(time_bins.value, time)[0]

            power_of_ten = int(np.floor(np.log10(time)))
            front_part = time / 10**power_of_ten
            if front_part == 1.0:
                time_label = r'10^{%d}' % (power_of_ten)
            else:
                time_label = r'%.1f \times 10^{%d}' % (
                    front_part, power_of_ten)

            color = next(color_cycle)
            spectra = np.asanyarray(
                [fnu[key][see_day_idx].value for key in fnu.keys()])
            sim_lines[tidx], = ax.plot(
                args.nu, spectra, color=color, label=r'$t={} \rm day$'.format(time_label))

            if args.example_data is not None:
                example_data = read_afterglow_library_data(args.example_data)
                nearest_day = util.find_nearest(
                    example_data['tday'].value, time)[1] * units.day
                ax.plot(
                    example_data['freq'],
                    example_data['spectra'][nearest_day],
                    'o',
                    color=color,
                    markersize=0.5)

            if args.extra_files is not None:
                for dfile in args.extra_files:
                    dat = read_simbi_afterglow(dfile)
                    nearest_day = util.find_nearest(dat['tday'].value, time)[0]
                    spectra = np.asanyarray(
                        [dat['fnu'][key][nearest_day].value for key in dat['fnu'].keys()])
                    ax.plot(dat['freq'], spectra, color=color, markersize=0.5)
    else:
        sim_lines = [0] * len(args.nu)
        for nidx, freq in enumerate(args.nu):
            power_of_ten = int(np.floor(np.log10(freq)))
            front_part = freq / 10**power_of_ten
            if front_part == 1.0:
                freq_label = r'10^{%d}' % (power_of_ten)
            else:
                freq_label = r'%f \times 10^{%fid}' % (
                    front_part, power_of_ten)

            color = next(color_cycle)
            if args.mode != 'checkpoint':
                sim_lines[nidx], = ax.plot(
                    time_bins, fnu[freq], color=color, label=r'$\nu={} \rm Hz$'.format(freq_label))

            if args.example_data is not None:
                marks = cycle(['o', 's'])
                for file in args.example_data:
                    example_data = read_afterglow_library_data(file)
                    nu_unit = freq * units.Hz
                    ax.plot(
                        example_data['tday'],
                        example_data['fnu'][nu_unit],
                        next(marks),
                        color=color,
                        markersize=1)

            if args.extra_files is not None:
                for dfile in args.extra_files:
                    dat = read_simbi_afterglow(dfile)
                    nu_unit = freq * units.Hz
                    doot, = ax.plot(
                        dat['tday'], dat['fnu'][nu_unit], color=color, label=r'$\nu={} \rm Hz$'.format(freq_label))
                    if not args.mode != 'checkpoint':
                        sim_lines[nidx] = doot
    if args.xlims is not None:
        tbound1, tbound2 = np.asanyarray(args.xlims) * units.day
    else:
        tbound1 = time_bins[0] if 'time_bins' in locals() else dat['tday'][0]
        tbound2 = time_bins[-1] if 'time_bins' in locals() else dat['tday'][-1]

    if args.title is not None:
        if dim == 1:
            ax.set_title(
                r'$ \rm Light \  curve \ for \ spherical \ BMK \ Test$')
        else:
            ax.set_title(r'$ \rm Light \ curve \ for \ conical \ BMK \ Test$')

    ylims = args.ylims if args.ylims else (1e-11, 1e4)
    ax.set_ylim(*ylims)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'$\rm Flux \ Density \ [\rm mJy]$')
    if args.spectra:
        if args.xlims is not None:
            ax.set_xlim(*args.xlims)
        ax.set_xlabel(r'$\nu_{\rm obs} [\rm Hz]$')
    else:
        ax.set_xlim(tbound1.value, tbound2.value)
        ax.set_xlabel(r'$t_{\rm obs} [\rm day]$')

    ex_lines = []
    if args.example_data is not None:
        marks = cycle(['o', 's'])
        for label in args.example_labels:
            ex_lines += [mlines.Line2D([0],
                                       [0],
                                       marker=next(marks),
                                       color='w',
                                       label=label,
                                       markerfacecolor='grey',
                                       markersize=5)]

        ax.legend(handles=[*sim_lines, *ex_lines])
        # ax.axvline(3.5, linestyle='--', color='red')
    else:
        ax.legend()

    if args.save:
        file_str = f"{args.save}".replace(' ', '_')
        print(f'saving as {file_str}.pdf')
        fig.savefig(f'{file_str}.pdf')
        plt.show()
    else:
        plt.show()
