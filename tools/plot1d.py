#! /usr/bin/env python

# Read in a File and Plot it

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import utility as util 
import time
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const
import astropy.units as u 
import os
import utility as util
from datetime import datetime
from itertools import cycle

try:
    import cmasher as cmr 
except:
    pass 

derived       = ['D', 'momentum', 'energy', 'energy_rst', 'enthalpy', 'temperature', 'mass', 'mach']
field_choices = ['rho', 'c', 'p', 'gamma_beta', 'chi'] + derived
lin_fields    = ['chi', 'gamma_beta']
def plot_profile(args, fields, mesh, setup, ax = None, overplot = False, subplot = False, case = 0):
    ncols = len(args.filename) * len(args.fields)
    vmin, vmax = args.clims 
    cinterval  = np.linspace(vmin, vmax, ncols)
    cmap       = plt.cm.get_cmap(args.cmap)
    colors     = util.get_colors(cinterval, cmap, vmin, vmax)
    ccycler    = cycle(colors)
    linestyles = ['--', '-', '-.']
    linecycler = cycle(linestyles)
    
    r      = mesh['r']
    tend   = setup['time']
    if args.units:
        tend *= util.time_scale 
    
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,8))
    
    field_labels = util.get_field_str(args)
    for idx, field in enumerate(args.fields):
        unit_scale = 1.0
        if args.units:
            if field == 'rho' or field == 'D':
                unit_scale = util.rho_scale
            elif field == 'p' or field == 'energy':
                unit_scale = util.pre_scale
            
        if field in derived:
            var = util.prims2var(fields, field)
        else:
            var = fields[field]
        
        if args.labels:
            label = r'$\rm {}, t={:.1f}$'.format(args.labels[case], tend)
        else:
            label = r'$t-{:.1f}$'.format(tend)
            
        if len(args.fields) > 1:
            label = field_labels[idx] + ' ' + label
            
        ax.plot(r, var * unit_scale, color=colors[case], label=label, linestyle=next(linecycler))
        if case == 0:
            if args.fields[0] == 'gamma_beta':
                max_idx = np.argmax(var)
                ax.plot(r[max_idx:], var[max_idx] * (r[max_idx:] / r[max_idx]) ** (-3/2), label='$\propto r^{-3/2}$', color='black', linestyle='--')

    ax.tick_params(axis='both')
    if args.log:
        ax.set_xscale('log')
        if args.fields[0] not in lin_fields:
            ax.set_yscale('log')
    elif not setup['linspace']:
        ax.set_xscale('log')
    
    ax.set_xlabel('$r$')
    if args.xlims is None:
        ax.set_xlim(r.min(), r.max()) if args.rmax == 0.0 else ax.set_xlim(r.min(), args.rmax)
    else:
        ax.set_xlim(*args.xlims)
    # Change the format of the field
    field_str = util.get_field_str(args)
    
    if (len(args.fields) == 1):
        ax.set_ylabel('{}'.format(field_str))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.axvline(0.60, color='black', linestyle='--')
    
    if args.legend:
        ax.legend()
        
    ########
    # Personal Calculations
    # TODO: Remove Later
    ########
    # r_outer = find_nearest(r, 0.55)[0]
    # r_slow  = find_nearest(r, 1.50)[0]
    # dV      = util.calc_cell_volume1D(mesh['r']) 
    # mout    = (4./3.) * np.pi * np.sum(dV[r_outer:r_slow] * fields['rho'][r_outer: r_slow])
    # print(mout)
    # zzz = input('')
    ########################
    
    
    if not subplot:   
        ax.set_title('{}'.format(args.setup[0]))
    if not overplot:
        ax.set_title('{} at t = {:.3f}'.format(args.setup[0], tend))
        return fig
    
def plot_hist(args, fields, mesh, setup, overplot=False, ax=None, subplot = False, case=0):
    colors = plt.cm.twilight_shifted(np.linspace(0.25, 0.75, len(args.filename)))
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend        = setup['time']
    edens_total = util.prims2var(fields, 'energy')
    r           = mesh['r']
    dV          = util.calc_cell_volume1D(r)
    
    if args.eks:
        mass   = dV * fields['W']**2 * fields['rho']
        energy = (fields['W'] - 1.0) * mass * util.e_scale.value
    elif args.hhist:
        energy = (fields['enthalpy'] - 1.0) *  dV * util.e_scale.value
    else:
        energy = edens_total * dV * util.e_scale.value


    u = fields['gamma_beta']
    gbs = np.logspace(np.log10(1.e-4), np.log10(u.max()), 128)
    
    energy = np.asarray([energy[u > gb].sum() for gb in gbs])
    # E_seg_rat  = energy[1:]/energy[:-1]
    # gb_seg_rat = gbs[1:]/gbs[:-1]
    # E_seg_rat[E_seg_rat == 0] = 1
    
    # slope = (energy[1:] - energy[:-1])/(gbs[1:] - gbs[:-1])
    # power_law_region = np.argmin(slope)
    # up_min           = find_nearest(gbs, 2 * gbs[power_law_region: ][0])[0]
    # upower           = gbs[up_min: ]
    
    # # Fix the power law segment, ignoring the sharp dip at the tail of the CDF
    # epower_law_seg   = E_seg_rat [up_min: np.argmin(E_seg_rat > 0.8)]
    # gbpower_law_seg  = gb_seg_rat[up_min: np.argmin(E_seg_rat > 0.8)]
    # segments         = np.log10(epower_law_seg) / np.log10(gbpower_law_seg)
    # alpha            = 1.0 - np.mean(segments)
    # E_0 = energy[up_min] * upower[0] ** (alpha - 1)
    # print('Avg power law index: {:.2f}'.format(alpha))
    if args.labels is None:
        hist = ax.hist(gbs, bins=gbs, weights=energy, label= r'$E_T$', histtype='step', color=colors[case], rwidth=1.0, linewidth=3.0)
        # ax.plot(upower, E_0 * upower**(-(alpha - 1)), '--')
    else:
        hist = ax.hist(gbs, bins=gbs, weights=energy, label=r'${}$, t={:.2f}'.format(args.labels[case], tend), color=colors[case], histtype='step', rwidth=1.0, linewidth=3.0)
        # ax.plot(upower, E_0 * upower**(-(alpha - 1)), '--', label = r'${}$ fit'.format(args.labels[case]))


    sorted_energy = np.sort(energy)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(sorted_energy[1], 1.5*ets.max())
    ax.set_xlabel(r'$\Gamma\beta $')
    if args.eks:
        ax.set_ylabel(r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$')
    elif args.hhist:
        ax.set_ylabel(r'$H ( > \Gamma \beta) \ [\rm{erg}]$')
    else:
        ax.set_ylabel(r'$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$')
    ax.tick_params('both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    

    if args.fill_scale is not None:
        fill_below_intersec(gbs, energy, args.fill_scale*energy.max(), colors[case])
    if not subplot:
        ax.set_title(r'setup: {}'.format(args.setup[0]))
        
    if not overplot:
        ax.set_title(r'setup: {}, t ={:.2f}'.format(args.setup[0], tend))
        return fig

def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog='This Only Supports H5 Files Right Now')
    
    parser.add_argument('filename', metavar='Filename', nargs='+', help='A Data Source to Be Plotted')
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str, help='The name of the setup you are plotting (e.g., Blandford McKee)')
    parser.add_argument('--fields', dest = 'fields', metavar='Field Variable(s)', nargs='+', help='The name of the field variable(s) you\'d like to plot', choices=field_choices, default='rho')
    parser.add_argument('--rmax', dest = 'rmax', metavar='Radial Domain Max', default = 0.0, help='The domain range')
    parser.add_argument('--xlims', dest = 'xlims', metavar='Domain',default = None, help='The domain range', nargs='+', type=float)
    parser.add_argument('--fill_scale', dest = 'fill_scale', metavar='Filler maximum', type=float, default = None, help='Set the y-scale to start plt.fill_between')
    parser.add_argument('--log', dest='log', action='store_true', default=False, help='Logarithmic Radial Grid Option')
    parser.add_argument('--ehist', dest='ehist', action='store_true',default=False, help='Plot the energy_vs_gb histogram')
    parser.add_argument('--eks', dest='eks', action='store_true', default=False,help='Plot the kinetic energy on the histogram')
    parser.add_argument('--hhist', dest='hhist', action='store_true',default=False,help='Plot the enthalpy on the histogram')
    parser.add_argument('--labels', dest='labels', nargs='+',help='map labels to filenames')
    parser.add_argument('--save', dest='save', default=None, help='If you want save the fig')
    parser.add_argument('--first_order', dest='forder', action='store_true', default=False, help='True if this is a grid using RK1')
    parser.add_argument('--plots', dest='plots', type = int, default=1,help=r'Number of subplots you\'d like')
    parser.add_argument('--units', dest='units', action='store_true', default=False,help='True if you would like units scale (default is solar units)')
    parser.add_argument('--tex', dest='tex', default=False, action='store_true', help='set if want Latex typesetting')
    parser.add_argument('--fig_size', dest='fig_size', default=(4,3.5), type=float, help='size of figure', nargs=2)
    parser.add_argument('--cmap', dest='cmap', default='viridis', type=str, help='matplotlib color map')
    parser.add_argument('--clims', dest='clims', default=[0, 1], type=float, nargs='+', help='color limits')
    parser.add_argument('--legend', dest='legend', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    if args.tex:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Time New Roman"]})
    
    fig_size = args.fig_size
    if len(args.filename) > 1:
        if args.plots == 1:
            fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
            ax = fig.add_subplot(1, 1, 1)
            for idx, file in enumerate(args.filename):
                fields, setup, mesh = util.read_1d_file(file)
                if args.ehist or args.eks or args.hhist:
                    plot_hist(args, fields, mesh, setup, ax = ax, overplot= True, case = idx)
                else:
                    plot_profile(args, fields, mesh, setup, ax = ax, overplot=True, case = idx)
        else:
            fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            for idx, file in enumerate(args.filename):
                fields, setup, mesh = util.read_1d_file(file)
                plot_hist(args, fields,mesh, setup,  ax = ax1, overplot= True, subplot = True, case = idx)
                plot_profile(args, fields, mesh, setup, ax = ax2, overplot=True, subplot = True, case = idx)
                
            fig.suptitle('{}'.format(args.setup[0]))
            
        if args.labels != None:
            ax.legend()
            
    else:
        fields, setup, mesh = util.read_1d_file(args.filename[0])
        if args.ehist or args.hhist or args.eks:
            fig = plot_hist(args, fields, mesh, setup)
        else:
            fig = plot_profile(args, fields, mesh, setup)
        
    
    
    if args.save is not None:
        fig.savefig('{}.pdf'.format(args.save), bbox_inches='tight')
    else:
        plt.show()
    
if __name__ == '__main__':
    main()
