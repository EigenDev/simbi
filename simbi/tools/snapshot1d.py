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
from utility import DEFAULT_SIZE, SMALL_SIZE
from visual import lin_fields, derived
try:
    import cmasher as cmr 
except:
    pass 

def plot_profile(args, fields, mesh, setup, ncols: int, ax = None, overplot = False, subplot = False, case = 0):
    vmin = args.cbar[0] or 0.0 
    vmax = args.cbar[1] or 1.0
    cinterval  = np.linspace(vmin, vmax, ncols)
    cmap       = plt.cm.get_cmap(args.cmap)
    colors     = util.get_colors(cinterval, cmap, vmin, vmax)
    ccycler    = cycle(colors)
    linestyles = ['--', '-', '-.']
    linecycler = cycle(linestyles)
    
    r      = mesh['x1']
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
        
        if 'v' in field:
            field = 'v1'
            
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
        # if case == 0:
        #     if args.fields[0] == 'gamma_beta':
        #         max_idx = np.argmax(var)
        #         ax.plot(r[max_idx:], var[max_idx] * (r[max_idx:] / r[max_idx]) ** (-3/2), label='$\propto r^{-3/2}$', color='black', linestyle='--')

    ax.tick_params(axis='both')
    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
        # if args.fields[0] not in lin_fields:
        #     ax.set_yscale('log')
    elif not setup['linspace']:
        ax.set_xscale('log')
    
    ax.set_xlabel('$r$')
    if args.xlims is None:
        ax.set_xlim(r.min(), r.max()) if args.xmax == 0.0 else ax.set_xlim(r.min(), args.xmax)
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
    # dV      = util.calc_cell_volume1D(mesh['x1']) 
    # mout    = (4./3.) * np.pi * np.sum(dV[r_outer:r_slow] * fields['rho'][r_outer: r_slow])
    # print(mout)
    # zzz = input('')
    ########################
    
    
    if not subplot:   
        ax.set_title('{}'.format(args.setup))
    if not overplot:
        ax.set_title('{} at t = {:.3f}'.format(args.setup, tend))
        return fig
    
def plot_hist(args, fields, mesh, setup, overplot=False, ax=None, subplot = False, case=0):
    colors = plt.cm.twilight_shifted(np.linspace(0.25, 0.75, len(args.files)))
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend        = setup['time']
    edens_total = util.prims2var(fields, 'energy')
    r           = mesh['x1']
    dV          = util.calc_cell_volume1D(r)
    
    if args.kinetic:
        mass   = dV * fields['W']**2 * fields['rho']
        energy = (fields['W'] - 1.0) * mass * util.e_scale.value
    elif args.enthalpy:
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
    if args.xlims:
        ax.set_xlim(*args.xlims)
    # ax.set_ylim(sorted_energy[1], 1.5*ets.max())
    ax.set_xlabel(r'$\Gamma\beta $')
    if args.kinetic:
        ax.set_ylabel(r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$')
    elif args.enthalpy:
        ax.set_ylabel(r'$H ( > \Gamma \beta) \ [\rm{erg}]$')
    else:
        ax.set_ylabel(r'$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$')
    ax.tick_params('both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    

    if args.fill_scale is not None:
        fill_below_intersec(gbs, energy, args.fill_scale*energy.max(), colors[case])
    if not subplot:
        ax.set_title(r'setup: {}'.format(args.setup))
        
    if not overplot:
        ax.set_title(r'setup: {}, t ={:.2f}'.format(args.setup, tend))
        return fig

def snapshot(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    
    fig_size = args.fig_dims
    flist, _ = util.get_file_list(args.files)
    ncols    = len(flist) * len(args.fields)
    if len(flist) > 1:
        if args.nplots == 1:
            fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
            ax = fig.add_subplot(1, 1, 1)
            for idx, file in enumerate(flist):
                fields, setup, mesh = util.read_file(args, file, ndim=1)
                if args.hist:
                    plot_hist(args, fields, mesh, setup, ax = ax, overplot= True, case = idx)
                else:
                    plot_profile(args, fields, mesh, setup,ncols,  ax = ax, overplot=True, case = idx)
        else:
            fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            for idx, file in enumerate(flist):
                fields, setup, mesh = util.read_file(args, file, ndim=1)
                plot_hist(args, fields,mesh, setup,  ax = ax1, overplot= True, subplot = True, case = idx)
                plot_profile(args, fields, mesh, setup, ncols, ax = ax2, overplot=True, subplot = True, case = idx)
                
            fig.suptitle('{}'.format(args.setup))
            
        if args.labels != None:
            ax.legend()
            
    else:
        fields, setup, mesh = util.read_file(args, flist[0], ndim=1)
        if args.hist:
            fig = plot_hist(args, fields, mesh, setup)
        else:
            fig = plot_profile(args, fields, mesh, setup, ncols)
        
    
    
    if args.save is not None:
        fig.savefig('{}.pdf'.format(args.save), bbox_inches='tight')
    else:
        plt.show()
