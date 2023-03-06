# Tool for making movies from h5 files in directory

import numpy as np 
import matplotlib.pyplot as plt #! /usr/bin/env python
import matplotlib.ticker as tkr
import utility as util
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const
import os
from visual import derived, lin_fields
from ..detail import get_subparser
try:
    import cmasher as cmr 
except ImportError:
    pass 

from cycler import cycler
from utility import DEFAULT_SIZE, SMALL_SIZE, BIGGER_SIZE
from matplotlib.animation import FuncAnimation

def plot_profile(fig, ax, filename, args):
    fields, setup, mesh = util.read_file(args, filename, ndim=1)
    x1, t = mesh['x1'], setup['time']
    if args.units:
        t *= util.time_scale 
    
    field_labels = util.get_field_str(args)
    label = None
    for idx, field in enumerate(args.fields):
        unit_scale = 1.0
        if args.units:
            if field == 'rho' or field == 'D':
                unit_scale = util.rho_scale
            elif field == 'p' or field == 'energy':
                unit_scale = util.edens_scale
        
        if field == 'v':
            field = 'v1'
            
        if field in derived:
            var = util.prims2var(fields, field)
        else:
            var = fields[field]

        if args.scale_down:
            var /= args.scale_down[idx]
            
        if len(args.fields) > 1:
            label = field_labels[idx]
        
        ax.plot(x1, var * unit_scale, label=label)

    
    if args.xlims is None:
        ax.set_xlim(x1.min(), x1.max()) if args.xmax == 0.0 else ax.set_xlim(r.min(), args.xmax)
    else:
        ax.set_xlim(*args.xlims)
    
    if args.ylims:
        ax.set_ylim(*args.ylims)
           
    if args.legend:
        if (h := ax.get_legend_handles_labels()[0]): 
            ax.legend()
    ax.set_title('{} at t = {:.2f}'.format(args.setup, t))

def plot_hist(args, fields, overplot=False, ax=None, case=0):
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend = fields['t']
    edens_total = util.prims2var(fields, 'energy')
    r           = fields['x1']
    
    if fields['is_linspace']:
        rvertices = 0.5 * (r[1:] + r[:-1])
    else:  
        rvertices = np.sqrt(r[1:] * r[:-1])
        
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, rvertices.shape[0], r[-1])
    dr = rvertices[1:] - rvertices[:-1]
    dV          =  ( (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3) )
    
    etotal = edens_total * (4 * np.pi * dV) * e_scale.value
    mass   = dV * fields['W'] * fields['rho']
    e_k    = (fields['W'] - 1.0) * mass * e_scale.value

    u = fields['gamma_beta']
    w = np.diff(u).max()*1e-1
    n = int(np.ceil( (u.max() - u.min() ) / w ) )
    gbs = np.logspace(np.log10(1.e-4), np.log10(u.max()), n)
    eks = np.asanyarray([e_k[u > gb].sum() for gb in gbs])
    ets = np.asanyarray([etotal[u > gb].sum() for gb in gbs])
    
    E_seg_rat  = ets[1:]/ets[:-1]
    gb_seg_rat = gbs[1:]/gbs[:-1]
    E_seg_rat[E_seg_rat == 0] = 1
    
    slope = (ets[1:] - ets[:-1])/(gbs[1:] - gbs[:-1])
    power_law_region = np.argmin(slope)
    up_min           = find_nearest(gbs, 2 * gbs[power_law_region: ][0])[0]
    upower           = gbs[up_min: ]
    
    # Fix the power law segment, ignoring the sharp dip at the tail of the CDF
    epower_law_seg   = E_seg_rat [up_min: np.argmin(E_seg_rat > 0.8)]
    gbpower_law_seg  = gb_seg_rat[up_min: np.argmin(E_seg_rat > 0.8)]
    segments         = np.log10(epower_law_seg) / np.log10(gbpower_law_seg)
    alpha            = 1.0 - np.mean(segments)
    
    print('Avg power law index: {:.2f}'.format(alpha))
    bins    = np.arange(min(gbs), max(gbs) + w, w)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))
    
    E_0 = ets[up_min] * upower[0] ** (alpha - 1)
    if args.labels is None:
        hist = ax.hist(gbs, bins=gbs, weights=ets, label= r'$E_T$', histtype='step', rwidth=1.0, linewidth=3.0)
        # ax.plot(upower, E_0 * upower**(-(alpha - 1)), '--')
    else:
        hist = ax.hist(gbs, bins=gbs, weights=ets, label=r'${}$, t={:.2f}'.format(args.labels[case], tend), histtype='step', rwidth=1.0, linewidth=3.0)
        # ax.plot(upower, E_0 * upower**(-(alpha - 1)), '--', label = r'${}$ fit'.format(args.labels[case]))
    
    
    ax.set_title(r'setup: {}'.format(args.setup[0]), fontsize=20)
    ax.legend(fontsize=15)
    if not overplot:
        ax.set_title(r'setup: {}, t ={:.2f}'.format(args.setup[0], tend), fontsize=20)
        return fig



def movie(parser: argparse.ArgumentParser):
    plot_parser = get_subparser(parser, 1)
    plot_parser.add_argument('--scale_down', dest='scale_down', default=None, type=float, help='list of values to scale down fields', nargs='+')
    plot_parser.add_argument('--frame_range', dest='frame_range', default = [None, None], nargs=2, type=int)
    args = parser.parse_args()

    vmin = args.cbar[0] or 0.0
    vmax = args.cbar[1] or 1.0
    cinterval   = np.linspace(vmin, vmax, len(args.fields))
    cmap        = plt.cm.get_cmap(args.cmap)
    colors      = util.get_colors(cinterval, cmap, vmin, vmax)
    plt.rc('axes', prop_cycle=(cycler(color=colors)))
    
    
    flist, frame_count = util.get_file_list(args.files)
    flist              = flist[args.frame_range[0]: args.frame_range[1]]
    frame_count        = len(flist)
        
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    setup = util.read_file(args, flist[0], ndim=1)[1]
    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif not setup['linspace']:
        ax.set_xscale('log')
    
    if args.hist:
        ax.set_ylabel(r'$E( > \Gamma \beta) \ [\rm{erg}]$')
        ax.set_xlabel(r'$\Gamma\beta$')
    else:
        field_strs = util.get_field_str(args)
        if len(args.fields) == 1:
            ax.set_ylabel(f'{field_strs}')
        ax.set_xlabel('x')
    
    if args.dbg:
        plt.style.use('dark_background')
    
    def init_mesh(filename):
        return plot_profile(fig, ax, filename, args)
        
    def update(frame, fargs):
        '''
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        '''

        ax.cla()
        # Not strictly neccessary, just so we know we are stealing these from
        # the global scope
        pcolor_mesh = plot_profile(fig, ax, flist[frame], fargs)

        return pcolor_mesh

    # Initialize plot
    inital_im = init_mesh(flist[0])

    animation = FuncAnimation(
        # Your Matplotlib Figure object
        fig,
        # The function that does the updating of the Figure
        update,
        # Frame information (here just frame number)
        np.arange(frame_count),
        #blit = True,
        # Extra arguments to the animate function
        fargs=[args],
        # repeat=False,
        # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
        interval= 1000 / 10
    )

    if args.save is not None:
        animation.save('{}.mp4'.format(args.save).replace(' ', '_'), progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\r', flush=True))
        # animation.save('science/{}_{}.mp4'.format(args.setup[0], args.fields))
    else:
        plt.show()
