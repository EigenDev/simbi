#! /usr/bin/env python

# Tool for making movies from h5 files in directory

import numpy as np 
import matplotlib.pyplot as plt #! /usr/bin/env python
import matplotlib.ticker as tkr
import utility as util
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const

from utility import DEFAULT_SIZE, SMALL_SIZE, BIGGER_SIZE
from matplotlib.animation import FuncAnimation

import os, os.path

derived = ['gamma_beta', 'temperature']
field_choices = ['rho', 'v', 'p'] + derived

def get_frames(dir):
    # Get number of files in dir
    total_frames = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

    return total_frames, frames

def plot_profile(fig, ax, filepath, filename, args):
    fields = util.read_1d_file(filepath + filename)
    r, t      = fields['r'], fields['t']
    x1min, x1max = fields['xlims']
    
    if args.field[0] in derived:
        var = util.prims2var(fields, args.field[0])
    else:
        var = fields[args.field]
        
    ax.plot(r, var)
    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    if args.units:
        xlabel = r'$r/R_\odot$'
    else:
        xlabel = 'r'
        
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_title('{} at t = {:.2f} s'.format(args.setup[0], t), fontsize=30)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim(x1min, x1max) if args.rmax == 0.0 else ax.set_xlim(x1min, args.rmax)

    # Change the format of the field
    field_str = util.get_field_str(args)
    ax.set_ylabel('{}'.format(field_str), fontsize=20)
    return ax

def plot_hist(args, fields, overplot=False, ax=None, case=0):
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend = fields['t']
    edens_total = util.prims2var(fields, 'energy')
    r           = fields['r']
    
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
    eks = np.asarray([e_k[u > gb].sum() for gb in gbs])
    ets = np.asarray([etotal[u > gb].sum() for gb in gbs])
    
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
    
   
    # if case == 0:
    #     ax.hist(gbs_1d, bins=gbs_1d, weights=ets_1d, alpha=0.8, label= r'1D Sphere', histtype='step', linewidth=3.0)
    
    sorted_energy = np.sort(ets)
    plt.xscale('log')
    plt.yscale('log')
    # ax.set_ylim(sorted_energy[1], 1.5*ets.max())
    ax.set_xlabel(r'$\Gamma\beta $', fontsize=20)
    ax.set_ylabel(r'$E( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
    ax.tick_params('both', labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_title(r'setup: {}'.format(args.setup[0]), fontsize=20)
    ax.legend(fontsize=15)
    if not overplot:
        ax.set_title(r'setup: {}, t ={:.2f}'.format(args.setup[0], tend), fontsize=20)
        return fig



def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog='This Only Supports H5 Files Right Now')
    
    parser.add_argument('data_dir', metavar='dir', nargs='+',
                        help='A data directory to retrieve the h5 files')
    
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--field', dest = 'field', metavar='Field Variable', nargs='+',
                        help='The name of the field variable you\'d like to plot',
                        choices=field_choices, default='rho')
    
    parser.add_argument('--rmax', dest = 'rmax', metavar='Radial Domain Max',
                        default = 0.0, help='The domain range')
    parser.add_argument('--tex', dest='tex', action='store_true', default=False,
                        help='Use latex typesetting')
    parser.add_argument('--log', dest='log', action='store_true', default=False,
                        help='Logarithmic Radial Grid Option')
    parser.add_argument('--units', dest='units', default=False, action='store_true')
    
    parser.add_argument('--save', dest='save', action='store_true',
                        default=False)

   
    args = parser.parse_args()
    if args.tex:
        plt.rc('font',   size=DEFAULT_SIZE)          # controls default text sizes
        plt.rc('axes',   titlesize=DEFAULT_SIZE)     # fontsize of the axes title
        plt.rc('axes',   labelsize=DEFAULT_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('ytick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('legend', fontsize=DEFAULT_SIZE)      # legend fontsize
        plt.rc('figure', titlesize=DEFAULT_SIZE)    # fontsize of the figure title
        
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": DEFAULT_SIZE
            }
        )
    fig, ax = plt.subplots(1, 1, figsize=(15,8))
    
    frame_count, flist = get_frames(args.data_dir[0])
    
    def init_mesh(filename):
        return plot_profile(fig, ax, args.data_dir[0], filename, args)
        
    def update(frame, fargs):
        '''
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        '''

        ax.cla()
        # Not strictly neccessary, just so we know we are stealing these from
        # the global scope
        pcolor_mesh = plot_profile(fig, ax, args.data_dir[0], flist[frame], fargs)

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
        interval= 1000 / 2
    )

    if args.save:
        animation.save('{}.mp4'.format(args.setup[0]).replace(' ', '_'))
        # animation.save('science/{}_{}.mp4'.format(args.setup[0], args.field))
    else:
        plt.show()

    
    
if __name__ == '__main__':
    main()

import h5py 