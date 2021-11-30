#! /usr/bin/env python

# Tool for making movies from h5 files in directory

import numpy as np 
import matplotlib.pyplot as plt #! /usr/bin/env python
import matplotlib.ticker as tkr
import time
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const

from matplotlib.animation import FuncAnimation

import os, os.path

field_choices = ['rho', 'v', 'p', 'gamma_beta', 'temperature']

R_0 = const.R_sun.cgs 
c   = const.c.cgs
m   = const.M_sun.cgs
 
rho_scale  = m / (4./3. * np.pi * R_0 ** 3) 
e_scale    = m * const.c.cgs.value**2
pre_scale  = e_scale / (4./3. * np.pi * R_0**3)
vel_scale  = c 
time_scale = R_0 / c

def get_field_str(args):
    if args.field == "rho":
        if args.units:
            return r'$\rho$ [g cm$^{-3}$]'
        else:
            return r'$\rho$'
    elif args.field == "gamma_beta":
        return r"$\Gamma \ \beta$"
    elif args.field == "energy":
        return r"$\tau$"
    else:
        return args.field
    
def prims2cons(fields, cons):
    if cons == "D":
        return fields['rho'] * fields['W']
    elif cons == "S":
        return fields['rho'] * fields['W']**2 * fields['v']
    elif cons == "energy":
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p'] - fields['rho']*fields['W']
    
def get_frames(dir):
    # Get number of files in dir
    total_frames = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

    return total_frames, frames

def plot_profile(fig, ax, filepath, filename, args):
    
    field_dict = {}
    with h5py.File(filepath + filename, 'r') as hf:
        
        ds = hf.get("sim_info")
        
        rho         = hf.get("rho")[:]
        v           = hf.get("v")[:]
        p           = hf.get("p")[:]
        nx          = ds.attrs["Nx"]
        t           = ds.attrs["current_time"]
        xmax        = ds.attrs["xmax"]
        xmin        = ds.attrs["xmin"]
        
        
        if args.forder:
            rho = rho[1:-1]
            v   = v[1:-1]
            p   = p  [1:-1]
            xactive = nx - 2
        else:
            rho = rho[2:-2]
            v   = v [2:-2]
            p   = p  [2:-2]
            xactive = nx - 4
            
        W    = 1/np.sqrt(1 - v**2)
        beta = v
        
        e = 3*p/rho 
        c = const.c.cgs.value
        a = (4 * const.sigma_sb.cgs.value / c)
        m = const.m_p.cgs.value
        T = (3 * p * c ** 2  / a)**(1./4.)
        
        
        field_dict["rho"]         = rho
        field_dict["v"]           = v
        field_dict["p"]           = p
        field_dict["gamma_beta"]  = W*beta
        field_dict["temperature"] = T
        
        
    xnpts = xactive

    if (args.log):
        r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
    else:
        r = np.linspace(xmin, xmax, xactive)
        
    ax.plot(r, field_dict[args.field])
    
    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel('$r/R_\odot$', fontsize=30)


    if args.log and args.field != "gamma_beta":
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        ax.yaxis.set_major_formatter(logfmt)
        
    ax.set_title('{} at t = {:.2f} s'.format(args.setup[0], t), fontsize=30)
    
    ax.tick_params(axis='both', labelsize=20)
    
    ax.set_xlim(xmin, xmax) if args.rmax == 0.0 else ax.set_xlim(xmin, args.rmax)

    # Change the format of the field
    if args.field:
        if   args.field == "rho":
            field_str = r'$\log \rho$'
        elif args.field == "gamma_beta":
            field_str = r"$\Gamma \ \beta$"
        elif args.field == "temperature":
            field_str = r"$\log$ T [K]"
        else:
            field_str = arg.sfield
    else:
        if  args.field == "rho":
            field_str = r' $\rho$'
        elif args.field == "gamma_beta":
            field_str = r" $\Gamma \ \beta$"
        elif args.field == "temperature":
            field_str = " T [K]"
        else:
            field_str = args.field
    
    
    ax.set_ylabel('{}'.format(field_str), fontsize=20)
    
    return ax

def plot_hist(args, fields, overplot=False, ax=None, case=0):
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend = fields["t"]
    edens_total = prims2cons(fields, "energy")
    r           = fields["r"]
    
    if fields["is_linspace"]:
        rvertices = 0.5 * (r[1:] + r[:-1])
    else:  
        rvertices = np.sqrt(r[1:] * r[:-1])
        
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, rvertices.shape[0], r[-1])
    dr = rvertices[1:] - rvertices[:-1]
    dV          =  ( (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3) )
    
    etotal = edens_total * (4 * np.pi * dV) * e_scale.value
    mass   = dV * fields["W"] * fields["rho"]
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
    
    print("Avg power law index: {:.2f}".format(alpha))
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
        epilog="This Only Supports H5 Files Right Now")
    
    parser.add_argument('data_dir', metavar='dir', nargs='+',
                        help='A data directory to retrieve the h5 files')
    
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--field', dest = "field", metavar='Field Variable', nargs='?',
                        help='The name of the field variable you\'d like to plot',
                        choices=field_choices, default="rho")
    
    parser.add_argument('--rmax', dest = "rmax", metavar='Radial Domain Max',
                        default = 0.0, help='The domain range')
    
    parser.add_argument('--log', dest='log', action='store_true',
                        default=False,
                        help='Logarithmic Radial Grid Option')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1')
    
    parser.add_argument('--save', dest='save', action='store_true',
                        default=False)

   
    args = parser.parse_args()
    fig = plt.figure(figsize=(15,8))
    
    for field in args.field:
        ax  = fig.add_subplot(111)
    
    frame_count, flist = get_frames(args.data_dir[0])
    
    def init_mesh(filename):
        p = plot_profile(fig, ax, args.data_dir[0], filename, args)
        
        return p
        
    def update(frame, fargs):
        """
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        """

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
        animation.save("{}.mp4".format(args.setup[0]).replace(" ", "_"))
        # animation.save("science/{}_{}.mp4".format(args.setup[0], args.field))
    else:
        plt.show()

    
    
if __name__ == "__main__":
    main()

import h5py 