#! /usr/bin/env python

# Read in a File and Plot it

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import time
import scipy.special as spc
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const
import astropy.units as u 
import os

from datetime import datetime

cons = ['D', 'momentum', 'energy']
field_choices = ['rho', 'v', 'p', 'gamma_beta', 'temperature'] + cons
col = plt.cm.jet([0.25,0.75])  

R_0 = const.R_sun.cgs 
c   = const.c.cgs
m   = const.M_sun.cgs
 
rho_scale  = m / (4./3. * np.pi * R_0 ** 3) 
e_scale    = m * const.c.cgs.value**2
pre_scale  = e_scale / (4./3. * np.pi * R_0**3)
vel_scale  = c 
time_scale = R_0 / c

def find_nearest(arr, val):
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]
 
def fill_below_intersec(x, y, constraint, color):
    # colors = plt.cm.plasma(np.linspace(0.25, 0.75, len(x)))
    ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:],y[ind:], color=color, alpha=0.1, interpolate=True)
    
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


def plot_profile(args, field_dict, ax = None, overplot = False, subplot = False, case = 0):
    
    colors = plt.cm.twilight_shifted(np.linspace(0.25, 0.75, len(args.filename)))
    r = field_dict["r"]
    tend = field_dict['t']
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,8))
    
    unit_scale = 1.0
    if (args.units):
        if args.field == "rho":
            unit_scale = rho_scale
        elif args.field == "p":
            unit_scale = pre_scale
        
    if args.field in cons:
        var = prims2cons(field_dict, args.field)
    else:
        var = field_dict[args.field]
        
    if args.labels is None:
        ax.plot(r, var * unit_scale, color=colors[case])
    else:
        ax.plot(r, var * unit_scale, color=colors[case], label=r'${}$, t={:.2f}'.format(args.labels[case], tend))

    ax.tick_params(axis='both', labelsize=15)
    if (args.log):
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel('$r/R_\odot$', fontsize=20)
    if args.xlim is None:
        ax.set_xlim(r.min(), r.max()) if args.rmax == 0.0 else ax.set_xlim(r.min(), args.rmax)
    else:
        xmin, xmax = eval(args.xlim)
        ax.set_xlim(xmin, xmax)
    # Change the format of the field
    field_str = get_field_str(args)
        
    ax.set_ylabel('{}'.format(field_str), fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axvline(0.60, color='black', linestyle='--')
    
    ########
    # Personal Calculations
    # TODO: Remove Later
    ########
    r_outer = find_nearest(r, 0.55)[0]
    r_slow  = find_nearest(r, 1.50)[0]
    if field_dict["is_linspace"]:
        rvertices = 0.5 * (r[1:] + r[:-1])
    else:  
        rvertices = np.sqrt(r[1:] * r[:-1])
        
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, rvertices.shape[0], r[-1])
    dr = rvertices[1:] - rvertices[:-1]
    dV          =  ( (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3) )
    mout    = (4./3.) * np.pi * np.sum(dV[r_outer:r_slow] * field_dict["rho"][r_outer: r_slow])
    # print(mout)
    # zzz = input('')
    ########################
    
    
    if not subplot:   
        ax.set_title('{}'.format(args.setup[0]), fontsize=20)
    if not overplot:
        ax.set_title('{} at t = {:.3f}'.format(args.setup[0], tend), fontsize=20)
        return fig
    
def plot_hist(args, fields, overplot=False, ax=None, subplot = False, case=0):
    colors = plt.cm.twilight_shifted(np.linspace(0.25, 0.75, len(args.filename)))
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
    dV =  ( (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3) )
    
    if args.eks:
        mass   = 4.0 * np.pi * dV * fields["W"]**2 * fields["rho"]
        energy = (fields['W'] - 1.0) * mass * e_scale.value
    elif args.hhist:
        energy = (fields['enthalpy'] - 1.0) *  4.0 * np.pi * dV * e_scale.value
    else:
        energy = edens_total * 4.0 * np.pi * dV * e_scale.value


    u = fields['gamma_beta']
    w = 0.01 #np.diff(u).max()*1e-1
    n = int(np.ceil( (u.max() - u.min() ) / w ) )
    gbs = np.logspace(np.log10(1.e-4), np.log10(u.max()), n)
    
    energy = np.asarray([energy[u > gb].sum() for gb in gbs])
    
    E_seg_rat  = energy[1:]/energy[:-1]
    gb_seg_rat = gbs[1:]/gbs[:-1]
    E_seg_rat[E_seg_rat == 0] = 1
    
    slope = (energy[1:] - energy[:-1])/(gbs[1:] - gbs[:-1])
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
    
    E_0 = energy[up_min] * upower[0] ** (alpha - 1)
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
    ax.set_xlabel(r'$\Gamma\beta $', fontsize=20)
    if args.eks:
        ax.set_ylabel(r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
    elif args.hhist:
        ax.set_ylabel(r'$H ( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
    else:
        ax.set_ylabel(r'$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
    ax.tick_params('both', labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    

    if args.fill_scale is not None:
        fill_below_intersec(gbs, energy, args.fill_scale*energy.max(), colors[case])
    if not subplot:
        ax.set_title(r'setup: {}'.format(args.setup[0]), fontsize=20)
        # ax.legend(fontsize=15)
    if not overplot:
        ax.set_title(r'setup: {}, t ={:.2f}'.format(args.setup[0], tend), fontsize=20)
        return fig

def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog="This Only Supports H5 Files Right Now")
    
    parser.add_argument('filename', metavar='Filename', nargs='+',
                        help='A Data Source to Be Plotted')
    
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--field', dest = "field", metavar='Field Variable', nargs='?',
                        help='The name of the field variable you\'d like to plot',
                        choices=field_choices, default="rho")
    
    parser.add_argument('--rmax', dest = "rmax", metavar='Radial Domain Max',
                        default = 0.0, help='The domain range')
    
    parser.add_argument('--xlim', dest = "xlim", metavar='Domain',
                        default = None, help='The domain range')
    
    parser.add_argument('--fill_scale', dest = "fill_scale", metavar='Filler maximum', type=float,
                        default = None, help='Set the y-scale to start plt.fill_between')
    
    parser.add_argument('--log', dest='log', action='store_true',
                        default=False,
                        help='Logarithmic Radial Grid Option')
    
    parser.add_argument('--ehist', dest='ehist', action='store_true',
                        default=False,
                        help='Plot the energy_vs_gb histogram')
    
    parser.add_argument('--eks', dest='eks', action='store_true',
                        default=False,
                        help='Plot the kinetic energy on the histogram')
    
    parser.add_argument('--hhist', dest='hhist', action='store_true',
                        default=False,
                        help='Plot the enthalpy on the histogram')
    
    parser.add_argument('--labels', dest='labels', nargs='+',
                        help='map labels to filenames')

    parser.add_argument('--save', dest='save',
                        default=None,
                        help='If you want save the fig')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1')
    
    parser.add_argument('--plots', dest='plots', type = int,
                        default=1,
                        help=r'Number of subplots you\'d like')
    
    parser.add_argument('--units', dest='units', action='store_true',
                        default=False,
                        help='True if you would like units scale (default is solar units)')

   
    args = parser.parse_args()
    field_dict = {}
    for idx, file in enumerate(args.filename):
        field_dict[idx] = {}
        with h5py.File(file, 'r') as hf:
            
            ds = hf.get("sim_info")
            
            rho         = hf.get("rho")[:]
            v           = hf.get("v")[:]
            p           = hf.get("p")[:]   
            nx          = ds.attrs["Nx"]
            t           = ds.attrs["current_time"] * time_scale
            xmax        = ds.attrs["xmax"]
            xmin        = ds.attrs["xmin"]
            
            # added these attributes after some time, so fallbacks included
            try:
                ad_gamma = ds.attrs["adbiatic_gamma"]
            except:
                ad_gamma = 4./3.
                
            
            try:
                is_linspace = ds.attrs["linspace"]
            except:
                is_linspace = False

            
            if args.forder:
                rho = rho[1:-1]
                v   = v  [1:-1]
                p   = p  [1:-1]
                xactive = nx - 2
            else:
                rho = rho[2:-2]
                v   = v  [2:-2]
                p   = p  [2:-2]
                xactive = nx - 4
                
            W    = 1/np.sqrt(1 - v**2)
            beta = v
            
            e = 3*p/rho 
            c = const.c.cgs.value
            a = (4 * const.sigma_sb.cgs.value / c)
            k = const.k_B.cgs.value
            m = const.m_p.cgs.value
            me = const.m_e.cgs.value
            T = (3 * p * c ** 2  / a)**(1./4.)
            
            h = 1.0 + ad_gamma * p / (rho * (ad_gamma - 1.0))
            
            if (is_linspace):
                r = np.linspace(xmin, xmax, xactive)
            else:
                r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
                
            # post process the time into days, weeks, 
            if (t.value > u.hour.to(u.s)):
                t = t.to(u.hour)
            if (t.value > u.day.to(u.s)):
                t = t.to(u.day)
            elif (t.value > u.week.to(u.s)):
                t = t.to(u.week)
            
            field_dict[idx]["rho"]         = rho
            field_dict[idx]["v"]           = v
            field_dict[idx]["p"]           = p
            field_dict[idx]["gamma_beta"]  = W*beta
            field_dict[idx]["temperature"] = T
            field_dict[idx]["enthalpy"]    = h
            field_dict[idx]["W"]           = W
            field_dict[idx]["t"]           = t 
            field_dict[idx]["xmin"]        = xmin
            field_dict[idx]["xmax"]        = xmax
            field_dict[idx]["xactive"]     = xactive
            field_dict[idx]["r"]           = r
            field_dict[idx]["is_linspace"]           = is_linspace
        
    if len(args.filename) > 1:
        if args.plots == 1:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1, 1, 1)
            for idx, file in enumerate(args.filename):
                if args.ehist or args.eks or args.hhist:
                    plot_hist(args, field_dict[idx], ax = ax, overplot= True, case = idx)
                else:
                    plot_profile(args, field_dict[idx], ax = ax, overplot=True, case = idx)
        else:
            fig = plt.figure(figsize=(30,10))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            for idx, file in enumerate(args.filename):
                plot_hist(args, field_dict[idx], ax = ax1, overplot= True, subplot = True, case = idx)
                plot_profile(args, field_dict[idx], ax = ax2, overplot=True, subplot = True, case = idx)
                
            fig.suptitle("{}".format(args.setup[0]), fontsize=40)
        if args.labels != None:
            ax.legend(fontsize = 15)
            
    else:
        if args.ehist or args.hhist or args.eks:
            fig = plot_hist(args, field_dict[0])
        else:
            fig = plot_profile(args, field_dict[0])
        
    
    
    if args.save is not None:
        fig.savefig("{}.png".format(args.save), bbox_inches='tight')
    else:
        plt.show()
    
if __name__ == "__main__":
    main()
