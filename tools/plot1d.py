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

from datetime import datetime
import os

field_choices = ['rho', 'v', 'p', 'gamma_beta', 'temperature', 'vpdf']

def prims2cons(fields, cons):
    if cons == "D":
        return fields['rho'] * fields['W']
    elif cons == "energy":
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p'] - fields['rho']*fields['W']


def plot_profile(args, field_dict, ax = None, overplot = False, case = 0):
    
    r = field_dict["r"]
    
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,8))
    
    if args.labels is None:
        if (args.log):
            ax.loglog(r, field_dict[args.field])
        else:
            ax.plot(r, field_dict[args.field])
    else:
        if (args.log):
            ax.loglog(r, field_dict[args.field], label = args.labels[case])
        else:
            ax.plot(r, field_dict[args.field], label = args.labels[case])
        
    
    ax.set_title('{} at t = {:.2f} s'.format(args.setup[0], field_dict["t"]), fontsize=20)


    ax.tick_params(axis='both', labelsize=20)
    
    
    ax.set_xlabel('$r/R_\odot$', fontsize=20)
    ax.set_xlim(r.min(), r.max()) if args.rmax == 0.0 else ax.set_xlim(r.min(), args.rmax)

    # Change the format of the field
    if args.field == "rho":
        field_str = r'$\rho$'
    elif args.field == "gamma_beta":
        field_str = r"$\Gamma \ \beta$"
    elif args.field == "vpdf":
        field_str = r"$\exp[- (v_{\rm lab} - v_H)^2/v_{\rm rms}^2]$"
    else:
        field_str = args.field
        
    
    ax.set_ylabel('{}'.format(field_str), fontsize=20)
    
def plot_hist(args, fields, overplot=False, ax=None, case=0):
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend = fields["t"]
    e_scale = 2e33 * const.c.cgs.value**2
    
    edens_total = prims2cons(fields, "energy")
    r           = fields["r"]
    
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, rvertices.shape[0], r[-1])
    dr = rvertices[1:] - rvertices[:-1]
        
    dV          =  ( (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3) )
    
    etotal = edens_total * (4 * np.pi * dV) * e_scale
    mass   = dV * fields["W"] * fields["rho"]
    e_k    = (fields['W'] - 1.0) * mass * e_scale
    
    u = fields['gamma_beta']
    w = np.diff(u).max()*1e-1
    n = int(np.ceil( (u.max() - u.min() ) / w ) )
    gbs = np.logspace(np.log10(1.e-4), np.log10(u.max()), n)
    eks = np.asarray([e_k[np.where(u > gb)].sum() for gb in gbs])
    ets = np.asarray([etotal[np.where(u > gb)].sum() for gb in gbs])
    
    bins    = np.arange(min(gbs), max(gbs) + w, w)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))

    if args.labels is None:
        ax.hist(gbs, bins=gbs, weights=ets, label= r'$E_T$', histtype='step', rwidth=1.0, linewidth=3.0)
    else:
        ax.hist(gbs, bins=gbs, weights=ets, label=r'${}$'.format(args.labels[case]), histtype='step', rwidth=1.0, linewidth=3.0)
    
    # if case == 0:
    #     ax.hist(gbs_1d, bins=gbs_1d, weights=ets_1d, alpha=0.8, label= r'1D Sphere', histtype='step', linewidth=3.0)
    
    sorted_energy = np.sort(ets)
    plt.xscale('log')
    plt.yscale('log')
    #ax.set_ylim(sorted_energy[1], 1.5*ets.max())
    ax.set_xlabel(r'$\Gamma\beta $', fontsize=20)
    ax.set_ylabel(r'$E( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
    ax.tick_params('both', labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(r'setup: {}, t ={:.2f} s'.format(args.setup[0], tend), fontsize=20)
    ax.legend(fontsize=15)
    if not overplot:
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
    
    parser.add_argument('--log', dest='log', action='store_true',
                        default=False,
                        help='Logarithmic Radial Grid Option')
    
    parser.add_argument('--ehist', dest='ehist', action='store_true',
                        default=False,
                        help='Plot the energy_vs_gb histogram')
    
    parser.add_argument('--labels', dest='labels', nargs='+',
                        help='map labels to filenames')

    parser.add_argument('--save', dest='save', action='store_true',
                        default=False,
                        help='True if you want save the fig')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1')

   
    args = parser.parse_args()
    field_dict = {}
    for idx, file in enumerate(args.filename):
        field_dict[idx] = {}
        with h5py.File(file, 'r+') as hf:
            
            ds = hf.get("sim_info")
            
            rho         = hf.get("rho")[:]
            v           = hf.get("v")[:]
            p           = hf.get("p")[:]
            nx          = ds.attrs["Nx"]
            t           = ds.attrs["current_time"]
            xmax        = ds.attrs["xmax"]
            xmin        = ds.attrs["xmin"]
            try:
                ad_gamma = ds.attrs["adbiatic_gamma"]
            except:
                ad_gamma = 4./3.

            
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
            
            if (args.log):
                r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
            else:
                r = np.linspace(xmin, xmax, xactive)
            
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
        
    if len(args.filename) > 1:
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        for idx, file in enumerate(args.filename):
            if args.ehist:
                plot_hist(args, field_dict[idx], ax = ax, overplot= True, case = idx)
            else:
                plot_profile(args, field_dict[idx], ax = ax, overplot=True, case = idx)
        if args.labels != None:
            ax.legend()
            
    else:
        if args.ehist:
            plot_hist(args, field_dict[0])
        else:
            plot_profile(args, field_dict[0])
        
    plt.show()
    
    if args.save:
        fig.savefig("plots/{}.png".format(args.setup[0]))
    
if __name__ == "__main__":
    main()
