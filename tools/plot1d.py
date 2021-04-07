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

    parser.add_argument('--save', dest='save', action='store_true',
                        default=False,
                        help='True if you want save the fig')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1')

   
    args = parser.parse_args()
    field_dict = {}
    with h5py.File(args.filename[0], 'r+') as hf:
        
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
        
        
        
        y    = np.argmax(v)
        vgas = v[:y]
        vm = np.sqrt(2 * k * 1e3 / m)
        vh = np.sqrt(2 * k * 6e3 / m)
        
        vgas = W[:y] * (1 + beta[:y])*vgas
        
        maxwell  =  1/(np.sqrt(np.pi) * vm) * np.exp(-(vgas * c - vh)**2 /vm**2)
        
        
        field_dict["rho"]         = rho
        field_dict["v"]           = v
        field_dict["p"]           = p
        field_dict["gamma_beta"]  = W*beta
        field_dict["temperature"] = T
        field_dict["vpdf"]= maxwell
        
        
    xnpts = rho.size 
    fig, ax= plt.subplots(1, 1, figsize=(10,8))
    
    if (args.log):
        if args.field != 'vpdf':
            r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
            ax.loglog(r, field_dict[args.field])
        else:
            ax.semilogx(vgas * c, field_dict[args.field])
    else:
        if args.field != 'vpdf':
            r = np.linspace(xmin, xmax, xactive)
            ax.plot(r, field_dict[args.field])
        else:
            ax.plot(vgas * c, field_dict[args.field])
        
    
    tend = t
    
    ax.set_title('{} at t = {:.2f} s'.format(args.setup[0], t), fontsize=20)


    ax.tick_params(axis='both', labelsize=20)
    
    if args.field != "vpdf":
        ax.set_xlabel('R', fontsize=20)
        ax.set_xlim(xmin, xmax) if args.rmax == 0.0 else ax.set_xlim(xmin, args.rmax)
    else:
        ax.set_xlabel(r'$V_{\rm lab}$ [cm/s]', fontsize=20)
        ax.set_xlim(c * vgas.min(), c * vgas.max()) if args.rmax == 0.0 else ax.set_xlim(c * v.min(), c * args.rmax)

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
        
    plt.show()
    
    if args.save:
        fig.savefig("plots/{}.png".format(args.setup[0]), dpi=1200)
    
if __name__ == "__main__":
    main()
