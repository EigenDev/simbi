#! /usr/bin/env python

# Read in a File and Plot it

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import time
import scipy.ndimage
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const

from datetime import datetime
import os

field_choices = ['rho', 'v1', 'v2', 'p', 'gamma_beta', 'temperature']
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
    
    parser.add_argument('--cbar_range', dest = "cbar", metavar='Range of Color Bar',
                        default ='None, None', help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cmap', dest = "cmap", metavar='Color Bar Colarmap',
                        default = 'magma', help='The colorbar cmap you\'d like to plot')
    
    parser.add_argument('--log', dest='log', action='store_true',
                        default=False,
                        help='Logarithmic Radial Grid Option')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1')
    
    parser.add_argument('--rev_cmap', dest='rcmap', action='store_true',
                        default=False,
                        help='True if you want the colormap to be reversed')

    parser.add_argument('--save', dest='save', action='store_true',
                        default=False,
                        help='True if you want save the fig')

   
    args = parser.parse_args()
    vmin, vmax = eval(args.cbar)
    field_dict = {}
    with h5py.File(args.filename[0], 'r+') as hf:
        
        ds = hf.get("sim_info")
        
        rho         = hf.get("rho")[:]
        v1          = hf.get("v1")[:]
        v2          = hf.get("v2")[:]
        p           = hf.get("p")[:]
        nx          = ds.attrs["NX"]
        ny          = ds.attrs["NY"]
        t           = ds.attrs["current_time"]
        xmax        = ds.attrs["xmax"]
        xmin        = ds.attrs["xmin"]
        ymax        = ds.attrs["ymax"]
        ymin        = ds.attrs["ymin"]
        
        
        rho = rho.reshape(ny, nx)
        v1  = v1.reshape(ny, nx)
        v2  = v2.reshape(ny, nx)
        p   = p.reshape(ny, nx)
        
        if args.forder:
            rho = rho[1:-1, 1: -1]
            v1  = v1 [1:-1, 1: -1]
            v2  = v2 [1:-1, 1: -1]
            p   = p  [1:-1, 1: -1]
            xactive = nx - 2
            yactive = ny - 2
        else:
            rho = rho[2:-2, 2: -2]
            v1  = v1 [2:-2, 2: -2]
            v2  = v2 [2:-2, 2: -2]
            p   = p  [2:-2, 2: -2]
            xactive = nx - 4
            yactive = ny - 4
            
        W    = 1/np.sqrt(1 - v1**2 + v2**2)
        beta = np.sqrt(v1**2 + v2**2)
        
        e = 3*p/rho 
        c = const.c.cgs.value
        a = (4 * const.sigma_sb.cgs.value / c)
        m = const.m_p.cgs.value
        T = (3 * p * c ** 2  / a)**(1./4.)
        
        
        field_dict["rho"]         = rho
        field_dict["v1"]          = v1 
        field_dict["v2"]          = v2 
        field_dict["p"]           = p
        field_dict["gamma_beta"]  = W*beta
        field_dict["temperature"] = T
        
        
    ynpts, xnpts = rho.shape 

    if (args.log):
        r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
        norm = colors.LogNorm(vmin=rho.min(), vmax=3.*rho.min())
    else:
        r = np.linspace(xmin, xmax, xactive)
        # norm = colors.LinearNorm(vmin=None, vmax=None)
        
    # r = np.logspace(np.log10(0.01), np.log10(0.5), xnpts)
    theta = np.linspace(ymin, ymax, yactive)
    theta_mirror = - theta[::-1]
    theta_mirror[-1] *= -1.
    
    rr, tt = np.meshgrid(r, theta)
    rr, t2 = np.meshgrid(r, theta_mirror)
    
    W = 1/np.sqrt(1- v1**2 + v2**2)
    
    ad_gamma = 4/3
    D = rho * W 
    h = 1 + ad_gamma*p/(rho*(ad_gamma - 1))
    tau = rho *h *W**2 - p - rho * W 
    S1 = D*h*W**2*v1
    S2 = D*h*W**2*v2
    S = np.sqrt(S1**2 + S2**2)
    E = tau + D 
    # E[:, np.where(r > 0.16)] = E[0][]
    norm  = colors.LogNorm(vmin = vmin, vmax = vmax)
    vnorm = colors.LogNorm(vmin = W.min(), vmax = W.max())
    enorm = colors.LogNorm(vmin = 1, vmax = 9e1)
    snorm = colors.LogNorm(vmin=1.e-5, vmax=1e2)
    
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
    
    fig, ax= plt.subplots(1, 1, figsize=(15,8), subplot_kw=dict(projection='polar'), constrained_layout=True)

    tend = t
    c1 = ax.pcolormesh(tt, rr, field_dict[args.field], cmap=color_map, shading='auto', norm = norm)
    c2 = ax.pcolormesh(t2[::-1], rr, field_dict[args.field],  cmap=color_map, shading='auto', norm = norm)

    fig.suptitle('SIMBI: {} at t = {:.2f} s'.format(args.setup[0], t), fontsize=20, y=0.95)

    cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
    if args.log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)
    ax.set_position( [0.1, -0.18, 0.8, 1.43])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.grid(True, alpha=0.1)
    ax.xaxis.grid(True, alpha=0.1)
    ax.tick_params(axis='both', labelsize=10)
    cbaxes.tick_params(axis='x', labelsize=10)
    ax.axes.xaxis.set_ticklabels([])
    ax.set_rmax(xmax) if args.rmax == 0.0 else ax.set_rmax(args.rmax)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    # Change the format of the field
    if args.field == "rho":
        field_str = r'$\rho$'
    elif args.field == "gamma_beta":
        field_str = r"$\Gamma \ \beta$"
    else:
        field_str = args.field
    
    if args.log:
        cbar.ax.set_xlabel('Log [{}]'.format(field_str), fontsize=20)
    else:
        cbar.ax.set_xlabel('[{}]'.format(args.field), fontsize=20)
        
    plt.show()
    
    if args.save:
        fig.savefig("plots/2D/SR/{}.png".format(args.setup[0]), dpi=1200)
    
if __name__ == "__main__":
    main()
