#! /usr/bin/env python

# Read in a File and Plot it

import numpy as np 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import matplotlib.colors as colors
import argparse 
import h5py 

import pandas as pd

from datetime import datetime
import os

prim_choices = ['rho', 'v1', 'v2', 'p']
def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog="This Only Supports H5 Files Right Now")
    
    parser.add_argument('filename', metavar='Filename', nargs='+',
                        help='A Data Source to Be Plotted')
    
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--prim', dest = "prim", metavar='Primitive Variable', nargs='?',
                        help='The name of the primitive variable you\'d like to plot',
                        choices=prim_choices, default="rho")
    
    parser.add_argument('--cbar_range', dest = "cbar", metavar='Range of Color Bar', nargs=2,
                        default = [None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cmap', dest = "cmap", metavar='Color Bar Colarmap', nargs=1,
                        default = 'magma', help='The colorbar cmap you\'d like to plot')
    
    parser.add_argument('--log', dest='log', action='store_true',
                        default=False,
                        help='Logarithmic Radial Grid Option')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1 accuracy')

   
    args = parser.parse_args()
    vmin, vmax = args.cbar
    prim_dict = {}
    with h5py.File(args.filename[0], 'r+') as hf:
        kek = [key for key in hf.keys()]
        # attr = [at for at in hf[kek[0]].keys() ]
        ds = hf[kek[0]]
        
        rho         = ds.attrs["rho"] 
        v1          = ds.attrs["v1"] 
        v2          = ds.attrs["v2"]  
        p           = ds.attrs["p"] 
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
        
        prim_dict["rho"] = rho
        prim_dict["v1"]  = v1 
        prim_dict["v2"]  = v2 
        prim_dict["p"]   = p
        
        
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
    
    color_map = plt.cm.get_cmap(args.cmap[0])
    reversed_color_map = color_map.reversed()
    
    fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)

    tend = t
    c1 = ax.pcolormesh(tt, rr, prim_dict[args.prim], cmap=color_map, shading='auto', norm = norm)
    c2 = ax.pcolormesh(t2[::-1], rr, prim_dict[args.prim],  cmap=color_map, shading='auto', norm = norm)

    fig.suptitle('SIMBI: {} at t = {:.2f} s'.format(args.setup[0], t), fontsize=20)

    cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
    cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)
    ax.set_position( [0.1, -0.18, 0.8, 1.43])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.grid(True, alpha=0.95)
    ax.xaxis.grid(True, alpha=0.95)
    ax.tick_params(axis='both', labelsize=20)
    cbaxes.tick_params(axis='x', labelsize=20)
    ax.axes.xaxis.set_ticklabels([])
    # ax.set_rmax(0.3)
    # ax.set_rmin(r.min())
    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    if args.log:
        cbar.ax.set_xlabel('Log [{}]'.format(args.prim), fontsize=20)
    else:
        cbar.ax.set_xlabel('[{}]'.format(args.prim), fontsize=20)
        
    plt.show()
    
if __name__ == "__main__":
    main()
