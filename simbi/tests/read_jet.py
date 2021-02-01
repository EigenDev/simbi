#! /usr/bin/env python

# Read in a File and Plot it

import numpy as np 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import matplotlib.colors as colors
import argparse 
import h5py 

from datetime import datetime

prim_choices = ['rho', 'v1', 'v2', 'p']
def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).'
        epiloh="This Only Supports H5 Files Right Now")
    
    parser.add_argument('filename', metavar='Filename', nargs='+',
                        help='A Data Source to Be Plotted')
    
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--prim', metavar='Primitive Variable', nargs='?',
                        help='The name of the primitive variable you\'d like to plot',
                        choice=prim_choices, argument_default="rho")
    
    parser.add_argument('--cbar_range', metavar='Range of Color Bar', type=list, nargs='?',
                        help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--log', dest='log', action='store_false',
                        default=False,
                        help='Logarithmic Radial Grid Option')

    args = parser.parse_args()
    
    with h5py.File(args.filename, 'r+') as hf:
        rho         = hf.get('rho')[:] 
        v_r         = hf.get('vr')[:] 
        vt          = hf.get('vt')[:] 
        p           = hf.get('p')[:] 
        rmin        = hf.get('rmin')[:]
        rmax        = hf.get('rmax')[:]
        ttheta_min  = hf.get('theta_min')[:]
        ttheta_min  = hf.get('theta_max')[:]
        tend        = hf.get('t')[:]
        
    ynpts, xnpts = rho.shape 

    if (args.log):
        r = np.logspace(np.log10(rmin), np.log10(rmax), xnpts)
        norm = colors.LogNorm(vmin=rho.min(), vmax=3.*rho.min())
    else:
        r = np.linspace(rmin, rmax, xnpts)
        norm = colors.LinearNorm(vmin=args., vmax=3.*rho.min())
        
    rr, tt = np.meshgrid(r, theta)
    rr, t2 = np.meshgrid(r, theta_mirror)
    theta_mirror = - theta[::-1]
    theta_mirror[-1] *= -1.

    
    
    
    fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)

    tend = t
    c1 = ax.pcolormesh(tt, rr, args.prim, cmap='gist_rainbow', shading='auto', norm = rnorm)
    c2 = ax.pcolormesh(t2[::-1], rr, rho,  cmap='gist_rainbow', shading='auto', norm=rnorm)

    fig.suptitle('SIMBI: {}'.format(args.setup), fontsize=20)

    cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
    cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)
    ax.set_position( [0.1, -0.18, 0.8, 1.43])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.grid(True, alpha=0.95)
    ax.xaxis.grid(True, alpha=0.95)
    ax.tick_params(axis='both', labelsize=10)
    cbaxes.tick_params(axis='x', labelsize=10)
    ax.axes.xaxis.set_ticklabels([])
    ax.set_rmax(0.06)
    ax.set_rmin(r.min())
    ax.set_thetamin(-90)
    ax.set_thetamax(90)


    cbar.ax.set_xlabel('Density', fontsize=20)

    plt.show()
    
if __name__ == "__main__":
    main()
