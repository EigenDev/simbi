#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import time
import argparse
import matplotlib.colors as mcolors
from numpy.random import default_rng
try:
    import cmasher
except:
    print("Can't find CMasher, so defaulting to matplotlib colors")

from pysimbi import Hydro 
from astropy import units as u 

def main():
    parser = argparse.ArgumentParser(description="KH Instability Test")
    parser.add_argument('--gamma', '-g',      help = 'adbatic gas index', dest='gamma', type=float, default=5/3)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.1)
    parser.add_argument('--nzones', '-n',     help = 'number of x,y zones', dest='nzones', type=int, default=512)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.1)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/') 
    parser.add_argument('--hllc',             help = 'HLLC flag', dest='hllc', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cmap', '-c',       help = 'colormap for output plot', dest='cmap', type=str, default='gist_ncar')
    parser.add_argument('--forder',           help= ' First order flag', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    xmin = -0.5
    xmax = 0.5
    ymin = -0.5
    ymax = 0.5

    xnpts = args.nzones
    ynpts = xnpts

    rhoL = 2.0
    vxT  = 0.5
    pL   = 2.5

    rhoR = 1.0
    vxB  = - 0.5
    pR   = 2.5

    x = np.linspace(xmin, xmax, xnpts)
    y = np.linspace(ymin, ymax, ynpts)


    rho = np.zeros(shape=(ynpts, xnpts), dtype= float)
    rho[np.where(np.abs(y) < 0.25)] = rhoL 
    rho[np.where(np.abs(y) > 0.25)] = rhoR

    vx = np.zeros(shape=(ynpts, xnpts), dtype= float)
    vx[np.where(np.abs(y) > 0.25)]  = vxT
    vx[np.where(np.abs(y) < 0.25)]  = vxB

    vy = np.zeros(shape=(ynpts, xnpts), dtype= float)

    p = np.zeros(shape=(ynpts, xnpts), dtype= float)
    p[np.where(np.abs(y) > 0.25)] = pL 
    p[np.where(np.abs(y) < 0.25)] = pR

    # Seed the KH instability with random velocities
    rng     = default_rng()
    sin_arr = 0.01*np.sin(2*np.pi*x)
    vx_rand = rng.choice(sin_arr, size=vx.shape)
    vy_rand = rng.choice(sin_arr, size=vy.shape)

    vx += vx_rand
    vy += vy_rand

    xx, yy = np.meshgrid(x, y)

    fig, ax= plt.subplots(1, 1, figsize=(12,6), constrained_layout=False)

    sim_params = {
            'tend': args.tend,
            'first_order': args.forder,
            'compute_mode': args.mode,
            'boundary_condition': 'periodic',
            'cfl': args.cfl,
            'hllc': args.hllc,
            'linspace': True,
            'plm_theta': args.plm_theta,
            'data_directory': args.data_dir,
            'chkpt_interval': args.chint,
    }
    # HLLC Simulation
    SodHLLC = Hydro(gamma = args.gamma, initial_state=(rho, p, vx, vy), 
                Npts=(xnpts, ynpts), geometry=((xmin, xmax),(ymin, ymax)), n_vars=4)

    t1 = (time.time()*u.s).to(u.min)
    hllc_result = SodHLLC.simulate(**sim_params)

    print("The 2D KH Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))

    rho, vx, vy, pre, chi = hllc_result

    rnorm = mcolors.LogNorm(vmin=0.9, vmax=2.1)
    ax.grid(False)
    c1 = ax.pcolormesh(xx, yy, rho, cmap=args.cmap, edgecolors='none', shading ='auto', vmin=0.9, vmax=2.1)

    fig.suptitle('SIMBI: KH Instability Test at t={} s on {} x {} grid.'.format(args.tend, xnpts, ynpts), fontsize=20)


    cbar = fig.colorbar(c1, orientation='vertical')
    ax.tick_params(axis='both', labelsize=10)
    cbar.ax.set_ylabel('Density', fontsize=20)
    plt.show()

if __name__ == '__main__':
    main()










