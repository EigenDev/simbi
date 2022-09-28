#! /usr/bin/env python

# Code to test out the Sedov-Taylor Explosion for an  axissymmetric sphere

import numpy as np 
import matplotlib.pyplot as plt
import time
import argparse
import sys
from pysimbi import Hydro 
from astropy import units as u

if sys.version_info <= (3,9):
    action = 'store_false'
else:
    action = argparse.BooleanOptionalAction
    
def main():
    parser = argparse.ArgumentParser(description='Sedov-Taylor test problem')
    parser.add_argument('--gamma', '-g',      help = 'adbatic gas index', dest='gamma', type=float, default=5/3)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.1)
    parser.add_argument('--npolar', '-n',     help = 'number of polar zones', dest='npolar', type=int, default=512)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.8)
    parser.add_argument('--forder', '-f',     help = 'first order flag', dest='forder', action='store_true', default=False)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--omega',            help = 'density power law index', dest='omega', type=float, default=0.0)
    parser.add_argument('--bc', '-bc',        help = 'boundary condition', dest='boundc', type=str, default='outflow', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/') 
    parser.add_argument('--hllc',             help = 'HLLC flag', dest='hllc', action=action, default=True)
    
    args = parser.parse_args()
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def rho0(n, theta):
        return 1.0 - 0.95*np.cos(n*theta)


    # Constants
    p_init  = 1.e-6
    r_init  = 0.01
    nu      = 3.
    epsilon = 1.

    rho_init = rho0(0, np.pi)
    v_init   = 0.
    ntheta   = args.npolar
    rmin     = 0.01
    rmax     = 10.0
    
    theta_min = 0
    theta_max = np.pi

    theta        = np.linspace(theta_min, theta_max, ntheta)
    theta_mirror = -theta[::-1]

    # Choose xnpts carefully such that the grid zones remain roughly square
    dtheta = theta_max/ntheta
    nr = int(np.ceil(1 + np.log10(rmax/rmin)/dtheta ))

    r = np.logspace(np.log10(rmin), np.log10(rmax), nr) 

    r_right = np.sqrt(r[1:nr] * r[0:nr-1])
    dr = rmin * 1.5 
    
    p_zones = find_nearest(r, dr)[0]
    p_zones = int(p_zones)

    p_c = (args.gamma - 1.)*(3*epsilon/((nu + 1)*np.pi*dr**nu))

    print("Central Pressure:", p_c)
    print("Dimensions: {} x {}".format(ntheta, nr))
    zzz = input("Press any key to continue...")
    rho = np.zeros((ntheta , nr), float)
    rho[:] = 1.0 * r ** (-args.omega)

    p              = p_init * rho 
    p[:, :p_zones] = p_c

    vx = np.zeros_like(p)
    vy = vx.copy()

    tend   = 1.0
    dr     = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    dt     = 0.1 * np.minimum(dr, r[0]*dtheta)

    bm = Hydro(gamma = args.gamma, initial_state=(rho, p, vx, vy), 
                Npts=(nr, ntheta), 
                geometry=((rmin, rmax),(theta_min, theta_max)), 
                n_vars=4, regime="classical", coord_system="spherical")


    sim_params = {
        'tend': args.tend,
        'first_order': args.forder,
        'compute_mode': args.mode,
        'boundary_condition': args.boundc,
        'cfl': args.cfl,
        'hllc': args.hllc,
        'linspace': False,
        'plm_theta': args.plm_theta,
        'data_directory': args.data_dir,
        'chkpt_interval': args.chint
    }
    t1 = (time.time()*u.s).to(u.min)
    sol = bm.simulate(**sim_params)

    print("The 2D Sedov-Taylor Simulation for N = {} took {:.3f}".format(ntheta, (time.time()*u.s).to(u.min) - t1))

    rr, tt = np.meshgrid(r, theta)
    rr, t2 = np.meshgrid(r, theta_mirror)

    fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)
    c1 = ax.pcolormesh(tt, rr, sol[0], cmap='inferno', shading = "auto")
    c2 = ax.pcolormesh(t2[::-1], rr, sol[0], cmap='inferno', shading = "auto")

    fig.suptitle('Spherical Explosion at t={} s on {} x {} grid'.format(args.tend, nr, ntheta), fontsize=15)
    cbar = fig.colorbar(c1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmax(rmax)
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.grid(True, alpha=0.4)
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_thetamin(0)
    ax.set_thetamax(360)

    cbar.ax.set_ylabel('Velocity', fontsize=20)
    plt.show()
    
if __name__ == "__main__":
    main()
