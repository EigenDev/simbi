#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
import matplotlib.pyplot as plt
import time
import argparse
from pysimbi import Hydro 

from astropy import units as u


def main():
    parser = argparse.ArgumentParser(description='Mignone and Bodo Test Problem 1/2 Params')
    parser.add_argument('--gamma', '-g',  dest='gamma', type=float, default=4/3)
    parser.add_argument('--tend', '-t',   dest='tend', type=float, default=0.1)
    parser.add_argument('--npolar', '-n', dest='npolar', type=int, default=400)
    parser.add_argument('--chint',        dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',          dest='cfl', type=float, default=0.8)
    parser.add_argument('--forder', '-f', dest='forder', action='store_true', default=False)
    parser.add_argument('--plm',          dest='plm', type=float, default=1.5)
    parser.add_argument('--omega',        dest='omega', type=float, default=0.0)
    parser.add_argument('--bc', '-bc',    dest='boundc', type=str, default='reflecting', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',   dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dr', '-d',   dest='data_dir', type=str, default='data/') 
    
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
    rmax     = 1.0
    
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


    p              = p_init 
    p[:, :p_zones] = p_c 

    vx = np.zeros((ntheta ,nr), np.double)
    vy = np.zeros((ntheta ,nr), np.double)

    bm = Hydro(gamma = args.gamma, initial_state=(rho, p, vx, vy), 
                Npts=(nr, ntheta), 
                geometry=((rmin, rmax),(theta_min, theta_max)), 
                n_vars=4, regime="relativistic", coord_system="spherical")


    t1 = (time.time()*u.s).to(u.min)
    sol = bm.simulate(tend=args.tend, first_order= args.forder, compute_mode=args.mode, boundary_condition=args.boundc,
                        cfl=args.cfl, hllc=True, linspace=False, plm_theta=args.plm, data_directory=args.data_dir, chkpt_interval=args.chint)

    print("The 2D BM Simulation for N = {} took {:.3f}".format(ntheta, (time.time()*u.s).to(u.min) - t1))

    W     = 1/np.sqrt(1 - (sol[1]**2 + sol[2]**2))
    beta  = (1 - 1 / W**2)**0.5
    ufour = W * beta
    rr, tt = np.meshgrid(r, theta)
    rr, t2 = np.meshgrid(r, theta_mirror)

    fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)
    c1 = ax.pcolormesh(tt, rr, ufour, cmap='inferno', shading = "aufourto")
    c2 = ax.pcolormesh(t2[::-1], rr, ufour, cmap='inferno', shading = "auto")

    fig.suptitle('Spherical Explosion at t={} s on {} x {} grid'.format(args.tend, nr, ntheta), fontsize=15)
    # ax.set_title(r'$\rho(\theta) = 1.0 - 0.95\cos(n \ \theta)$ with n = {}'.format(n), fontsize=10)
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

    cbar.ax.set_ylabel('4-Velocity', fontsize=20)
    plt.show()

if __name__ == "__main__":
    main()