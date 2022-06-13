#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
import matplotlib.pyplot as plt
import time
import argparse
from pysimbi import Hydro 

from astropy import units as u


def volume(r: np.ndarray):
    rcop = r.copy()
    rvertices = np.sqrt(rcop[1:]*rcop[:-1])
    rvertices = np.insert(rvertices, 0, rcop[0])
    rvertices = np.insert(rvertices, rvertices.shape[0], rcop[-1])
    return 4.0 * np.pi * (1.0/3.0) * (rvertices[1:]**3 - rvertices[:-1]**3)


def volume(r: np.ndarray, theta: np.ndarray):
    rcop = r.copy()
    rvertices = np.sqrt(rcop[:, 1:]*rcop[:, :-1])
    rvertices = np.insert(rvertices, 0, rcop[:, 0], axis=1)
    rvertices = np.insert(rvertices, rvertices.shape[-1], rcop[:, -1], axis=1)
    
    tcop      = theta.copy()
    tvertices = 0.5 * (tcop[1:] + tcop[:-1])
    tvertices = np.insert(tvertices, 0, tcop[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], tcop[-1], axis=0)
    dcos      = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])
    
    return 2.0 * np.pi * dcos * (1.0/3.0) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3)

def main():
    parser = argparse.ArgumentParser(description='Mignone and Bodo Test Problem 1/2 Params')
    parser.add_argument('--gamma', '-g',  dest='gamma', type=float, default=4/3)
    parser.add_argument('--tend', '-t',   dest='tend', type=float, default=0.1)
    parser.add_argument('--npolar', '-n', dest='npolar', type=int, default=400)
    parser.add_argument('--chint',        dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',          dest='cfl', type=float, default=0.8)
    parser.add_argument('--forder', '-f', dest='forder', action='store_true', default=False)
    parser.add_argument('--plm',          dest='plm', type=float, default=1.5)
    parser.add_argument('--e_scale',      dest='e_scale', type=float, default=1.0)
    parser.add_argument('--omega',        dest='omega', type=float, default=2.0)
    parser.add_argument('--bc', '-bc',    dest='boundc', type=str, default='reflecting', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',   dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   dest='data_dir', type=str, default='data/') 
    parser.add_argument('--vouter', dest='vouter', default=0.1, type=float, help='velocity of outer voundary in units of c')
    args = parser.parse_args()
    def find_nearest(array, value):
        array = np.asarray(array)
        idx   = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def rho0(n, theta):
        return 1.0 - 0.95*np.cos(n*theta)


    # Constants
    r_init    = 0.01
    exp_scale = args.e_scale

    rho_init = rho0(0, np.pi)
    v_init   = 0.
    ntheta   = args.npolar
    rmin     = 0.01
    rmax     = 10.0
    
    theta_min = 0
    theta_max = np.pi

    theta        = np.linspace(theta_min, theta_max, ntheta)
    theta_mirror = -theta

    # Choose xnpts carefully such that the grid zones remain roughly square
    dtheta = theta_max/ntheta
    nr = int(np.ceil(1 + np.log10(rmax/rmin)/dtheta ))
    r           = np.geomspace(rmin, rmax, nr) 
    rr, thetta  = np.meshgrid(r, theta)
    dV          = volume(rr, thetta)
    dr          = rmin * 1.5
    rho         = np.ones((ntheta , nr), float) * (r / r[0])**(-args.omega)
    chi         = np.zeros_like(rho)
    p_zones     = find_nearest(r, dr)[0]
    epsilon_0   = np.sum(rho[:, :p_zones] * dV[:, :p_zones])
    epsilon_exp = exp_scale * epsilon_0

    p_c              = (args.gamma - 1.) * (epsilon_exp / dV[:, : p_zones].sum())
    jet_angle        = 0.1 
    p                = rho * 1e-6
    p[:, :p_zones]   = p_c 
    chi[:, :p_zones] = 1.0
    
    h      = 1.0 + args.gamma * p / (rho * (args.gamma - 1.0))
    tau    = rho * h - p - rho
    energy = (tau * dV)

    print("Central Pressure:", p_c)
    print("Dimensions: {} x {}".format(ntheta, nr))
    zzz = input("Press any key to continue...")

    rho0 = rho[0, 0]
    r0   = r[0]
    k    = args.omega
    def a(t):
        return 1.0 
    def adot(t):
        return args.vouter                                                  
    
    def rho_outer(r, theta):
        return rho0 * (r / r0)**(-args.omega)
        
    def s1(r, theta):
        return 0.0 
    
    def s2(r, theta):
        return 0.0
    
    def edens(r, theta):
        rho = rho0 * (r / r0)**(-args.omega)
        p   = rho * 1e-6
        h   = 1.0 + p * args.gamma / (rho * (args.gamma - 1))
        return rho * h - p - rho 
    
    vx = np.zeros((ntheta ,nr))
    vy = np.zeros((ntheta ,nr))

    bm = Hydro(gamma = args.gamma, initial_state=(rho, p, vx, vy), 
                Npts=(nr, ntheta), 
                geometry=((rmin, rmax),(theta_min, theta_max)), 
                n_vars=4, regime="relativistic", coord_system="spherical")


    sim_params = {
        'tend': args.tend,
        'first_order': args.forder,
        'compute_mode': args.mode,
        'boundary_condition': args.boundc,
        'scalars': chi,
        'cfl': args.cfl,
        'hllc': True,
        'linspace': False,
        'plm_theta': args.plm,
        'data_directory': args.data_dir,
        'chkpt_interval': args.chint,
        'adot': adot,
        'a': a,
        'dens_outer': rho_outer,
        'mom_outer':  [s1, s2],
        'edens_outer': edens
    }
    t1 = (time.time()*u.s).to(u.min)
    sol = bm.simulate(**sim_params)

    print("The 2D BM Simulation for N = {} took {:.3f}".format(ntheta, (time.time()*u.s).to(u.min) - t1))

    W     = 1/np.sqrt(1 - (sol[1]**2 + sol[2]**2))
    beta  = (1 - 1 / W**2)**0.5
    ufour = W * beta
    rr, tt = np.meshgrid(r, theta)
    rr, t2 = np.meshgrid(r, theta_mirror)

    fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)
    c1 = ax.pcolormesh(tt, rr, ufour, cmap='inferno', shading = "auto")
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