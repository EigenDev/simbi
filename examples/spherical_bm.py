#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
import matplotlib.pyplot as plt
import time
import argparse
from simbi import Hydro, print_problem_params 
import sys
from astropy import units as u


if sys.version_info <= (3,9):
    action = 'store_false'
else:
    action = argparse.BooleanOptionalAction
    
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
    parser.add_argument('--gamma', '-g',      help = 'adiabatic gas index', dest='gamma', type=float, default=1.4)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.4)
    parser.add_argument('--npolar', '-n',     help = 'number of polar zones', dest='npolar', type=int, default=128)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.1)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/') 
    parser.add_argument('--hllc',             help = 'HLLC flag', dest='hllc', action=action, default=True)
    parser.add_argument('--forder',           help = 'First order flag', action='store_true', default=False)
    parser.add_argument('--bc',               help = 'Boundary condition', default='outflow', type=str, choices=['periodic', 'outflow'])
    parser.add_argument('--e_scale',          help = 'energy scale in units of 1e53 erg', dest='e_scale', type=float, default=1.0)
    parser.add_argument('--omega',            help = 'density power law index', dest='omega', type=float, default=2.0)
    args = parser.parse_args()
    
    print_problem_params(args, parser)
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
    m_rest      = np.sum(rho[:, :p_zones] * dV[:, :p_zones])
    epsilon_exp = exp_scale * m_rest   

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
    zzz = input("Press Enter key to continue...")

    vx = np.zeros((ntheta ,nr))
    vy = np.zeros((ntheta ,nr))

    bm = Hydro(gamma = args.gamma, initial_state=(rho, vx, vy, p), 
                resolution=(nr, ntheta), 
                geometry=((rmin, rmax),(theta_min, theta_max)), 
                regime="relativistic", coord_system="spherical")


    sim_params = {
        'tend': args.tend,
        'first_order': args.forder,
        'compute_mode': args.mode,
        'boundary_condition': args.bc,
        'cfl': args.cfl,
        'hllc': args.hllc,
        'linspace': False,
        'plm_theta': args.plm_theta,
        'data_directory': args.data_dir,
        'chkpt_interval': args.chint,
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
    ax.grid(False)
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