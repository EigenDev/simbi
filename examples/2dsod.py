#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import time
import argparse
import sys
from pysimbi import Hydro, print_problem_params 
from astropy import units as u

if sys.version_info <= (3,9):
    action = 'store_false'
else:
    action = argparse.BooleanOptionalAction


def main():
    parser = argparse.ArgumentParser(description='2D Sod Shock Tube Params')
    parser.add_argument('--gamma', '-g',      help = 'adiabatic gas index', dest='gamma', type=float, default=5/3)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.1)
    parser.add_argument('--npolar', '-n',     help = 'number of polar zones', dest='npolar', type=int, default=512)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.1)
    parser.add_argument('--forder', '-f',     help = 'first order flag', dest='forder', action='store_true', default=False)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--bc', '-bc',        help = 'boundary condition', type=str, default='outflow', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/')     
    args = parser.parse_args()
    
    print_problem_params(args, parser)
    zzz = input("Press Enter key to continue...")
    
    theta_min = 0
    theta_max = np.pi/2
    rmin  = 0.1
    rmax  = 1.1
    rmid  = (rmax - rmin) / 2
    ynpts = args.npolar 

    # Choose dtheta carefully such that the grid zones remain roughly square
    dtheta = (theta_max - theta_min) / ynpts
    xnpts  = int(1 + np.log10(rmax/rmin)/dtheta)

    rhoL = 1.0
    vL   = 0.0
    pL   = 1.0

    rhoR = 0.125
    vR   = 0.0
    pR   = 0.1

    r = np.geomspace(rmin, rmax, xnpts)
    r1 = r[1]
    r0 = r[0]
    ri = (r1*r0)**0.5

    theta = np.linspace(theta_min, theta_max, ynpts)

    rho = np.zeros(shape=(ynpts, xnpts), dtype= float)
    rho[:,r < rmid] = rhoL 
    rho[:,r > rmid] = rhoR

    p   = np.zeros(shape=(ynpts, xnpts), dtype= float)
    p[:,r < rmid] = pL 
    p[:,r > rmid] = pR

    vr = np.zeros(shape=(ynpts, xnpts), dtype= float)
    vt = np.zeros(shape=(ynpts, xnpts), dtype= float)

    print("Dim: {}x{}".format(ynpts, xnpts))

    sim_params = {
            'tend': args.tend,
            'first_order': args.forder,
            'compute_mode': args.mode,
            'boundary_condition': args.bc,
            'cfl':  args.cfl,
            'hllc': False,
            'linspace': False,
            'plm_theta': args.plm_theta,
            'data_directory': args.data_dir,
            'chkpt_interval': args.chint,
    }
    
    SodHLLE = Hydro(gamma = args.gamma, initial_state=(rho, p, vr, vt), regime="classical", coord_system="spherical",
                dimensions=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4)

    t1 = (time.time()*u.s).to(u.min)
    hlle_result = SodHLLE.simulate(**sim_params)


    # HLLC Simulation
    SodHLLC = Hydro(gamma = args.gamma, initial_state=(rho, p, vr, vt), regime="classical", coord_system="spherical",
                dimensions=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4)

    sim_params['hllc'] = True 
    t1 = (time.time()*u.s).to(u.min)
    hllc_result = SodHLLC.simulate(**sim_params)

    print("The 2D SOD Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))

    rhoE = hlle_result[0]
    rhoC = hllc_result[0]

    # Plot Radial Density at Theta = 0
    plt.semilogx(r, rhoE[0], label='HLLE')
    plt.semilogx(r, rhoC[0], label='HLLC')

    plt.xlim(r[0], r[-1])
    plt.xlabel('R')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()