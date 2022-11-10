#! /usr/bin/env python 

import numpy as np 
import time
import argparse
import matplotlib.pyplot as plt 
import sys 
from pysimbi import Hydro, print_problem_params

if sys.version_info <= (3,9):
    action = 'store_false'
else:
    action = argparse.BooleanOptionalAction
    
def main():
    parser = argparse.ArgumentParser(description='Mignone and Bodo Test Problem 1/2 Params')
    parser.add_argument('--gamma', '-g',      help = 'adiabatic gas index', dest='gamma', type=float, default=1.4)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.4)
    parser.add_argument('--nzones', '-n',     help = 'number of x,y zones', dest='nzones', type=int, default=100)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.1)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/') 
    parser.add_argument('--hllc',             help = 'HLLC flag', dest='hllc', action=action, default=True)
    parser.add_argument('--forder',           help= 'First order flag', action='store_true', default=False)
    parser.add_argument('--bc',               help= 'Boundary condition', default='outflow', type=str, choices=['periodic', 'outflow'])
    parser.add_argument('--prob2',            help='flag for problem 2', default=False, action='store_true')
    args = parser.parse_args()
    
    print_problem_params(args, parser)
    zzz = input("Press Enter key to continue...")
    if args.prob2:
        rhol = 1 
        rhor = 10 
        vl   = -0.6 
        vr   = 0.5
        pl   = 10
        pr   = 20 
    else:
        rhol = 1  
        rhor = 1  
        vl   = 0.9
        vr   = 0  
        pl   = 1  
        pr   = 10 

    init = ((rhol, vl, pl), (rhor, vr, pr))

    fig, axs = plt.subplots(3, 1, figsize=(9,9), sharex=True)

    hydro2 = Hydro(gamma=args.gamma, initial_state = init,
            resolution=args.nzones, geometry=(0.0,1.0,0.5), regime="relativistic")

    hydro = Hydro(gamma=args.gamma, initial_state = init,
            resolution=args.nzones, geometry=(0.0,1.0,0.5), regime="relativistic")

    sim_params = {
        'tend': args.tend,
        'first_order': args.forder,
        'compute_mode': args.mode,
        'boundary_condition': args.bc,
        'cfl': args.cfl,
        'hllc': True,
        'linspace': True,
        'plm_theta': args.plm_theta,
        'data_directory': args.data_dir,
        'chkpt_interval': args.chint,
    }
    h = hydro2.simulate(**sim_params)
    sim_params['hllc'] = False
    u = hydro.simulate(**sim_params)

    x = np.linspace(0, 1, args.nzones)

    axs[0].plot(x, u[0], 'o', fillstyle='none', label='RHLLE')
    axs[0].plot(x, h[0], label='RHLLC')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_ylabel('Density', fontsize=20)

    axs[1].plot(x, u[1], 'o', fillstyle='none', label='RHLLE')
    axs[1].plot(x, h[1], label='RHLLC')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylabel('Velocity', fontsize=20)
    axs[1].set_xlabel('X', fontsize=20)

    axs[2].plot(x, u[2], 'o', fillstyle='none', label='RHLLE')
    axs[2].plot(x, h[2], label='RHLLC')
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].set_ylabel('Pressure', fontsize=20)

    axs[0].set_title('N={} at t = {} (Mignone & Bodo 2005, Problem {})'.format(args.nzones, args.tend, 2 if args.prob2 else 1), fontsize=10)

    fig.subplots_adjust(hspace=0.01)

    axs[0].set_xlim(0.0, 1.0)
    axs[0].legend(fontsize=15)

    plt.show()

if __name__ == "__main__":
    main()