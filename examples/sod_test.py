#! /usr/bin/env python

import numpy as np 
import time
import matplotlib.pyplot as plt 
import argparse 
import sys 
from pysimbi import Hydro, print_problem_params
    
def main():
    parser = argparse.ArgumentParser(description='Sod Shock Tube Params')
    parser.add_argument('--gamma', '-g',      help = 'adiabatic gas index', dest='gamma', type=float, default=1.4)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.4)
    parser.add_argument('--nzones', '-n',     help = 'number of x,y zones', dest='nzones', type=int, default=100)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.1)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/') 
    parser.add_argument('--forder',           help= 'First order flag', action='store_true', default=False)
    parser.add_argument('--bc',               help= 'Boundary condition', default='outflow', type=str, choices=['periodic', 'outflow'])
    args = parser.parse_args()
    
    print_problem_params(args, parser)
    zzz = input("Press Enter key to continue...")
    sod   = ((1.0,0.0,1.0),(0.1, 0.0, 0.125))
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    hydro = Hydro(gamma=args.gamma, initial_state = sod,
            resolution=args.nzones, geometry=(0.0,1.0,0.5))

    hydro2 = Hydro(gamma=args.gamma, initial_state = sod,        
            resolution=args.nzones, geometry=(0.0,1.0,0.5))
            

    sim_params = {
        'tend': args.tend,
        'first_order': args.forder,
        'compute_mode': args.mode,
        'boundary_condition': args.bc,
        'cfl': args.cfl,
        'hllc': False,
        'linspace': True,
        'plm_theta': args.plm_theta,
        'data_directory': args.data_dir,
        'chkpt_interval': args.chint,
    }
    t1 = time.time()
    hlle = hydro.simulate(**sim_params)
    print("Time for HLLE Simulation: {:.2f} sec".format(time.time() - t1))

    t2 = time.time()
    sim_params['hllc'] = True
    hllc = hydro2.simulate(**sim_params)
    print("Time for HLLC Simulation: {:.2f} sec".format(time.time() - t2))

    u = hllc[1]
    v = hlle[1]


    x = np.linspace(0, 1, args.nzones)
    fig.suptitle("Sod Shock Tube Problem at t = {} with N = {}".format(args.tend, args.nzones))
    ax.plot(x, hlle[0], 'r--', fillstyle='none', label='HLLE')
    ax.plot(x, hllc [0], 'b', label='HLLC')
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)


    ax.legend()
    ax.set_xlim(0, 1)
    plt.show()

if __name__ == '__main__':
    main()