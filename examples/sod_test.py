#! /usr/bin/env python

import numpy as np 
import time
import matplotlib.pyplot as plt 
import argparse 

from pysimbi import Hydro

def main():
    parser = argparse.ArgumentParser(description='Sod Shock Tube Params')
    parser.add_argument('--gamma', '-g',  dest='gamma', type=float, default=1.4)
    parser.add_argument('--tend', '-t',   dest='tend', type=float, default=0.1)
    parser.add_argument('--nzones', '-n', dest='nzones', type=int, default=100)
    parser.add_argument('--chint',        dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',          dest='cfl', type=float, default=0.1)
    parser.add_argument('--forder', '-f', dest='forder', action='store_true', default=False)
    parser.add_argument('--bc', '-bc',    dest='boundc', type=str, default='outflow', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',   dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   dest='data_dir', type=str, default='data/')                     

    args = parser.parse_args()
    sod   = ((1.0,1.0,0.0),(0.1,0.125,0.0))
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    hydro = Hydro(gamma=args.gamma, initial_state = sod,
            Npts=args.nzones, geometry=(0.0,1.0,0.5), n_vars=3)

    hydro2 = Hydro(gamma=args.gamma, initial_state = sod,        
            Npts=args.nzones, geometry=(0.0,1.0,0.5), n_vars=3)
            

    t1 = time.time()
    hlle = hydro.simulate(tend=args.tend, first_order=args.forder, hllc=False, cfl=args.cfl, data_directory=args.data_dir,
                          compute_mode=args.mode, boundary_condition=args.boundc, chkpt_interval=args.chint)
    print("Time for HLLE Simulation: {:.2f} sec".format(time.time() - t1))

    t2 = time.time()
    hllc = hydro2.simulate(tend=args.tend, first_order=args.forder, hllc=True, cfl=args.cfl, data_directory=args.data_dir,
                           compute_mode=args.mode, boundary_condition=args.boundc, chkpt_interval=args.chint)
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