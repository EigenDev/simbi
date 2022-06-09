#! /usr/bin/env python

import numpy as np 
import time
import argparse
import matplotlib.pyplot as plt 

from pysimbi import Hydro

def main():
    parser = argparse.ArgumentParser(description='Mignone and Bodo Test Problem 1/2 Params')
    parser.add_argument('--gamma', '-g',  dest='gamma', type=float, default=1.4)
    parser.add_argument('--tend', '-t',   dest='tend', type=float, default=1.0)
    parser.add_argument('--nzones', '-n', dest='nzones', type=int, default=400)
    parser.add_argument('--chint',        dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',          dest='cfl', type=float, default=0.8)
    parser.add_argument('--forder', '-f', dest='forder', action='store_true', default=False)
    parser.add_argument('--plm',          dest='plm', type=float, default=1.5)
    parser.add_argument('--omega',        dest='omega', type=float, default=0.0)
    parser.add_argument('--bc', '-bc',    dest='boundc', type=str, default='outflow', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',   dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   dest='data_dir', type=str, default='data/') 
    parser.add_argument('--tex', dest='tex', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.tex:
        plt.rc('text', usetex=True)
        
    stationary = ((1.4, 1.0, 0.0), (1.0, 1.0, 0.0))
    fig, ax = plt.subplots(1, 1, figsize=(3.5,2.78))

    hydro = Hydro(gamma=args.gamma, initial_state = stationary,
            Npts=args.nzones, geometry=(0.0,1.0,0.5), n_vars=3)

    hydro2 = Hydro(gamma=args.gamma, initial_state = stationary,
            Npts=args.nzones, geometry=(0.0,1.0,0.5), n_vars=3)

    t1 = time.time()
    poll = hydro.simulate(tend=args.tend, first_order=args.forder, hllc=False, cfl=args.cfl, compute_mode=args.mode, boundary_condition=args.boundc)
    print("Time for HLLE Simulation: {} sec".format(time.time() - t1))

    t2 = time.time()
    bar = hydro2.simulate(tend=args.tend, first_order=args.forder, hllc=True, cfl=args.cfl, compute_mode=args.mode, boundary_condition=args.boundc)
    print("Time for HLLC Simulation: {} sec".format(time.time() - t2))

    u = bar[1]
    v = poll[1]


    x = np.linspace(0, 1, args.nzones)
    fig.suptitle(r"$\rm{{Stationary  \ Wave \  Problem \ at}} \ t = {} \ \rm{{with}} \ N = {}$".format(args.tend, args.nzones))
    ax.plot(x, poll[0], 'r--', fillstyle='none', label='HLLE')
    ax.plot(x, bar [0], 'b', label='HLLC')
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$\rho / \rho_0$')


    ax.legend()
    ax.set_xlim(0, 1)
    fig.savefig('stationary.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()