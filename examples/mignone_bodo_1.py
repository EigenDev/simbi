#! /usr/bin/env python 

import numpy as np 
import time
import argparse
import matplotlib.pyplot as plt 

from pysimbi import Hydro

def main():
    parser = argparse.ArgumentParser(description='Mignone and Bodo Test Problem 1/2 Params')
    parser.add_argument('--gamma', '-g',  dest='gamma', type=float, default=1.4)
    parser.add_argument('--tend', '-t',   dest='tend', type=float, default=0.4)
    parser.add_argument('--nzones', '-n', dest='nzones', type=int, default=400)
    parser.add_argument('--chint',        dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',          dest='cfl', type=float, default=0.8)
    parser.add_argument('--forder', '-f', dest='forder', action='store_true', default=False)
    parser.add_argument('--prob2',        dest='prob2', action='store_true', default=False)
    parser.add_argument('--bc', '-bc',    dest='boundc', type=str, default='outflow', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',   dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   dest='data_dir', type=str, default='data/') 
    
    args = parser.parse_args()

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

    init = ((rhol, pl, vl), (rhor, pr, vr))

    fig, axs = plt.subplots(3, 1, figsize=(9,9), sharex=True)

    hydro2 = Hydro(gamma=args.gamma, initial_state = init,
            Npts=args.nzones, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

    hydro = Hydro(gamma=args.gamma, initial_state = init,
            Npts=args.nzones, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

    h = hydro2.simulate(tend=args.tend, first_order=args.forder,  cfl=args.cfl, hllc=True , compute_mode=args.mode, boundary_condition=args.boundc)
    u = hydro.simulate(tend=args.tend,  first_order=args.forder,  cfl=args.cfl, hllc=False, compute_mode=args.mode, boundary_condition=args.boundc)

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