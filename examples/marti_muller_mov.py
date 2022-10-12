#! /usr/bin/env python 

import numpy as np 
import time
import argparse
import matplotlib.pyplot as plt 
import sys
from pysimbi import Hydro

if sys.version_info <= (3,9):
    action = 'store_false'
else:
    action = argparse.BooleanOptionalAction
    
def main():
    parser = argparse.ArgumentParser(description='Marti and Muller Test Problem 1 Params')
    parser.add_argument('--gamma', '-g',      help = 'adbatic gas index', dest='gamma', type=float, default=5/3)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.1)
    parser.add_argument('--nzones', '-n',     help = 'number of x,y zones', dest='nzones', type=int, default=512)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.1)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/') 
    parser.add_argument('--hllc',             help = 'HLLC flag', dest='hllc', action=action, default=True)
    parser.add_argument('--adot',             help = 'speed of moving mesh faces in units of c', dest='adot', type=float, default=0.1)
    parser.add_argument('--forder',           help= 'First order flag', action='store_true', default=False)
    parser.add_argument('--bc',               help= 'Boundary condition', dest='boundc', default='outflow', type=str, choices=['periodic', 'outflow'])
    args = parser.parse_args()
    
    print("\nProblem paramters:\n")
    for arg in vars(args):
        print(f"{str(arg).ljust(30, '.')} {getattr(args, arg)}")
    print("\n")
    
    fig, axs = plt.subplots(3, 1, figsize=(9,9), sharex=True)
    xmin, xmax = 0.0, 1.0 
    xmid       = (xmax - xmin) * 0.5 
    rhoL = 10.0 
    pL   = 13.33 
    vL   = 0.0
    rhoR = 1.0 
    pR   = 1e-10
    vR   = 0.0
    
    def a(t):
        return 1.0
    
    def adot(t):
        return args.adot
    
    def rho(x):
        if x < 1.5:
            return rhoR
        else:
            return rhoR * 0.1
    def s(x):
        return 0.0 
    
    def tau(x):
        rho = rhoR if x < 1.5 else rhoR * 0.1 
        p   = pR if x < 1.5 else pR * 0.1 
        h   = 1.0 + p * (args.gamma) / (rho * (args.gamma - 1))
        return rho * h - p - rho 
    
    hydro2 = Hydro(gamma=args.gamma, initial_state = ((rhoL,pL,vL),(rhoR,pR,vR)),
            Npts=args.nzones, geometry=(xmin,xmax,xmid), n_vars=3, regime="relativistic")

    hydro = Hydro(gamma=args.gamma, initial_state = ((rhoL,pL,vL),(rhoR,pR,vR)),
            Npts=args.nzones, geometry=(xmin,xmax, xmid), n_vars=3, regime="relativistic")
    
    sim_params = {
            'tend': args.tend,
            'first_order': args.forder,
            'compute_mode': args.mode,
            'boundary_condition': args.boundc,
            'cfl': args.cfl,
            'hllc': False,
            'linspace': True,
            'plm_theta': args.plm_theta,
            'data_directory': args.data_dir,
            'chkpt_interval': args.chint,
            'a': a,
            'adot': adot,
            'dens_outer': rho,
            'mom_outer': s,
            'edens_outer': tau
    }
    
    h = hydro2.simulate(**sim_params)
    sim_params['hllc'] = True
    u = hydro.simulate(**sim_params)


    x  = np.linspace(xmin, xmax, args.nzones)
    x += adot(args.tend) / a(args.tend) * args.tend
    axs[0].plot(x, u[0], 'o', fillstyle='none', label='RHLLC')
    axs[0].plot(x, h[0], label='RHLLE')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_ylabel('Density', fontsize=20)

    axs[1].plot(x, u[1], 'o', fillstyle='none', label='RHLLC')
    axs[1].plot(x, h[1], label='RHLLE')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylabel('Velocity', fontsize=20)
    axs[1].set_xlabel('X', fontsize=20)

    axs[2].plot(x, u[2], 'o', fillstyle='none', label='RHLLE')
    axs[2].plot(x, h[2], label='RHLLC')
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].set_ylabel('Pressure', fontsize=20)

    axs[0].set_title('1D Relativistic Blast Wave with N={} at t = {} (Marti & Muller 1999, Problem 1)'.format(args.nzones, args.tend), fontsize=10)
    fig.subplots_adjust(hspace=0.01)

    axs[0].set_xlim(x.min(), x.max())
    axs[0].legend(fontsize=15)

    plt.show()

if __name__ == "__main__":
    main()