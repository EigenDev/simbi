#! /usr/bin/env python 

import numpy as np 
import time
import argparse
import matplotlib.pyplot as plt 

from pysimbi import Hydro

def main():
    parser = argparse.ArgumentParser(description='Marti and Muller Test Problem 1 Params')
    parser.add_argument('--gamma', '-g',  dest='gamma', type=float, default=1.4)
    parser.add_argument('--tend', '-t',   dest='tend', type=float, default=0.4249  )
    parser.add_argument('--nzones', '-n', dest='nzones', type=int, default=264)
    parser.add_argument('--chint',        dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',          dest='cfl', type=float, default=0.1)
    parser.add_argument('--forder', '-f', dest='forder', action='store_true', default=False)
    parser.add_argument('--bc', '-bc',    dest='boundc', type=str, default='outflow', choices=['outflow', 'inflow', 'reflecting', 'periodic'])
    parser.add_argument('--mode', '-m',   dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   dest='data_dir', type=str, default='data/')  
    parser.add_argument('--plm_theta', dest='plm_theta', type=float, help='piecwise linear reconstruction slope', default=1.5)
    args = parser.parse_args()
    fig, axs = plt.subplots(3, 1, figsize=(9,9), sharex=True)
    xmin, xmax = 0.0, 1.0 
    xmid       = (xmax - xmin) * 0.5 
    rhoL = 10.0 
    pL   = 13.33 
    vL   = 0.0
    rhoR = 1.0 
    pR   = 1e-10
    vR   = 0.0
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
    }

    h = hydro2.simulate(**sim_params)
    sim_params['hllc'] = True
    u = hydro.simulate(**sim_params)


    x = np.linspace(xmin, xmax, args.nzones)
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