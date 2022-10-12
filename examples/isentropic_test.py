#! /usr/bin/env python

# Code to test out the convergence of hydro code

import numpy as np 
import matplotlib.pyplot as plt
import argparse
import sys
from pysimbi import Hydro, print_problem_params

if sys.version_info <= (3,9):
    action = 'store_false'
else:
    action = argparse.BooleanOptionalAction
    
alpha_max = 1.0 
alpha_min = 1e-3

def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < alpha_min or f >= alpha_max:
        raise argparse.ArgumentTypeError("Argument must be < " + str(alpha_max) + " and > " + str(alpha_min))
    return f

def main():
    parser = argparse.ArgumentParser(description='Sod Shock Tube Params')
    parser.add_argument('--gamma', '-g',      help = 'adbatic gas index', dest='gamma', type=float, default=5/3)
    parser.add_argument('--tend', '-t',       help = 'simulation end time', dest='tend', type=float, default=0.1)
    parser.add_argument('--npolar', '-n',     help = 'number of polar zones', dest='npolar', type=int, default=512)
    parser.add_argument('--chint',            help = 'checkpoint interval', dest='chint', type=float, default=0.1)
    parser.add_argument('--cfl',              help = 'Courant-Friedrichs-Lewy number', dest='cfl', type=float, default=0.1)
    parser.add_argument('--plm_theta',        help = 'piecewise linear reconstruction parameter', dest='plm_theta', type=float, default=1.5)
    parser.add_argument('--mode', '-m',       help = 'compute mode [gpu,cpu]', dest='mode', type=str, default='cpu', choices=['gpu', 'cpu'])    
    parser.add_argument('--data_dir', '-d',   help = 'data directory', dest='data_dir', type=str, default='data/') 
    parser.add_argument('--hllc',             help = 'HLLC flag', dest='hllc', action=action, default=True)
    parser.add_argument('--alpha',            help = 'wave amplitude', type=range_limited_float_type, default=0.5)
    args = parser.parse_args()
    
    print_problem_params(args, parser)
    zzz = input("Press Enter key to continue...")
    # Define Constants 
    gamma   = args.gamma
    alpha   = args.alpha
    rho_ref = 1.0
    p_ref   = 1.0

    mode = args.mode
    def func(x):
        return np.sin(2*np.pi*x)

    def rho(alpha, x):
        return 1 + alpha*func(x)

    def cs(rho, pressure):
        return np.sqrt(gamma*pressure/rho)

    def pressure(gamma, rho):
        return p_ref*(rho/rho_ref)**gamma

    def velocity(gamma, rho, pressure):
        return 2/(gamma - 1.)*(cs(rho, pressure) - cs(rho_ref, p_ref))

    ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
    rk2 = {}
    rk1 = {}
    for npts in ns:
        x = np.linspace(0, 1, npts, dtype=float)
        r = rho(alpha, x)
        p = pressure(gamma, r)
        v = velocity(gamma, r, p)
        tend = 0.1
        first_o  = Hydro(gamma, initial_state=(r,p,v), Npts=npts, geometry=(0, 1.0))
        second_o = Hydro(gamma, initial_state=(r,p,v), Npts=npts, geometry=(0, 1.0))
        
        cfl = 10.0/npts
        rk1[npts] = first_o.simulate(tend=tend , chkpt_interval=args.chint, plm_theta=args.plm_theta, hllc=args.hllc, first_order=True, boundary_condition='periodic', cfl=cfl,  compute_mode=mode, data_directory=args.data_dir)
        rk2[npts] = second_o.simulate(tend=tend, chkpt_interval=args.chint, plm_theta=args.plm_theta, hllc=args.hllc, first_order=False, boundary_condition='periodic', cfl=cfl, compute_mode=mode, data_directory=args.data_dir)

        
    epsilon = []
    beta = []

    r_sol = rk1[ns[-1]][0]
    s_sol = rk2[ns[-1]][0]


    for idx, key in enumerate(rk2.keys()):
        r_1 = rk1[key][0]
        p_1 = rk1[key][2]
        
        r_2 = rk2[key][0]
        p_2 = rk2[key][2]

        # epsilons for the first/higher order methods
        first_eps = np.sum(np.absolute(p_1 * r_1**(-gamma) - 1.))
        high_eps  = np.sum(np.absolute(p_2 * r_2**(-gamma) - 1.))
        
        # Divide by the reference Npts
        first_ratio = first_eps/ns[idx]
        high_ratio  = high_eps/ns[idx]
        
        epsilon.append(first_ratio)
        beta.append(high_ratio)

    ns = np.array(ns)
    epsilon = np.array(epsilon)
    beta = np.array(beta)

    fig, ax = plt.subplots(1,1,figsize=(15,13))

    # Plot everything except the true N=4096 solution
    ax.loglog(ns, epsilon,'-d', label='First Order')
    ax.loglog(ns, beta,'-s', label='Higher Order')
    ax.loglog(ns, 1/ns,'--', label='$N^{-1}$')
    ax.set_title('T = {} s'.format(tend))
    ax.set_ylabel(r'$\sum 1/N|P/\rho^\gamma - 1|$', fontsize=15)
    ax.set_xlabel('N', fontsize=15)
    ax.legend()
    plt.show()
              
if __name__ == "__main__":
    main()