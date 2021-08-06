#! /usr/bin/env python

# Code to test out the convergence of hydro code

import numpy as np 
import matplotlib.pyplot as  plt
from pysimbi import Hydro

# Define Constants 
gamma   = 4/3
alpha   = 0.5 
rho_ref = 1.0
p_ref   = 1.0

K = p_ref*rho_ref**(-gamma)


def func(x):
    return np.sin(2*np.pi*x)

def rho(alpha, x):
    return 2.0 + alpha*func(x)

def cs(rho, pressure):
    return np.sqrt(gamma*pressure/rho)

def pressure(gamma, rho):
    return p_ref*(rho/rho_ref)**gamma

def velocity(gamma, rho, pressure):
    a = np.sqrt(gamma - 1) + cs(rho, pressure)
    b = np.sqrt(gamma - 1) - cs(rho, pressure)
    
    c = np.sqrt(gamma - 1) + cs(rho_ref, p_ref)
    d = np.sqrt(gamma - 1) - cs(rho_ref, p_ref)
    upsilon = 2/(np.sqrt(gamma - 1))
    x = np.sign(a/b) * np.abs(a/b)**upsilon
    y = np.sign(c/d)*np.abs(c/d)**(-upsilon)
    #print(x)
    xi = x*y
    
    return (xi - 1)/(xi + 1)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
rk2 = {}
rk1 = {}
for npts in ns:
    x = np.linspace(0, 1, npts, dtype=float)
    r = rho(alpha, x)
    p = pressure(gamma, r)
    v = velocity(gamma, r, p)
    v *= -1.
    
    # Get velocity at center of the wave
    center, coordinate = find_nearest(x, 0.5)
    v_wave = v[center]
    lx = x[-1] - x[0]
    dx = lx/npts
    dt = 1.e-4
    
    tend = 0.1
    
    #print('End Time:', tend)
    first_o = Hydro(gamma, initial_state=(r,p,v), Npts=npts, geometry=(0, 1.0), regime="relativistic")
    second_o = Hydro(gamma, initial_state=(r,p,v), Npts=npts, geometry=(0, 1.0), regime="relativistic")
    
    rk1[npts] = first_o.simulate(tend=tend,dt=dt, periodic=True)
    rk2[npts] = second_o.simulate(tend=tend,dt=dt, first_order=False, periodic=True)

    
epsilon = []
beta = []

r_sol = rk1[ns[-1]][0]
s_sol = rk2[ns[-1]][0]


for idx, key in enumerate(rk2.keys()):
    r_1 = rk1[key][0]
    p_1 = rk1[key][2]

    r_2 = rk2[key][0]
    p_2 = rk2[key][2]
    exp = rk1[key][0]
    exp2 = rk2[key][0]
    
    # Slice points to divvy up solution
    # arrays to match length of N < N_max values
    s_1 = int(ns[-1]/exp.size)
    s_2 = int(ns[-1]/exp2.size)
    
    # epsilons for the first/higher order methods
    first_eps = np.sum(np.absolute(p_1*r_1**(-gamma) - 1.0))
    high_eps = np.sum(np.absolute(p_2*r_2**(-gamma) - 1.0))
    
    # Divide by the reference Npts
    first_ratio = first_eps/ns[idx]
    high_ratio = high_eps/ns[idx]
    
    epsilon.append(first_ratio)
    beta.append(high_ratio)

ns = np.array(ns)
epsilon = np.array(epsilon)
beta = np.array(beta)

fig, ax = plt.subplots(1,1,figsize=(15,13))

# Plot everything except the true N=4096 solution
ax.loglog(ns[: -1], epsilon[: -1],'-d', label='First Order')
ax.loglog(ns[: -1], beta[: -1],'-s', label='Higher Order')
# ax.loglog(ns[: -1], 1/ns[: -1],'--', label='$N^{-1}$')
ax.set_title(r"""Relativistic isentropic wave at t = {} s""".format(tend), fontsize=20)
ax.set_ylabel(r'$\sum 1/N|P/\rho^\gamma - K|$', fontsize=20)
ax.set_xlabel('N', fontsize=15)
ax.legend()
plt.show()
              