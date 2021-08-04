#! /usr/bin/env python

# Code to test out the convergence of hydro code

import numpy as np 
import matplotlib.pyplot as  plt
from pysimbi import Hydro

# Define Constants 
gamma = 5/3
alpha = 0.5 
rho_ref = 1.0
p_ref = 2.0

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
sims = {}
results = {}
for npts in ns:
    x = np.linspace(0, 1, npts, dtype=float)
    r = rho(alpha, x)
    p = pressure(gamma, r)
    #p = np.ones(npts, float)
    #v = np.zeros(npts, float)
    #v[: ] = 0.4
    
    v = velocity(gamma, r, p)
    v *= -1.
    #print(v)
    #zzz = input('')
    
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
    
    results[npts] = first_o.simulate(tend=tend,dt=dt, periodic=True)
    sims[npts] = second_o.simulate(tend=tend,dt=dt, first_order=False, periodic=True)
    
    # f_p = first_o.cons2prim(results[npts])
    # s_p = second_o.cons2prim(sims[npts])
    
    #h = 1/npts
    #l1 = h*np.sum(np.absolute(sims[npts][1]/(sims[npts][0]**gamma) - K))
    #print(l1)
    #zzz = input('')
    #fig, ax = plt.subplots(1,1, figsize=(15,11))
    #
    #ax.plot(x, results[npts][1],'rs',fillstyle='none', label='First Order')
    #ax.plot(x, sims[npts][1], 'b-', label='Higher Order' )
    ## ax.plot(x, p/(r**gamma), '.')
    ## ax.plot(x, r, 'g--', label='Initial')
    #ax.set_xlim(x[0], x[-1])
    #ax.set_xlabel('X', fontsize=15)
    #ax.set_ylabel('Density', fontsize=15)
    #ax.legend()
    #plt.show()
    
    # pause = input('Save Figure? (y/n)')
    # if pause == 'y':
    #     fig.savefig('density_{}.pdf'.format(npts))
    #plt.clf()
    
epsilon = []
beta = []

r_sol = results[ns[-1]][0]
s_sol = sims[ns[-1]][0]


for idx, key in enumerate(sims.keys()):
    r_1 = results[key][0]
    p_1 = results[key][1]
    
    # p_1 = pressure(gamma, r_1)
    
    r_2 = sims[key][0]
    p_2 = sims[key][1]
    # p_2 = pressure(gamma, r_2)
    
    exp = results[key][0]
    exp2 = sims[key][0]
    
    # Slice points to divvy up solution
    # arrays to match length of N < N_max values
    s_1 = int(ns[-1]/exp.size)
    s_2 = int(ns[-1]/exp2.size)
    
    # True Solutions Divided up evenly
    r_ref = r_sol[::s_1]
    s_ref = s_sol[::s_2]
    
    
    # epsilons for the first/higher order methods
    first_eps = np.sum(np.absolute(p_1*r_1**(-gamma) - K))
    high_eps = np.sum(np.absolute(p_2*r_2**(-gamma) - K))
    
    
    
    # Divide by the reference Npts
    first_ratio = first_eps/ns[idx]
    high_ratio = high_eps/ns[idx]
    
    # print('First Order Eps: {}'.format(first_ratio))
    # print('Second Order Eps: {}'.format(high_ratio))
    # zzz = input('')
    
    epsilon.append(first_ratio)
    beta.append(high_ratio)

ns = np.array(ns)
epsilon = np.array(epsilon)
beta = np.array(beta)

fig, ax = plt.subplots(1,1,figsize=(15,13))

# Plot everything except the true N=4096 solution
ax.loglog(ns[: -1], epsilon[: -1],'-d', label='First Order')
ax.loglog(ns[: -1], beta[: -1],'-s', label='Higher Order')
ax.loglog(ns[: -1], 1/ns[: -1],'--', label='$N^{-1}$')
ax.set_title(r"""Relativistic isentropic wave at t = {} s""".format(tend), fontsize=20)
ax.set_ylabel(r'$\sum 1/N|P/\rho^\gamma - K|$', fontsize=20)
ax.set_xlabel('N', fontsize=15)
ax.legend()
plt.savefig('isentropic_rel_converge.pdf')
plt.show()
              