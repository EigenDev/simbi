#! /usr/bin/env python 

import numpy as np 
import time
import matplotlib.pyplot as plt 

from pysimbi import Hydro

gamma = 4/3 
tend  = 0.4
N     = 400
dt    = 1.e-4
mode  = 'cpu'

rhol = 1  #1 
rhor = 1  #10 
vl   = 0.9#-0.6 
vr   = 0  #0.5
pl   = 1  #10
pr   = 10 #20 

init = ((rhol, pl, vl), (rhor, pr, vr))

fig, axs = plt.subplots(3, 1, figsize=(9,9), sharex=True)

hydro2 = Hydro(gamma=gamma, initial_state = init,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro = Hydro(gamma=gamma, initial_state = init,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")



h = hydro2.simulate(tend=tend, first_order=False,  cfl=0.8, hllc=True , compute_mode=mode, boundary_condition="outflow")
u = hydro.simulate(tend=tend,  first_order=False,  cfl=0.8, hllc=False, compute_mode=mode, boundary_condition="outflow")


x = np.linspace(0, 1, N)

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



axs[0].set_title('N={} at t = {} (Mignone & Bodo 2005, Problem 1)'.format(N, tend), fontsize=10)

fig.subplots_adjust(hspace=0.01)

axs[0].set_xlim(0.0, 1.0)
axs[0].legend(fontsize=15)

plt.show()