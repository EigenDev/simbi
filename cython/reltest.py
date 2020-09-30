#! /usr/bin/env python 

import numpy as np 
import time
import matplotlib.pyplot as plt 

from simbi import Hydro

from state import PyStateSR

u = np.zeros((3,1000))
u2 = np.zeros((4,128,128))
u[0] = 1
u[1] = 2
u[2] = 3 

u2[0] = 1
u2[1] = 2
u2[2] = 3 
u2[3] = 7

gamma = 5/3 
tend = 0.4249
N = 264
dt = 1.e-4

fig, axs = plt.subplots(3, 1, figsize=(15,30), sharex=True)

hydro = Hydro(gamma=gamma, initial_state = ((10.0,13.33,0.0),(1.0,1.e-10,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro2 = Hydro(gamma=gamma, initial_state = ((10.0,13.33,0.0),(1.0,1.e-10,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

u = hydro.simulate(tend=tend, first_order=True)
h = hydro2.simulate(tend=tend, first_order=False)

x = np.linspace(0, 1, N)
#fig, ax = plt.subplots(1, 1, figsize=(15, 11))

axs[0].plot(x, u[0], 'o', fillstyle='none', label='First Order')
axs[0].plot(x, h[0], label='Higher Order')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].set_ylabel('Density', fontsize=30)

axs[1].plot(x, u[2], 'o', fillstyle='none', label='First Order')
axs[1].plot(x, h[2], label='Higher Order')
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_ylabel('Velocity', fontsize=30)

axs[2].plot(x, u[1], 'o', fillstyle='none', label='First Order')
axs[2].plot(x, h[1], label='Higher Order')
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)
axs[2].set_ylabel('Pressure', fontsize=30)
axs[2].set_xlabel('X', fontsize=30)

axs[0].set_title('1D Relativistic Blast Wave with N={} at t = {} (Marti & Muller 1999, Problem 1)'.format(N, tend), fontsize=24)

fig.subplots_adjust(hspace=0.01)

axs[0].legend(fontsize=30)
plt.show()
fig.savefig('plots/relativisitc_blast_wave_test_p1.pdf')
