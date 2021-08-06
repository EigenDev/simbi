#! /usr/bin/env python 

import numpy as np 
import time
import matplotlib.pyplot as plt 

from pysimbi_gpu import Hydro

gamma = 4/3 
tend = 0.4249
N = 2**12
dt = 1.e-6
data_dir = "../data/"

fig, axs = plt.subplots(3, 1, figsize=(15,30), sharex=True)

hydro2 = Hydro(gamma=gamma, initial_state = ((10.0,13.33,0.0),(1.0,1.e-10,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro = Hydro(gamma=gamma, initial_state = ((10.0,13.33,0.0),(1.0,1.e-10,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")



h = hydro2.simulate(tend=tend, first_order=False,  CFL=0.4, hllc=True , data_directory=data_dir)
u = hydro.simulate(tend=tend,  first_order=False,  CFL=0.4, hllc=False, data_directory=data_dir)


x = np.linspace(0, 1, N)
#fig, ax = plt.subplots(1, 1, figsize=(15, 11))

# print("RHLLC Density: {}", h[0])

axs[0].plot(x, u[0], 'o', fillstyle='none', label='RHLLE')
axs[0].plot(x, h[0], label='RHLLC')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].set_ylabel('Density', fontsize=20)

axs[1].plot(x, u[2], 'o', fillstyle='none', label='RHLLE')
axs[1].plot(x, h[2], label='RHLLC')
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].set_ylabel('Velocity', fontsize=20)

axs[2].plot(x, u[1], 'o', fillstyle='none', label='RHLLE')
axs[2].plot(x, h[1], label='RHLLC')
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)
axs[2].set_ylabel('Pressure', fontsize=20)
axs[2].set_xlabel('X', fontsize=20)

axs[0].set_title('1D Relativistic Blast Wave with N={} at t = {} (Marti & Muller 1999, Problem 1)'.format(N, tend), fontsize=20)

fig.subplots_adjust(hspace=0.01)

axs[0].set_xlim(0.0, 1.0)
axs[0].legend(fontsize=15)

del hydro 
del hydro2
plt.show()
#fig.savefig('plots/relativisitc_blast_wave_test_p1.pdf')
