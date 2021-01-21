#! /usr/bin/env python

import numpy as np 
import time
import matplotlib.pyplot as plt 

from simbi import Hydro

from state import PyState, PyState2D

gamma = 1.4 
tend = 0.1
N = 200
dt = 1.e-4


stationary = ((1.4, 1.0, 0.0), (1.0, 1.0, 0.0))
sod = ((1.0,1.0,0.0),(0.1,0.125,0.0))
fig, axs = plt.subplots(1, 2, figsize=(20,10), sharex=True)

hydro = Hydro(gamma=1.4, initial_state = sod,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3)

hydro2 = Hydro(gamma=1.4, initial_state = sod,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3)

t1 = time.time()
poll = hydro.simulate(tend=tend, dt=dt, first_order=False, hllc=False)
print("Time for HLLE Simulation: {} sec".format(time.time() - t1))

t2 = time.time()
bar = hydro2.simulate(tend=tend, dt=dt, first_order=False, hllc=True)
print("Time for HLLC Simulation: {} sec".format(time.time() - t2))

u = hydro.cons2prim(bar)[2]
v = hydro.cons2prim(poll)[2]

#print(bar[0])
x = np.linspace(0, 1, N)
fig.suptitle("Stationary Wave Problem at t = {} with N = {}".format(tend, N))
axs[0].plot(x[::], poll[0][::], 'r--', fillstyle='none', label='HLLE')
axs[0].plot(x, bar[0], 'b', label='HLLC')
axs[0].set_xlabel('X', fontsize=20)
axs[0].set_ylabel('Density', fontsize=20)

axs[1].plot(x[::], v[::],  'r--', fillstyle='none', label = 'HLLE')
axs[1].plot(x, u, 'b', label='HLLC')
axs[1].set_xlabel('X', fontsize=20)
axs[1].set_ylabel('Velocity', fontsize=20)


axs[0].legend()
axs[0].set_xlim(0, 1)
#plt.savefig('plots/hllc_hll_comparison.pdf')
plt.show()
