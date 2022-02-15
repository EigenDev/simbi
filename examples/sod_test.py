#! /usr/bin/env python

import numpy as np 
import time
import matplotlib.pyplot as plt 

from pysimbi import Hydro

gamma = 1.4
tend  = 0.1
N     = 256
dt    = 1.e-4
mode  = 'gpu'

sod   = ((1.0,1.0,0.0),(0.1,0.125,0.0))
fig, ax = plt.subplots(1, 1, figsize=(10,10))

hydro = Hydro(gamma=gamma, initial_state = sod,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3)

hydro2 = Hydro(gamma=gamma, initial_state = sod,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3)


t1 = time.time()
hlle = hydro.simulate(tend=tend, dt=dt, first_order=False, hllc=False, cfl=0.4, compute_mode=mode, boundary_condition='outflow')
print("Time for HLLE Simulation: {:.2f} sec".format(time.time() - t1))

t2 = time.time()
hllc = hydro2.simulate(tend=tend, dt=dt, first_order=False, hllc=True, cfl=0.4, compute_mode=mode, boundary_condition='outflow')
print("Time for HLLC Simulation: {:.2f} sec".format(time.time() - t2))

u = hllc[1]
v = hlle[1]


x = np.linspace(0, 1, N)
fig.suptitle("Sod Shock Tube Problem at t = {} with N = {}".format(tend, N))
ax.plot(x, hlle[0], 'r--', fillstyle='none', label='HLLE')
ax.plot(x, hllc [0], 'b', label='HLLC')
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Density', fontsize=20)


ax.legend()
ax.set_xlim(0, 1)
plt.show()
