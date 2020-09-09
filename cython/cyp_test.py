#! /usr/bin/env python

import numpy as np 
import time
import matplotlib.pyplot as plt 

from simbi import Hydro

from state import PyState, PyState2D

u = np.zeros((3,1000))
u2 = np.zeros((4,128,128))
u[0] = 1
u[1] = 2
u[2] = 3 

u2[0] = 1
u2[1] = 2
u2[2] = 3 
u2[3] = 7

gamma = 1.4 
tend = 0.1
N = 500 
dt = 1.e-4

a = PyState(u, gamma)
b = PyState2D(u2, gamma)

fig, axs = plt.subplots(1, 2, figsize=(20,10))

hydro = Hydro(gamma=1.4, initial_state = ((1.0,1.0,0.0),(0.1,0.125,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3)

hydro2 = Hydro(gamma=1.4, initial_state = ((1.0,1.0,0.0),(0.1,0.125,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3)

t1 = time.time()
poll = hydro.simulate(tend=tend, dt=dt, first_order=True)
print("Time for FO Simulation: {} sec".format(time.time() - t1))

t2 = time.time()
bar = hydro2.simulate(tend=tend, dt=dt, first_order=False)
print("Time for SO Simulation: {} sec".format(time.time() - t2))

u = hydro.cons2prim(bar)[2]
v = hydro.cons2prim(poll)[2]

#print(bar[0])
x = np.linspace(0, 1, N)
fig.suptitle("Sod Problem at t = {} with N = {}".format(tend, N))
axs[0].plot(x[::], poll[0][::], '--', fillstyle='none', label='First Order')
axs[0].plot(x, bar[0], label='Higher Order')
axs[0].set_xlabel('X', fontsize=20)
axs[0].set_ylabel('Density', fontsize=20)

axs[1].plot(x[::], v[::],  '--', fillstyle='none', label = 'Frist Order')
axs[1].plot(x, u, label='Higher Order')
axs[1].set_xlabel('X', fontsize=20)
axs[1].set_ylabel('Velocity', fontsize=20)

axs[0].legend()

plt.savefig('plots/sod_cartesian.pdf')
plt.show()
