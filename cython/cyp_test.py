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

a = PyState(u, gamma)
b = PyState2D(u2, gamma)

hydro = Hydro(gamma=1.4, initial_state = ((1.0,1.0,0.0),(0.1,0.125,0.0)),
        Npts=500, geometry=(0.0,1.0,0.5), n_vars=3)

hydro2 = Hydro(gamma=1.4, initial_state = ((1.0,1.0,0.0),(0.1,0.125,0.0)),
        Npts=500, geometry=(0.0,1.0,0.5), n_vars=3)

t1 = time.time()
poll = hydro.simulate(first_order=True)
print("Time for FO Simulation: {} sec".format(time.time() - t1))

t2 = time.time()
bar = hydro2.simulate(first_order=False)
print("Time for SO Simulation: {} sec".format(time.time() - t2))

#print(bar[0])

plt.plot(poll[0], 'o', fillstyle='none', label='First Order')
plt.plot(bar[0], label='Higher Order')
plt.legend()
plt.show()
