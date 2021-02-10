#! /usr/bin/env python 

import numpy as np 
import time
import matplotlib.pyplot as plt 

from simbi_py import Hydro

from state import PyStateSR

gamma = 4/3 
tend = 0.4
N = 100
dt = 1.e-4



hydro2 = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")



h = hydro2.simulate(tend=tend, first_order = True, CFL=0.4, hllc=True)
u = hydro.simulate(tend=tend, first_order = True, CFL=0.4, hllc=False)

x = np.linspace(0,1, N)

fig, axs = plt.subplots(1, 2, figsize=(8,10))

axs[0].plot(x, u[0], label="RHLLE")
axs[0].plot(x, h[0], label='RHLLC')

N = 400
dt = 1.e-4



hydro2 = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")



h = hydro2.simulate(tend=tend, first_order = False, CFL=0.4, hllc=True)
u = hydro.simulate(tend=tend, first_order = False, CFL=0.4, hllc=False)

x = np.linspace(0,1, N)

axs[1].plot(x, u[0], label="RHLLE")
axs[1].plot(x, h[0], label='RHLLC')

axs[1].legend()
plt.show()