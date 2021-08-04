#! /usr/bin/env python 

import numpy as np 
import time
import matplotlib.pyplot as plt 

from pysimbi_gpu import Hydro
gamma = 5/3 
tend = 0.4
N = 100
dt = 1.e-4
data_dir = "../data/"


hydro2 = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")



h = hydro2.simulate(tend=tend, first_order = True, CFL=0.8, hllc=True, data_directory=data_dir)
u = hydro.simulate(tend=tend, first_order = True, CFL=0.8, hllc=False, data_directory=data_dir)

x = np.linspace(0,1, N)

fig, axs = plt.subplots(1, 2, figsize=(8,8))

axs[0].plot(x, u[0], label="RHLLE")
axs[0].plot(x, h[0], label='RHLLC')
axs[0].set_ylabel("Density", fontsize=30)

N = 400
dt = 1.e-4



hydro2 = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro = Hydro(gamma=gamma, initial_state = ((1.0,1.0,0.9),(1.0,10.0,0.0)),
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")



h = hydro2.simulate(tend=tend, first_order = False, CFL=0.8, hllc=True, data_directory=data_dir)
u = hydro.simulate(tend=tend, first_order = False, CFL=0.8, hllc=False, data_directory=data_dir)

x = np.linspace(0,1, N)

axs[1].plot(x, u[0], label="RHLLE")
axs[1].plot(x, h[0], label='RHLLC')

axs[1].legend(fontsize=20)
fig.suptitle("Mignone & Bodo (2005) Test Problem 1. ")
plt.show()