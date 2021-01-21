#! /usr/bin/env python 

import numpy as np 
import time
import matplotlib.pyplot as plt 

from simbi import Hydro

from state import PyStateSR

gamma = 4/3 
tend = 1.0
N = 100
dt = 1.e-4


stationary = ((1.4, 1.0, 0.0), (1.0, 1.0, 0.0))
hydro2 = Hydro(gamma=gamma, initial_state = stationary,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")

hydro = Hydro(gamma=gamma, initial_state = stationary,
        Npts=N, geometry=(0.0,1.0,0.5), n_vars=3, regime="relativistic")



h = hydro2.simulate(tend=tend, first_order=False, CFL=0.4, hllc=True)
u = hydro.simulate(tend=tend, first_order=False, CFL=0.4, hllc=False)


x = np.linspace(0, 1, N)
fig, ax = plt.subplots(1, 1, figsize=(15, 11))

# print("RHLLC Density: {}", h[0])

ax.plot(x, u[0], 'o', fillstyle='none', label='RHLLE')
ax.plot(x, h[0], label='RHLLC')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Density', fontsize=20)


ax.set_title('1D Relativistic Stationary Wave with N={} at t = {}'.format(N, tend), fontsize=20)

fig.subplots_adjust(hspace=0.01)

ax.set_xlim(0.0, 1.0)
ax.legend(fontsize=15)

del hydro 
del hydro2
plt.show()
#fig.savefig('plots/relativisitc_blast_wave_test_p1.pdf')


