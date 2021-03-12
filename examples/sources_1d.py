#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt 
from simbi_py import * 


gamma = 4/3 
gamma_0 = 50 
eta_0 = 100 
N = 256
rmin = 0.01
rmax = 1.



tend = 1.0

r = np.logspace(np.log10(rmin), np.log10(rmax), N)

rho = np.ones(N)
v   = np.zeros(N)
p   = 1.e-6*rho

S_0 = np.zeros(N)
S_0[np.where(r < 2*rmin)] = 1.e-1
S_r = S_0*(1 - 1/gamma_0**2)
S_D = S_0/eta_0



dt = 1.e-8

fake = Hydro(gamma = gamma, initial_state=(rho, p, v), 
              Npts=N, geometry=(rmin, rmax), n_vars=3, regime="relativistic")

u = fake.simulate(tend=tend, dt=dt, CFL = 0.4, hllc = True, linspace=False, first_order=False, sources=(S_D, S_r, S_0), coordinates=b"spherical")

dens = u[0]
pres = u[1]
vel  = u[2]

fig, axs = plt.subplots(3,1, figsize=(10, 8), sharex=True)
axs[0].semilogx(r, dens)
#axs[0].semilogx(r, rho)
axs[0].set_ylabel('Density', fontsize=15)
axs[0].set_title('Toy 1D Spherical Source Terms at t = {} s and N = {}'.format(tend, N))

axs[1].semilogx(r, vel)
axs[1].set_ylabel('Velocity', fontsize=15)

axs[2].semilogx(r, pres)
axs[2].set_ylabel('Pressure', fontsize=15)
axs[2].set_xlim(rmin, rmax)

plt.subplots_adjust(hspace = 0.1)
fig.savefig('plots/dummmy_sources_spherical.pdf')
plt.show()
