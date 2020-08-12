#!/usr/bin/env python


import numpy as np 
import matplotlib.pyplot as plt 
import math 
from simbi import Hydro 

gamma = 5/3 
epsilon = 1.
nu = 3.
r_min = 0.1
r_max = 1.
N = 1100
dr = 0.01
pressure_zone = int(dr*N)

p_init = 3*(gamma-1.)*epsilon/((nu + 1)*np.pi*dr**nu)

p = np.zeros(N+1, float)
p[0: pressure_zone] = p_init
p[pressure_zone: ] = 1.e-5
rho = np.ones(N+1, float)
v = np.zeros(N+1, float)

tend = 0.003
dt = 1.e-6

sod = (1.0,1.0,0.0),(0.1,0.125,0.0)
sedov = (rho, p, v)

# Object used for the linearly spaced grid
sedov = Hydro(gamma = gamma, initial_state=(rho, p ,v), 
              Npts=N+1, geometry=(r_min, r_max), n_vars=3)

# Idem for the log-spaced grid
#sedov2 = Hydro(gamma = gamma, initial_state=(rho, p, v), 
#              Npts=N+1, geometry=(r_min, r_max), n_vars=3)

# Simulate with linearly-spaced radial zones. (Calls Simulate1D in the hydro.cpp file)
u = sedov.simulate(tend=tend, first_order=False, dt=dt, linspace = True)
# get the pressure and velocity
p, v = sedov.cons2prim(u)[1: ]

# Plot stuff
fig, ax = plt.subplots(1, 1, figsize=(13,11))

r = np.linspace(r_min, r_max, N+1)
ax.plot(r, u[0], '-', label='Density')

#o = sedov2.simulate(tend=tend, first_order=False, dt=dt, linspace = False)
#r = np.logspace(np.log10(r_min), np.log10(r_max), N+1)

#ax.plot(r, o[0], 'bo', fillstyle='none', label='Log Spacing')

#ax.plot(r, v, label='Velocity')
#ax.plot(r, p, label='Pressure')

ax.set_xlabel("R", fontsize=15)
ax.set_title("1D Sedov after t={} s at N = {}".format(tend, N), fontsize=20)
ax.legend()
fig.savefig("Sedov_spherical.pdf")
plt.show()
