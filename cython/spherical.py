#!/usr/bin/env python


import numpy as np 
import matplotlib.pyplot as plt 
import math 
from simbi import Hydro 
import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


gamma = 5/3 
epsilon = 1.
p_amb = 1.e-5
rho_amb = 1.
mach = 1000.
cs = np.sqrt(gamma*p_amb/rho_amb)
v_exp = mach*cs 
nu = 3.

r_min = 0.1
r_max = 1.
N = 1100
dr = 0.01
pressure_zone = int(dr*N)

p_form = 3*(gamma-1.)*epsilon/((nu+1)*np.pi*dr**3)
p_init = (gamma-1.)*rho_amb*v_exp**2

v_exp = np.sqrt(p_form/((gamma-1.)*rho_amb))
p = np.zeros(N+1, float)
p[0: pressure_zone] = p_form
p[pressure_zone: ] = p_amb
rho = np.ones(N+1, float)
v = np.zeros(N+1, float)

r_explosion = 0.9/0.1
tend = r_explosion/(v_exp)
tend = round(tend, 2)
#print("Tend: {:.2f}".format(tend))
#tend = 0.02

dt = 1.e-6

sod = (1.0,1.0,0.0),(0.1,0.125,0.0)
sedov = (rho, p, v)

# Object used for the linearly spaced grid
sedov = Hydro(gamma = gamma, initial_state=(rho, p ,v), 
              Npts=N+1, geometry=(r_min, r_max), n_vars=3)


# Simulate with linearly-spaced radial zones
u = sedov.simulate(tend=tend, first_order=False, dt=dt, linspace = False)

# get the pressure and velocity
p, v = sedov.cons2prim(u)[1: ]

# Plot stuff
fig, ax = plt.subplots(1, 1, figsize=(13,11))

p_bar = p/np.max(p)
v_bar = v/np.max(v)

r = np.logspace(np.log10(r_min), np.log10(r_max), N+1)
ax.plot(r, u[0], '-', label='Density')
ax.plot(r, 2*v_bar, label=r'$2 \times v$')
ax.plot(r, 4*p_bar, label=r'$4 \times p$')

# Make the plot pretty
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(r_min, r_max)
ax.set_xlabel("R", fontsize=15)
ax.set_title("1D Sedov after t={:.3f} s at N = {}".format(tend, N), fontsize=20)
ax.legend()
fig.savefig("Sedov_spherical_3.pdf")
plt.show()