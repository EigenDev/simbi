#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import time
from pysimbi import Hydro 
from astropy import units as u, constants as const


gamma = 5/3
theta_min = 0
theta_max = np.pi/2
rmin  = 0.1
rmax  = 1.1
rmid  = (rmax - rmin) / 2
ynpts = 256 

# Choose dtheta carefully such that the grid zones remain roughly square
dtheta = (theta_max - theta_min) / ynpts
xnpts = int(1 + np.log10(rmax/rmin)/dtheta)

rhoL = 1.0
vL   = 0.0
pL   = 1.0

rhoR = 0.125
vR   = 0.0
pR   = 0.1

r = np.geomspace(rmin, rmax, xnpts)
r1 = r[1]
r0 = r[0]
ri = (r1*r0)**0.5

theta = np.linspace(theta_min, theta_max, ynpts)

rho = np.zeros(shape=(ynpts, xnpts), dtype= float)
rho[:,r < rmid] = rhoL 
rho[:,r > rmid] = rhoR

p   = np.zeros(shape=(ynpts, xnpts), dtype= float)
p[:,r < rmid] = pL 
p[:,r > rmid] = pR

vr = np.zeros(shape=(ynpts, xnpts), dtype= float)
vt = np.zeros(shape=(ynpts, xnpts), dtype= float)

mode = 'gpu'
tend = 0.2
dtheta = theta_max/ynpts
cs = (gamma * p / gamma)**0.5
dt = 0.1 * (ri * dtheta)

print("dt: {}".format(dt) )
print("Rmin: {}".format(rmin))
print("Dim: {}x{}".format(ynpts, xnpts))

SodHLLE = Hydro(gamma = gamma, initial_state=(rho, p, vr, vt), regime="classical", coord_system="spherical",
              Npts=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
hlle_result = SodHLLE.simulate(tend=tend, first_order=False, dt=dt, compute_mode=mode,
                   linspace=False, cfl=0.1, hllc=False)


# HLLC Simulation
SodHLLC = Hydro(gamma = gamma, initial_state=(rho, p, vr, vt), regime="classical", coord_system="spherical",
              Npts=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4, boundary_condition='reflecting')

t1 = (time.time()*u.s).to(u.min)
hllc_result = SodHLLC.simulate(tend=tend, first_order=False, dt=dt, compute_mode=mode,
                    linspace=False, cfl=0.1, hllc=True, boundary_condition='reflecting')

print("The 2D SOD Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))

rhoE = hlle_result[0]
rhoC = hllc_result[0]

# Plot Radial Density at Theta = 0
plt.semilogx(r, rhoE[0], label='HLLE')
plt.semilogx(r, rhoC[0], label='HLLC')

plt.xlim(r[0], r[-1])
plt.xlabel('R')
plt.ylabel('Density')
plt.legend()
plt.show()