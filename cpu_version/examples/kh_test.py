#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import matplotlib.colors as colors

from pysimbi import Hydro 
from astropy import units as u, constants as const


gamma = 1.4
xmin = -0.5
xmax = 0.5
ymin = -0.5
ymax = 0.5

xnpts = 512
ynpts = xnpts

rhoL = 2.0
vxT = 0.5
pL = 2.5

rhoR = 1.0
vxB = - 0.5
pR = 2.5

x = np.linspace(xmin, xmax, xnpts)
y = np.linspace(ymin, ymax, ynpts)


rho = np.zeros(shape=(ynpts, xnpts), dtype= float)
rho[np.where(np.abs(y) < 0.25)] = rhoL 
rho[np.where(np.abs(y) > 0.25)] = rhoR

vx = np.zeros(shape=(ynpts, xnpts), dtype= float)
vx[np.where(np.abs(y) > 0.25)]  = vxT
vx[np.where(np.abs(y) < 0.25)]  = vxB

vy = np.zeros(shape=(ynpts, xnpts), dtype= float)

p = np.zeros(shape=(ynpts, xnpts), dtype= float)
p[np.where(np.abs(y) > 0.25)] = pL 
p[np.where(np.abs(y) < 0.25)] = pR

# Seed the KH instability with random velocities
seed = np.random.seed(0)
sin_arr = 0.01*np.sin(2*np.pi*x)
vx_rand = np.random.choice(sin_arr, size=vx.shape)
vy_rand = np.random.choice(sin_arr, size=vy.shape)

vx += vx_rand
vy += vy_rand

tend = 2.0

dt = 1.e-4
xx, yy = np.meshgrid(x, y)

fig, ax= plt.subplots(1, 1, figsize=(8,10), constrained_layout=False)

# HLLC Simulation
SodHLLC = Hydro(gamma = gamma, initial_state=(rho, p, vx, vy), 
              Npts=(xnpts, ynpts), geometry=((xmin, xmax),(ymin, ymax)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
hllc_result = SodHLLC.simulate(tend=tend, first_order=False, dt=dt, 
                               linspace=True, CFL=0.4, data_directory="data/kh/",
                               hllc=False, periodic = True, chkpt_interval= 0.01)

print("The 2D KH Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))

rho, vx, vy, pre = hllc_result

rnorm = colors.LogNorm(vmin=0.9, vmax=2.1)

c1 = ax.pcolormesh(xx, yy, rho, cmap='gist_rainbow', edgecolors='none', shading ='auto', vmin=0.9, vmax=2.1)

fig.suptitle('SIMBI: KH Instability Test at t={} s on {} x {} grid.'.format(tend, xnpts, ynpts), fontsize=20)


cbar = fig.colorbar(c1, orientation='vertical')
ax.tick_params(axis='both', labelsize=10)




cbar.ax.set_xlabel('Density', fontsize=20)
plt.show()












