#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import matplotlib.colors as colors

from simbi_py import Hydro 
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
vxB = -0.5
pR = 2.5

x = np.linspace(xmin, xmax, xnpts)
y = np.linspace(ymin, ymax, ynpts)


rho = np.zeros(shape=(ynpts, xnpts), dtype= float)
rho[np.where(np.abs(y) < 0.25)] = rhoL 
rho[np.where(np.abs(y) > 0.25)] = rhoR

vx = np.zeros(shape=(ynpts, xnpts), dtype= float)
vx[np.where(np.abs(y) > 0.25), :] = vxT
vx[np.where(np.abs(y) < 0.25), :] = vxB

vy = np.zeros(shape=(ynpts, xnpts), dtype= float)

p = np.zeros(shape=(ynpts, xnpts), dtype= float)
p[np.where(np.abs(y) > 0.25)] = pL 
p[np.where(np.abs(y) < 0.25)] = pR

# Seed the KH instability with random velocities
seed = np.random.seed(0)
sin_arr = 0.1*np.sin(2*np.pi*x)
vx_rand = np.random.choice(sin_arr, size=vx.shape)
vy_rand = np.random.choice(sin_arr, size=vy.shape)

vx += vx_rand
vy += vy_rand

tend = 2.0

dt = 1.e-4
S = np.zeros(shape = rho.shape)
xx, yy = np.meshgrid(x, y)

fig, ax= plt.subplots(1, 1, figsize=(8,10), constrained_layout=False)
# ax.pcolormesh(xx, yy, vx, cmap='gist_rainbow', edgecolors='none', shading ='auto')
# plt.show()
# fig.clf()
# HLLC Simulation
SodHLLC = Hydro(gamma = gamma, initial_state=(rho, p, vx, vy), 
              Npts=(xnpts, ynpts), geometry=((xmin, xmax),(ymin, ymax)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
hllc_result = SodHLLC.simulate(tend=tend, first_order=False, dt=dt, 
                               sources=(S, S, S, S),
                               linspace=True, CFL=0.8,
                               hllc=True, periodic = True)

print("The 2D KH Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))

rho, mx, my, E = hllc_result
print(rho)



rnorm = colors.LogNorm(vmin=0.9, vmax=2.1)

c1 = ax.pcolormesh(xx, yy, rho, cmap='gist_rainbow', edgecolors='none', shading ='auto', vmin=0.9, vmax=2.1)

fig.suptitle('SIMBI: KH Instability Test at t={} s on {} x {} grid.'.format(tend, xnpts, ynpts), fontsize=20)


#divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.05)


# cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.02]) 
cbar = fig.colorbar(c1, orientation='vertical')
#cbar2 = fig.colorbar(c1)

# ax.set_position( [0.1, -0.18, 0.8, 1.43])
#ax.yaxis.grid(True, alpha=0.95)
#ax.xaxis.grid(True, alpha=0.95)
ax.tick_params(axis='both', labelsize=10)
# cbaxes.tick_params(axis='x', labelsize=10)
# ax.axes.xaxis.set_ticklabels([])




#np.savetxt('blandford_mckee_test.txt', sol)
cbar.ax.set_xlabel('Density', fontsize=20)
#plt.tight_layout()
# plt.tight_layout()
plt.show()
fig.savefig("plots/2D/Newtonian/kh_instability_{}_by_{}_at_{}.pdf".format(ynpts, xnpts, tend))












