#! /usr/bin/env python

# Code to test out the Sedov Explosion

import numpy as np 
import matplotlib.pyplot as plt
import time
from simbi_py import Hydro 
from state import PyState2D
from astropy import units as u

# Constants
gamma = 5/3
p_init = 1.e-5
r_init = 0.01
nu = 3.
epsilon = 1.0
p_c = (gamma - 1.)*(3*epsilon/((nu + 1)*np.pi*r_init**nu))
rho_init = 1.
v_init = 0.
N = 128

def circular_mask(h, w, center=None, radius=None):
    
    if center is None: #Get the center of the grid
        center = (int(h/2), int(w/2))
        
    if radius is None: # Smallest distance from the center and image walls
        radius = min(*center, h - center[0], w-center[1])
        
    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius 
    
    return mask
    
    

               
p = np.zeros((N+1,N+1), np.double)
w, h = p.shape
p[:, :] = p_c

mask = circular_mask(h, w, radius=1)
pr = p.copy()
pr[~mask] = p_init

# print(pr)
# zzz = input('')

rho = np.zeros((N+1,N+1), float)
rho[:, :] = rho_init 

vx = np.zeros((N+1,N+1), np.double)
vy = np.zeros((N+1,N+1), np.double)

vx[:, :] = v_init
vy[:, :] = v_init

tend = 0.01

sedov = Hydro(gamma = gamma, initial_state=(rho, pr, vx, vy), 
              Npts=(N+1, N+1), geometry=((-1., 1.),(-1.,1.)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
sol = sedov.simulate(tend=tend, first_order=False, dt=1.e-5, hllc=False)
print("The 2D Sedov Simulation for N = {} took {:.3f}".format(N, (time.time()*u.s).to(u.min) - t1))

pressure = sedov.cons2prim(sol)[1]

x = np.linspace(-1., 1, N + 1)
y = np.linspace(-1.,1,  N + 1)

xx, yy = np.meshgrid(x, y)

fig, ax= plt.subplots(1, 1, figsize=(15,10))
c1 = ax.contourf(xx, yy, pressure, cmap='plasma')
ax.set_title('Pressure at t={} s on {} x {} grid'.format(tend, N, N), fontsize=20)
cbar = fig.colorbar(c1)

plt.gca().set_aspect('equal', adjustable='box')
#c2 = axes[1].contourf(xx, yy, sol[0], cmap='plasma')
#axes[1].set_xlim(-0.1, 0.1)
#axes[1].set_ylim(-0.1, 0.1)
#axes[1].set_title('Density at t={} s and N = {}'.format(tend, N), fontsize=20)
#cbar2 = fig.colorbar(c2)

#cbar.ax.set_ylabel('Pressure', fontsize=15)
plt.show()
# fig.savefig('plots/2D_sedov_pressure.pdf')
