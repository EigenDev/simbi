#! /usr/bin/env python

# Code to test out the Sedov Explosion

import numpy as np 
import matplotlib.pyplot as plt
import time
from pysimbi import Hydro 
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
N = 256

def circular_mask(h, w, center=None, radius=None):
    
    if center is None: #Get the center of the grid
        center = (int(h/2), int(w/2))
        
    if radius is None: # Smallest distance from the center and image walls
        radius = min(*center, h - center[0], w-center[1])
        
    X, Y = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius 
    
    return mask
    
    

               
p = np.zeros((N,N), np.double)
w, h = p.shape
p[:, :] = p_c

mask = circular_mask(h, w, radius=1)
pr = p.copy()
pr[~mask] = p_init

# print(pr)
# zzz = input('')

rho = np.zeros((N,N), float)
rho[:, :] = rho_init 

vx = np.zeros((N,N), np.double)
vy = np.zeros((N,N), np.double)

vx[:, :] = v_init
vy[:, :] = v_init

tend = 0.01

sedov = Hydro(gamma = gamma, initial_state=(rho, pr, vx, vy), 
              Npts=(N, N), geometry=((-1., 1.),(-1.,1.)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
sol = sedov.simulate(tend=tend, first_order=False, dt=1.e-5, hllc=True)
print("The 2D Sedov Simulation for N = {} took {:.3f}".format(N, (time.time()*u.s).to(u.min) - t1))

rho = sol[0]

x = np.linspace(-1., 1, N)
y = np.linspace(-1.,1,  N)

xx, yy = np.meshgrid(x, y)

fig, ax= plt.subplots(1, 1, figsize=(15,10))
c1 = ax.pcolormesh(xx, yy, rho, cmap='plasma', shading='auto')
ax.set_title('Density at t={} s on {} x {} grid'.format(tend, N, N), fontsize=20)
cbar = fig.colorbar(c1)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
