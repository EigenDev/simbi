#! /usr/bin/env python

# Code to test out the Sedov Explosion

import numpy as np 
import matplotlib.pyplot as plt
import time
from pysimbi import Hydro 
from state import PyState2D
from astropy import units as u


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def rho0(n, theta):
    return 2.0 + np.sin(n*theta)


# Constants
gamma = 5/3
p_init = 1.e-5
r_init = 0.01
nu = 3.
epsilon = 1.0

rho_init = rho0(0, np.pi)
v_init = 0.
N = 128
rmin = 0.01
rmax = 1 
N_init = 5 

N_exp = 5


r = np.linspace(rmin, rmax, N + 1)
r_right = 0.5*(r[1:N] + r[0:N-1])
dr = r_right[N_exp]

theta = np.linspace(0., np.pi,  N + 1)
theta_mirror = np.linspace(np.pi, 2*np.pi, N + 1)

delta_r = dr - rmin
p_zones = find_nearest(r, (rmin + dr))[0]
p_zones = int(p_zones)
#p_zones = (int(rmin*N))
#print(p_zones)


p_c = (gamma - 1.)*(3*epsilon/((nu + 1)*np.pi*dr**nu))

print("Central Pressure:", p_c)


               
p = np.zeros((N+1,N+1), np.double)
p[:, :p_zones] = p_c 
p[:, p_zones:] = p_init

n = 2.0
rho = np.zeros((N+1,N+1), float)
rho[:] = 1.0 #(rho0(n, theta)).reshape(N+1, 1)


# print(rho)
# zzz = input()
vx = np.zeros((N+1,N+1), np.double)
vy = np.zeros((N+1,N+1), np.double)

vx[:, :] = v_init
vy[:, :] = v_init



tend = 0.2
dt = 1.e-5

sedov = Hydro(gamma = gamma, initial_state=(rho, p, vx, vy), 
              Npts=(N+1, N+1), geometry=((rmin, rmax),(0.,np.pi)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
sol = sedov.simulate(tend=tend, first_order=False, dt=dt, coordinates="spherical", hllc=True)
print("The 2D Sedov Simulation for N = {} took {:.3f}".format(N, (time.time()*u.s).to(u.min) - t1))

density = sol[0]

rr, tt = np.meshgrid(r, theta)
rr, t2 = np.meshgrid(r, theta_mirror)

fig, ax= plt.subplots(1, 1, figsize=(15,10), subplot_kw=dict(projection='polar'))
c1 = ax.contourf(tt, rr, density, cmap='plasma')
c2 = ax.contourf(t2, rr, np.flip(density, axis=0), cmap='plasma')
fig.suptitle('Sedov Explosion at t={} s on {} x {} grid'.format(tend, N, N), fontsize=20)
ax.set_title(r'$\rho(\theta) = 2.0 + \sin(n \theta)$ with n = {}'.format(n), fontsize=25)
cbar = fig.colorbar(c1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rmax(1)
ax.set_thetamin(0)
ax.set_thetamax(360)
#plt.gca().set_aspect('equal', adjustable='box')
#c2 = axes[1].contourf(xx, yy, sol[0], cmap='plasma')
#axes[1].set_xlim(-0.1, 0.1)
#axes[1].set_ylim(-0.1, 0.1)
#axes[1].set_title('Density at t={} s and N = {}'.format(tend, N), fontsize=20)
#cbar2 = fig.colorbar(c2)

cbar.ax.set_ylabel('Density', fontsize=30)
plt.show()
fig.savefig('plots/2D_sedov_density_spherical.pdf')
