#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
import matplotlib.pyplot as plt
import time
from state import PyStateSR2D
from simbi_py import Hydro 

from astropy import units as u


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def rho0(n, theta):
    return 1.0 - 0.95*np.cos(n*theta)


# Constants
gamma = 4/3
p_init = 1.e-6
r_init = 0.01
nu = 3.
epsilon = 2.

rho_init = rho0(0, np.pi)
v_init = 0.
ntheta = 256
rmin = 0.05
rmax = 1
N_exp = 5


theta_min = 0
theta_max = np.pi

theta = np.linspace(theta_min, theta_max, ntheta)
theta_mirror = np.linspace(np.pi, 2*np.pi, ntheta)

# Choose xnpts carefully such that the grid zones remain roughly square
dtheta = theta_max/ntheta
nr = int(np.ceil(1 + np.log10(rmax/rmin)/dtheta ))

r = np.logspace(np.log10(rmin), np.log10(rmax), nr) 

r_right = np.sqrt(r[1:nr] * r[0:nr-1])
dr = rmin * 1.5 


p_zones = find_nearest(r, dr)[0]
p_zones = int(p_zones)

p_c = (gamma - 1.)*(3*epsilon/((nu + 1)*np.pi*dr**nu))

print("Central Pressure:", p_c)
print("Dimensions: {} x {}".format(ntheta, nr))

n = 2.0
omega = 2.0
rho = np.zeros((ntheta , nr), float)
rho[:] = 1.0 * r ** (-omega) #(rho0(n, theta)).reshape(ntheta, 1)

p = rho * 1e-6
p[:, :p_zones] = p_c 
# p[:, p_zones:] = p_init

# print(rho)
# zzz = input()
vx = np.zeros((ntheta ,nr), np.double)
vy = np.zeros((ntheta ,nr), np.double)



tend = 1.0
dt = 1.e-8
# with PackageResource() as bm:
#     bm.Hydro()
bm = Hydro(gamma = gamma, initial_state=(rho, p, vx, vy), 
            Npts=(nr, ntheta), 
            geometry=((rmin, rmax),(theta_min, theta_max)), 
            n_vars=4, regime="relativistic")


t1 = (time.time()*u.s).to(u.min)
sol = bm.simulate(tend=tend, first_order= False, dt=dt,
                  coordinates="spherical", CFL=0.4, 
                  hllc=True, linspace=False, theta=1.5)

print("The 2D BM Simulation for N = {} took {:.3f}".format(ntheta, (time.time()*u.s).to(u.min) - t1))

#density = b.cons2prim(sol)[0]

W = 1/np.sqrt(1 - sol[1]**2 + sol[2]**2)
print(sol[1].max())
rr, tt = np.meshgrid(r, theta)
rr, t2 = np.meshgrid(r, theta_mirror)

fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)
c1 = ax.pcolormesh(tt, rr, sol[0], cmap='inferno', shading = "auto")
c2 = ax.pcolormesh(t2, rr, sol[0], cmap='inferno', shading = "auto")

fig.suptitle('Spherical Explosion at t={} s on {} x {} grid'.format(tend, nr, ntheta), fontsize=15)
# ax.set_title(r'$\rho(\theta) = 1.0 - 0.95\cos(n \ \theta)$ with n = {}'.format(n), fontsize=10)
cbar = fig.colorbar(c1)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rmax(1)
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.grid(True, alpha=0.4)
ax.xaxis.grid(True, alpha=0.4)
ax.set_thetamin(0)
ax.set_thetamax(360)

cbar.ax.set_ylabel('Lorentz Factor', fontsize=20)

plt.show()
# fig.savefig('plots/2D/SR/2D_bm_0.1_.pdf', bbox_inches="tight")
