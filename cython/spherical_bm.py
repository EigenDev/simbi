#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
import matplotlib.pyplot as plt
import time
from state import PyStateSR2D
from simbi import Hydro 

from astropy import units as u


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def rho0(n, theta):
    return 1.0 - 0.95*np.cos(n*theta)


# Constants
gamma = 4/3
p_init = 1.e-10
r_init = 0.01
nu = 3.
epsilon = 10.
jet_ang = 4
theta_jet = jet_ang*np.pi/180
dOmega = 2*np.pi*theta_jet**2
rho_init = rho0(0, np.pi)
v_init = 0.
N = 100
rmin = 0.01
rmax = 1
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


p_c = 2*(gamma - 1.)*(3*epsilon/((nu + 1)*np.pi*dr**nu))

print("Central Pressure:", p_c)

print(p_zones)
p = np.zeros((N+1,N+1), np.double)
p[:, :p_zones] = p_c 
p[:, p_zones:] = p_init


n = 2.0
rho = np.zeros((N+1,N+1), float)
rho[:] = (rho0(n, theta)).reshape(N+1, 1)


# print(rho)
# zzz = input()
vx = np.zeros((N+1,N+1), np.double)
vy = np.zeros((N+1,N+1), np.double)



tend = 0.01
dt = 1.e-8
# with PackageResource() as bm:
#     bm.Hydro()
bm = Hydro(gamma = gamma, initial_state=(rho, p, vx, vy), 
              Npts=(N+1, N+1), geometry=((rmin, rmax),(0.,np.pi)), n_vars=4, regime="relativistic")

t1 = (time.time()*u.s).to(u.min)
sol = bm.simulate(tend=tend, first_order=False, dt=dt, coordinates=b"spherical", CFL=0.1)
print("The 2D BM Simulation for N = {} took {:.3f}".format(N, (time.time()*u.s).to(u.min) - t1))

#density = b.cons2prim(sol)[0]

W_r = 1/np.sqrt(1 - sol[2]**2)
print(sol[2].max())
rr, tt = np.meshgrid(r, theta)
rr, t2 = np.meshgrid(r, theta_mirror)

fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)
c1 = ax.contourf(tt, rr, W_r, 50, cmap='inferno')
c2 = ax.contourf(t2, rr, np.flip(W_r, axis=0), 50, cmap='inferno')

fig.suptitle('Blandford-McKee Problem at t={} s on {} x {} grid'.format(tend, N, N), fontsize=15)
ax.set_title(r'$\rho(\theta) = 1.0 - 0.95\cos(n \ \theta)$ with n = {}'.format(n), fontsize=10)
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
#plt.gca().set_aspect('equal', adjustable='box')

del bm

#np.savetxt('blandford_mckee_test.txt', sol)
cbar.ax.set_ylabel('Radial $\Gamma$', fontsize=20)
# plt.tight_layout()
plt.show()
#fig.savefig('plots/2D_bm_lorentzr__test_spherical_.eps', bbox_inches="tight")
