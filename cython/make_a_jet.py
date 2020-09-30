#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
from simbi import Hydro 
from astropy import units as u, constants as const




# Stellar and Engine Parameters
c = const.c.to(u.cm/u.s)        # Speed of light converted to cm/s
m_0 = 2e33*u.g                  # solar radius
R_0 = 7e10*u.cm                 # Characteristic Length Scale
rho_c = 3e7*  m_0/R_0**3        # Central Density
R_1 = 0.0017*R_0                # First Break Radius
R_2 = 0.0125 * R_0              # Second Break Radius
R_3 = 0.65 * R_0                # Outer Radius
k1 = 3.24                       # First Break Slope
k2 = 2.57                       # Second Break Slope
n = 16.7                        # Atmosphere Cutoff Slope
rho_wind = 1e-9 * m_0/R_0**3    # Wind Density

theta_0 = 0.1                   # Injection Angle
gamma_0 = 50                    # Injected Lorentz Factor
eta_0 = 100                     # Energy-to-Mass Ratio
r_0 = 0.01 * R_0                # Nozzle size
L_0 = (2e-3 * m_0 * c**3 / R_0).to(u.erg/u.s)   # Engine Power (One-Sided)
tau_0 = 4.3 * R_0/c             # Enginge Duration


print("Engine Power: {power:.3e}".format(power = L_0))
print("Central Density: {density:.3e}".format(density = rho_c))
print("Wind Density: {density:.3e}".format(density = rho_wind))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def rho0(r, c_scale=1, wind_scale=1):
    
    r = np.asarray(r)
    null_vec = np.zeros(r.size)
    central_scale = rho_c* (c_scale/rho_c)
    env_denom =  1 + (r*R_0/R_1)**k1 / (1 + (r*R_0/R_2)**k2 )
    star_envelope = (np.maximum(1 - r * R_0/R_3, null_vec))**n / env_denom 

    wind_density = rho_wind*(wind_scale/rho_wind)*(r*R_0/R_3)**(-2.)
    
    return central_scale*star_envelope + wind_density
    
# Physical Constants
gamma = 4/3
N_0 = 4*np.pi*r_0**3 * (1. - np.exp(-2./(theta_0**2)))*theta_0**2
N = 50
rmin = 5e-4
rmax = 1.2
theta_min = 0
theta_max = np.pi/2

r = np.logspace(np.log10(rmin), np.log10(rmax), N)
#r = np.linspace(rmin, rmax, N)
theta = np.linspace(theta_min, theta_max,  N)
theta_mirror = np.linspace(-theta_max, theta_min, N)

rr, tt = np.meshgrid(r, theta)
rr, t2 = np.meshgrid(r, theta_mirror)

# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# #print(rho0(r))
# ax.loglog(r, rho0(r, rho_c, rho_wind))
# ax.set_xlabel('$r/R_0$', fontsize=20)
# ax.set_ylabel('Density [g cm$^{-3}$ ]', fontsize=20)
# ax.set_title('Stellar density based on Duffell & MacFadyen (2018)', fontsize = 30)
# fig.savefig('plots/MESA_fit.pdf')
# plt.show()
# zzz = input('')

# Initial Conditions
rho_norm = m_0/R_0**3
rho_init = rho0(r, rho_c, rho_wind)/rho_norm
p_init =  1.e-6*rho_init 
g = (rr*R_0/r_0)* np.exp(-(rr*R_0/r_0)**2 / 2) * np.exp( (np.cos(tt) - 1) / theta_0**2 ) / (N_0/(R_0**3))

# plt.loglog(r, g[0])
# plt.ylim(1e-3, 1e7)
# plt.xlim(rmin, rmax)
# plt.show()
L_norm = ( (m_0*c**2)/(R_0/c) ).to(u.erg/u.s)
L = L_0/L_norm
S_0 =  2e-4 * g
S_r = S_0*np.sqrt(1 - 1/gamma_0**2)
S_D = S_0 / eta_0

p = np.zeros((N,N), float)
p[:] = p_init

rho = np.zeros((N,N), float)
rho[:] = rho_init

vr = np.zeros((N,N), float)
vt = np.zeros((N,N), float)

tend = 0.5
dt = 1.e-5

Jet = Hydro(gamma = gamma, initial_state=(rho, p, vr, vt), 
              Npts=N, geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4, regime="relativistic")

t1 = (time.time()*u.s).to(u.min)
sol = Jet.simulate(tend=tend, first_order=False, dt=dt, coordinates=b"spherical", sources=(S_0, S_D, S_r),
                   linspace=False)
print("The 2D Jet Simulation for N = {} took {:.3f}".format(N, (time.time()*u.s).to(u.min) - t1))

W_r = np.log10(1/np.sqrt(1 - sol[2]**2))
#g = np.log10(g)
print(np.max(W_r))
#W_r = scipy.ndimage.zoom(W_r, 1)
dens = np.log10(sol[0])

fig, ax= plt.subplots(1, 1, figsize=(14,10), subplot_kw=dict(projection='polar'))
c1 = ax.contourf(tt, rr, dens, 100, cmap='inferno')
c2 = ax.contourf(t2, rr, np.flip(W_r, axis=0), 100, cmap='inferno')

fig.suptitle('Jet Propagation at t={} s on {} x {} grid. Based on Duffell & MacFadyen (2018)'.format(tend, N, N), fontsize=20)
cbar = fig.colorbar(c2)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.yaxis.grid(False)
ax.xaxis.grid(False)
ax.set_xticks(np.array([0, -45, -90, np.nan, np.nan, np.nan, 90, 45])/180*np.pi)
#ax.set_rlim(rmin)
#ax.set_rmax(0.01)
#ax.set_rscale('log')
ax.set_thetamin(-90)
ax.set_thetamax(90)
#plt.gca().set_aspect('equal', adjustable='box')

#np.savetxt('blandford_mckee_test.txt', sol)
cbar.ax.set_ylabel('Radial $\Gamma$', fontsize=20)
# plt.tight_layout()
plt.show()
fig.savefig('plots/2D_jet_propagation_break.pdf', bbox_inches="tight")
