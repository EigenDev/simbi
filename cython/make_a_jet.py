#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
#import matplotlib 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import matplotlib.colors as colors

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
N_0 = 4*np.pi*(r_0/R_0)**3 * (1. - np.exp(-2./(theta_0**2)))*theta_0**2
xnpts = 64

rmin = 5.e-3 #((r_0/R_0)/1e3).value
rmax = 1.0
theta_min = 0
theta_max = np.pi/2

r = np.logspace(np.log10(rmin), np.log10(rmax), xnpts)

r1 = r[1]
r0 = r[0]
ri = 0.5*(r1 + r0)
volavg = 0.75*(r1**4 - r0**4)/(r1**3 - r0**3)
# Choose dtheta carefully such that the grid zones remain roughly square
dtheta = (r1 - r0)/volavg
print(theta_max/dtheta)
ynpts = int(theta_max/dtheta)
#r = np.linspace(rmin, rmax, xnpts)
#theta = np.logspace(np.log10(theta_min), np.log10(theta_max),  ynpts)
theta = np.arange(theta_min, theta_max,  dtheta)
theta = np.linspace(theta_min, theta_max, ynpts)
ynpts = theta.size
theta_mirror = - theta[::-1]
theta_mirror[-1] *= -1.


rr, tt = np.meshgrid(r, theta)
rr, t2 = np.meshgrid(r, theta_mirror)

print(dtheta)
print(theta)
print(tt)

#print(ynpts)
print("Grid Dimensions: {} x {} ".format(r.size, theta.size) )
#print(r)
zzz = input('')
# Initial Conditions
rho_norm = m_0/R_0**3
rho_init = rho0(r, rho_c, rho_wind)/rho_norm
p_init =  1.e-6*rho_init 
g = (rr*R_0/r_0)* np.exp(-(rr*R_0/r_0)**2 / 2) * np.exp( (np.cos(tt) - 1) / theta_0**2 ) / (N_0)

L_norm = ( (m_0*c**2)/(R_0/c) ).to(u.erg/u.s)
L = L_0/L_norm
scale = 1.
S_0 =  L*g*scale
S_r = S_0*np.sqrt(1 - 1/gamma_0**2)
S_D = S_0/eta_0
S_theta = np.zeros(S_r.shape)


p = np.zeros((ynpts, xnpts), float)
p[:] = p_init


rho = np.zeros((ynpts, xnpts), float)
rho[:] = rho_init

vr = np.zeros((ynpts, xnpts), float)
vt = np.zeros((ynpts, xnpts), float)

tend = 3.0
dtheta = theta_max/ynpts
dt = 0.4*(rmin*dtheta)
#dt = 0.1*(rmax/xnpts)
print("dt: {}".format(dt) )
print("Rmin: {}".format(rmin))

Jet = Hydro(gamma = gamma, initial_state=(rho, p, vr, vt), 
              Npts=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4, regime="relativistic")

t1 = (time.time()*u.s).to(u.min)
sol = Jet.simulate(tend=tend, first_order=True, dt=dt, coordinates=b"spherical", sources=(S_D, S_r,S_theta ,S_0),
                   linspace=False, CFL=0.1)

print("The 2D Jet Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))

v = np.sqrt(sol[2]**2 + sol[3]**2)    
W = 1/np.sqrt(1 - v**2)
dens = sol[0]
D = np.log10(W*dens)
#source = np.log10(g[0])

# fig, axs = plt.subplots(2, 1, figsize=(10f, 15), sharex=True)


# a = W_r[0]/np.max(W_r[0])
# axs[0].loglog(r, W_r[0])
# axs[1].semilogx(r, dens[0]/dens[0].max())
# #axs[1].semilogx(r, source/source.max())
# plt.show()


norm = colors.LogNorm(vmin=dens.min(), vmax=dens.max())
norm2 = colors.LogNorm(vmin=W.min(), vmax=W.max())

fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=False)


c1 = ax.pcolormesh(tt, rr, D, cmap='gist_rainbow', shading = 'gouraud')
c2 = ax.pcolormesh(t2[::-1], rr, W,  cmap='gist_rainbow', shading = 'gouraud')

fig.suptitle('SIMBI: Jet Propagation at t={} s on {} x {} grid.'.format(tend, xnpts, ynpts), fontsize=20)


#divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.05)


cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.02]) 
cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)

ax.set_position( [0.1, -0.18, 0.8, 1.43])
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.yaxis.grid(True, alpha=0.95)
ax.xaxis.grid(True, alpha=0.95)
ax.tick_params(axis='both', labelsize=10)
cbaxes.tick_params(axis='x', labelsize=10)
ax.axes.xaxis.set_ticklabels([])
#ax.set_rmax(0.1)
ax.set_thetamin(-90)
ax.set_thetamax(90)



#np.savetxt('blandford_mckee_test.txt', sol)
cbar.ax.set_xlabel('Lorentz Factor', fontsize=20)
#plt.tight_layout()
# plt.tight_layout()

plt.show()

for idx, val in enumerate(['rho', 'p', 'vr','vt']):
    np.savetxt("data/jet_data_{}.txt".format(val), sol[idx])
    
del Jet
del sol 

fig.savefig('plots/2D_jet_{}_by_{}.pdf'.format(xnpts, ynpts), pad_inches = 0)
    
#fig.savefig('plots/2D_jet_propagation_break.png', bbox_inches="tight")
