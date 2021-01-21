#! /usr/bin/env python

# Code to test out the BM Explosion for an axially symmetric sphere

import numpy as np 
#import matplotlib 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import matplotlib.colors as colors
from datetime import datetime
import h5py 

from simbi import Hydro 
from astropy import units as u, constants as const




# Stellar and Engine Parameters
G = const.G.cgs                 # Newtonian G constant in cgs
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
rho_wind = 1.e-9 * m_0/R_0**3   # Wind Density
E_0 = (m_0*c**2).to(u.erg)      # Rest Mass Energy

theta_0 = 0.1                   # Injection Angle
mach = 0.05                     # Mach Number
eta_0 = 100.                    # Energy-to-Mass Ratio
r_0 = 0.015 * R_0                # Nozzle size
U = 1.e51*u.erg                 # Thermal Energy of Gas
L_0 = U*(mach**2 + 1)/(E_0)     # Engine Power (One-Sided)
tau_0 = 4.3 * R_0/c             # Enginge Duration
c_s = 1.                        # Initial Sound Speed

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
gamma = 5/3
N_0 = 4*np.pi*(r_0/R_0)**3 * (1. - np.exp(-2./(theta_0**2)))*theta_0**2
xnpts = 32

rmin = 0.01 #((r_0/R_0)/1e3).value
rmax = 0.3
theta_min = 0
theta_max = np.pi/2

r = np.logspace(np.log10(rmin), np.log10(rmax), xnpts)

r1 = r[1]
r0 = r[0]
ri = 0.5*(r1 + r0)
volavg = 0.75*(r1**4 - r0**4)/(r1**3 - r0**3)
# Choose dtheta carefully such that the grid zones remain roughly square
dtheta = (r1 - r0)/volavg
#print(theta_max/dtheta)
ynpts = int(theta_max/dtheta)
#r = np.linspace(rmin, rmax, xnpts)
#theta = np.logspace(np.log10(theta_min), np.log10(theta_max),  ynpts)
#theta = np.arange(theta_min, theta_max,  dtheta)
theta = np.linspace(theta_min, theta_max, ynpts)
ynpts = theta.size
theta_mirror = - theta[::-1]
theta_mirror[-1] *= -1.


rr, tt = np.meshgrid(r, theta)
rr, t2 = np.meshgrid(r, theta_mirror)

# print(dtheta)
# print(theta)
# print(tt)

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
vscale = 0.5
S_0 =  L_0*g*scale
S_r = S_0*vscale
S_D = S_0/eta_0
S_theta = np.zeros(S_r.shape)


p = np.zeros((ynpts, xnpts), float)
p[:] = p_init


rho = np.zeros((ynpts, xnpts), float)
rho[:] = rho_init

vr = np.zeros((ynpts, xnpts), float)
vt = np.zeros((ynpts, xnpts), float)

tend =  2.0
dtheta = theta_max/ynpts
dt = 0.4*(rmin*dtheta)

print("Engine Power: {power:.3e}".format(power = scale * U/u.s))
print("Central Density: {density:.3e}".format(density = rho_c))
print("Wind Density: {density:.3e}".format(density = rho_wind))

print("dt: {:.3e}".format(dt) )
print("Rmin: {}".format(rmin))
print("Rmax: {}".format(rmax))
print("Power Scale: {}".format(scale))
print("Velocity Scale: {}".format(vscale))
print("End Time: {}".format(tend))
print("Jet Angle: {}".format(theta_0))
print("Mass Load: {}".format(eta_0))
Jet = Hydro(gamma = gamma, initial_state=(rho, p, vr, vt), 
              Npts=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
sol = Jet.simulate(tend=tend, first_order=False, dt=dt, coordinates=b"spherical", sources=(S_D, S_r,S_theta ,S_0),
                   linspace=False, CFL=0.8, hllc=True)

# tt2 = (time.time()*u.s).to(u.min) - t1
# if (t2 <= 60*u.min):
#     print("The 2D Jet Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))
# else:
#     print("The 2D Jet Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.h) - t1))

rho = sol[0]
v_r = sol[2]
v_t = sol[3]
v = np.sqrt(v_r**2 + v_t**2)

print("V Max: {:.3f}".format(v.max()))

#source = np.log10(g[0])

# fig, axs = plt.subplots(2, 1, figsize=(10f, 15), sharex=True)


# a = W_r[0]/np.max(W_r[0])
# axs[0].loglog(r, W_r[0])
# axs[1].semilogx(r, dens[0]/dens[0].max())
# #axs[1].semilogx(r, source/source.max())
# plt.show()


norm = colors.LogNorm(vmin=rho.min(), vmax=rho.max())
norm2 = colors.LogNorm(vmin=v.min(), vmax=v.max())

fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=False)


c1 = ax.pcolormesh(tt, rr, rho, cmap='gist_rainbow', shading='auto', norm = norm)
c2 = ax.pcolormesh(t2[::-1], rr, v,  cmap='gist_rainbow', shading='auto', norm=norm2)

fig.suptitle('SIMBI: HLLC at t={} s on {} x {} grid.'.format(tend, xnpts, ynpts), fontsize=20)


#divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.05)


cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.02]) 
cbar = fig.colorbar(c1, orientation='horizontal', cax=cbaxes)
#cbar2 = fig.colorbar(c1)

ax.set_position( [0.1, -0.18, 0.8, 1.43])
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.yaxis.grid(True, alpha=0.95)
ax.xaxis.grid(True, alpha=0.95)
ax.tick_params(axis='both', labelsize=10)
cbaxes.tick_params(axis='x', labelsize=10)
ax.axes.xaxis.set_ticklabels([])
ax.set_rmin(rmin)
ax.set_thetamin(-90)
ax.set_thetamax(90)



#np.savetxt('blandford_mckee_test.txt', sol)
cbar.ax.set_xlabel('Density', fontsize=20)
#plt.tight_layout()
# plt.tight_layout()

plt.show()

zzz = input('Save Data?: y/n ')
now = datetime.today().strftime("%Y-%m-%d_%H%M%S")
if zzz == "y":
    print("Writing Data to File....")
    with h5py.File('data/sr/jet_output_{}_grid_{}_by_{}_on_{}.h5'.format(tend,ynpts, xnpts, now), 'w') as hf:
        hf.create_dataset('rho_' , data=sol[0])
        hf.create_dataset('vr' , data=sol[2])
        hf.create_dataset('vt' , data=sol[3])
        hf.create_dataset('p' , data=sol[1])
        hf.create_dataset('r', data=r)
        hf.create_dataset('t', data=theta)
        
    fig.savefig('plots/2D/Newtonian/newton_jet_hllc_{}_by_{}_at_{}.pdf'.format(xnpts, ynpts, now), pad_inches = 0)

    
# for idx, val in enumerate(['rho', 'p', 'vr','vt']):
#     np.savetxt("data/nr_jet_data_{}.txt".format(val), sol[idx])
    
del Jet
del sol 

# fig.savefig('plots/2D/Newtonian/newton_jet_hllc_{}_by_{}_at_{}.pdf'.format(xnpts, ynpts, now), pad_inches = 0)
    
#fig.savefig('plots/2D_jet_propagation_break.png', bbox_inches="tight")
