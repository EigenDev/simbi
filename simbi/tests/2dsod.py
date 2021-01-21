#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import matplotlib.colors as colors

from simbi_py import Hydro 
from astropy import units as u, constants as const


gamma = 1.4
theta_min = 0.
theta_max = np.pi/2
rmin = 0.1
rmax = 1.1
xnpts = 64 

rhoL = 1.4
vL = 0.0
pL =1.

rhoR = 0.1
vR = 0.0
pR = 0.125


rhoL = 1.4
vL = 0.0
pL =1.0

rhoR = 1.0
vR = 0.0
pR = 1.0

r = np.logspace(np.log10(rmin), np.log10(rmax), xnpts)

r1 = r[1]
r0 = r[0]
ri = 0.5*(r1 + r0)

# Choose dtheta carefully such that the grid zones remain roughly square
dtheta = (r1 - r0)/ri 
#print(theta_max/dtheta)
ynpts = int(theta_max/dtheta)

theta = np.linspace(theta_min, theta_max, ynpts)
theta_mirror = - theta[::-1]
theta_mirror[-1] *= -1.

rho = np.zeros(shape=(ynpts, xnpts), dtype= float)
rho[:, np.where(r < 0.5)] = rhoL 
rho[:, np.where(r > 0.5)] = rhoR

vr = np.zeros(shape=(ynpts, xnpts), dtype= float)
vr[:, np.where(r < 0.5)] = vL 
vr[:, np.where(r > 0.5)] = vR

vt = np.zeros(shape=(ynpts, xnpts), dtype= float)
vt[:, np.where(r < 0.5)] = vL 
vt[:, np.where(r > 0.5)] = vR

p = np.zeros(shape=(ynpts, xnpts), dtype= float)
p[:, np.where(r < 0.5)] = pL 
p[:, np.where(r > 0.5)] = pR

tend = 0.4
dtheta = theta_max/ynpts
dt = 0.4*(rmin*dtheta)
#dt = 0.1*(rmax/xnpts)
print("dt: {}".format(dt) )
print("Rmin: {}".format(rmin))
print("Dim: {}x{}".format(ynpts, xnpts))

S = np.zeros(shape = rho.shape)

SodHLLE = Hydro(gamma = gamma, initial_state=(rho, p, vr, vt), 
              Npts=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
hlle_result = SodHLLE.simulate(tend=tend, first_order=False, dt=dt, coordinates=b"spherical", sources=(S, S, S, S),
                   linspace=False, CFL=0.9, hllc=False)


# HLLC Simulation
SodHLLC = Hydro(gamma = gamma, initial_state=(rho, p, vr, vt), 
              Npts=(xnpts, ynpts), geometry=((rmin, rmax),(theta_min,theta_max)), n_vars=4)

t1 = (time.time()*u.s).to(u.min)
hllc_result = SodHLLC.simulate(tend=tend, first_order=False, dt=dt, coordinates=b"spherical", sources=(S, S, S, S),
linspace=False, CFL=0.9, hllc=True)

print("The 2D SOD Simulation for ({}, {}) grid took {:.3f}".format(xnpts, ynpts, (time.time()*u.s).to(u.min) - t1))

rhoE = hlle_result[0]
rhoC = hllc_result[0]
v_r = np.abs( rhoE[2])

# Plot Radial Density at Theta = 0
for idx, thta in enumerate(rhoE):
    if idx == 0:
        plt.plot(r, rhoE[idx], label='HLLE')
        plt.plot(r, rhoC[idx], label='HLLC')
    else:
        plt.plot(r, rhoE[idx])
        plt.plot(r, rhoC[idx])
        
plt.xlabel('R')
plt.ylabel('Density')
plt.legend()
plt.savefig('plots/2D/2D_sod_hllc_vs_hlle.pdf')
plt.show()














"""
norm = colors.LogNorm(vmin=rho.min(), vmax=rho.max())
norm2 = colors.LogNorm(vmin=v_r.min(), vmax=v_r.max())

fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=False)


c1 = ax.pcolormesh(tt, rr, rho, cmap='gist_rainbow', shading='gouraud', norm = norm)
c2 = ax.pcolormesh(t2[::-1], rr, v_r,  cmap='gist_rainbow', shading='gouraud', norm=norm2)

fig.suptitle('SIMBI: Non-Relatavistic Jet Propagation at t={} s on {} x {} grid.'.format(tend, xnpts, ynpts), fontsize=20)


#divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.05)


cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.02]) 
cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)
cbar2 = fig.colorbar(c1)

ax.set_position( [0.1, -0.18, 0.8, 1.43])
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.yaxis.grid(True, alpha=0.95)
ax.xaxis.grid(True, alpha=0.95)
ax.tick_params(axis='both', labelsize=10)
cbaxes.tick_params(axis='x', labelsize=10)
ax.axes.xaxis.set_ticklabels([])
#ax.set_rmin(rmin)
ax.set_thetamin(-90)
ax.set_thetamax(90)



#np.savetxt('blandford_mckee_test.txt', sol)
cbar.ax.set_xlabel('Radial Velocity', fontsize=20)
#plt.tight_layout()
# plt.tight_layout()

plt.show()

"""