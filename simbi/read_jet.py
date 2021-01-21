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



# f = h5py.File('data/jet_output_116_by_128_at_2020-12-15_103926.h5', 'r+')
# f.close()
with h5py.File('data/jet_output_2.0_grid_235_by_512_on_2021-01-09_034434.h5', 'r+') as hf:
    rho = hf.get('rho_')[:] #hf['rho_']
    v_r = hf.get('vr')[:] #hf['vr']
    vt = hf.get('vt')[:] #hf['vt']
    p = hf.get('p')[:] 
    r = hf.get('r')[:]
    theta = hf.get('t')[:]
    data = type('data', (object,), dict(rho=rho,
                                        p = p,
                                        vr = v_r,
                                        vt = vt))


# rho = np.loadtxt('data/nr_jet_data_rho.txt')
# v_r = np.loadtxt('data/nr_jet_data_vr.txt')

rho = np.abs(rho)
v_r = np.abs(v_r)
p = np.abs(p)

nan_mask = np.argwhere(np.isnan(rho)) #rho[np.where(np.isnan(rho))]


# rho = np.loadtxt('jet_data_rho_256.txt')
# v_r = np.loadtxt('jet_data_vr_256.txt')

ynpts, xnpts = rho.shape 

#rmin = 0.01
#rmax = 0.1
#r = np.logspace(np.log10(rmin), np.log10(rmax), rho[0].size)
#theta = np.linspace(0, np.pi/2, rho[:, 0].size)
theta_mirror = - theta[::-1]
theta_mirror[-1] *= -1.

# print("Rmin: {}".format(r[0]))
# print("Rmax: {}".format(r[-1]))

rr, tt = np.meshgrid(r, theta)
rr, t2 = np.meshgrid(r, theta_mirror)

print(rho.min())
rnorm = colors.LogNorm(vmin=rho.min(), vmax=3.*rho.min())
vnorm = colors.LogNorm(vmin=1.e-2, vmax=6.e-2)
pnorm = colors.LogNorm(vmin=1e-5, vmax=10.)

fig, ax= plt.subplots(1, 1, figsize=(8,10), subplot_kw=dict(projection='polar'), constrained_layout=True)

tend = 2.0 
c1 = ax.pcolormesh(tt, rr, rho, cmap='gist_rainbow', shading='auto', norm = rnorm)
c2 = ax.pcolormesh(t2[::-1], rr, rho,  cmap='gist_rainbow', shading='auto', norm=rnorm)

fig.suptitle('HLLC at t={} s on {} x {} grid.'.format(tend, xnpts, ynpts), fontsize=20)


#divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.05)


cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)
#cbar2 = fig.colorbar(c1)
# ax.set_title('HLLC at t={} s on {} x {} grid.'.format(tend, xnpts, ynpts), fontsize=20)
ax.set_position( [0.1, -0.18, 0.8, 1.43])
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.yaxis.grid(True, alpha=0.95)
ax.xaxis.grid(True, alpha=0.95)
ax.tick_params(axis='both', labelsize=10)
cbaxes.tick_params(axis='x', labelsize=10)
ax.axes.xaxis.set_ticklabels([])
ax.set_rmax(0.06)
ax.set_rmin(r.min())
ax.set_thetamin(-90)
ax.set_thetamax(90)



#np.savetxt('blandford_mckee_test.txt', sol)
cbar.ax.set_xlabel('Density', fontsize=20)
#plt.tight_layout()
# plt.tight_layout()

plt.show()
    
# for idx, val in enumerate(['rho', 'p', 'vr','vt']):
#     np.savetxt("data/nr_jet_data_{}.txt".format(val), sol[idx])

# fig.savefig('plots/2D/Newtonian/read_newton_jet_hllc.pdf', pad_inches = 0, bbox_inches='tight')
    
#fig.savefig('plots/2D_jet_propagation_break.png', bbox_inches="tight")
