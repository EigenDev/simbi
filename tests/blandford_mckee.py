#! /usr/bin/env python

# The Relativisitc Blast Wave Problem with the Blandford-McKee SOlution
#
import numpy as np 
import matplotlib.pyplot as plt 
from simbi_py import Hydro 


# Define some constant
rho0 = 1.
r0 = 1.
N = 500
omega = 0.
dOmega = 4*np.pi
c = 1.
W = 1
gamma = 4/3
rshell = ( (3 - omega)/dOmega ) ** (1/(3 - omega))
r = np.linspace(0.02, 1, N)
E = (dOmega/(3 - omega))*(rho0*r0**omega)*rshell**(3 - omega)*W**2*c**2
l = ((3-omega)*E/(rho0*r0**omega*c**2))**(1/(3-omega))


def W(v):
    #v = np.asarray(v)
    return 1/(np.sqrt(1 - v**2))

def energy(gamma, v):
    lorentz_gamma = W(v)
    
def R(t):
    return (2*l**(3-omega)/dOmega)**(1/(4-omega))*t**(1/(4-omega))
def lorentz_gamma(t):
    return (l**(3-omega)/(2**(3-omega)*dOmega))**(1/(2*(4-omega)))*t**(-(3-omega)/(2*(4-omega)))


tinit = 0.03
rho = rho0*(r/r0)**(-omega)

rmin = 0.01
rmax = l
r = np.logspace(np.log10(rmin), np.log10(rmax), N)
r = np.linspace(rmin, rmax, N)
chi = (1 + 2*(4-omega)*lorentz_gamma(tinit)**2)*(1 - r/tinit)
p_chi = 2/3 * lorentz_gamma(tinit)**2 * chi[np.where(chi > 1)]**(-(17-4*omega)/(12 - 3*omega))
rho_chi = 2*rho0 * lorentz_gamma(tinit)**2 * chi[np.where(chi > 1)]**(-(10-3*omega)/(2*(4-omega)))
yamma = lorentz_gamma(tinit)/np.sqrt(2*chi[np.where(chi > 1)])

vels  = np.sqrt(1 - 1/(lorentz_gamma(tinit)**2))

print(vels)
p = np.zeros(N, float)
v = np.zeros(N, float)
rho = np.ones(N, float)
gammas = np.ones(N, float)

p[np.where(chi > 1)] = p_chi
v [np.where(chi > 1)] = vels
rho[np.where(chi > 1)] =  rho_chi
gammas[np.where(chi > 1)] = yamma
gammas[np.where(chi < 1)] = 1/np.sqrt(2)


p[np.where(chi < 1)] = 1.e-10

#print(p)

dt = 1.e-8
tend = 1.
bm = Hydro(gamma, initial_state=(rho, p, v), Npts=N, geometry=(rmin, rmax), regime="relativistic")

u = bm.simulate(tend = tend,dt=dt, coordinates=b"spherical", linspace=True, first_order=False, CFL=0.1)

yamma = W(u[2])/np.sqrt(2)

fig, ax = plt.subplots(1, 1, figsize=(10, 15))

ax.semilogx(r, rho*gammas**2, '--',  label='$t_0$')
ax.semilogx(r, u[0]*yamma**2, label='$(t_0 + dt)$')
ax.semilogx(r, u[2], label='$v$')
ax.set_xlabel('R', fontsize=20)
ax.set_ylabel(r'$\rho\gamma^2$', fontsize=20)
ax.set_title('1D Blandford-McKee Blast Wave at t = {:.3f} [s] with N = {}'.format(tinit + tend, N), fontsize=20)
ax.legend(fontsize=15)
plt.show()
fig.savefig('plots/BM_test.pdf')
#Define 

