#! /usr/bin/env python

# 1D Spherical Sedov example problem 

import numpy as np
import matplotlib.pyplot as plt 
from simbi_py import Hydro 

rmin = 0.1
rmax = 1
gamma = 5/3
n = 1000
energy = 2
p_amb = 1.e-6 
omega = 3
tend = 3.0
r   = np.logspace(np.log10(rmin), np.log10(rmax), n)

n_exp = 2
dr  = r[n_exp]
p_exp = (gamma - 1.0)*energy/(4 * np.pi * dr**3 )

print(p_exp)
rho = 10 * np.ones(n) * r**(-omega)
v   = np.zeros(n)
p   = np.ones(n)

p[:n_exp] = p_exp 
p[n_exp:] = p_amb 

sedov = Hydro(gamma, (rho, p, v), n, (rmin, rmax), 3, "spherical")
solution = sedov.simulate(first_order=False, tend=tend, linspace=False)

rho = solution[0]
m   = solution[1]
e   = solution[2]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))


ax.loglog(r, rho)
ax.set_title("1D Sedov Sherical Explosion at t={:.1f}".format(tend), fontsize=30)
ax.set_xlabel("r", fontsize=20)
ax.set_ylabel("rho", fontsize=20)
plt.show()

