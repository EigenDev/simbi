#!/usr/bin/env python


import numpy as np 
import matplotlib.pyplot as plt 
import math 
import matplotlib as mpl
import time 
import astropy.units as u
from simbi import Hydro 





def R(rho, eps, nu, t):
    """
    Returns the characteristic radius 
    """
    return (eps*t**2/rho)**(1/(nu+2))
    
# Set up initial conditions
gamma = 1.4                                 # adiabatic index
epsilon = 1                                # Initial Explosion Energy
p_amb = 1.e-5                               # Ambient Pressure
rho0 = 1.                                   # Ambient Density
mach = 1500.                                # Mach Number
cs = np.sqrt(gamma*p_amb/rho0)              # Sound speed of the medium
v_exp = mach*cs                             # The veolicty of the explosion 
nu = 3.                                     # Parameter that determines geometry of explosion

r_min = 0.015                             # Minimum radial coordinate
r_max = 1.5                                 # Maximum radial coordinate
N = 900                                    # Number of zones
N_explosion = 5
dr = 0.01                                # Initial explosion radius
pressure_zone = int(dr*N)                   # The radius of the initial pressure

# r = np.logspace
p_form = 3*(gamma-1.)*epsilon/((nu+1)*np.pi*dr**nu)  # Initial pressure
p_init = (gamma-1.)*rho0*v_exp**2                   # Another form of the initial pressure


v_exp = np.sqrt(p_form/((gamma-1.)*rho0))
p = np.zeros(N, float)
p[0] = p_form
p[1: ] = p_amb
rho = np.ones(N, float)
v = np.zeros(N, float)

r_explosion = 0.9/0.1
tend = r_explosion/(v_exp)
#print(tend)
#zzz = input('')
tend = round(tend, 2)
#print("Tend: {}".format(tend))
#tend = 0.02


dt = 1.e-7

sod = (1.0,1.0,0.0),(0.1,0.125,0.0)
sed_init = (rho, p, v)

def init_pressure(gamma, epsilon, nu, dr):
    return 3*(gamma-1.)*epsilon/((nu+1)*np.pi*dr**nu) 
    

def sod():
    tend = 0.1
    fig, ax = plt.subplots(1, 1, figsize=(13,11))
    
    r_min = 0.1
    r_max = 1.
    r = [np.linspace(r_min, r_max, N), np.logspace(np.log(r_min), np.log(r_max), N, base=np.exp(1))]
    y = 0
    order = ['First Order', 'Higher Order']
    linny = [':', '--']
    for i in [True, False]:
        # Object used for the linearly spaced grid
        sedov = Hydro(gamma = gamma, initial_state=((1.0,1.0,0.0),(0.1,0.125,0.0)), 
                    Npts=N, geometry=(r_min, r_max, 0.5), n_vars=3)
    
        s = sedov.simulate(tend=tend, first_order=i, dt=dt, CFL=0.4, linspace=False, coordinates=b'spherical')
        
        ax.plot(r[1], s[0], linestyle=linny[y], label=order[y])
        
        y ^= 1
    
    # Make the plot pretty
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(r_min, r_max)
    ax.set_xlabel("R", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_title("1D Sod after t={:.1f}s with N = {}".format(tend, N), fontsize=20)
    ax.legend(fontsize=15)
    
    fig.savefig('sod.pdf')
    
    plt.show()
    


def loglin_sod():
    tend = 0.1
    fig, ax = plt.subplots(1, 1, figsize=(13,11))
    
    r_min = 0.1
    r_max = 1.
    
    r = [np.linspace(r_min, r_max, N), np.logspace(np.log(r_min), np.log(r_max), N, base=np.exp(1))]
    y = 0
    space = ['Linspace', 'Logspace']
    linny = [':', '--']
    for i in [True, False]:
        # Object used for the linearly spaced grid
        sedov = Hydro(gamma = gamma, initial_state=((1.0,1.0,0.0),(0.1,0.125,0.0)), 
                    Npts=N, geometry=(r_min, r_max, 0.5), n_vars=3)
    
        s = sedov.simulate(tend=tend, first_order=False, dt=dt, linspace=i)
        
        ax.plot(r[0], s[0], linestyle=linny[y], label=space[y])
        
        y ^= 1
    
    # Make the plot pretty
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(r_min, r_max)
    ax.set_xlabel("R", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)
    ax.set_title("1D Sod after t={:.1f}s with N = {}".format(tend, N), fontsize=20)
    ax.legend(fontsize=15)
    
    plt.show()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def energy(state):
    h = state[2]*(4/3*np.pi*r_max**3)
        
def plot_logvlin():
    fig, ax = plt.subplots(1, 1, figsize=(13,11))
    y = 0
    tend = 0.05
    for i in [True, False]:
        sedov = Hydro(gamma = gamma, initial_state=(rho, p ,v), 
              Npts=N, geometry=(r_min, r_max), n_vars=3)
        
        u = sedov.simulate(tend=tend, first_order=True, dt=dt, linspace=i)
        
        r = [np.linspace(r_min, r_max, N), np.logspace(np.log(r_min), np.log(r_max), N, base=np.exp(1))]
        linny = [':', '--']
        label = ['linspace', 'logspace']
        
        ax.plot(r[y], u[0], linestyle=linny[y], label=label[y])

        # Make the plot pretty
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(r_min, r_max)
        ax.set_xlabel("R", fontsize=15)
        ax.set_ylabel("Density", fontsize=15)
        ax.set_title("1D Sedov after t={:.3f} s at N = {}".format(tend, N), fontsize=20)
        ax.legend(fontsize=15)
        
        y ^= 1
    rs = (epsilon*tend**2/rho0)**(1/5)
    ax.axvline(rs)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    fig.savefig('Log_v_linear.pdf')
    plt.show()
        

        
    
        
    
    
# Object used for the linearly spaced grid
sedov = Hydro(gamma = gamma, initial_state=(rho, p ,v), 
            Npts=N, geometry=(r_min, r_max), n_vars=3)
def plot_rpv():
    
    re = 0.8
    dt = 1.e-6
    tend = (beta*re)**(5/2)*(rho0/epsilon)**(1/2)
    # Simulate with linearly-spaced radial zones
    u = sedov.simulate(tend=tend, first_order=False, dt=dt, linspace = True)

    # get the pressure and velocity
    p, v = sedov.cons2prim(u)[1: ]

    # Plot stuff
    fig, ax = plt.subplots(1, 1, figsize=(13,11))

    p_bar = p/np.max(p)
    v_bar = v/np.max(v)

    r = np.linspace(r_min, r_max, N)
    ax.plot(r, u[0], '-', label='Density')
    ax.plot(r, 2*v_bar, label=r'$2 \times v$')
    ax.plot(r, 4*p_bar, label=r'$4 \times p$')

    # Make the plot pretty
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.set_xticklabels(fontsize=15)
    #ax.set_yticklabels(fontsize=15)
    ax.set_xlim(r_min, r_max)
    ax.set_xlabel("R", fontsize=15)
    ax.set_title("1D Sedov after t={:.3f} s at N = {}".format(tend, N), fontsize=20)
    ax.legend(fontsize=15)
    
    rs = beta**(-1.)*tend**(2/5)
    r_peak = r[np.argmax(u[0])]
    print("R Max: {}".format(r_peak))
    print("R Shock: {}".format(rs))

    fig.savefig("Sedov_spherical_cpp.pdf")
    plt.show()

def alpha(nu, omega, gamma):
    return 32*np.pi/(3*(gamma**2 - 1)*(nu+2-omega)**2)

def calc_t(alpha, rho0, energy, r_s, nu, omega):
    return (alpha*rho0/energy)**(1/2)*(r_s)**((nu + 2 - omega)/2)

def plot_rho_vs_xi(omega = 0, spacing = 'linspace', dt=1.e-6):
    
    def R(epsilon,t, alpha, rho0=1., nu=3, omega=0):
        return (epsilon*t**2/(alpha*rho0))**(1/(nu+2-omega))
    
    def Rdot(epsilon, t, Rs):
        return (2*Rs/((nu+2-omega)*t))
    
    def D(gamma, rho, rho0):
        return ( (gamma-1.)/(gamma + 1.) ) * (rho/rho0)
    
    def V(gamma, v, Rdot):
        return ( (gamma+ 1)/2)*(v/Rdot)
    
    def P(gamma, p, rho0, Rdot):
        return ( (gamma + 1)/2)*(p/(rho0*Rdot**2))
    
    
    if spacing == 'linspace':
        r = np.linspace(r_min, r_max, N)
        space = True
    else:
        r = np.logspace(np.log(r_min), np.log(r_max), N, base=np.exp(1))
        space = False
        
    N_exp = 5
    dr = r[N_exp]
    p_exp = init_pressure(gamma, epsilon, nu, dr)
    
    delta_r = dr - r_min
    p_zones = find_nearest(r, (r_min + dr))[0]
    p_zones = int(p_zones/2.5)

    print(p_zones)
    #zzz = input('')
    p = np.zeros(N, float)
    p[: p_zones] = p_exp
    p[p_zones: ] = p_amb
    
    rho = rho0*r**(-omega)
    v = np.zeros(N, float)
        
    alph = alpha(nu, omega, gamma)
    re = 1.0
    rsedov = 10*(r_min + dr)
    tend = calc_t(alph, rho0, epsilon, re,  nu, omega)
    tmin = calc_t(alph, rho0, epsilon, rsedov, nu,  omega)

    ts = np.array([tmin, 2*tmin, tend])
    # Plot stuff
    fig = plt.figure(figsize=(15,9))
 
    for idx, t in enumerate(ts):
        ax = fig.add_subplot(1, 3, idx+1)
        
        sedov = Hydro(gamma = gamma, initial_state=(rho, p , v), 
                Npts=N, geometry=(r_min, r_max), n_vars=3)
        
        t1 = (time.time()*u.s).to(u.min)
        us = sedov.simulate(tend=t, first_order=False, CFL=0.4, dt=dt, linspace = space, coordinates=b'spherical')
        print("Simulation for t={} took {:.3f}".format(t,  (time.time()*u.s).to(u.min) - t1))
        pressure, vel = sedov.cons2prim(us)[1:]

        rs = R(epsilon, t, alpha=alph, nu=nu, omega=omega)
        
        rdot = Rdot(epsilon, t, rs)
        
        d = D(gamma, us[0], rho)
        vx = V(gamma, vel, rdot)
        px = P(gamma, pressure, rho, rdot)
        
        xi = r/(rs)
        max_idx = find_nearest(xi, 1)[0]
        print("R Max: {} at idx {}".format(r[np.argmax(us[0])], np.argmax(us[0])))
        print("R Shock: {} at idx {}".format(rs, max_idx))
        #zzz = input('')
        
        ax.plot(xi[: max_idx], d[: max_idx], ':', label=r'$D(\xi)$'.format(t))
        ax.plot(xi[: max_idx], vx[: max_idx], '.', label=r'$V(\xi)$'.format(t))
        ax.plot(xi[: max_idx], px[: max_idx], '--', label=r'$P(\xi)$'.format(t))
        
        # Make the plot pretty
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title("t = {:.3f}".format(t), fontsize=12)
        
        ax.set_xlim(xi[0], 1)
        ax.set_xlabel(r"$ \xi $", fontsize = 15)
        #ax.set_xlim(0, 1)
    
    
    
    #plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.suptitle("1D Sedov Self Similarity with N = {}, $\omega = {:.3f}, \gamma = {:.3f} $".format(N, omega, gamma),
                  fontsize=20)
    ax.legend(fontsize=15)

    fig.savefig("Sedov_densty_xi_linspace.pdf")
    plt.show()
 
#loglin_sod()
plot_rho_vs_xi(spacing='linspace', omega=0)