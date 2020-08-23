#!/usr/bin/env python


import numpy as np 
import matplotlib.pyplot as plt 
import math 
from simbi import Hydro 
import matplotlib as mpl




def R(rho, eps, nu, t):
    """
    Returns the characteristic radius 
    """
    return (eps*t**2/rho)**(1/(nu+2))
    
# Set up initial conditions
gamma = 1.4                                 # adiabatic index
epsilon = 1.                                # Initial Explosion Energy
p_amb = 1.e-5                               # Ambient Pressure
rho0 = 1.                                   # Ambient Density
mach = 1000.                                # Mach Number
cs = np.sqrt(gamma*p_amb/rho0)              # Sound speed of the medium
v_exp = mach*cs                             # The veolicty of the explosion 
nu = 3.                                     # Parameter that determines geometry of explosion
beta = 0.868

r_min = 0.1                                # Minimum radial coordinate
r_max = 1.                                  # Maximum radial coordinate
N = 1000                                    # Number of zones
dr = 0.01                                # Initial explosion radius
pressure_zone = int(dr*N)                   # The radius of the initial pressure

p_form = 3*(gamma-1.)*epsilon/((nu+1)*np.pi*dr**nu)  # Initial pressure
p_init = (gamma-1.)*rho0*v_exp**2                   # Another form of the initial pressure


v_exp = np.sqrt(p_form/((gamma-1.)*rho0))
p = np.zeros(N+1, float)
p[: pressure_zone] = p_form
p[pressure_zone: ] = p_amb
rho = np.ones(N+1, float)
v = np.zeros(N+1, float)

r_explosion = 0.9/0.1
tend = r_explosion/(v_exp)
#print(tend)
#zzz = input('')
tend = round(tend, 2)
#print("Tend: {}".format(tend))
#tend = 0.02


dt = 1.e-6

sod = (1.0,1.0,0.0),(0.1,0.125,0.0)
sed_init = (rho, p, v)

# Object used for the linearly spaced grid
sedov = Hydro(gamma = gamma, initial_state=(rho, p ,v), 
              Npts=N+1, geometry=(r_min, r_max), n_vars=3)

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
              Npts=N+1, geometry=(r_min, r_max), n_vars=3)
        
        u = sedov.simulate(tend=tend, first_order=True, dt=dt, linspace=i)
        
        r = [np.linspace(r_min, r_max, N+1), np.logspace(np.log(r_min), np.log(r_max), N+1, base=np.exp(1))]
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
        

        
    
        
    

def plot_rpv():
    # Simulate with linearly-spaced radial zones
    u = sedov.simulate(tend=tend, first_order=False, dt=dt, linspace = False)

    # get the pressure and velocity
    p, v = sedov.cons2prim(u)[1: ]

    # Plot stuff
    fig, ax = plt.subplots(1, 1, figsize=(13,11))

    p_bar = p/np.max(p)
    v_bar = v/np.max(v)

    r = np.logspace(np.log10(r_min), np.log10(r_max), N+1)
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

    fig.savefig("Sedov_spherical_4.pdf")
    plt.show()

def plot_rho_vs_xi():
    
    def R(epsilon,t, beta):
        return (1/beta)*(epsilon*t**2/rho0)**(1/5)
    
    def Rdot(epsilon, t, Rs):
        return (2*Rs/(5*t))
    
    def D(gamma, rho, rho0):
        return ( (gamma-1.)/(gamma + 1.) ) *(rho/rho0)
    
    def V(gamma, v, Rdot):
        return ( (gamma+ 1)/2)*(v/Rdot)
    
    def P(gamma, p, rho0, Rdot):
        return ( (gamma + 1)/2)*(p/(rho0*Rdot**2))
        
    
    ts = np.array([0.05, 0.1, tend])
    
    # Plot stuff
    fig, ax = plt.subplots(1, 1, figsize=(13,11))
    
    r = np.logspace(np.log10(r_min), np.log10(r_max), N+1)
    #r2 = np.linspace(r_min, r_max, N + 1)
    
    for t in ts:
        sedov = Hydro(gamma = gamma, initial_state=(rho, p , v), 
                Npts=N+1, geometry=(r_min, r_max), n_vars=3)
        
        u = sedov.simulate(tend=t, first_order=True, dt=dt, linspace = False)
        #pressure, v = sedov.cons2prim(u)[1:]
        
        rs = R(epsilon, t, beta)
        
        rdot = Rdot(epsilon, t, rs)
        d = D(gamma, u[0], rho0)
        
        xi = r/(2*rs)
        max_idx = find_nearest(xi, 1)[0]
        print("R Max: {} at idx {}".format(r[np.argmax(u[0])], np.argmax(u[0])))
        print("R Shock: {} at idx {}".format(rs, max_idx))
        #zzz = input('')
        
        ax.plot(xi[: ], d[: ], label='t = {:.3f} s'.format(t))
        #ax.set_xlim(0, 1)
    
    # Make the plot pretty
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #plt.setp(ax.get_xticklabels(), fontsize=15)
    #ax.set_xlim(0, 1.5)
    ax.set_xlabel(r"$ \xi $", fontsize = 25)
    ax.set_ylabel(r"$D(\xi)$", fontsize = 25)
    ax.set_title("1D Sedov Self Similarity with N = {}".format(N), fontsize=20)
    ax.legend(fontsize=15)

    fig.savefig("Sedov_densty_xi.pdf")
    plt.show()
    
plot_logvlin()