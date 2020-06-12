#! /usr/bin/env python

# A Hydro Code Useful for solving 1D structure problems
# Marcus DuPont
# New York University
# 06/10/2020

import numpy as np 
import matplotlib.pyplot as plt 

# Solving the 1D problem first
# dU/dt + dF/dt = 0
# U =[rho, rho*v,E)]^T
# F = [rho*v,rho*v**2+P,(E+P)*v]^T

# We solve this by employing the Forward-Euler method of the form:
# dU/dt = f(U,t) = -dF/dt = - (F[i+0.5] - F[i-1/2])/dx
# We then Taylor expand U(t) about time t to write
# U(t+dt) ~ U(t) + dt dU/dt = U(t) -dt((F[i+0.5] - F[i-1/2])/dx)
# And we finally solve for U(t)
# This method is called the Forward-Time Centered-Space (FTCS) method

# Here, since F (the flux) is a different variable from U, we must employ a different
# method to solve for it. We will use the approximate Riemann solver titled HLL
# source: https://www.scirp.org/pdf/JAMP_2015082610464230.pdf

# Here, I only write the solution
# F_HLL = (alpha_p * F_L + alpha_m*F_R - alpha_p*alpha_m*(U_R - U_L))/(alpha_p + alpha_m)
# where alpha_p + alpha_m are related to the minimal and maximal eigenvalues of the Jacobians
# of the left and right states in the form
# alpha_pm = MAX{0,pm eigen(U_L), pm eigen(U_R)}
# Here, the minal and maximum eigenvalues eigen_pm are given by
# eigen_pm = v pm c_s
# where c_s = sqrt(gamma*P/rho) is the sound speed. 

#Define the variables
N = 500                                 # Number of Grid Divisions
rho = np.empty(N+1, float)              # fluid density
v = 0.0                                 # fluid velocity in x-direction
h = np.empty(N+1, float)                # total energy
gamma = 1.4                             # adiabatic index 
p = (gamma-1.)*(h- 0.5*rho*v**2.)       # pressure EoS
dx = 1/N                                # step size
dt = 1.e-4                              # initial time step


def calc_pressure(gamma, energy, rho, velocity):
    """
    Calculate the ideal gass pressure given the adiabatic
    index, velocity, energy, and fluid density
    
    Parameters:
        gamma (float): The adiabatic index
        energy (float): The fluid energy
        rho (float): The fluid density
        velocity (float): The fluid (1d!) velocity
    """
    pressure = (gamma - 1)*(energy - 0.5*rho*velocity**2)
    
    return pressure

def calc_sound_speed(gamma, pressure, rho):
    """
    Compute the sound speed of the gas
    
    Parameters:
        gamma (float): Adiabatic index
        pressure (float): Fluid pressure
        rho (float): Fluid density
        
    Returns:
        c_s (float): Sound speed
    """
    
    c_s = np.sqrt(gamma*pressure/rho)
    return c_s

def eigenvals(left_state = (0.0,0.0,0.0), right_state = (0.0,0.0,0.0)):
    """
    Calculate the eigenvalues of the state tensors
    given the left and right conditions
    
    Parameters:
        left_state (tuple): The left state of the grid
        right_state (tuple): The right state of the grid
        
    Returns:
        lam (dict): The eigenvalue dictionary containing the necessary
        plus/minus left/right values.
    """
    
    # The state tensor decomposed into respective
    # variables
    rho_l, m_l, h_l = left_state
    rho_r, m_r, h_r = right_state
    
    v_l = m_l/rho_l
    v_r = m_r/rho_r
    
    p_l = calc_pressure(gamma, h_l, rho_l, v_l)
    p_r = calc_pressure(gamma, h_r, rho_r, v_r)
    
    # Compute Sound Speed
    c_s_right = calc_sound_speed(gamma, p_r, rho_r)
    c_s_left = calc_sound_speed(gamma, p_l, rho_l)
    
    
    # Initialize Dictionary to store plus/minus 
    # left/right eigenvalues
    lam = {}
    
    lam['left'] = {}
    lam['right'] = {}
    
    lam['left']['plus'] = v_l + c_s_left
    lam['left']['minus'] = v_l - c_s_left
    
    lam['right']['plus'] = v_r + c_s_right
    lam['right']['minus'] = v_r - c_s_right
    
    return lam 
    
def cons2prim(state = (1.0,0.0,0.1)):
    """
    Converts the state vector into the 
    respective primitive variables: 
    fluid density, momentum density,
    and energy
    
    Parameters:
        state (tuple): The state vector 
    
    Returns:
        rho (float): fluid density
        moment_dens (float): momentum density
        energy (float): fluid energy
    """
    rho = state[0]
    v = state[1]/rho 
    energy = state[2]
    
    return rho, v, energy
        
    
    
    
# Initialize u-vector array
u = np.empty(shape = (N+1,3), dtype=float)
f = np.empty(shape = (N+1,3), dtype=float)

# Initial Conditions
p_l = 1.0
rho_l = 1.0 
v_l = 0.0
h_l = p_l/(gamma - 1) + 0.5*rho_l*v_l**2
epsilon_l = rho_l*v_l**2 + p_l
beta_l = (h_l + p_l)*v_l

p_r = 0.125 
rho_r = 0.1 
v_r = 0.0 
h_r = p_r/(gamma - 1) + 0.5*rho_r*v_r**2
epsilon_r = rho_r*v_r**2 + p_r
beta_r = (h_r + p_r)*v_r


u[: int(N/2)] = np.array([rho_l, rho_l*v_l, h_l])                   # Left State
u[int(N/2): ] = np.array([rho_r, rho_r*v_r, h_r])                   # Right State
f[: int(N/2)] = np.array([rho_l*v_l, epsilon_l, beta_l])            # Left Flux
f[int(N/2): ] = np.array([rho_r*v_r, epsilon_r, beta_r])            # Right Flux


tend = 0.1
t = 0

# The state and flux profile tensors
up = u.copy()
fp = f.copy()

while t < tend:
    # Calculate the new values of U and F
    #print(f)
    for i in range(1,N):
        
        #Right cell interface
        u_l = u[i]
        u_r = u[i+1]
        f_l = f[i]
        f_r = f[i+1]
    
        lam = eigenvals(left_state = u_l, right_state = u_r)
        
        alpha_plus = max(0, lam['left']['plus'], 
                        lam['right']['plus'])
        
        alpha_minus = max(0, -lam['left']['minus'], 
                        -lam['right']['minus'])
        
        # The HLL flux calculated for f_[i+1/2]
        f1 = ( (alpha_plus*f_l + alpha_minus*f_r - 
                alpha_minus*alpha_plus*(u_r - u_l) ) /
                (alpha_minus + alpha_plus) )
    
        # Left cell interface
        u_l = u[i-1]
        u_r = u[i]
        
        f_l = f[i-1]
        f_r = f[i]
        
        lam = eigenvals(left_state = u_l, right_state = u_r)
        
        alpha_plus = max(0, lam['left']['plus'], 
                        lam['right']['plus'])
        
        alpha_minus = max(0, -lam['left']['minus'], 
                        -lam['right']['minus'])
        
        
        # The HLL flux calculated for f_[i-1/2]
        f2 = ( (alpha_plus*f_l + alpha_minus*f_r - 
                alpha_minus*alpha_plus*(u_r - u_l) ) /
                (alpha_minus + alpha_plus) )
        

        # Update the state vector using FTCS
        up[i] = u[i] - dt*(f1-f2)/dx
        
        #Calculate new flux from new primitive variables
        rho, v, energy = cons2prim(up[i])
        
        p = calc_pressure(gamma, energy, rho, v)
        momentum_dens = rho*v 
        energy_dens = rho*v**2 + p 
        beta = (energy + p)*v 

        fp[i] = np.array([momentum_dens, energy_dens,
                          beta])
        
    u, up = up, u
    f, fp = fp, f
    t += dt
        

        

# Boundaries
x_left = 0.0
x_right = 1.0

#X-Grid
x_arr = np.linspace(x_left, x_right, N+1)

# Separate the state vector into a human 
# readable dictionary
state = {}
state['density'] = u[:,0]
state['momentum_dens'] = u[:,1]
state['energy'] = u[:,2]

#Plot the results
plt.plot(x_arr, state['density'], label='Density')
plt.plot(x_arr, state['momentum_dens'], label='Momentum Density')
plt.plot(x_arr, state['energy'], label='Energy')
plt.xlabel('X')

plt.legend()
plt.show()