#! /usr/bin/env python

# A Hydro Code Useful for solving 1D structure problems
# Marcus DuPont
# New York University
# 06/10/2020

import numpy as np 
import matplotlib.pyplot as plt 
import math 
import sys

from state import PyState, PyState2D, PyStateSR, PyStateSR2D

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
# Here, the minimal and maximum eigenvalues eigen_pm are given by
# eigen_pm = v pm c_s
# where c_s = sqrt(gamma*P/rho) is the sound speed. 



class Hydro:
    
    def __init__(self, gamma, initial_state, Npts,
                 geometry=None, n_vars = 3, coord_system = 'cartesian',
                 regime = "classical"):
        """
        The initial conditions of the hydrodynamic system (1D for now)
        
        Parameters:
            gamma (float): Adiabatic Index
            initial_state (tuple or array): The initial conditions of the problem in the following format
                                Ex. state = ((1.0, 1.0, 0.0), (0.1,0.125,0.0)) for Sod Shock Tube
                                    state = (rho, pressure, velocity)
            Npts (int): Number of grid slices to make
            geometry (tuple): The first starting point, the last, and an optional midpoint in the grid
                                Ex. geometry = (0.0, 1.0, 0.5) for Sod Shock Tube
            n_vars (int): Number of variables in the problem
            
        Return:
            None
        """
        
        # hydro = Hydro(gamma=1.4, initial_state = ((1.0,0.0,1.0),(0.125,0.0,0.1)),
        # Npts=500, geometry=(0.0,1.0,0.5), n=3) 
        
        self.regime = regime
        discontinuity = False
        
        #Check dimensions of state
        if len(initial_state) == 2:
            print('Initializing the 1D Discontinuity...')
            
            discontinuity = True
            left_state = initial_state[0]
            right_state = initial_state[1]
            
            self.left_state = left_state
            self.right_state = right_state 
            
            if len(left_state) != len(right_state):
                print("ERROR: The left and right states must have the same number of variables")
                print('Left State:', left_state)
                print('Right State:', right_state)
                sys.exit()
                
            elif len(left_state) > 4 and len(right_state) > 4:
                print("Your state arrays contain too many variables. This version takes a maximum\n"
                    "of 4 state variables")
                
            elif len(left_state) == 3 and len(right_state) == 3:
                self.dimensions = 1
                
            elif len(left_state) == 4 and len(right_state) == 4:
                self.dimensions = 2
        
        self.gamma = gamma 
        self.geometry = geometry
        # self.dt = dt 
        self.Npts = Npts 
        self.n_vars = n_vars
        # step size
                                        
        # Initial Conditions
        
        # Check for Discontinuity
        if discontinuity:
            # Primitive Variables on LHS
            rho_l = self.left_state[0]
            p_l = self.left_state[1]
            v_l = self.left_state[2]
            
            # Primitive Variables on RHS
            rho_r = self.right_state[0]
            p_r = self.right_state[1]
            v_r = self.right_state[2]
        
            if self.regime == "classical":
                # Calculate Energy Density on LHS
                energy_l = p_l/(self.gamma - 1) + 0.5*rho_l*v_l**2
                
                # Calculate Energy Density on RHS
                energy_r = p_r/(self.gamma - 1) + 0.5*rho_r*v_r**2
            else:
                W_l = 1/np.sqrt(1 - v_l**2)
                W_r = 1/np.sqrt(1 - v_r**2)
                h_l = 1 + self.gamma*p_l/((self.gamma - 1)*rho_l)
                h_r = 1 + self.gamma*p_r/((self.gamma - 1)*rho_r)
                
                D_l = rho_l*W_l 
                D_r = rho_r*W_r 
                
                S_l = rho_l*h_l*W_l**2 * v_l
                S_r = rho_r*h_r*W_r**2 * v_r 
                
                tau_l = rho_l*h_l*W_l**2 - p_l - W_l*rho_l
                tau_r = rho_r*h_r*W_r**2 - p_r - W_r*rho_r
            

            # Initialize conserved u-tensor and flux tensors (defaulting to 2 ghost cells)
            self.u = np.empty(shape = (3, self.Npts + 2), dtype=float)

            left_bound = self.geometry[0]
            right_bound = self.geometry[1]
            midpoint = self.geometry[2]
            
            lx = right_bound - left_bound
            self.dx = lx/self.Npts
            
            size = abs(right_bound - left_bound)
            breakpoint = size/midpoint                                          # Define the fluid breakpoint
            slice_point = int((self.Npts+2)/breakpoint)                             # Define the array slicepoint
            
            if self.regime == "classical":
                self.u[:, : slice_point] = np.array([rho_l, rho_l*v_l, energy_l]).reshape(3,1)              # Left State
                self.u[:, slice_point: ] = np.array([rho_r, rho_r*v_r, energy_r]).reshape(3,1)              # Right State
            else:
                #Create the Lorentz factor array to account for each fluid cell and plit it accordingly
                self.W = np.zeros(self.Npts + 2)
                self.W[: slice_point] = W_l
                self.W[slice_point: ] = W_r
                
                
                self.u[:, : slice_point] = np.array([D_l, S_l, tau_l]).reshape(3,1)              # Left State
                self.u[:, slice_point: ] = np.array([D_r, S_r, tau_r]).reshape(3,1)              # Right State
                
        elif len(initial_state) == 3:
            self.dimensions = 1
            
            
            left_bound = self.geometry[0]
            right_bound = self.geometry[1]
            
            length = right_bound - left_bound
            self.dx = length/self.Npts
            
            self.n_vars = n_vars
            
            self.init_rho = initial_state[0]
            self.init_pressure = initial_state[1]
            
            if regime == "classical":
                self.init_v = initial_state[2]
                self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 
                                    0.5*self.init_rho*self.init_v**2 )
                
            else:
                self.init_v = initial_state[2]
                self.W = np.asarray(1/np.sqrt(1 - self.init_v**2))
                self.init_h = 1 + self.gamma*self.init_pressure/((self.gamma - 1)*self.init_rho)
                
                self.initD = self.init_rho*self.W
                self.initS = self.init_h*self.init_rho*self.W**2*self.init_v
                self.init_tau = (self.init_rho*self.init_h*self.W**2 - self.init_pressure
                                  - self.init_rho*self.W)
            
            self.u= None 
            
            
            
        elif len(initial_state) == 4:
            # TODO: Make this work
            self.dimensions = 2
            print('Initializing 2D Setup...')
            print('')
            
            left_x, right_x = geometry[0]
            left_y, right_y = geometry[1]
            
            lx = right_x - left_x
            ly = right_y - left_y
            
            self.xNpts, self.yNpts = Npts 
            
            # self.dx = lx/self.Npts
            # self.dy = ly/self.Npts
            
            self.n_vars = n_vars 
            
            if self.regime == "classical":
                self.init_rho = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_vx = initial_state[2]
                self.init_vy = initial_state[3]
                
                total_v = np.sqrt(self.init_vx**2 + self.init_vy**2)
                
                self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 
                                    0.5*self.init_rho*total_v**2 )
                
                
            else:
                self.init_rho = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_v1 = initial_state[2]
                self.init_v2 = initial_state[3]
                total_v = np.sqrt(self.init_v1**2 + self.init_v2**2)
                
                self.W = np.asarray(1/np.sqrt(1 - total_v**2))
                
                self.init_h = 1 + self.gamma*self.init_pressure/((self.gamma - 1)*self.init_rho)
                
                self.initD = self.init_rho*self.W
                self.initS1 = self.init_h*self.init_rho*self.W**2*self.init_v1
                self.initS2 = self.init_h*self.init_rho*self.W**2*self.init_v2 
                
                self.init_tau = (self.init_rho*self.init_h*(self.W)**2 - self.init_pressure
                                  - self.init_rho*(self.W))
            
            
            
                                                   
            
            
            
            self.u = None 
            
    def calc_flux(self, rho, pressure, velocity, x_direction=True):
        """
        Calculate the new flux tensor given the necessary
        primitive parameters
        
        Parameters:
            rho (array_like): Fluid density
            pressure (array_like): Fluid pressure
            velocity (array_like): Fluid velocity. If vector, input as velocity = (vx,vy)
            x_direction (boolean): Check if calculate x-direction flux. If False, will calculate 
                                   y-direction flux instead.
                                   
        Returns:
            flux (array_like): The calculate flux tensor
        """
        # Check if velocity is multi-dimensional
        if self.dimensions == 1:
            energy = self.calc_energy(self.gamma, rho, pressure, velocity)
            momentum_dens = rho*velocity
            energy_dens = rho*velocity**2 + pressure
            beta = (energy + pressure)*velocity 
            
            flux = np.array([momentum_dens, energy_dens,
                                        beta])
            
            return flux
        else:
            v_x = velocity[0]
            v_y = velocity[1]
            total_velocity = np.sqrt(v_x**2 + v_y**2)
            
            energy = self.calc_energy(self.gamma, rho, pressure, total_velocity)
            
            if x_direction:
                momentum_x = rho*v_x
                energy_dens_x = rho*v_x**2 + pressure 
                convect_x = rho*v_x*v_y 
                beta_x = (energy + pressure)*v_x
                
                flux = np.array([momentum_x, energy_dens_x,
                                 convect_x, beta_x])
                
                return flux
            else:
                momentum_y = rho*v_y
                energy_dens_y = rho*v_y**2 + pressure 
                convect_y = rho*v_x*v_y 
                beta_y = (energy + pressure)*v_y
                
                flux = np.array([momentum_y, convect_y,
                                 energy_dens_y, beta_y])
                
                return flux
        
        
    def calc_state(self, gamma, rho, pressure, velocity):
        """
        Calculate the new state tensor given the parameters
        """
        #state = np.empty(shape = (3, self.Npts + ghosts), dtype=float)
        if self.dimensions == 1:
            energy = self.calc_energy(gamma, rho, pressure, velocity)
            
            u = np.array([rho, rho*velocity, energy])
            
            return u
        else:
            vx = velocity[0]
            vy = velocity[1]
            
            total_velocity = np.sqrt(vx**2 + vy**2)
            energy = self.calc_energy(gamma, rho, pressure, total_velocity)
            
            u = np.array([rho, rho*vx, rho*vy, energy])
            
            return u
    
    def calc_eigenvals(self, left_state = (0.0,0.0,0.0), right_state = (0.0,0.0,0.0)):
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
        
        if self.dimensions == 1:
            rho_l, momentum_l, energy_l = left_state
            rho_r, momentum_r, energy_r = right_state
            
            v_l = momentum_l/rho_l
            v_r = momentum_r/rho_r
            
            p_l = self.calc_pressure(self.gamma, rho_l, energy_l, v_l)
            p_r = self.calc_pressure(self.gamma, rho_r, energy_r, v_r)
            
            # Compute Sound Speed
            c_s_right = self.calc_sound_speed(self.gamma, p_r, rho_r)
            c_s_left = self.calc_sound_speed(self.gamma, p_l, rho_l)
            
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
        
        else:
            
            rho_l, momentum_x_l, momentum_y_l, energy_l = left_state
            rho_r, momentum_x_r, momentum_y_r, energy_r = right_state
            
            v_x_l = momentum_x_l/rho_l 
            v_y_l = momentum_y_l/rho_l
            
            v_x_r = momentum_x_r/rho_r
            v_y_r = momentum_y_r/rho_r 
            
            v_tot_l = np.sqrt(v_x_l**2 + v_y_l**2)
            v_tot_r = np.sqrt(v_x_r**2 + v_y_r**2)
            
            p_l = self.calc_pressure(self.gamma, rho_l, energy_l, v_tot_l)
            p_r = self.calc_pressure(self.gamma, rho_r, energy_r, v_tot_r)
            
            # Compute Sound Speed
            c_s_right = self.calc_sound_speed(self.gamma, p_r, rho_r)
            c_s_left = self.calc_sound_speed(self.gamma, p_l, rho_l)
            
            # Initialize Dictionary to store plus/minus 
            # left/right eigenvalues
            lam = {}
            
            lam['left'] = {}
            lam['right'] = {}
            
            lam['left']['plus'] = v_tot_l + c_s_left
            lam['left']['minus'] = v_tot_l - c_s_left
            
            lam['right']['plus'] = v_tot_r + c_s_right
            lam['right']['minus'] = v_tot_r - c_s_right
            
            return lam 
        
    
    def calc_hll_flux(self, left_state, right_state, 
                      left_flux, right_flux):
        """
        Computes the HLL flux for the left/right shell
        interface
        """
        lam = self.calc_eigenvals(left_state = left_state, 
                             right_state = right_state)
        
        null = np.zeros(lam['left']['plus'].shape)
                    
        alpha_plus = np.maximum.reduce([null, lam['left']['plus'], 
                        lam['right']['plus']])
        
        alpha_minus = np.maximum.reduce([null, -lam['left']['minus'], 
                        -lam['right']['minus']])
        
        #print(alpha_plus)
        #zzz = input('')
        
        f_hll = ( (alpha_plus*left_flux + alpha_minus*right_flux - 
                alpha_minus*alpha_plus*(right_state - left_state) ) /
                (alpha_minus + alpha_plus) )
        
        return f_hll 
    
    def calc_energy(self, gamma, rho, pressure, velocity):
        """
        Calculate the ideal gas energy given the adiabatic
        index, velocity, pressure, and fluid density
        
        Parameters:
            gamma (float): The adiabatic index
            pressure (float): The fluid pressure
            rho (float): The fluid density
            velocity (float): The fluid (1d!) velocity
            
        ReturnsL
            energy (float): The fluid energy
        """
        energy = pressure/(gamma - 1.) + 0.5*rho*velocity**2
        
        return energy
    
    def calc_pressure(self, gamma, rho, energy, velocity):
        """
        Calculate the ideal gas pressure given the adiabatic
        index, velocity, energy, and fluid density
        
        Parameters:
            gamma (float): The adiabatic index
            pressure (float): The fluid pressure
            rho (float): The fluid density
            velocity (float): The fluid (1d!) velocity
            
        ReturnsL
            pressure (float): The fluid pressure
        """
        
        pressure = (gamma - 1.)*(energy - 0.5*rho*velocity**2)
            
        return pressure

    def calc_sound_speed(self, gamma, pressure, rho):
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
        
    def cons2prim(self, state = (1.0,0.0,0.1)):
        """
        Converts the state vector into the 
        respective primitive variables: 
        fluid density, momentum density,
        and energy
        
        Parameters:
            state (array or tuple): The state vector 
        
        Returns:
            rho (float): fluid density
            moment_dens (float): momentum density
            pressure (float): fluid pressure
        """
        # Check dimensions of the states variable
        if self.dimensions == 1:
        
            rho = state[0]
            v = state[1]/rho 
            energy = state[2]
            
            pressure = self.calc_pressure(self.gamma, rho, energy, v)
            
            return np.array([rho, pressure, v])
        else:
            rho = state[0]
            v_x = state[1]/rho 
            v_y = state[2]/rho 
            energy = state[3]
            
            v = np.sqrt(v_x**2 + v_y**2)
            
            pressure = self.calc_pressure(self.gamma, rho, energy, v)
            
            return np.array([rho, pressure, v_x, v_y]) 
    
    #@nb.jit # numba this function
    def u_dot(self, state, first_order = True, theta = 1.5, periodic=False, coordinates='cartesian'):
        """
        """
        
        def sign(y):
            """
            The mathematical sign function
            returning either +1 or -1
            based on the inputs
            
            Parameters:
                y (float/array): Some number
                
            Returns:
                sgn (float/array): Sign of the number y
            """
            if isinstance(y, (int, float)):
                sgn = math.copysign(1,y)
                return sgn
            elif isinstance(y, (list, np.ndarray)):
                try:
                    y = np.array(y)
                except:
                    pass
                
                sgn = np.sign(y)
                return sgn
                
            else:
                raise ValueError('The sign function only takes natural numbers')
            
       
        def minmod(x, y ,z):
            """
            A flux delimiter function
            
            Parameters:
                x (float): Some Number
                y (float): Idem 
                z (float): Idem
                
            Returns:
                values (float): Either x,y,z based on standard minmod requirements
            """
            # Must vectorize each parameter to compute the minimum
            v = [np.abs(arr) for arr in [x,y,z]]
            
            return (
                0.25*np.abs(sign(x) + sign(y))*
                    (sign(x) + sign(z))*np.asarray(v).min(0)
            )
         
        if self.dimensions == 1:
            if first_order:
                # Right cell interface
                if periodic:
                    u_l = state 
                    u_r = np.roll(state, 1, axis=1)
                    
                else:
                    u_l = state[:, 1: self.Npts + 1]
                    u_r = state[:, 2: self.Npts + 2]
                
                prims_l = self.cons2prim(state = u_l)
                prims_r = self.cons2prim(state = u_r)
                
                pc = prims_l[1]
                
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)
                
                # The HLL flux calculated for f_[i+1/2]
                f1 = self.calc_hll_flux(u_l, u_r, f_l, f_r)

                # Left cell interface
                if periodic:
                    u_l = np.roll(state, -1, axis=1) 
                    u_r = state
                    
                else:
                    u_l = state[:, 0: self.Npts]
                    u_r = state[:, 1: self.Npts + 1]
                
                prims_l = self.cons2prim(state = u_l)
                prims_r = self.cons2prim(state = u_r)
                
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)
                
                # The HLL flux calculated for f_[i-1/2]
                f2 = self.calc_hll_flux(u_l,u_r,f_l,f_r)
                
                if (coordinates == 'cartesian'):
                    return -(f1 - f2)/self.dx
                else:
                    r_max = self.geometry[1]
                    r_min = self.geometry[0]
                    
                    #delta_logr = np.log(r_max/r_min)/self.Npts
                    
                    r = np.linspace(r_min, r_max, self.Npts)
                    r = np.insert(r, 0, r[0])
                    r = np.insert(r, -1, r[-1])
                    
                    r_left = 0.5*(r[0: self.Npts] + r[1: self.Npts+1])
                    r_right = 0.5*(r[2:self.Npts+2] + r[1:self.Npts+1])
                    volAvg = 0.75*( (r_right**4 - r_left**4) / (r_right**3 - r_left**3))
                    dr = r_right - r_left
                    
                    L = np.zeros((self.n_vars, self.Npts))
                    L[0] = -(r_right**2*f1[0] - r_left**2*f2[0])/(volAvg**2 *dr)
                    L[1] = -(r_right**2*f1[1] - r_left**2*f2[1])/(volAvg**2 *dr) + 2*pc/volAvg
                    L[2] = -(r_right**2*f1[2] - r_left**2*f2[2])/(volAvg**2 *dr)
                    
                    
                    return L
            else:
                # Calculate the primitives at the central interface
                prims = self.cons2prim(state = state)
                    
                # Right cell interface
                if periodic:
                    left_most = np.roll(prims, -2, axis=1)
                    left_mid = np.roll(prims, -1, axis=1)
                    center = prims
                    right_mid = np.roll(prims, 1, axis=1)
                    right_most = np.roll(prims, 2, axis=1)
                else:
                    left_most = prims[:, 0:self.Npts]
                    left_mid = prims[:, 1:self.Npts + 1]
                    center = prims[:, 2:self.Npts + 2]
                    right_mid = prims[:,3: self.Npts + 3]
                    right_most = prims[:, 4: self.Npts + 4]
                
                
                prims_l = ( center + 0.5*
                        minmod(theta*(center - left_mid),
                                0.5*(right_mid - left_mid),
                                theta*(right_mid - center))
                            )
                
                
                prims_r = (right_mid - 0.5 *
                        minmod(theta*(right_mid -center),
                                0.5*(right_most - center),
                                theta*(right_most - right_mid))
                            )
                
                pc = prims_l[1]
                # Calculate the reconstructed left and right 
                # states using the higher order primitives
                u_l = self.calc_state(self.gamma, *prims_l)
                u_r = self.calc_state(self.gamma, *prims_r)
               
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)


                # The HLL flux calculated for f_[i+1/2]
                f1 = self.calc_hll_flux(u_l, u_r, f_l, f_r)
                
                # Left cell interface
                prims_l = (left_mid + 0.5 *
                        minmod(theta*(left_mid - left_most),
                                0.5*(center -left_most),
                                theta*(center - left_mid))
                            )
                
                
                prims_r = (center - 0.5 *
                        minmod(theta*(center - left_mid),
                                0.5*(right_mid - left_mid),
                                theta*(right_mid - center))
                            )
                
                
                u_l = self.calc_state(self.gamma, *prims_l)
                u_r = self.calc_state(self.gamma, *prims_r)
                
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)
                
                # The HLL flux calculated for f_[i-1/2]
                f2 = self.calc_hll_flux(u_l, u_r, f_l, f_r)
                
            if (coordinates == 'cartesian'):
                return -(f1 - f2)/self.dx
            else:
                r_max = self.geometry[1]
                r_min = self.geometry[0]
                
                #delta_logr = np.log(r_max/r_min)/self.Npts
                
                r = np.linspace(r_min, r_max, self.Npts)
                r = np.insert(r, 0, r[0])
                r = np.insert(r, -1, r[-1])
                
                r_left = 0.5*(r[0: self.Npts] + r[1: self.Npts+1])
                r_right = 0.5*(r[2:self.Npts+2] + r[1:self.Npts+1])
                volAvg = 0.75*( (r_right**4 - r_left**4) / (r_right**3 - r_left**3))
                dr = r_right - r_left
                
                #print(dr)
                #zzz = input('')
                
                L = np.zeros((self.n_vars, self.Npts))
                L[0] = -(r_right**2*f1[0] - r_left**2*f2[0])/(volAvg**2 *dr)
                L[1] = -(r_right**2*f1[1] - r_left**2*f2[1])/(volAvg**2 *dr) + 2*pc/volAvg
                L[2] = -(r_right**2*f1[2] - r_left**2*f2[2])/(volAvg**2 *dr)
                
                
                return L
        else:
            # TODO: Do the Higher Order 2-D Advection Problem
            # No Reason to do first order in 2D so ignore it
            
            # Calculate the primitives at the central interface
            prims = self.cons2prim(state = state)
                
            # Right cell interface
            if periodic:
                left_most = np.roll(prims, -2, axis=2)
                left_mid = np.roll(prims, -1, axis=2)
                center = prims
                right_mid = np.roll(prims, 1, axis=2)
                right_most = np.roll(prims, 2, axis=2)
            else:
                x_left_most = prims[:, 2: self.Npts, 0:self.Npts - 2]          # C_[i-2, j]
                y_left_most = prims[:, 0: self.Npts-2, 2: self.Npts]
                x_left_mid = prims[:, 2:self.Npts, 1: self.Npts-1]
                y_left_mid = prims[:, 1:self.Npts-1, 2:self.Npts]
                center = prims[:, 2:self.Npts, 2:self.Npts]
                x_right_mid = prims[:,  2: self.Npts, 3:self.Npts+1]
                y_right_mid = prims[:, 3:self.Npts+1, 2: self.Npts]
                x_right_most = prims[:, 2: self.Npts, 4:self.Npts+2]
                y_right_most = prims[:, 4:self.Npts+2, 2: self.Npts]      # C_[i, j+2]
                
            x_prims_l = ( center + 0.5*
                    minmod(theta*(center - x_left_mid),
                            0.5*(x_right_mid - x_left_mid),
                            theta*(x_right_mid - center))
                        )
            
            
            x_prims_r = (x_right_mid - 0.5 *
                    minmod(theta*(x_right_mid -center),
                            0.5*(x_right_most - center),
                            theta*(x_right_most - x_right_mid))
                        )
            
            
            y_prims_l = ( center + 0.5*
                    minmod(theta*(center - y_left_mid),
                            0.5*(y_right_mid - y_left_mid),
                            theta*(y_right_mid - center))
                        )
            
            
            y_prims_r = (y_right_mid - 0.5 *
                    minmod(theta*(y_right_mid -center),
                            0.5*(y_right_most - center),
                            theta*(y_right_most - y_right_mid))
                        )
            
            # Calculate the reconstructed left and right 
            # states using the higher order primitives
            ux_l = self.calc_state(self.gamma, rho=x_prims_l[0], 
                                  pressure = x_prims_l[1], 
                                  velocity = (x_prims_l[2], x_prims_l[3]))
            
            ux_r = self.calc_state(self.gamma, rho=x_prims_r[0], 
                                  pressure = x_prims_r[1], 
                                  velocity = (x_prims_r[2], x_prims_r[3]))
            
            uy_l = self.calc_state(self.gamma, rho=y_prims_l[0], 
                                  pressure = y_prims_l[1], 
                                  velocity = (y_prims_l[2], y_prims_l[3]))
            
            uy_r = self.calc_state(self.gamma, rho=y_prims_r[0], 
                                  pressure = y_prims_r[1], 
                                  velocity = (y_prims_r[2], y_prims_r[3]))
            
            # The fluxes in the x-direction(f) and y-direction (g)
            f_l = self.calc_flux(rho = x_prims_l[0], pressure=x_prims_l[1],
                                 velocity=(x_prims_l[2], x_prims_l[3]))
            
            g_l = self.calc_flux(rho = y_prims_l[0], pressure=y_prims_l[1],
                                 velocity=(y_prims_l[2], y_prims_l[3]), 
                                 x_direction = False)
            
            f_r = self.calc_flux(rho = x_prims_r[0], pressure=x_prims_r[1],
                                 velocity=(x_prims_r[2], x_prims_r[3]))
            
            g_r = self.calc_flux(rho = y_prims_r[0], pressure= y_prims_r[1],
                                 velocity=(y_prims_r[2], y_prims_r[3]),
                                 x_direction = False)
            
            
            
            #print("Rho (R): {}".format(y_prims_r[0, 3, 5]))
            #print("Rho (L): {}".format(y_prims_l[0, 3, 5]))
            #print("P (R): {}".format(y_prims_r[1, 3, 5]))
            #print("P (L): {}".format(y_prims_l[1, 3, 5]))
            #print("Vx (R): {}".format(y_prims_r[2, 3, 5]))
            #print("Vx (L): {}".format(y_prims_l[2, 3, 5]))
            #print("Vy (R): {}".format(y_prims_r[3, 3, 5]))
            #print("Vy (L): {}".format(y_prims_l[, 3, 5]))
            #print("F (R): {}".format(f_l[3, 3, 5]))
            #print("F (L): {}".format(f_r[3, 3, 5]))
            #print("G (R): {}".format(g_r[3, 3, 5]))
            #print("G (L): {}".format(g_l[3, 3, 5]))
            #zzz = input('')

            # The HLL flux calculated for f_[i+1/2, j]
            f1 = self.calc_hll_flux(ux_l, ux_r, f_l, f_r)
            
            # The HLL flux calculated for g_[i, j+1/2]
            g1 = self.calc_hll_flux(uy_l, uy_r, g_l, g_r)
            
            print("i,j + 1/2")
            print("Ux (R): {}".format(ux_r[3, 3, 5]))
            print("Ux (L): {}".format(ux_l[3, 3, 5]))
            print("Uy (R): {}".format(uy_r[3, 3, 5]))
            print("Uy (L): {}".format(uy_l[3, 3, 5]))
            print("F (R): {}".format(f_r[3, 3, 5]))
            print("F (L): {}".format(f_l[3, 3, 5]))
            print("G (R): {}".format(g_r[3, 3, 5]))
            print("G (L): {}".format(g_l[3, 3, 5]))
            
            # Left cell interface
            x_prims_l = ( x_left_mid + 0.5*
                    minmod(theta*(x_left_mid - x_left_most),
                            0.5*(center - x_left_most),
                            theta*(center - x_left_mid))
                        )
            
            
            x_prims_r = (center - 0.5 *
                    minmod(theta*(center -x_left_mid),
                            0.5*(x_right_mid - x_left_mid),
                            theta*(x_right_mid - center))
                        )
            
            
            y_prims_l = ( y_left_mid + 0.5*
                    minmod(theta*(y_left_mid - y_left_most),
                            0.5*(center - y_left_most),
                            theta*(center - y_left_mid))
                        )
            
            
            y_prims_r = (center - 0.5 *
                    minmod(theta*(center - y_left_mid),
                            0.5*(y_right_mid - y_left_mid),
                            theta*(y_right_mid - center))
                        )
            
            
            
            # Calculate the reconstructed left and right 
            # states using the higher order primitives
            ux_l = self.calc_state(self.gamma, rho=x_prims_l[0], 
                                  pressure = x_prims_l[1], 
                                  velocity = (x_prims_l[2], x_prims_l[3]))
            
            ux_r = self.calc_state(self.gamma, rho=x_prims_r[0], 
                                  pressure = x_prims_r[1], 
                                  velocity = (x_prims_r[2], x_prims_r[3]))
            
            uy_l = self.calc_state(self.gamma, rho=y_prims_l[0], 
                                  pressure = y_prims_l[1], 
                                  velocity = (y_prims_l[2], y_prims_l[3]))
            
            uy_r = self.calc_state(self.gamma, rho=y_prims_r[0], 
                                  pressure = y_prims_r[1], 
                                  velocity = (y_prims_r[2], y_prims_r[3]))
            
            # The fluxes in the x-direction (f) and y-direction (g)
            f_l = self.calc_flux(rho = x_prims_l[0], pressure=x_prims_l[1],
                                 velocity=(x_prims_l[2], x_prims_l[3]))
            
            g_l = self.calc_flux(rho = y_prims_l[0], pressure=y_prims_l[1],
                                 velocity=(y_prims_l[2], y_prims_l[3]), 
                                 x_direction=False)
            
            f_r = self.calc_flux(rho = x_prims_r[0], pressure=x_prims_r[1],
                                 velocity=(x_prims_r[2], x_prims_r[3]))
            
            g_r = self.calc_flux(rho = y_prims_r[0], pressure= y_prims_r[1],
                                 velocity=(y_prims_r[2], y_prims_r[3]),
                                 x_direction = False)
            
            # The HLL flux calculated for f_[i-1/2, j]
            f2 = self.calc_hll_flux(ux_l, ux_r, f_l, f_r)
            
            # The HLL flux calculated for g_[i, j-1/2]
            g2 = self.calc_hll_flux(uy_l, uy_r, g_l, g_r)
            
            print("")
            print("i,j - 1/2")
            print("Ux (R): {}".format(ux_r[3, 3, 5]))
            print("Ux (L): {}".format(ux_l[3, 3, 5]))
            print("Uy (R): {}".format(uy_r[3, 3, 5]))
            print("Uy (L): {}".format(uy_l[3, 3, 5]))
            print("F (R): {}".format(f_r[3, 3, 5]))
            print("F (L): {}".format(f_l[3, 3, 5]))
            print("G (R): {}".format(g_r[3, 3, 5]))
            print("G (L): {}".format(g_l[3, 3, 5]))
            
            #print("F1: {}".format(f1[3, 3, 5]))
            #print("G1: {}".format(g1[3, 3, 5]))
            #print("F2: {}".format(f2[3, 3, 5]))
            #print("G2: {}".format(g2[3, 3, 5]))
            #zzz = input('')
           
            L = -(f1 - f2)/self.dx - (g1 -g2)/self.dy
            
            return L
            
        
    def adaptive_timestep(self, u, CFL=0.4):
        """
        Returns the adjustable timestep based
        on the Courant number C = alpha* delta_t/delta_x < 1
        """
        min_dt = 0
        p, v = self.cons2prim(u)[1:, 1:self.Npts+1]
        cs = self.calc_sound_speed(self.gamma, p, v)
        
        r_max = self.geometry[0]
        r_min = self.geometry[1]
        
        r = np.linspace(r_max, r_min, self.Npts)
        r = np.insert(r, 0, r[0])
        r = np.insert(r, -1, r[-1])
        
        r_left = 0.5*(r[0: self.Npts] + r[1: self.Npts+1])
        r_right = 0.5*(r[2:self.Npts+2] + r[1:self.Npts+1])
        dr = r_right - r_left
        min_dt = CFL*min(dr/np.maximum(np.abs(v + cs), np.abs(v - cs)))
        
        
        return min_dt
    
    def _initialize_simulation(self):
        """
        Initialize the hydro simulation based on 
        init params
        """
        
        self._results = Hydro(
            gamma = self.gamma,
            left_state = self.left_state,
            right_state = self.right_state,
            Npts = self.Npts,
            geometry = self.geometry,
            dt = self.dt, 
            dimensions = self.dimensions
        )
    
    #@nb.jit # numba this function
    def simulate(self, tend=0.1, dt = 1.e-4, 
                 first_order=True, periodic=False, linspace=True,
                 coordinates=b"cartesian", CFL=0.4, sources = None):
        """
        Simulate the hydro setup
        
        Parameters:
            tend (float): The desired time to end the simulation
            first_order (bollean): The switch the runs the FTCS method
            or the RK3 method in time.
            
        Returns:
            u (array): The conserved variable tensor
        """
        # Initialize conserved u-tensor
        
        self.u = np.asarray(self.u)
        
        # Check if u-tensor is empty. If it is, generate an array.
        if self.dimensions == 1:
            if not self.u.any():
                if periodic:
                    if self.regime == "classical":
                        self.u = np.empty(shape = (self.n_vars, self.Npts), dtype = float)
                        
                        self.u[:, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                                self.init_energy])
                    else:
                        self.u = np.empty(shape = (self.n_vars, self.Npts), dtype = float)
                        
                        self.u[:, :] = np.array([self.initD, self.initS, 
                                                self.init_tau])
                        
                else:
                    if first_order:
                        if self.regime == "classical":
                            self.u = np.empty(shape = (self.n_vars, self.Npts), dtype=float)
                            self.u[:, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                                self.init_energy])
                            
                            # Add boundary ghosts
                            right_ghost = self.u[:, -1]
                            left_ghost = self.u[:, 0]
                            
                            self.u = np.insert(self.u, self.u.shape[-1], right_ghost , axis=1)
                            self.u = np.insert(self.u, 0, left_ghost , axis=1)
                            
                        else:
                            self.u = np.empty(shape = (self.n_vars, self.Npts), dtype=float)
                            self.u[:, :] = np.array([self.initD, self.initS, 
                                                self.init_tau])
                            
                            # Add boundary ghosts
                            right_ghost = self.u[:, -1]
                            left_ghost = self.u[:, 0]
                            
                            self.u = np.insert(self.u, self.u.shape[-1], right_ghost , axis=1)
                            self.u = np.insert(self.u, 0, left_ghost , axis=1)
                            
                        
                    else:
                        if self.regime == "classical":
                            self.u = np.empty(shape = (self.n_vars, self.Npts), dtype=float)
                            self.u[:, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                                self.init_energy])
                            
                            # Add boundary ghosts
                            right_ghost = self.u[:, -1]
                            left_ghost = self.u[:, 0]
                            
                            self.u = np.insert(self.u, self.u.shape[-1], 
                                            (right_ghost, right_ghost) , axis=1)
                            
                            self.u = np.insert(self.u, 0,
                                            (left_ghost, left_ghost) , axis=1)
                        else:
                            self.u = np.empty(shape = (self.n_vars, self.Npts), dtype=float)
                            self.u[:, :] = np.array([self.initD, self.initS, 
                                                self.init_tau])
                            
                            # Add boundary ghosts
                            right_ghost = self.u[:, -1]
                            left_ghost = self.u[:, 0]
                            
                            
                            
                            self.u = np.insert(self.u, self.u.shape[-1], 
                                            (right_ghost, right_ghost) , axis=1)
                            
                            self.u = np.insert(self.u, 0,
                                            (left_ghost, left_ghost) , axis=1)
                            
                            self.W = np.insert(self.W, self.W.shape[-1],
                                              right_ghost)
                            self.W = np.insert(self.W, 0, left_ghost)
                            
                    
            else:
                if not first_order:
                    # Add the extra ghost cells for i-2, i+2
                    right_ghost = self.u[:, -1]
                    left_ghost = self.u[:, 0]
                    self.u = np.insert(self.u, self.u.shape[-1], right_ghost , axis=1)
                    self.u = np.insert(self.u, 0, left_ghost , axis=1)
                    
                    print(self.W[0])
                    right_gamma = self.W[-1]
                    left_gamma = self.W[0]

                    
                    self.W = np.insert(self.W, -1, right_gamma, axis=0)
                    self.W = np.insert(self.W, 0, left_gamma)
                    
                    
                    #zzz = input('')
        else:
            if not self.u.any():
                if periodic:
                    self.u = np.empty(shape = (self.n_vars, self.yNpts, self.xNpts), dtype = float)
                    
                    self.u[:, :, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                            self.init_energy])
                else:
                    if first_order:
                        if self.regime == "classical":
                            self.u = np.empty(shape = (self.n_vars, self.yNpts, self.xNpts), dtype=float)
                            self.u[:, :, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                                self.init_energy])
                            
                            # Add boundary ghosts
                            right_ghost = self.u[:, :, -1]
                            left_ghost = self.u[:, :, 0]
                            
                            self.u = np.insert(self.u, self.u.shape[-1], right_ghost , axis=2)
                            self.u = np.insert(self.u, 0, left_ghost , axis=2)
                            
                            upper_ghost = self.u[:, 0]
                            bottom_ghost = self.u[:, -1]
                            
                            self.u = np.insert(self.u, self.u.shape[1], bottom_ghost , axis=1)
                            self.u = np.insert(self.u, 0, upper_ghost , axis=1)
                        else:
                            self.u = np.empty(shape = (self.n_vars, self.yNpts, self.xNpts), dtype=float)
                            self.u[:, :, :] = np.array([self.initD, self.initS1,
                                                        self.initS2, self.init_tau])
                            
                            # Add boundary ghosts
                            bottom_ghost = self.u[:, -1]
                            upper_ghost = self.u[:, 0]
                            
                            bottom_gamma = self.W[-1]
                            upper_gamma = self.W[0]
                            
                            self.u = np.insert(self.u, self.u.shape[1], 
                                            bottom_ghost , axis=1)
                            
                            self.u = np.insert(self.u, 0,
                                            upper_ghost , axis=1)
                            
                            self.W = np.insert(self.W, self.W.shape[0], 
                                            bottom_gamma , axis=0)
                            
                            self.W = np.insert(self.W, 0,
                                            upper_gamma , axis=0)
                            
                            left_ghost = self.u[:, :, 0]
                            right_ghost = self.u[:, :, -1]
                            
                            left_gamma = self.W[ :, 0]
                            right_gamma = self.W[ :,  -1]
                            
                            
                            self.u = np.insert(self.u, 0, 
                                            left_ghost , axis=2)
                            
                            self.u = np.insert(self.u, self.u.shape[2],
                                            right_ghost , axis=2)
                            
                            self.W = np.insert(self.W, 0, 
                                            left_gamma , axis=1)
                            
                            self.W = np.insert(self.W, self.W.shape[1],
                                            right_gamma, axis=1)
                            
                        
                    else:
                        if self.regime == "classical":
                            self.u = np.empty(shape = (self.n_vars, self.yNpts, self.xNpts), dtype=float)
                            self.u[:, :, :] = np.array([self.init_rho, self.init_rho*self.init_vx,
                                                        self.init_rho*self.init_vy, self.init_energy])
                            
                            # Add boundary ghosts
                            bottom_ghost = self.u[:, -1]
                            upper_ghost = self.u[:, 0]
                            
                            
                            self.u = np.insert(self.u, self.u.shape[1], 
                                            (bottom_ghost, bottom_ghost) , axis=1)
                            
                            self.u = np.insert(self.u, 0,
                                            (upper_ghost, upper_ghost) , axis=1)
                            
                            left_ghost = self.u[:, :, 0]
                            right_ghost = self.u[:, :, -1]
                            
                            self.u = np.insert(self.u, 0, 
                                            (left_ghost, left_ghost) , axis=2)
                            
                            self.u = np.insert(self.u, self.u.shape[2],
                                            (right_ghost, right_ghost) , axis=2)
                        else:
                            self.u = np.empty(shape = (self.n_vars, self.yNpts, self.xNpts), dtype=float)
                            self.u[:, :, :] = np.array([self.initD, self.initS1,
                                                        self.initS2, self.init_tau])
                            
                            # Add boundary ghosts
                            bottom_ghost = self.u[:, -1]
                            upper_ghost = self.u[:, 0]
                            
                            bottom_gamma = self.W[-1]
                            upper_gamma = self.W[0]
                            
                            self.u = np.insert(self.u, self.u.shape[1], 
                                            (bottom_ghost, bottom_ghost) , axis=1)
                            
                            self.u = np.insert(self.u, 0,
                                            (upper_ghost, upper_ghost) , axis=1)
                            
                            self.W = np.insert(self.W, self.W.shape[0], 
                                            (bottom_gamma, bottom_gamma) , axis=0)
                            
                            self.W = np.insert(self.W, 0,
                                            (upper_gamma, upper_gamma) , axis=0)
                            
                            left_ghost = self.u[:, :, 0]
                            right_ghost = self.u[:, :, -1]
                            
                            left_gamma = self.W[ :, 0]
                            right_gamma = self.W[ :,  -1]
                            
                            
                            self.u = np.insert(self.u, 0, 
                                            (left_ghost, left_ghost) , axis=2)
                            
                            self.u = np.insert(self.u, self.u.shape[2],
                                            (right_ghost, right_ghost) , axis=2)
                            
                            self.W = np.insert(self.W, 0, 
                                            (left_gamma, left_gamma) , axis=1)
                            
                            self.W = np.insert(self.W, self.W.shape[1],
                                            (right_gamma, right_gamma) , axis=1)
                            
                    
            else:
                if not first_order:
                    # Add the extra ghost cells for i-2, i+2
                    right_ghost = self.u[:, :, -1]
                    left_ghost = self.u[:, :, 0]
                    
                    right_W_ghost = self.W[-1]
                    left_W_ghost = self.W[0]
                    
                    self.u = np.insert(self.u, self.u.shape[-1], right_ghost , axis=2)
                    self.u = np.insert(self.u, 0, left_ghost , axis=2)
                    
                    self.W = np.insert(self.W, self.W.shape[-1], right_W_ghost)
                    self.W = np.insert(self.W, 0, right_W_ghost)
            
        
        u = self.u 
        
        # Copy state tensor
        cons_p = u.copy()

        t = 0
        if self.dimensions == 1:
            if first_order:
                print("Computing First Order...")
                r_min = self.geometry[0]
                r_max = self.geometry[1]
                if linspace:
                    r_arr = np.linspace(r_min, r_max, self.Npts)
                else:
                    r_arr = np.logspace(np.log(r_min), np.log(r_max), self.Npts, base=np.exp(1))
                    
                if self.regime == "classical":
                    a = PyState(u, self.gamma, CFL, r = r_arr, coord_system = coordinates)
                    u = a.simulate(tend=tend, dt=dt, linspace=linspace, periodic=periodic)
                else:
                    print("FO W: ", self.W.size)
                    a = PyStateSR(u, self.gamma, CFL, r = r_arr, coord_system = coordinates)
                    u = a.simulate(tend=tend, dt=dt, linspace=linspace, sources=sources, periodic=periodic, lorentz_gamma=self.W)
                    
                
                """
                while t < tend:
                    if periodic:
                        cons_p = u + dt*self.u_dot(u, periodic=True, coordinates = 'cartesian')
                    else:
                        cons_p[:, 1: self.Npts+1] = u[:, 1: self.Npts+1] + dt*self.u_dot(u, coordinates = 'cartesian')
                    
                    
                    # cons_p[0][0] = cons_p[0][1]
                    # cons_p[1][0] = - cons_p[1][1]
                    # cons_p[2][0] = cons_p[2][1]
                    
                    # if (t > 0):
                    #     dt = self.adaptive_timestep(cons_p)
                        
                    u, cons_p = cons_p, u
                    

                    t += dt
                """
                
                
                
                
                
            else:
                ########################## 
                # RK3 ORDER IN TIME
                ##########################
                print('Computing Higher Order...')
                r_min = self.geometry[0]
                r_max = self.geometry[1]
                if linspace:
                    r_arr = np.linspace(r_min, r_max, self.Npts)
                else:
                    r_arr = np.logspace(np.log10(r_min), np.log10(r_max), self.Npts)
                    
                if self.regime == "classical":
                    a = PyState(u, self.gamma, CFL, r = r_arr, coord_system = coordinates)
                    u = a.simulate(tend=tend, first_order=False,  dt=dt, linspace=linspace, periodic=periodic)
                else:
                    print(self.W.size)
                    print(u.shape)
                    a = PyStateSR(u, self.gamma, CFL, r = r_arr, coord_system = coordinates)
                    u = a.simulate(tend=tend, first_order=False, sources=sources, dt=dt, linspace=linspace, periodic=periodic, lorentz_gamma=self.W)
                   
                #a = PyState(u, self.gamma, CFL, r = r_arr, coord_system = coordinates)
                #u = a.simulate(tend=tend, first_order=False, dt=dt, linspace=linspace, periodic=periodic)
                
            
                """
                u_1 = u.copy()
                u_2 = u.copy()
                while t < tend:
                    # First Version Of U
                    if periodic:
                        u_1 = u + dt*self.u_dot(u, first_order=False, periodic=True)
                        
                        u_2 = 0.75*u + 0.25*u_1 + 0.25*dt*self.u_dot(u, first_order= False, 
                                                                    periodic=True)
                        
                        cons_p = (1/3)*u + (2/3)*u_2 + (2/3)*dt*self.u_dot(u_2, first_order=False,
                                                                        periodic=True)
                    else:
                        u_1[:, 2:self.Npts+2] = u[:, 2: self.Npts+2] + dt*self.u_dot(u, first_order=False, 
                                                                                 coordinates='cartesian')
                        
                        
                        # Second Version Of U
                        u_2[:, 2:self.Npts+2] = ( 0.75*u[:, 2:self.Npts+2] + 0.25*u_1[:, 2:self.Npts+2] 
                                                + 0.25*dt*self.u_dot(u_1, first_order=False,
                                                                     coordinates='cartesian') )
                        
                        
                        # Final U 
                        cons_p[:, 2: self.Npts+2] = ( (1/3)*u[:, 2: self.Npts+2] + (2/3)*u_2[:, 2:self.Npts+2] + 
                                                    (2/3)*dt*self.u_dot(u_2, first_order=False, 
                                                                        coordinates='cartesian') )
                        
                    
                    u, cons_p = cons_p, u
                    
                    
                
                    
                    # Update timestep
                    # dt = self.adaptive_timestep(dx, [alpha_plus, alpha_minus])
                    t += dt
                """
                
                
                
                
                
                
                
        else:
            if (first_order):
                print("Comptuing First Order...")
                if (linspace):
                    x1 = np.linspace(self.geometry[0][0], self.geometry[0][1], self.xNpts)
                    x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
                else:
                    x1 = np.logspace(np.log10(self.geometry[0][0]), np.log10(self.geometry[0][1]), self.xNpts)
                    x0 = x1[0]
                    xn = x1[1]
                    xi = 0.5*(x0 + xn)
                    volavg = 0.75*(xi**4 - x0**4)/(xi**3 - x0**3)
                    dx2 = (xi - x0)/volavg
                    #x2 = np.arange(self.geometry[1][0], self.geometry[1][1], dx2)
                    x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
                if not sources:
                    if self.regime == "classical":
                        b = PyState2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates)
                        u = b.simulate(tend, dt=dt, linspace=linspace)
                        
                    else:
                        b = PyStateSR2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates)
                        u = b.simulate(tend, dt=dt, first_order=first_order, lorentz_gamma = self.W, linspace=linspace)
                else:
                    if self.regime == "classical":
                        b = PyState2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates)
                        u = b.simulate(tend, dt=dt)
                        
                    else:
                        b = PyStateSR2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates, cfl=CFL)
                        u = b.simulate(tend=tend, dt=dt, first_order=first_order, lorentz_gamma = self.W, sources = sources,
                                    linspace=linspace)
            else:
                print('Computing Higher Order...')
                if (linspace):
                    x1 = np.linspace(self.geometry[0][0], self.geometry[0][1], self.xNpts)
                    x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
                else:
                    x1 = np.logspace(np.log10(self.geometry[0][0]), np.log10(self.geometry[0][1]), self.xNpts)
                    x0 = x1[0]
                    xn = x1[1]
                    xi = 0.5*(x0 + xn)
                    volavg = 0.75*(xi**4 - x0**4)/(xi**3 - x0**3)
                    dx2 = (xi - x0)/volavg
                    #x2 = np.arange(self.geometry[1][0], self.geometry[1][1], dx2)
                    x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
                if not sources:
                    if self.regime == "classical":
                        b = PyState2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates)
                        u = b.simulate(tend, dt=dt, linspace=linspace)
                        
                    else:
                        b = PyStateSR2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates)
                        u = b.simulate(tend, dt=dt, lorentz_gamma = self.W, linspace=linspace)
                else:
                    if self.regime == "classical":
                        b = PyState2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates)
                        u = b.simulate(tend, dt=dt)
                        
                    else:
                        b = PyStateSR2D(u, self.gamma, x1=x1, x2=x2, coord_system=coordinates, cfl=CFL)
                        u = b.simulate(tend=tend, dt=dt, lorentz_gamma = self.W, sources = sources,
                                    linspace=linspace, first_order=False)
                
            
                
            
           
            
            
            """
            u_1 = u.copy()
            u_2 = u.copy()
            while t < tend:
                #o = self.u_dot(u, first_order=False)[3]
                #print(o)
                
                
                # o = self.cons2prim(u_1)
                # rint("Rho: {}".format(o[0,5,5]))
                # print("Pressure: {}".format(o[1,5,5]))
                # print("Vx: {}".format(o[2,5,5]))
                # print("Vy: {}".format(o[3,5,5]))
                # zzz = input('')
                
                u_1[:, 2:self.Npts, 2:self.Npts] = ( u[:, 2: self.Npts, 2:self.Npts] 
                                                    + dt*self.u_dot(u, first_order=False) )

                
                #print(u[2, 2:self.Npts + 2, 2:self.Npts + 2])
                #print(u_1[2, 2:self.Npts + 2, 2:self.Npts + 2])
                #zzz = input('')
                
                # Second Version Of U
                # u_2[:, 2:self.Npts, 2:self.Npts] = ( 0.75*u[:, 2:self.Npts:, 2:self.Npts] +
                #                                     0.25*u_1[:, 2:self.Npts, 2:self.Npts]
                #                        + 0.25*dt*self.u_dot(u_1, first_order=False) )
                
                
                # Final U 
                # cons_p[:, 2: self.Npts, 2:self.Npts] = ( (1/3)*u[:, 2: self.Npts, 2:self.Npts] 
                #                                         + (2/3)*u_2[:, 2:self.Npts, 2:self.Npts] + 
                #                            (2/3)*dt*self.u_dot(u_2, first_order=False) )
                
                
                
                u, u_1 = u_1, u
                
                t += dt
            """
            
        
        # Return the final state tensor, purging the ghost cells
        if first_order:
            if periodic:
                return u
            else:
                if self.dimensions == 1:
                    return u[:, 1: -1]
                else:
                    return u[:, 1:-1, 1:-1]
        else:
            if periodic:
                return u
            else:
                if self.dimensions == 1:
                    return u[:, 2: -2]
                else:
                    return u[:, 2:-2, 2:-2]
        
        
        
    def __del__(self):
        print("Destroying Object")

class PackageResource:
    def __enter__(self):
        class Student(Hydro):
            pass
        
        self.package_obj = Student()
        return self.package_obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.package_obj.cleanup()