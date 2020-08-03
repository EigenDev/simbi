#! /usr/bin/env python

# A Hydro Code Useful for solving 1D structure problems
# Marcus DuPont
# New York University
# 06/10/2020

import numpy as np 
import matplotlib.pyplot as plt 
import math 
import sys
import numba as nb

from state import PyState, PyState2D

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
                 geometry=None, n_vars = 3):
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
            
            # Calculate Energy on LHS
            energy_l = p_l/(self.gamma - 1) + 0.5*rho_l*v_l**2
            
            
            # Primitive Variables on RHS
            rho_r = self.right_state[0]
            p_r = self.right_state[1]
            v_r = self.right_state[2]
        
            # Calculate Energy on RHS
            energy_r = p_r/(self.gamma - 1) + 0.5*rho_r*v_r**2
            

            # Initialize conserved u-tensor and flux tensors (defaulting to 2 ghost cells)
            self.u = np.empty(shape = (3, self.Npts + 2), dtype=float)

            left_bound = self.geometry[0]
            right_bound = self.geometry[1]
            midpoint = self.geometry[2]
            
            lx = right_bound - left_bound
            self.dx = lx/self.Npts
            
            size = abs(right_bound - left_bound)
            breakpoint = size/midpoint                                          # Define the fluid breakpoint
            slice_point = int(self.Npts/breakpoint)                             # Define the array slicepoint
            
            self.u[:, : slice_point] = np.array([rho_l, rho_l*v_l, energy_l]).reshape(3,1)              # Left State
            self.u[:, slice_point: ] = np.array([rho_r, rho_r*v_r, energy_r]).reshape(3,1)              # Right State
            
        elif len(initial_state) == 3:
            self.dimensions = 1
            
            left_bound = self.geometry[0]
            right_bound = self.geometry[1]
            
            lx = right_bound - left_bound
            self.dx = lx/self.Npts
            
            self.n_vars = n_vars
            
            self.init_rho = initial_state[0]
            self.init_pressure = initial_state[1]
            self.init_v = initial_state[2]
            
            self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 
                                    0.5*self.init_rho*self.init_v**2 )
            
            # Define state variable to be defined later
            self.u = None
            self.eos = eos
            
            
        elif len(initial_state) == 4:
            # TODO: Make this work
            self.dimensions = 2
            print('Initializing 2D Setup...')
            print('')
            
            left_x, right_x = geometry[0]
            left_y, right_y = geometry[1]
            
            lx = right_x - left_x
            ly = right_y - left_y
            
            self.dx = lx/self.Npts
            self.dy = ly/self.Npts
            
            self.n_vars = n_vars 
            
            self.init_rho = initial_state[0]
            self.init_pressure = initial_state[1]
            self.init_vx = initial_state[2]
            self.init_vy = initial_state[3]
                                                   
            
            total_v = self.init_vx**2 + self.init_vy**2
            self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 
                                    0.5*self.init_rho*total_v**2 )
            
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
                    
        alpha_plus = max(0, *lam['left']['plus'].flatten(), 
                        *lam['right']['plus'].flatten())
        
        alpha_minus = max(0, *-lam['left']['minus'].flatten(), 
                        *-lam['right']['minus'].flatten())
        
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
    def u_dot(self, state, first_order = True, theta = 1.5, periodic=False):
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
                    u_l = state[:, 1: self.Npts]
                    u_r = state[:, 2: self.Npts + 1]
                
                prims_l = self.cons2prim(state = u_l)
                prims_r = self.cons2prim(state = u_r)
                
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)
                
                # The HLL flux calculated for f_[i+1/2]
                f1 = self.calc_hll_flux(u_l, u_r, f_l, f_r)

                # Left cell interface
                if periodic:
                    u_l = np.roll(state, -1, axis=1) 
                    u_r = state
                    
                else:
                    u_l = state[:, 0: self.Npts - 1]
                    u_r = state[:, 1: self.Npts]
                
                prims_l = self.cons2prim(state = u_l)
                prims_r = self.cons2prim(state = u_r)
                
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)
                
                # The HLL flux calculated for f_[i-1/2]
                f2 = self.calc_hll_flux(u_l,u_r,f_l,f_r)
                
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
                    left_most = prims[:, 0:self.Npts-2]
                    left_mid = prims[:, 1:self.Npts - 1]
                    center = prims[:, 2:self.Npts]
                    right_mid = prims[:,3: self.Npts + 1]
                    right_most = prims[:, 4: self.Npts + 2]
                
                
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
                
            return -(f1 - f2)/self.dx
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
                x_left_most = prims[:, 0: self.Npts-2, 2:self.Npts]          # C_[i-2, j]
                y_left_most = prims[:, 2: self.Npts, 0: self.Npts-2]
                x_left_mid = prims[:, 1:self.Npts -1, 2: self.Npts]
                y_left_mid = prims[:, 2:self.Npts, 1:self.Npts - 1]
                center = prims[:, 2:self.Npts, 2:self.Npts]
                x_right_mid = prims[:,  3: self.Npts +1, 2:self.Npts]
                y_right_mid = prims[:, 2:self.Npts, 3: self.Npts + 1]
                x_right_most = prims[:, 4: self.Npts+2, 2:self.Npts]
                y_right_most = prims[:, 2:self.Npts, 4: self.Npts + 2]      # C_[i, j+2]
                
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

            # The HLL flux calculated for f_[i+1/2, j]
            f1 = self.calc_hll_flux(ux_l, ux_r, f_l, f_r)
            
            # The HLL flux calculated for g_[i, j+1/2]
            g1 = self.calc_hll_flux(uy_l, uy_r, g_l, g_r)
            
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
           
            L = -(f1 - f2)/self.dx - (g1 -g2)/self.dy
            
            return L
            
        
    def adaptive_timestep(self, dx, alphas):
        """
        Returns the adjustable timestep based
        on the Courant number C = alpha* delta_t/delta_x < 1
        """
        max_dt = dx/max(*alphas)
        magnitude = int(np.log10(max_dt))
        new_dt = max_dt - 10**magnitude
        
        return new_dt
    
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
                 first_order=True, periodic=False):
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
                    self.u = np.empty(shape = (self.n_vars, self.Npts), dtype = float)
                    
                    self.u[:, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                            self.init_energy])
                else:
                    if first_order:
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
                if not first_order:
                    # Add the extra ghost cells for i-2, i+2
                    right_ghost = self.u[:, -1]
                    left_ghost = self.u[:, 0]
                    self.u = np.insert(self.u, self.u.shape[-1], right_ghost , axis=1)
                    self.u = np.insert(self.u, 0, left_ghost , axis=1)
        else:
            if not self.u.any():
                if periodic:
                    self.u = np.empty(shape = (self.n_vars, self.Npts), dtype = float)
                    
                    self.u[:, :, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                            self.init_energy])
                else:
                    if first_order:
                        self.u = np.empty(shape = (self.n_vars, self.Npts, self.Npts), dtype=float)
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
                        self.u = np.empty(shape = (self.n_vars, self.Npts, self.Npts), dtype=float)
                        self.u[:, :, :] = np.array([self.init_rho, self.init_rho*self.init_vx,
                                                    self.init_rho*self.init_vy, self.init_energy])
                        
                        # Add boundary ghosts
                        bottom_ghost = self.u[:, :, -1]
                        upper_ghost = self.u[:, :,  0]
                        
                        
                        self.u = np.insert(self.u, self.u.shape[-1], 
                                        (bottom_ghost, bottom_ghost) , axis=2)
                        
                        self.u = np.insert(self.u, 0,
                                        (upper_ghost, upper_ghost) , axis=2)
                        
                        left_ghost = self.u[:, 0]
                        right_ghost = self.u[:, -1]
                        
                        self.u = np.insert(self.u, 0, 
                                        (left_ghost, left_ghost) , axis=1)
                        
                        self.u = np.insert(self.u, self.u.shape[1],
                                        (right_ghost, right_ghost) , axis=1)
                    
            else:
                if not first_order:
                    # Add the extra ghost cells for i-2, i+2
                    right_ghost = self.u[:, :, -1]
                    left_ghost = self.u[:, :, 0]
                    self.u = np.insert(self.u, self.u.shape[-1], right_ghost , axis=2)
                    self.u = np.insert(self.u, 0, left_ghost , axis=2)
            
        
        u = self.u 
        
        # Copy state tensor
        cons_p = u.copy()

        t = 0
        if self.dimensions == 1:
            if first_order:
                print("Computing First Order...")
                
                a = PyState(u, self.gamma)
                u = a.simulate(tend)
                """
                while t < tend:
                    if periodic:
                        cons_p = u + dt*self.u_dot(u, periodic=True)
                    else:
                        cons_p[:, 1: self.Npts] = u[:, 1: self.Npts] + dt*self.u_dot(u)
                    
                    u, cons_p = cons_p, u
                    
                    t += dt
                """
            else:
                ########################## 
                # RK3 ORDER IN TIME
                ##########################
                print('Computing Higher Order...')
                
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
                        u_1[:, 2:self.Npts] = u[:, 2: self.Npts] + dt*self.u_dot(u, first_order=False)
                        
                        
                        # Second Version Of U
                        u_2[:, 2:self.Npts] = ( 0.75*u[:, 2:self.Npts] + 0.25*u_1[:, 2:self.Npts] 
                                                + 0.25*dt*self.u_dot(u_1, first_order=False) )
                        
                        
                        # Final U 
                        cons_p[:, 2: self.Npts] = ( (1/3)*u[:, 2: self.Npts] + (2/3)*u_2[:, 2:self.Npts] + 
                                                    (2/3)*dt*self.u_dot(u_2, first_order=False) )
                        
                    
                    u, cons_p = cons_p, u
                
                    
                    # Update timestep
                    # dt = self.adaptive_timestep(dx, [alpha_plus, alpha_minus])
                    t += dt
        else:
            print('Computing Higher Order...')
            
            b = PyState2D(u, self.gamma)
            u = b.simulate(tend)
            """
            while t < tend:
                u_1 = u.copy()
                u_2 = u.copy()
                
                u_1[:, 2:self.Npts, 2:self.Npts] = ( u[:, 2: self.Npts, 2:self.Npts] 
                                                    + dt*self.u_dot(u, first_order=False) )
                        
                # Second Version Of U
                u_2[:, 2:self.Npts, 2:self.Npts] = ( 0.75*u[:, 2:self.Npts:, 2:self.Npts] +
                                                     0.25*u_1[:, 2:self.Npts, 2:self.Npts]
                                        + 0.25*dt*self.u_dot(u_1, first_order=False) )
                
                
                # Final U 
                cons_p[:, 2: self.Npts, 2:self.Npts] = ( (1/3)*u[:, 2: self.Npts, 2:self.Npts] 
                                                        + (2/3)*u_2[:, 2:self.Npts, 2:self.Npts] + 
                                            (2/3)*dt*self.u_dot(u_2, first_order=False) )
                
                u, cons_p = cons_p, u
                
                t += dt
            """
        
        # Return the final state tensor, purging the ghost cells
        if first_order:
            if periodic:
                return u
            else:
                return u[:, 1: -1]
        else:
            if periodic:
                return u
            else:
                if self.dimensions == 1:
                    return u[:, 2: -2]
                else:
                    return u[:, 2:-2, 2:-2]
        
        
        

        

