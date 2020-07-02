#! /usr/bin/env python

# A Hydro Code Useful for solving 1D structure problems
# Marcus DuPont
# New York University
# 06/10/2020

import numpy as np 
import matplotlib.pyplot as plt 
import math 
import sys

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
                 geometry, dt = 1.e-4, n=3):
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
            dt (float): initial time_step for the calculation
            n (int): Number of variables in the problem
            
        Return:
            None
        """
        
        # hydro = Hydro(gamma=1.4, initial_state = ((1.0,0.0,1.0),(0.125,0.0,0.1)),
        # Npts=500, geometry=(0.0,1.0,0.5), n=3) 
        
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
        self.dt = dt 
        self.Npts = Npts 
        
        
        

        # step size
        self.dx = 1/self.Npts
                                        
        # Initial Conditions
        
        # Check for Discontinuity
        if len(initial_state) == 2:
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
            

            # Initialize conserved u-tensor and flux tensors
            self.u = np.empty(shape = (3, self.Npts +1), dtype=float)

            left_bound = self.geometry[0]
            right_bound = self.geometry[1]
            midpoint = self.geometry[2]
            
            size = abs(right_bound - left_bound)
            breakpoint = size/midpoint                                          # Define the fluid breakpoint
            slice_point = int(self.Npts/breakpoint)                             # Define the array slicepoint
            
            self.u[:, : slice_point] = np.array([rho_l, rho_l*v_l, energy_l]).reshape(3,1)              # Left State
            self.u[:, slice_point: ] = np.array([rho_r, rho_r*v_r, energy_r]).reshape(3,1)              # Right State
            
        if len(initial_state) == 3:
            self.dimensions = 1
            
            self.n_vars = len(initial_state)
            
            self.init_rho = initial_state[0]
            self.init_pressure = initial_state[1]
            self.init_v = initial_state[2]
            
            self.init_energy =  ( self.init_pressure/(self.gamma - 1) + 
                                    0.5*self.init_rho*self.init_v**2 )
            
            
        elif len(initial_state) == 4:
            self.dimensions = 2
            
            n_vars = len(initial_state) 
            
            rho = initial_state[0]
            pressure = initial_state[1]
            v_x = initial_state[2]
            v_y = initial_state[3]
            
            energy = pressure/(self.gamma - 1.) + 0.5*rho*(v_x**2 + v_y**2)
            
            # Flux variables in x-direction
            epsilon_x = rho*v_x**2 + pressure 
            convect_x = rho*v_x*v_y 
            beta_x = (energy + pressure)*v_x
            
            # Flux variables in y-direction
            epsilon_y = rho*v_y**2 + pressure 
            convect_y = rho*v_x*v_y 
            beta_y = (energy + pressure)*v_y 
            
            # Initialize conserved u-tensor and flux tensors
            self.u = np.empty(shape = (n_vars, self.Npts + 1, self.Npts + 1), dtype=float)
            
            self.u[:, :] = np.array([rho, rho*v_x,rho*v_y, energy])
                
                
    def periodic_bc(self, u): 
        if self.dimensions == 1:
            # A sloppy implementation of periodic BCs
            
            u[:, 0] = u[:, -2]
            u[:,-1] = u[:,  1]
            
            return u     
            
    def calc_flux(self, rho, pressure, velocity, x_direction=True):
        """
        Calculate the new flux tensor given the necessary
        primitive parameters
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
            total_velocity = v_x**2 + v_y**2
            
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
        
        
    def calc_state(self, gamma, rho, pressure, velocity, ghosts=1):
        """
        Calculate the new state tensor given the parameters
        """
        state = np.empty(shape = (3, self.Npts + ghosts), dtype=float)
        
        energy = self.calc_energy(gamma, pressure, rho, velocity)
        
        u = np.array([rho, rho*velocity, energy])
        
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
            v_y_l = momentum_y_l/rho_r 
            
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
                    
        alpha_plus = max(0, *lam['left']['plus'], 
                        *lam['right']['plus'])
        
        alpha_minus = max(0, *-lam['left']['minus'], 
                        *-lam['right']['minus'])
        
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
        energy = pressure/(gamma - 1) + rho*velocity**2
        
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
        pressure = (gamma - 1.)*(energy - rho*velocity**2)
        
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
            rho == state[0]
            v_x = state[1]/rho 
            v_y = state[2]/rho 
            energy = state[4]
            
            v = np.sqrt(v_x**2 + v_y**2)
            
            pressure = self.calc_pressure(self.gamma, rho, energy, v)
            
            return np.array([rho, pressure, v_x, v_y])
        
    def sign(self, y):
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
            
    def minmod(self, x, y ,z):
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
        theta = 1.0 
        
        return (
            0.25*np.abs(self.sign(x) + self.sign(y))*
                (self.sign(x) + self.sign(z))*np.asarray(v).min(0)
        )  
    
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
                
                # Right cell interface
                if periodic:
                    u_l = state
                    u_r = np.roll(state, 1, axis=1)
                    
                    prims = self.cons2prim(state = state)
                    
                    left_most = np.roll(prims, -2, axis=1)
                    left_mid = np.roll(prims, -1, axis=1)
                    center = prims
                    right_mid = np.roll(prims, 1, axis=1)
                    right_most = np.roll(prims, 2, axis=1)
                else:
                    u_l = state[:, 2: self.Npts]
                    u_r = state[:, 3: self.Npts + 1]
                    
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
                
                #print('Minmods:', minmod(theta*(center - left_mid),
                #                0.5*(right_mid - left_mid),
                #                theta*(right_mid - center)))
                #print('Central Densities:', center[0])
                #print('Left Densities', prims_l[0])
                #zzz = input('')
                #plt.plot(center[0], label='Centered Values')
                #plt.plot(prims_l[0], label='Left of i+1/2')
                #plt.plot(prims_r[0], label='Right of i+1/2')
                #plt.ylabel('Density')
                #plt.legend()
                #plt.show()
                
                
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)

                # The HLL flux calculated for f_[i+1/2]
                f1 = self.calc_hll_flux(u_l, u_r, f_l, f_r)
                
                # Left cell interface
                if periodic:
                    u_l = np.roll(state, -1, axis=1)
                    u_r = state
                else:
                    u_l = state[:, 1: self.Npts - 1] 
                    u_r = state[:, 2: self.Npts]
                
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
                
                
                f_l = self.calc_flux(*prims_l)
                f_r = self.calc_flux(*prims_r)
            
                # zzz = input("")
                # The HLL flux calculated for f_[i-1/2]
                f2 = self.calc_hll_flux(u_l, u_r, f_l, f_r)
                
            return -(f1 - f2)/self.dx
        else:
            # TODO: Do the Higher Order 2-D Advection Problem
            
            pass
            
        
    def adaptive_timestep(self, dx, alphas):
        """
        Returns the adjustable timestep based
        on the Courant number C = alpha* delta_t/delta_x < 1
        """
        max_dt = dx/max(alphas)
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
    
    def simulate(self, tend=0.1, first_order=True, periodic=False):
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
        if periodic:
            self.u = np.empty(shape = (self.n_vars, self.Npts), dtype = float)
            
            self.u[:, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                    self.init_energy])
        else:
            if first_order:
                self.u = np.empty(shape = (self.n_vars, self.Npts + 1), dtype=float)
            else:
                self.u = np.empty(shape = (self.n_vars, self.Npts + 2), dtype=float)
                
            self.u[:, :] = np.array([self.init_rho, self.init_rho*self.init_v, 
                                    self.init_energy])
        
        u = self.u 
        dt = self.dt 
        dx = self.dx
        
        # Copy state and flux profile tensors
        cons_p = u.copy()

        t = 0
        if first_order:
            print("Computing First Order...")
            while t < tend:
                # Calculate the new values of U and F
                if periodic:
                    cons_p = u + dt*self.u_dot(u, periodic=True)
                else:
                    cons_p[:, 1: self.Npts] = u[:, 1: self.Npts] + dt*self.u_dot(u)

                #if periodic:
                #    cons_p = self.periodic_bc(cons_p)
                    
                u, cons_p = cons_p, u
            
                # Update timestep
                #dt = self.adaptive_timestep(dx, [alpha_plus, alpha_minus])
                t += dt
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
                    u_1 = u + dt*self.u_dot(u, first_order=True, periodic=True)
                    
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
        
        return u 
class Visual(Hydro):
    """
    Store plotting results from the simulation
    """
    
    def __init__(self, results, geometry, Npts):
        self.Npts = Npts
        self.geometry = geometry
        self._results = results
        
    def plot_results(self):
        """
        """
        # Boundaries
        x_left = self.geometry[0]
        x_right = self.geometry[1]

        #X-Grid
        x_arr = np.linspace(x_left, x_right, self.Npts+1)

        state = {}
        state['density'] = self._results[:,0]
        state['momentum_dens'] = self._results[:,1]
        state['energy'] = self._results[:,2]

        #print(x_arr, u[:,0])
        plt.plot(x_arr, state['density'])
        plt.plot(x_arr, state['momentum_dens'])
        plt.plot(x_arr, state['energy'])

        #plt.show()
        
        

        

