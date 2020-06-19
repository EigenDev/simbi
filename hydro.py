#! /usr/bin/env python

# A Hydro Code Useful for solving 1D structure problems
# Marcus DuPont
# New York University
# 06/10/2020

import numpy as np 
import matplotlib.pyplot as plt 
import math 

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



class Hydro:
    
    def __init__(self, gamma, left_state, right_state, Npts,
                 geometry, dt = 1.e-4, dimensions=1):
        """
        The initial conditions of the hydrodynamic system (1D for now)
        
        Parameters:
            gamma (float): Adiabatic Index
            left_state (tuple): The initial state on the left side of the interface. Must be
                                in the form (pressure, velocity, density)
            right_state (tuple): Idem right side of the interface
            Npts (int): Number of grid slices to make
            geometry (tuple): The first starting point, the last, and the midpoint in the grid
            dt (float): initial time_step for the calculation
            dimensions (int): The number of dimensions for this hydro simulation
            
        Return:
            None
        """
        
        # hydro = Hydro(gamma=1.4, left_state = (1.0,0.0,1.0), right_state=(0.125,0.0,0.1),
        # Npts=500, geometry=(0.0,1.0,0.5)) 
        
        self.gamma = gamma 
        self.geometry = geometry
        self.dt = dt 
        self.Npts = Npts 
        self.left_state = left_state
        self.right_state = right_state 
        self.dimensions = dimensions 

        # step size
        self.dx = 1/self.Npts
                                        
        # Initial Conditions
        # Primitive Variables on LHS
        p_l = self.left_state[0]
        v_l = self.left_state[1]
        rho_l = self.left_state[2]
        
        # Calculate Energy and Flux Variables on LHS
        energy_l = p_l/(self.gamma - 1) + 0.5*rho_l*v_l**2
        epsilon_l = rho_l*v_l**2 + p_l
        beta_l = (energy_l + p_l)*v_l
        
        # Primitive Variables on RHS
        p_r = self.right_state[0]
        v_r = self.right_state[1]
        rho_r = self.right_state[2]
    
        # Calculate Energy and Flux Variables on RHS
        energy_r = p_r/(self.gamma - 1) + 0.5*rho_r*v_r**2
        epsilon_r = rho_r*v_r**2 + p_r
        beta_r = (energy_r + p_r)*v_r

        # Initialize conserved u-tensor and flux tensors
        self.u = np.empty(shape = (3, self.Npts +1), dtype=float)
        self.f = np.empty(shape = (3, self.Npts +1), dtype=float)

        left_bound = self.geometry[0]
        right_bound = self.geometry[1]
        midpoint = self.geometry[2]
        
        size = abs(right_bound - left_bound)
        breakpoint = size/midpoint                                          # Define the fluid breakpoint
        slice_point = int(self.Npts/breakpoint)                             # Define the array slicepoint
        
        self.u[:, : slice_point] = np.array([rho_l, rho_l*v_l, energy_l]).reshape(3,1)              # Left State
        self.u[:, slice_point: ] = np.array([rho_r, rho_r*v_r, energy_r]).reshape(3,1)              # Right State
        self.f[:, : slice_point] = np.array([rho_l*v_l, epsilon_l, beta_l]).reshape(3,1)            # Left Flux
        self.f[:, slice_point: ] = np.array([rho_r*v_r, epsilon_r, beta_r]).reshape(3,1)            # Right Flux
        
    def calc_flux(self, pressure, rho, velocity):
        """
        Calculate the new flux tensor given the necessary
        primitive parameters
        """
        energy = self.calc_energy(self.gamma, pressure, rho, velocity)
        momentum_dens = rho*velocity
        energy_dens = rho*velocity**2 + pressure
        beta = (energy + pressure)*velocity 
        
        flux = np.array([momentum_dens, energy_dens,
                                    beta])
        
        return flux
        
    def calc_state(self, gamma, pressure, rho, velocity, ghosts=1):
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
        
        # The state tensor decomposed into respective
        # variables
        #try:
        #    left_state, right_state = np.array([[left_state], [right_state]])
        #except:
        #    print("Please input a tuple or array for the states")
        
        rho_l, m_l, energy_l = left_state
        rho_r, m_r, energy_r = right_state
        
        v_l = m_l/rho_l
        v_r = m_r/rho_r
        
        p_l = self.calc_pressure(self.gamma, energy_l, rho_l, v_l)
        p_r = self.calc_pressure(self.gamma, energy_r, rho_r, v_r)
        
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
    
    def calc_energy(self, gamma, pressure, rho, velocity):
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
    
    def calc_pressure(self, gamma, energy, rho, velocity):
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
        state = np.array(state)
        
        rho = state[0]
        v = state[1]/rho 
        energy = state[2]
        
        pressure = self.calc_pressure(self.gamma, energy, rho, v)
        
        #print(np.array([pressure, rho, v]))
        #zzz = input('')
        return np.array([pressure, rho, v])
    
    def u_dot(self, state, first_order = True, theta = 1.5):
        """
        """
        
        def sign(y):
            """
            The mathematical sign function
            returning either +1 or -1
            based on the inputs
            
            Parameters:
                y (float): Some number
                
            Returns:
                sgn (float): Sign of the number y
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
            v = [x,y,z]
            
            return (
                0.25*(sign(x) + sign(y))*
                    (sign(x) + sign(z))*np.asarray(v).min(0)
            )
         
        
        if first_order:
            # Right cell interface
            u_l = state[:, 1: self.Npts]
            u_r = state[:, 2: self.Npts + 1]
            
            prims_l = self.cons2prim(state = u_l)
            prims_r = self.cons2prim(state = u_r)
            
            f_l = self.calc_flux(*prims_l)
            f_r = self.calc_flux(*prims_r)
            
            # The HLL flux calculated for f_[i+1/2]
            f1 = self.calc_hll_flux(u_l, u_r, f_l, f_r)
    
            # Left cell interface
            u_l = state[:, 0: self.Npts - 1] 
            u_r = state[:, 1: self.Npts]
            
            prims_l = self.cons2prim(state = u_l)
            prims_r = self.cons2prim(state = u_r)
            
            f_l = self.calc_flux(*prims_l)
            f_r = self.calc_flux(*prims_r)
            
            # The HLL flux calculated for f_[i-1/2]
            f2 = self.calc_hll_flux(u_l,u_r,f_l,f_r)
            
        else:
            
            # Regenerate the state with the necessary ghost cells
            ghost_vec = state[:, -1].reshape(3,1)
            state = np.append(state, ghost_vec, axis=1)
            
            # Right cell interface
            u_l = state[:, 2: self.Npts]
            u_r = state[:, 3: self.Npts + 1]
            
            prims = self.cons2prim(state = state)
            
            left_most = prims[:, 0:self.Npts-2]
            left_mid = prims[:, 1:self.Npts - 1]
            center = prims[:, 2:self.Npts]
            right_mid = prims[:,3: self.Npts + 1]
            right_most = prims[:, 4: self.Npts + 2]
            
            # [i + 1/2] interface
            prims_l = ( center + 0.5*
                       minmod(theta*(center- left_mid),
                              0.5*(right_mid - left_mid),
                              theta*(right_mid - center))
                        )
           
            prims_r = (right_mid - 0.5 *
                       minmod(theta*(right_mid -center),
                              0.5*(right_most - center),
                              theta*(right_most - right_mid))
                        )
        

            f_l = self.calc_flux(*prims_l)
            f_r = self.calc_flux(*prims_r)
        

            # The HLL flux calculated for f_[i+1/2]
            f1 = self.calc_hll_flux(u_l, u_r, f_l, f_r)
            
            # Left cell interface
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
        

            # The HLL flux calculated for f_[i+1/2]
            f2 = self.calc_hll_flux(u_l, u_r, f_l, f_r)
            
        
        return -(f1 - f2)/self.dx
        
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
    
    def simulate(self, tend=0.1, first_order=True):
        """
        Simulate the hydro setup
        
        Parameters:
            tend (float): The desired time to end the simulation
            first_order (bollean): The switch the runs the FTCS method
            or the RK3 method in time.
            
        Returns:
            u (array): The conserved variable tensor
        """
        
        u = self.u 
        f = self.f 
        dt = self.dt 
        dx = self.dx
        
        theta = 1.5
        
        # Copy state and flux profile tensors
        cons_p = u.copy()
        flux_p = f.copy()

        t = 0
        while t < tend:
            # Calculate the new values of U and F
            
            if first_order:
                cons_p[:, 1: self.Npts] = u[:, 1: self.Npts] + dt*self.u_dot(u)
                
                # Calculate new flux from new primitive variables
                prims = self.cons2prim(state = cons_p)
                
                flux_p = self.calc_flux(*prims)
                
            else:
                ########################## 
                # RK3 ORDER IN TIME
                ##########################
                
                u_1 = cons_p.copy()
                u_2 = cons_p.copy()
                
                # First Order In U
                u_1[:,1:self.Npts] = u[:, 1: self.Npts] + dt*self.u_dot(u)
                
                # Second Order In U
                u_2[:, 1:self.Npts] = ( 0.75*u[:, 1:self.Npts] + 0.25*u_1[:, 1:self.Npts] 
                                       + 0.25*dt*self.u_dot(u_1) )
                
                
                # Final U 
                cons_p[:, 2: self.Npts] = ( u[:, 2: self.Npts]/3 + (2/3)*u_2[:, 2:self.Npts] + 
                                           (2/3)*dt*self.u_dot(u_2, first_order=False) )
                
                # Calculate new flux from new primitive variables
                prims = self.cons2prim(state = cons_p)
                
                flux_p = self.calc_flux(*prims)
                
            u, cons_p = cons_p, u
            f, flux_p = flux_p, f
            
            # Update timestep
            #dt = self.adaptive_timestep(dx, [alpha_plus, alpha_minus])
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
        
        

        

