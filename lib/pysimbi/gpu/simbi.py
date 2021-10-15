# A Hydro Code Useful for solving MultiD structure problems
# Marcus DuPont
# New York University
# 06/10/2020

import numpy as np 
import sys 
import h5py 
import pysimbi.initial_condition as simbi_ic 

try:
    from gpu_state import *
except ImportError:
    print("The gpu_state binary not configured. Be sure to install with --gpu flag enabled")

class Hydro:
    
    def __init__(self, 
                 gamma: float,
                 initial_state: tuple,
                 Npts: tuple,
                 geometry: tuple=None,
                 n_vars: int = 3,
                 coord_system:str = 'cartesian',
                 regime: str = "classical"):
        """
        The initial conditions of the hydrodynamic system (1D for now)
        
        Parameters:
            gamma (float):                  Adiabatic Index
            
            initial_state (tuple or array): The initial conditions of the problem in the following format
                                            Ex. state = ((1.0, 1.0, 0.0), (0.1,0.125,0.0)) for Sod Shock Tube
                                            state = (array_like rho, array_like pressure, array_like velocity)
                                    
            Npts (int, tuple):              Number of grid points in 1D/2D Coordinate Lattice
            
            geometry (tuple):               The first starting point, the last, and an optional midpoint in the grid
                                            Ex. geometry = (0.0, 1.0, 0.5) for Sod Shock Tube
                                            Ex. geometry = ((xmin, xmax), (ymin, ymax))
                                
            n_vars (int):                   Number of primitives in the problem
            
            coord_system (string):          The coordinate system the problem uses. Currently only supports Cartesian 
                                            and Spherical Coordinate Lattces
            
            regime (string):                The classical (Newtonian) or relativisitc regime to do the problem in
            
        Return:
            None
        """
        # TODO: Add an example Instantian Here for the Sod Problem
        
        self.regime = regime
        discontinuity = False
        
        #Check dimensions of state
        if len(initial_state) == 2:
            print('Initializing the 1D Discontinuity...')
            
            discontinuity = True
            left_state  = initial_state[0]
            right_state = initial_state[1]
            
            self.left_state  = left_state
            self.right_state = right_state 
            
            if len(left_state) != len(right_state):
                print("ERROR: The left and right states must have the same number of variables")
                print('Left State:',   left_state)
                print('Right State:', right_state)
                sys.exit()
                
            elif len(left_state) > 4 and len(right_state) > 4:
                print("Your state arrays contain too many variables. This version takes a maximum\n"
                    "of 4 state variables")
                
            elif len(left_state) == 3 and len(right_state) == 3:
                self.dimensions = 1
                
            elif len(left_state) == 4 and len(right_state) == 4:
                self.dimensions = 2
                
            elif len(left_state) == 5 and len(right_state) == 5:
                self.dimensions = 3
        
        self.gamma    = gamma 
        self.geometry = geometry
        self.Npts     = Npts 
        self.n_vars   = n_vars
                                        
        # Initial Conditions
        
        # Check for Discontinuity
        if discontinuity:
            # Primitive Variables on LHS
            rho_l = self.left_state[0]
            p_l   = self.left_state[1]
            v_l   = self.left_state[2]
            
            # Primitive Variables on RHS
            rho_r = self.right_state[0]
            p_r   = self.right_state[1]
            v_r   = self.right_state[2]
        
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

            left_bound  = self.geometry[0]
            right_bound = self.geometry[1]
            midpoint    = self.geometry[2]
            
            size = abs(right_bound - left_bound)
            breakpoint = size/midpoint                                              # Define the fluid breakpoint
            slice_point = int((self.Npts+2)/breakpoint)                             # Define the array slicepoint
            
            if self.regime == "classical":
                self.u[:, : slice_point] = np.array([rho_l, rho_l*v_l, energy_l]).reshape(3,1)              # Left State
                self.u[:, slice_point: ] = np.array([rho_r, rho_r*v_r, energy_r]).reshape(3,1)              # Right State
            else:                
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
            self.dimensions = 2
            print('Initializing 2D Setup...')
            print('')
            
            left_x, right_x = geometry[0]
            left_y, right_y = geometry[1]
            
            self.xNpts, self.yNpts = Npts 
            
            self.n_vars = n_vars 
            
            if self.regime == "classical":
                self.init_rho      = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_vx       = initial_state[2]
                self.init_vy       = initial_state[3]
                
                v2 = self.init_vx**2 + self.init_vy**2
                
                self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 
                                    0.5*self.init_rho*v2 )
                
                
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
            
        elif len(initial_state) == 5:
            self.dimensions = 3
            print('Initializing 3D Setup...')
            print('')
            
            left_x, right_x = geometry[0]
            left_y, right_y = geometry[1]
            left_z, right_z = geometry[2]
            
            self.xNpts, self.yNpts, self.zNpts = Npts 
            
            self.n_vars = n_vars 
            
            if self.regime == "classical":
                self.init_rho      = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_vx       = initial_state[2]
                self.init_vy       = initial_state[3]
                self.init_vz       = initial_state[4]
                
                v2 = self.init_vx**2 + self.init_vy**2 + self.init_xz**2
                
                self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 
                                    0.5*self.init_rho*v2 )
                
                
            else:
                self.init_rho = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_v1 = initial_state[2]
                self.init_v2 = initial_state[3]
                self.init_v3 = initial_state[4]
                total_v = np.sqrt(self.init_v1**2 + self.init_v2**2 + self.init_v3**2)
                
                self.W = np.asarray(1/np.sqrt(1 - total_v**2))
                
                self.init_h = 1 + self.gamma*self.init_pressure/((self.gamma - 1)*self.init_rho)
                
                self.initD = self.init_rho*self.W
                self.initS1 = self.init_h*self.init_rho*self.W**2*self.init_v1
                self.initS2 = self.init_h*self.init_rho*self.W**2*self.init_v2 
                self.initS3 = self.init_h*self.init_rho*self.W**2*self.init_v3 
                
                self.init_tau = (self.init_rho*self.init_h*(self.W)**2 - self.init_pressure
                                  - self.init_rho*(self.W))
                
            
            
            self.u = None 
    
    def _cleanup(self, state, first_order=True):
        """
        Cleanup the ghost cells from the final simulation
        results
        """
        if first_order:
            if self.dimensions == 1:
                return state[:, 1: -1]
            else:
                return state[:, 1:-1, 1:-1]
        else:
            if self.dimensions == 1:
                return state[:, 2: -2]
            else:
                return state[:, 2:-2, 2:-2]
        
    # TODO: Make this more Pythomic
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
    

    def simulate(self, 
                 tstart: float = 0,
                 tend: float = 0.1,
                 dt: float = 1.e-4,
                 plm_theta: float = 1.5,
                 first_order: bool = True,
                 periodic: bool = False,
                 linspace: bool = True,
                 coordinates: str="cartesian",
                 CFL: float = 0.4,
                 sources: np.ndarray = None,
                 hllc: bool =False,
                 chkpt: str = None,
                 chkpt_interval:float = 0.1,
                 data_directory:str = "data/",
                 engine_duration: float = 10.0):
        """
        Simulate the Hydro Setup
        
        Parameters:
            tend        (float): The desired time to end the simulation
            dt          (float): The desired timestep
            first_order (boolean): First order RK1 or the RK2 PLM.
            period      (boolean): Periodic BCs or not
            linspace    (boolean): Prompts a linearly spaced mesh or log spaced if False
            coordinate  (boolean): The coordinate system the simulation is taking place in
            CFL         (float):   The CFL number for min adaptive timestep
            sources     (array_like): The source terms for the simulations
            hllc        (boolean): Tells the simulation whether to perform HLLC or HLLE
            chkpt       (string): The path to the checkpoint file to read into the simulation
            
        Returns:
            u (array): The conserved/primitive variable array
        """
        
        #Convert strings to byte arrays
        data_directory = data_directory.encode('utf-8')
        coordinates    = coordinates.encode('utf-8')
        # Initialize conserved u-tensor
        
        self.u = np.asarray(self.u)
        self.t = 0
        
        if not chkpt:
            simbi_ic.initializeModel(self, first_order, periodic)
        else:
            simbi_ic.load_checkpoint(self, chkpt, self.dimensions)
            
        u = self.u 
        start_time = tstart if self.t == 0 else self.t
        
        
        if self.dimensions == 1:
            if (linspace):
                x1 = np.linspace(self.geometry[0], self.geometry[1], self.Npts)
            else:
                x1 = np.logspace(np.log10(self.geometry[0]), np.log10(self.geometry[1]), self.Npts)
            sources = np.zeros((3, x1.size), dtype=float) if not sources else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
            
            if first_order:
                print("Computing First Order...") 
            else:
                print('Computing Higher Order...')
            if self.regime == "classical":
                a = PyState(u, self.gamma, CFL, r = x1, coord_system = coordinates)
            else:
                a = PyStateSR(u, self.gamma, CFL, r = x1, coord_system = coordinates)
    
            u = a.simulate(sources = sources,
                tstart = tstart,
                tend = tend,
                dt = dt,
                plm_theta = plm_theta,
                engine_duration = engine_duration,
                chkpt_interval = chkpt_interval,
                data_directory = data_directory,
                first_order = first_order,
                periodic = periodic,
                linspace = linspace,
                hllc = hllc)  
                
        elif self.dimensions == 2:
            if (linspace):
                x1 = np.linspace(self.geometry[0][0], self.geometry[0][1], self.xNpts)
                x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
            else:
                x1 = np.logspace(np.log10(self.geometry[0][0]), np.log10(self.geometry[0][1]), self.xNpts)
                x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
            
            sources = np.zeros((4, x2.size, x1.size), dtype=float) if not sources else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
            
            if (first_order):
                print("Computing First Order...")
            
            else:
                print('Computing Higher Order...')
                
            if self.regime == "classical":
                b = PyState2D(u, self.gamma, cfl=CFL, x1=x1, x2=x2, coord_system=coordinates)
            else:
                b = PyStateSR2D(u, self.gamma, cfl=CFL, x1=x1, x2=x2, coord_system=coordinates)
                   
            u = b.simulate(sources = sources,
                tstart = tstart,
                tend = tend,
                dt = dt,
                plm_theta = plm_theta,
                engine_duration = engine_duration,
                chkpt_interval = chkpt_interval,
                data_directory = data_directory,
                first_order = first_order,
                periodic = periodic,
                linspace = linspace,
                hllc = hllc)  

        else:
            if (linspace):
                x1 = np.linspace(self.geometry[0][0], self.geometry[0][1], self.xNpts)
                x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
                x3 = np.linspace(self.geometry[2][0], self.geometry[2][1], self.zNpts)
            else:
                x1 = np.logspace(np.log10(self.geometry[0][0]), np.log10(self.geometry[0][1]), self.xNpts)
                x2 = np.linspace(self.geometry[1][0], self.geometry[1][1], self.yNpts)
                x3 = np.linspace(self.geometry[2][0], self.geometry[2][1], self.zNpts)
            
            sources = np.zeros((5, x3.size, x2.size, x1.size), dtype=float) if not sources else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
            
            if (first_order):
                print("Computing First Order...")
            
            else:
                print('Computing Higher Order...')
                
            if self.regime == "classical":
                pass
                # b = PyState3D(u, self.gamma, cfl=CFL, x1=x1, x2=x2, coord_system=coordinates)
            else:
                b = PyStateSR3D(u, self.gamma, cfl=CFL, x1=x1, x2=x2, x3=x3, coord_system=coordinates)
                   
            u = b.simulate(sources = sources,
                tstart = tstart,
                tend = tend,
                dt = dt,
                plm_theta = plm_theta,
                engine_duration = engine_duration,
                chkpt_interval = chkpt_interval,
                data_directory = data_directory,
                first_order = first_order,
                periodic = periodic,
                linspace = linspace,
                hllc = hllc)  
        
        # Return the final state tensor, purging the ghost cells
        if first_order:
            if periodic:
                return u
            else:
                if self.dimensions == 1:
                    return u[:, 1: -1]
                elif self.dimensions == 2:
                    return u[:, 1:-1, 1:-1]
                else:
                    return u[:, 1:-1, 1:-1, 1:-1]
        else:
            if periodic:
                return u
            else:
                if self.dimensions == 1:
                    return u[:, 2: -2]
                elif self.dimensions == 2:
                    return u[:, 2:-2, 2:-2]
                else:
                    return u[:, 2:-2, 2:-2, 2:-2]
        
    