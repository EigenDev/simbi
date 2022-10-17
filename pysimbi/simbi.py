# A Hydro Code Useful for solving MultiD structure problems
# Marcus DuPont
# New York University
# 06/10/2020
import pysimbi.helpers as helpers
import numpy as np 
import os
import sys 
import inspect
import pysimbi.initial_condition as simbi_ic 
import warnings
from typing import Callable

regimes             = ['classical', 'relativistic']
coord_systems       = ['spherical', 'cartesian'] # TODO: Implement Cylindrical
boundary_conditions = ['outflow', 'reflecting', 'inflow', 'periodic']

class Hydro:
    
    def __init__(self, 
                 gamma: float,
                 initial_state: tuple,
                 dimensions: tuple,
                 geometry: tuple=None,
                 n_vars: int = 3,
                 coord_system:str = 'cartesian',
                 regime: str = "classical",
                 setup = None):
        """
        The initial conditions of the hydrodynamic system (1D for now)
        
        Parameters:
            gamma (float):                  Adiabatic Index
            
            initial_state (tuple or array): The initial conditions of the problem in the following format
                                            Ex. state = ((1.0, 1.0, 0.0), (0.1,0.125,0.0)) for Sod Shock Tube
                                            state = (array_like rho, array_like pressure, array_like velocity)
                                    
            dimensions (int, tuple):              Number of grid points in 1D/2D Coordinate Lattice
            
            geometry (tuple):               The first starting point, the last, and an optional midpoint in the grid
                                            Ex. geometry = (0.0, 1.0, 0.5) for Sod Shock Tube
                                            Ex. geometry = ((x1min, x1max), (x2min, x2max))
                                
            n_vars (int):                   Number of primitives in the problem
            
            coord_system (string):          The coordinate system the problem uses. Currently only supports Cartesian 
                                            and Spherical Coordinate Lattces
            
            regime (string):                The classical (Newtonian) or relativisitc regime
            
        Return:
            None
        """
        if setup:
            coord_system                  = setup.coord_system 
            regime                        = setup.regime 
            initial_state                 = setup.initial_state 
            gamma                         = setup.gamma 
            dimensions                    = setup.dimensions 
            geometry                      = setup.geometry 
            self.linspace                 = setup.linspace 
            self.sources                  = setup.sources 
            self.scalars                  = setup.scalars 
            self.scale_factor             = setup.scale_factor 
            self.scale_factor_derivative  = setup.scale_factor_derivative
            self.edens_outer              = setup.edens_outer
            self.mom_outer                = setup.mom_outer 
            self.dens_outer               = setup.dens_outer 
        
        if coord_system not in coord_systems:
            raise ValueError(f"Invalid coordinate system. Expected one of: {coord_systems}")
        
        if regime not in regimes:
            raise ValueError(f"Invalid simulation regime. Expected one of: {regimes}")
        
        self.coord_system = coord_system
        self.regime       = regime
        discontinuity     = False
        
        #Check dimensions of state
        if len(initial_state) == 2:
            print('Initializing the 1D Discontinuity...', flush=True)
            
            discontinuity = True
            left_state  = initial_state[0]
            right_state = initial_state[1]
            
            self.left_state  = left_state
            self.right_state = right_state 
            
            if len(left_state) != len(right_state):
                print("ERROR: The left and right states must have the same number of variables", flush=True)
                print('Left State:',   left_state, flush=True)
                print('Right State:', right_state, flush=True)
                sys.exit()
                
            elif len(left_state) > 4 and len(right_state) > 4:
                print("Your state arrays contain too many variables. This version takes a maximum\n"
                    "of 4 state variables", flush=True)
                
            elif len(left_state) == 3 and len(right_state) == 3:
                self.dimensionality  = 1
                
            elif len(left_state) == 4 and len(right_state) == 4:
                self.dimensionality  = 2
                
            elif len(left_state) == 5 and len(right_state) == 5:
                self.dimensionality  = 3
        
        self.gamma          = gamma 
        self.geometry       = geometry
        self.dimensions     = dimensions 
                                        
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
            self.u = np.empty(shape = (3, self.dimensions), dtype=float)

            left_bound  = self.geometry[0]
            right_bound = self.geometry[1]
            midpoint    = self.geometry[2]
            
            size        = abs(right_bound - left_bound)
            break_pt    = size/midpoint                                              # Define the fluid breakpoint
            slice_point = int((self.dimensions+2)/break_pt)                             # Define the array slicepoint
            
            if self.regime == "classical":
                self.u[:, : slice_point] = np.array([rho_l, rho_l*v_l, energy_l]).reshape(3,1)              # Left State
                self.u[:, slice_point: ] = np.array([rho_r, rho_r*v_r, energy_r]).reshape(3,1)              # Right State
            else:                
                self.u[:, : slice_point] = np.array([D_l, S_l, tau_l]).reshape(3,1)              # Left State
                self.u[:, slice_point: ] = np.array([D_r, S_r, tau_r]).reshape(3,1)              # Right State
                
        elif len(initial_state) == 3:
            self.dimensionality  = 1
            
            self.init_rho      = initial_state[0]
            self.init_pressure = initial_state[1]
            
            if regime == "classical":
                self.init_v = initial_state[2]
                self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 
                                    0.5*self.init_rho*self.init_v**2 )
                
            else:
                self.init_v   = initial_state[2]
                self.W        = np.asarray(1/np.sqrt(1 - self.init_v**2))
                self.init_h   = 1 + self.gamma*self.init_pressure/((self.gamma - 1)*self.init_rho)
                self.initD    = self.init_rho*self.W
                self.initS    = self.init_h*self.init_rho*self.W**2*self.init_v
                self.init_tau = (self.init_rho*self.init_h*self.W**2 - self.init_pressure
                                  - self.init_rho*self.W)
            
            self.u = None 
            
        elif len(initial_state) == 4:
            self.dimensionality  = 2
            print('Initializing 2D Setup...', flush=True)
            print('',flush=True)
            self.xdimensions, self.ydimensions = dimensions 
            
            if self.regime == "classical":
                self.init_rho      = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_vx       = initial_state[2]
                self.init_vy       = initial_state[3]
                
                v2 = self.init_vx**2 + self.init_vy**2
                
                self.init_energy =  ( self.init_pressure/(self.gamma - 1.) +  0.5*self.init_rho*v2 )
            else:
                self.init_rho      = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_v1       = initial_state[2]
                self.init_v2       = initial_state[3]
                vsq                = self.init_v1**2 + self.init_v2**2
                
                self.W = 1/np.sqrt(1 - vsq)
                self.init_h = 1 + self.gamma*self.init_pressure/((self.gamma - 1)*self.init_rho)
                self.initD  = self.init_rho*self.W
                self.initS1 = self.init_h*self.init_rho*self.W**2*self.init_v1
                self.initS2 = self.init_h*self.init_rho*self.W**2*self.init_v2 
                
                self.init_tau = (self.init_rho*self.init_h*self.W**2 - self.init_pressure
                                  - self.init_rho*self.W)
            self.u = None 
            
        elif len(initial_state) == 5:
            self.dimensionality  = 3
            print('Initializing 3D Setup...', flush=True)
            print('', flush=True)
            
            left_x, right_x = geometry[0]
            left_y, right_y = geometry[1]
            left_z, right_z = geometry[2]
            
            self.xdimensions, self.ydimensions, self.zdimensions = dimensions  
            
            if self.regime == "classical":
                self.init_rho      = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_vx       = initial_state[2]
                self.init_vy       = initial_state[3]
                self.init_vz       = initial_state[4]
                
                vsq = self.init_vx**2 + self.init_vy**2 + self.init_xz**2
                
                self.init_energy =  ( self.init_pressure/(self.gamma - 1.) + 0.5*self.init_rho*vsq )
            else:
                self.init_rho      = initial_state[0]
                self.init_pressure = initial_state[1]
                self.init_v1       = initial_state[2]
                self.init_v2       = initial_state[3]
                self.init_v3       = initial_state[4]
                vsq                = self.init_v1**2 + self.init_v2**2 + self.init_v3**2
                
                self.W = 1/np.sqrt(1 - vsq)
                
                self.init_h = 1 + self.gamma*self.init_pressure/((self.gamma - 1)*self.init_rho)
                
                self.initD  = self.init_rho*self.W
                self.initS1 = self.init_h*self.init_rho*self.W**2*self.init_v1
                self.initS2 = self.init_h*self.init_rho*self.W**2*self.init_v2 
                self.initS3 = self.init_h*self.init_rho*self.W**2*self.init_v3 
                
                self.init_tau = (self.init_rho*self.init_h*(self.W)**2 - self.init_pressure - self.init_rho*(self.W))
                
            self.u = None 
    
    @classmethod
    def gen_from_setup(cls, setup):
        return cls(*[0]*7, setup=setup)
    
    def _cleanup(self, first_order=True):
        """
        Cleanup the ghost cells from the final simulation
        results
        """
        if first_order:
            if self.dimensionality  == 1:
                self.solution = self.solution[:, 1: -1]
            elif self.dimensionality  == 2:
                self.solution = self.solution[:, 1:-1, 1:-1]
            else:
                self.solution = self.solution[:, 1:-1, 1:-1, 1:-1]
        else:
            if self.dimensionality  == 1:
                self.solution = self.solution[:, 2: -2]
            elif self.dimensionality  == 2:
                self.solution = self.solution[:, 2:-2, 2:-2]
            else:
                self.solution = self.solution[:, 2:-2, 2:-2, 2:-2]
    
    def _print_params(self, frame):
        params = inspect.getargvalues(frame)
        print("="*80, flush=True)
        print("Simulation Parameters", flush=True)
        print("="*80, flush=True)
        for key, value in params.locals.items():
            if key != 'self':
                try:
                    if type(value) == float or type(value) == np.float64:
                        val_str = f"{value:.2f}"
                    else:
                        val_str = str(value)
                except:
                    if value == 'None':
                        val_str = 'None'
                    else:
                        val_str = 'function object'
                
                my_str = str(key).ljust(30, '.')
                print(f"{my_str} {val_str}", flush=True)
        system_dict = {
            'adiabatic_gamma' : self.gamma,
            'dimensions'      : self.dimensions,
            'geometry'        : self.geometry,
            'coord_system'    : self.coord_system,
            'regime'          : self.regime,
        }
        
        for key, val in system_dict.items():
            my_str = str(key).ljust(30, '.')
            if type(val) == float:
                val_str = f"{val:.2f}"
            elif type(val) == tuple:
                val_str = ''
                for tup in val:
                    if type(tup) == int:
                        val_str = str(val)
                        break
                    elif type(tup) == tuple:
                        val_str += '(' + ', '.join('{0:.3f}'.format(t) for t in tup) + ')'
            else:
                val_str = str(val)
                
            print(f"{my_str} {val_str}", flush=True)
        print("="*80, flush=True)
                
    def simulate(
        self, 
        tstart: float = 0.0,
        tend: float = 0.1,
        dlogt: float = 0.0,
        plm_theta: float = 1.5,
        first_order: bool = True,
        linspace: bool = True,
        cfl: float = 0.4,
        sources: np.ndarray = None,
        scalars: np.ndarray = 0,
        hllc: bool = False,
        chkpt: str = None,
        chkpt_interval:float = 0.1,
        data_directory:str = "data/",
        boundary_condition: str = "outflow",
        engine_duration: float = 10.0,
        compute_mode: str = 'cpu',
        quirk_smoothing: bool = True,
        scale_factor: Callable = None,
        scale_factor_derivative: Callable = None,
        dens_outer: Callable = None,
        mom_outer: Callable = None,
        edens_outer: Callable = None) -> np.ndarray:
        """
        Simulate the Hydro Setup
        
        Parameters:
            tstart      (flaot):         The start time of the simulation
            tend        (float):         The desired time to end the simulation
            dlogt       (float):         The desired logarithmic spacing in checkpoints
            plm_theta   (float):         The Piecewise Linear Reconstructed slope parameter
            first_order (boolean):       First order RK1 or the RK2 PLM.
            linspace    (boolean):       Prompts a linearly spaced mesh or log spaced if False
            cfl         (float):         The cfl number for min adaptive timestep
            sources     (array_like):    The source terms for the simulations
            scalars     (array_like):    The array of passive scalars
            hllc        (boolean):       Tells the simulation whether to perform HLLC or HLLE
            chkpt       (string):        The path to the checkpoint file to read into the simulation
            chkpt_interval (float):      The interval at which to save the checkpoints
            data_directory (string):     The directory at which to save the checkpoint files
            bounday_condition (string):  The outer conditions at the domain x1 boundaries
            engine_duration (float):     The duration the source terms will last in the simulation
            compute_mode (string):       The compute mode for simulation execution (cpu or gpu)
            quirksmoothing (bool):       The switch that controls the Quirk (1960) shock smoothing method
            scale_factor              (Callable):   The scalar function for moving mesh. Think cosmology
            scale_factor_derivative   (Callable):   The first derivative of the scalar function for moving mesh
            dens_outer     (Callable):   The density to be fed into outer zones if moving mesh
            mom_outer      (Callables):  idem but for momentum density
            edens_outer    (Callable):   idem but for energy density
            
        Returns:
            u (array): The hydro solution containing the primitive variables
        """
        self._print_params(inspect.currentframe())
        if scale_factor == None:
            scale_factor = lambda t: 1.0 
        if scale_factor_derivative == None:
            scale_factor_derivative = lambda t: 0.0
        
        if linspace:
            genspace = np.linspace 
        else:
            genspace = np.geomspace 
            
        mesh_motion = adot(1.0) / a(1.0) != 0
        if mesh_motion and self.coord_system != 'cartesian':
            if self.dimensions == 1:
                volume_factor = helpers.calc_cell_volume1D(x1)
            elif self.dimensions == 2:
                volume_factor = helpers.calc_cell_volume2D(x1, x2)
        else:
            volume_factor = 1.0
                
        if boundary_condition not in boundary_conditions:
            raise ValueError(f"Invalid boundary condition. Expected one of: {boundary_conditions}")
        
        if compute_mode == 'cpu':
            from cpu_ext import PyState, PyState2D, PyStateSR, PyStateSR3D, PyStateSR2D
        else:
            try:
                from gpu_ext import PyState, PyState2D, PyStateSR, PyStateSR3D, PyStateSR2D
            except Exception as e:
                warnings.warn("Error in loading GPU extension. Loading CPU instead...", GPUExtNotBuiltWarning)
                warnings.warn(f"For reference, the gpu_ext had the follow error: {e}", GPUExtNotBuiltWarning)
                from cpu_ext import PyState, PyState2D, PyStateSR, PyStateSR3D, PyStateSR2D
                
        self.u         = np.asarray(self.u)
        self.t         = 0
        self.chkpt_idx = 0
        
        if not chkpt:
            simbi_ic.initializeModel(self, first_order, boundary_condition, scalars, volume_factor=volume_factor)
        else:
            simbi_ic.load_checkpoint(self, chkpt, self.dimensionality , mesh_motion)
        
        periodic = boundary_condition == 'periodic'
        start_time  = tstart if self.t == 0 else self.t
        #Convert strings to byte arrays
        data_directory     = os.path.join(data_directory, '').encode('utf-8')
        coordinates        = self.coord_system.encode('utf-8')
        boundary_condition = boundary_condition.encode('utf-8')
        
        # Check whether the specified path exists or not
        if not os.path.exists(data_directory):
            # Create a new directory because it does not exist 
            os.makedirs(data_directory)
            print("The data directory provided does not exist. Creating the {data_directory} now!", flush=True)
        
        if first_order:
            print("Computing First Order Solution...", flush=True)
        else:
            print('Computing Second Order Solution...', flush=True)
          
        if self.dimensionality  == 1:
            sources = np.zeros(self.u.shape) if not sources else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
            kwargs  = {}
            if self.regime == "classical":
                state = PyState(self.u, self.gamma, cfl, x1 = x1, coord_system = coordinates)
            else:   
                state = PyStateSR(self.u, self.gamma, cfl, r = self.x1, coord_system = coordinates)
                kwargs = {'a': scale_factor, 'adot': scale_factor_derivative}
                if dens_outer and mom_outer and edens_outer:
                    kwargs['d_outer'] =  dens_outer
                    kwargs['s_outer'] =  mom_outer
                    kwargs['e_outer'] =  edens_outer
                
            self.solution = state.simulate(
                sources            = sources,
                tstart             = start_time,
                tend               = tend,
                dlogt              = dlogt,
                plm_theta          = plm_theta,
                engine_duration    = engine_duration,
                chkpt_interval     = chkpt_interval,
                chkpt_idx          = self.chkpt_idx,
                data_directory     = data_directory,
                boundary_condition = boundary_condition,
                first_order        = first_order,
                linspace           = linspace,
                hllc               = hllc,
                **kwargs)  
                
        elif self.dimensionality  == 2:            
            # ignore the chi term
            sources = np.zeros(self.u[:-1].shape, dtype=float) if not sources else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
            
            kwargs = {}
            if self.regime == "classical":
                state = PyState2D(self.u, self.gamma, cfl=cfl, x1=x1, x2=x2, coord_system=coordinates)
            else:
                kwargs = {'a': scale_factor, 'adot': scale_factor_derivative, 'quirk_smoothing': quirk_smoothing}
                if dens_outer and mom_outer and edens_outer:
                    kwargs['d_outer']  =  dens_outer
                    kwargs['s1_outer'] =  mom_outer[0]
                    kwargs['s2_outer'] =  mom_outer[1]
                    kwargs['e_outer']  =  edens_outer
                    
                state = PyStateSR2D(self.u, self.gamma, cfl=cfl, x1=x1, x2=x2, coord_system=coordinates)
                
            self.solution = state.simulate(
                sources         = sources,
                tstart          = start_time,
                tend            = tend,
                dlogt           = dlogt,
                plm_theta       = plm_theta,
                engine_duration = engine_duration,
                chkpt_interval  = chkpt_interval,
                chkpt_idx       = self.chkpt_idx,
                data_directory  = data_directory,
                boundary_condition = boundary_condition,
                first_order     = first_order,
                linspace        = linspace,
                hllc            = hllc,
                **kwargs)  

        else:
            x1 = genspace(*self.geometry[0], self.xNpts)
            x2 = np.linpace(*self.geometry[1], self.yNpts)
            x3 = np.linpace(*self.geometry[2], self.zNpts)
            
            sources = np.zeros(self.u.shape[:-1], dtype=float) if not sources else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
                
            if self.regime == "classical":
                # TODO: Implement Newtonian 3D
                pass
                # b = PyState3D(u, self.gamma, cfl=cfl, x1=x1, x2=x2, coord_system=coordinates)
            else:
                state = PyStateSR3D(self.u, self.gamma, cfl=cfl, x1=x1, x2=x2, x3=x3, coord_system=coordinates)
            
            self.solution = state.simulate(
                sources         = sources,
                tstart          = tstart,
                tend            = tend,
                dlogt           = dlogt,
                plm_theta       = plm_theta,
                engine_duration = engine_duration,
                chkpt_interval  = chkpt_interval,
                chkpt_idx       = self.chkpt_idx,
                data_directory  = data_directory,
                boundary_condition = boundary_condition,
                first_order     = first_order,
                linspace        = linspace,
                hllc            = hllc)  
        
        if not periodic:
            self._cleanup(first_order)
        return self.solution

def __enter__(self):
    return self 

def __exit__(self, exc_type, exc_value, traceback):
    pass

class GPUExtNotBuiltWarning(UserWarning):
    pass