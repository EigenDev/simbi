# A Hydro Code Useful for solving MultiD structure problems
# Marcus DuPont
# New York University
# 06/10/2020
import simbi.helpers as helpers
import numpy as np 
import numpy.typing as npt
import os
import inspect
import simbi.initial_condition as simbi_ic 
import warnings
from functools import wraps
from .key_types import *

available_regimes             = ['classical', 'relativistic']
available_coord_systems       = ['spherical', 'cartesian', 'cylindrical', 'planar_cylindrical', 'axis_cylindrical']
available_boundary_conditions = ['outflow', 'reflecting', 'inflow', 'periodic']

class Hydro:
    linspace: BoolOrNone                    = None 
    sources: SequenceOrNone                 = None 
    passive_scalars: SequenceOrNone         = None 
    scale_factor: CallableOrNone            = None 
    scale_factor_derivative: CallableOrNone = None
    discontinuity: bool                     = False
    dens_outer: CallableOrNone              = None 
    edens_outer: CallableOrNone             = None
    mom_outer: Optional[Union[Sequence[Callable[...,Any]], Callable[...,Any]]] = None 
    x1: Any = None
    x2: Any = None
    x3: Any = None
    coord_system: str
    regime: str
    solution: NDArray[Any]
    geometry: Any
    u: NDArray[Any]
    resolution: Sequence[int]
    
    
    def __init__(self, *,
                 gamma: float,
                 initial_state: Union[Sequence[Any], NDArray[Any]],
                 resolution: Union[int, Sequence[int],NDArray[Any]],
                 geometry: Union[Sequence[float], Sequence[Sequence[float]]],
                 coord_system:str = 'cartesian',
                 regime: str = "classical", **extras: Any) -> None:
        """
        The initial conditions of the hydrodynamic system (1D for now)
        
        Parameters:
            gamma (float):                  Adiabatic Index
            
            initial_state (tuple or array): The initial conditions of the problem in the following format
                                            Ex. state = ((1.0, 1.0, 0.0), (0.1,0.125,0.0)) for Sod Shock Tube
                                            state = (array_like rho, array_like pressure, array_like velocity)
                                    
            resolution (int, tuple):              Number of grid points in 1D/2D Coordinate Lattice
            
            geometry (tuple):               The first starting point, the last, and an optional midpoint in the grid
                                            Ex. geometry = (0.0, 1.0, 0.5) for Sod Shock Tube
                                            Ex. geometry = ((x1min, x1max), (x2min, x2max))
            
            coord_system (string):          The coordinate system the problem uses. Currently only supports Cartesian 
                                            and Spherical Coordinate Lattces
            
            regime (string):                The classical (Newtonian) or relativisitc regime
            
        Return:
            None
        """
        if coord_system not in available_coord_systems:
            raise ValueError(f"Invalid coordinate system. Expected one of: {available_coord_systems}")
        
        if regime not in available_regimes:
            raise ValueError(f"Invalid simulation regime. Expected one of: {available_regimes}")
        
        # Update any static vars with attributes obtained from some setup configuration
        clean_attributes = [x for x in extras.keys() if not x.startswith('__')]
        [setattr(self, attribute, extras[attribute]) for attribute in clean_attributes if attribute in dir(self)] # type: ignore
        
        # check if given simple nexted sequence to split across the grid
        if any(isinstance(i, (list, tuple, np.ndarray)) for i in initial_state):
            if all(len(v) == 3 for v in initial_state):
                self.dimensionality = 1
                self.discontinuity = True
            elif all(len(v) == 4 for v in initial_state):
                self.dimensionality = 2
                self.discontinuity = True
            elif all(len(v) == 5 for v in initial_state):
                self.dimensionality = 3
                self.discontinuity = True
            else:
                pass 
            
        self.coord_system  = coord_system
        self.regime        = regime
        initial_state      = helpers.pad_jagged_array(initial_state)
        if not isinstance(resolution, (Sequence, np.ndarray)):
            resolution = (resolution,)
        
        self.gamma = gamma 
        if len(initial_state) < 5:
            self.geometry            = cast(Sequence[float], geometry)
            self.resolution          = cast(Sequence[int], resolution)
            self.nvars               = (2 + 1 * (self.dimensionality != 1) + self.dimensionality)
            
            # Initialize conserved u-tensor and flux tensors (defaulting to 2 ghost cells)
            self.u = np.zeros(shape = (self.nvars, *np.asarray(self.resolution).flatten()[::-1]))
            if self.discontinuity:
                print(f'Initializing Problem With a {str(self.dimensionality)}D Discontinuity...', flush=True)
                
                if len(self.geometry) == 3:
                    geom_tuple: Any = (self.geometry,)
                else:
                    geom_tuple = self.geometry
                
                break_points = [val[2] for val in geom_tuple if len(val) == 3]
                if len(break_points) > self.dimensionality:
                    raise ValueError("Number of break points must be less than or equal to the number of dimensions")
                
                spacings    = [(geom_tuple[idx][1] - geom_tuple[idx][0]) / self.resolution[idx] for idx in range(len(geom_tuple))]
                pieces      = [round(break_points[idx] / spacings[idx]) for idx in range(len(geom_tuple))]
                
                partition_inds: list[Any]
                if len(break_points) == 1:
                    partition_inds = [np.s_[:, :pieces[0]], np.s_[:, pieces[0]:]]
                elif len(break_points) == 2:
                    partition_inds = [np.s_[:, :pieces[1], :pieces[0]], np.s_[:, :pieces[1], pieces[0]:], np.s_[:, pieces[1]:, :pieces[0]], np.s_[:, pieces[1]:, pieces[0]:]] 
                else:
                    partition_inds = [np.s_[:, :pieces[2], :pieces[1], :pieces[0]], np.s_[:, :pieces[2], :pieces[1], pieces[0]:], np.s_[:, :pieces[2], pieces[1]:, :pieces[0]], np.s_[:, :pieces[2], pieces[1]:, pieces[0]:],
                                      np.s_[:, pieces[2]:, :pieces[1], :pieces[0]], np.s_[:, pieces[2]:, :pieces[1], pieces[0]:], np.s_[:, pieces[2]:, pieces[1]:, :pieces[0]], np.s_[:, pieces[2]:, pieces[1]:, pieces[0]:]] 
                
                partitions = [self.u[sector] for sector in partition_inds]
                for idx, part in enumerate(partitions):
                    state  = initial_state[idx]
                    rho, *velocity, pressure = state
                    velocity = np.asarray(velocity)
                    
                    vsqr                = self.calc_vsq(velocity)
                    lorentz_factor      = self.calc_lorentz_factor(vsqr, regime)
                    internal_energy     = self.calc_internal_energy(vsqr, regime)
                    total_enthalpy      = self.calc_enthalpy(rho, pressure, internal_energy, self.gamma)
                    enthalpy_limit      = self.calc_spec_enthalpy(rho, pressure, internal_energy, gamma, regime)
                    
                    energy = self.calc_energy_density(rho, lorentz_factor, total_enthalpy, pressure)
                    dens   = self.calc_labframe_densiity(rho, lorentz_factor)
                    mom    = self.calc_labframe_momentum(rho, lorentz_factor, enthalpy_limit, velocity)
                    
                    if self.dimensionality == 1:
                        part[...] = np.array([dens, *mom, energy])[:,None]
                    else:
                        part[...] = (part[...].transpose() + np.array([dens, *mom, energy, 0.0])).transpose()
            else:
                self.dimensionality = np.asarray(initial_state[0]).ndim
                # velocity: Any 
                rho, *velocity, pressure = initial_state
                velocity        = np.asarray(velocity)
                vsqr            = self.calc_vsq(velocity)
                lorentz_factor  = self.calc_lorentz_factor(vsqr, regime)
                internal_energy = self.calc_internal_energy(vsqr, regime)
                total_enthalpy  = self.calc_enthalpy(rho, pressure, internal_energy, gamma)
                enthalpy_limit  = self.calc_spec_enthalpy(rho, pressure, internal_energy, gamma, regime)
                
                self.init_density   = self.calc_labframe_densiity(rho, lorentz_factor) 
                self.init_momentum  = self.calc_labframe_momentum(rho, lorentz_factor, enthalpy_limit, velocity)
                self.init_energy    = self.calc_labframe_energy(rho, lorentz_factor, total_enthalpy, pressure)
                
                if self.dimensionality == 2:
                    self.u[...] = np.array([self.init_density, *self.init_momentum, self.init_energy, np.zeros_like(self.init_density)])
                else:
                    self.u[...] = np.array([self.init_density, *self.init_momentum, self.init_energy])
        else:
            raise ValueError("Initial State contains too many variables")
    
    @classmethod
    def gen_from_setup(cls, setup: Any) -> Any:
        return cls(**{str(param): getattr(setup, param) for param in dir(setup)})
    
    @staticmethod
    def calc_lorentz_factor(vsquared: NDArray[Any], regime: str) -> FloatOrArray:
        return 1.0 if regime == 'classical' else (1 - np.asarray(vsquared))**(-0.5)
    
    @staticmethod
    def calc_enthalpy(rho: FloatOrArray, pressure: FloatOrArray, internal_energy: FloatOrArray, gamma: float) -> FloatOrArray:
        return internal_energy + gamma * pressure / (rho * (gamma - 1))

    @staticmethod
    def calc_spec_enthalpy(rho: FloatOrArray, pressure: FloatOrArray, internal_energy: FloatOrArray, gamma: float, regime: str) -> FloatOrArray:
        return 1.0 if regime == 'classical' else Hydro.calc_enthalpy(rho, pressure, internal_energy, gamma)
    
    @staticmethod
    def calc_internal_energy(vsquared: NDArray[Any], regime: str) -> FloatOrArray:
        return 1.0 + 0.5 * np.asarray(vsquared) if regime == 'classical' else 1
    
    @staticmethod
    def calc_vsq(velocity: Union[NDArray[Any], Sequence[float]])-> NDArray[Any]:
        return np.array(sum(vcomp * vcomp for vcomp in velocity))
    
    @staticmethod
    def calc_energy_density(rho: FloatOrArray, lorentz: FloatOrArray, enthalpy: FloatOrArray, pressure: FloatOrArray) -> FloatOrArray:
        return rho * lorentz * lorentz * enthalpy - pressure - rho * lorentz
    
    @staticmethod
    def calc_labframe_densiity(rho: FloatOrArray, lorentz: FloatOrArray) -> FloatOrArray:
        return rho * lorentz 
    
    @staticmethod
    def calc_labframe_momentum(rho: FloatOrArray, lorentz: FloatOrArray, enthalpy: FloatOrArray, velocity: NDArray[Any]) -> NDArray[Any]:
        return rho * lorentz * lorentz * enthalpy * velocity
    
    @staticmethod
    def calc_labframe_energy(rho: FloatOrArray, lorentz: FloatOrArray, enthalpy: FloatOrArray, pressure: FloatOrArray) -> FloatOrArray:
        return rho * lorentz * lorentz * enthalpy  - pressure - rho * lorentz
    
    def _cleanup(self, first_order: bool) -> None:
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
    
    
    def _print_params(self, frame: Any) -> None:
        params = inspect.getargvalues(frame)
        print("="*80, flush=True)
        print("Simulation Parameters", flush=True)
        print("="*80, flush=True)
        for key, param in params.locals.items():
            if key != 'self':
                if isinstance(param, (float, np.float64)):
                    val_str = f"{param:.2f}"
                elif callable(param):
                    val_str = f"user-defined {key} function"
                elif isinstance(param, tuple):
                    if any(callable(p) for p in param):
                        val_str = f"user-defined {key} function(s)"
                elif isinstance(param, (list, np.ndarray)):
                    if len(param) > 6:
                        val_str = f"user-defined {key} terms"
                    else:
                        val_str = f"{param}"
                else:
                    val_str = str(param)
                
                my_str = str(key).ljust(30, '.')
                print(f"{my_str} {val_str}", flush=True)
        system_dict = {
            'adiabatic_gamma' : self.gamma,
            'resolution'      : self.resolution,
            'geometry'        : self.geometry,
            'coord_system'    : self.coord_system,
            'regime'          : self.regime,
        }
        
        for key, val in system_dict.items():
            my_str = str(key).ljust(30, '.')
            if isinstance(val, float):
                val_str = f"{val:.2f}"
            elif isinstance(val, tuple):
                val_str = str(val)
                if isinstance(val[0], tuple):
                    val_str = ''
                    for elem in val:
                        val_str += '(' + ', '.join('{0:.3f}'.format(t) for t in elem) + ')'
            else:
                val_str = str(val)
                
            print(f"{my_str} {val_str}", flush=True)
        print("="*80, flush=True)
    
    
    
    def _place_boundary_sources(self, boundary_sources: Union[Sequence[Any], NDArray[Any]], first_order: bool) -> NDArray[Any]:
        boundary_sources = [np.array([val]).flatten() for val in boundary_sources]
        max_len = np.max([len(a) for a in boundary_sources])
        boundary_sources = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in boundary_sources])
        edges   = [0,-1] if first_order else [0, 1, -1, -2]
        view    = self.u[:self.dimensionality + 2]
        
        slices: list[Any]
        if view.ndim == 1:
            slices = [(...,i) for i in edges] 
        elif view.ndim == 2:
            slices = [np.s_[:, i, :] for i in edges] + [np.s_[..., i] for i in edges]
        else:
            slices = [np.s_[:, i, ...] for i in edges] + [np.s_[..., i, :] for i in edges] + [np.s_[..., i] for i in edges]
            
        order   = 1 if first_order else 2
        for boundary in range(self.dimensionality * len(edges)):
            source = boundary_sources[boundary // order]
            if any(val != 0 for val in source):
                view[slices[boundary]] = source[:, None]
            
        return boundary_sources 
    
    
    
    def _generate_the_grid(self, linspace: bool) -> None:
        genspace: Callable[...,Any] = np.linspace 
        if not linspace:
            genspace = np.geomspace 
        
        if self.dimensionality == 1:
            self.x1 = self.x1 or genspace(*self.geometry[:2], *self.resolution)
        elif self.dimensionality == 2:
            self.x1 = self.x1 or genspace(self.geometry[0][0],   self.geometry[0][1], self.resolution[0])
            self.x2 = self.x2 or np.linspace(self.geometry[1][0], self.geometry[1][1], self.resolution[1])
        else:
            self.x1 = self.x1 or genspace(self.geometry[0][0],    self.geometry[0][1], self.resolution[0])
            self.x2 = self.x2 or np.linspace(self.geometry[1][0],  self.geometry[1][1], self.resolution[1])
            self.x3 = self.x3 or np.linspace(self.geometry[2][0], self.geometry[2][1], self.resolution[2])
            
        self.x1 = np.asarray(self.x1)
        self.x2 = np.asarray(self.x2)
        self.x3 = np.asarray(self.x3)
        
    def _set_boundary_conditions(self, boundary_conditions: Union[Sequence[Any], str, NDArray[Any]]) -> None:
        self.boundary_conditions = boundary_conditions
        
        
    def _check_boundary_conditions(self, boundary_conditions: Union[Sequence[Any], str, NDArray[Any]]) -> None:
        if isinstance(boundary_conditions, str):
            boundary_conditions = [boundary_conditions]
        
        for bc in boundary_conditions:
            if bc not in available_boundary_conditions:
                raise ValueError(f"Invalid boundary condition. Expected one of: {available_boundary_conditions}. Instead got: {bc}")
            
        number_of_given_bcs = len(boundary_conditions)
        if number_of_given_bcs != 2 * self.dimensionality:
            if number_of_given_bcs == 1:
                boundary_conditions = list(boundary_conditions) * 2 * self.dimensionality
            elif number_of_given_bcs == self.dimensionality // 2:
                boundary_conditions = [boundary_conditions[idx] * 2 for idx in range(number_of_given_bcs)]
            else:
                raise ValueError("Please include at a number of boundary conditions equal to at least half the number of cell faces")
        
        self.boundary_conditions = boundary_conditions
        
        
    def simulate(
        self, 
        tstart: float = 0.0,
        tend: float = 0.1,
        dlogt: float = 0.0,
        plm_theta: float = 1.5,
        first_order: bool = True,
        linspace: bool = True,
        cfl: float = 0.4,
        sources: Optional[NDArray[Any] ]= None,
        passive_scalars: Optional[Union[NDArray[Any], int]] = None,
        hllc: bool = False,
        chkpt: Optional[str] = None,
        chkpt_interval:       float = 0.1,
        data_directory:       str = "data/",
        boundary_conditions: Union[Sequence[str], str] = "outflow",
        engine_duration: float = 10.0,
        compute_mode:     str = 'cpu',
        quirk_smoothing:  bool = True,
        constant_sources: bool = False,
        scale_factor:            Optional[Callable[...,float]] = None,
        scale_factor_derivative: Optional[Callable[...,float]] = None,
        dens_outer:              Optional[Callable[...,float]] = None,
        mom_outer:               Optional[Union[Callable[...,float], Sequence[Callable[...,float]]]] = None,
        edens_outer:             Optional[Callable[...,float]] = None,
        object_positions: Optional[Union[Sequence[Any], NDArray[Any]]] = None,
        boundary_sources: Optional[Union[Sequence[Any], NDArray[Any]]] = None) -> NDArray[Any]:
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
            passive_scalars  (array_like):    The array of passive passive_scalars
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
        self.u          = np.asarray(self.u)
        self.start_time: float = 0.0
        self.chkpt_idx: int    = 0
        if compute_mode == 'cpu':
            from cpu_ext import PyState, PyState2D, PyStateSR, PyStateSR3D, PyStateSR2D
        else:
            try:
                from gpu_ext import PyState, PyState2D, PyStateSR, PyStateSR3D, PyStateSR2D
            except ImportError as e:
                warnings.warn("Error in loading GPU extension. Loading CPU instead...", GPUExtNotBuiltWarning)
                warnings.warn(f"For reference, the gpu_ext had the follow error: {e}", GPUExtNotBuiltWarning)
                from cpu_ext import PyState, PyState2D, PyStateSR, PyStateSR3D, PyStateSR2D
        
        scale_factor = scale_factor or (lambda t: 1.0)
        scale_factor_derivative = scale_factor_derivative or (lambda t: 0.0)        
        self._generate_the_grid(linspace)
        
        mesh_motion = scale_factor_derivative(1.0) / scale_factor(1.0) != 0
        volume_factor: Any = 1.0
        if mesh_motion and self.coord_system != 'cartesian':
            if self.dimensionality == 1:
                volume_factor = helpers.calc_cell_volume1D(x1=self.x1)
            elif self.dimensionality == 2:
                volume_factor = helpers.calc_cell_volume2D(x1=self.x1, x2=self.x2, coord_system=self.coord_system)
        
        self._check_boundary_conditions(boundary_conditions)        
        if not chkpt:
            simbi_ic.initializeModel(self, first_order, volume_factor, passive_scalars)
        else:
            simbi_ic.load_checkpoint(self, chkpt, self.dimensionality, mesh_motion)
        
        if self.dimensionality == 1 and self.coord_system in ['planar_cylindrical', 'axis_cylindrical']:
            self.coord_system = 'cylindrical'
            
        periodic        = all(bc == 'periodic' for bc in boundary_conditions)
        self.start_time = self.start_time or tstart
        #Convert strings to byte arrays
        cython_data_directory      = os.path.join(data_directory, '').encode('utf-8')
        cython_coordinates         = self.coord_system.encode('utf-8')
        cython_boundary_conditions = np.array([bc.encode('utf-8') for bc in self.boundary_conditions])
        
        # Offset the start time from zero if wanting log 
        # checkpoints, but with initial time of zero
        if dlogt !=0 and self.start_time == 0:
            start_time = 1e-16 
            
        # Check whether the specified path exists or not
        if not os.path.exists(data_directory):
            # Create a new directory because it does not exist 
            os.makedirs(data_directory)
            print(f"The data directory provided does not exist. Creating the {data_directory} directory now!", flush=True)

        # Loading bar to have chance to check params
        helpers.print_progress()
        
        # Create boolean maks for object immersed boundaries (impermable)
        object_cells: NDArray[Any] = np.zeros_like(self.u[0], dtype=bool) if object_positions is None else np.asarray(object_positions, dtype=bool) 
        #####################################################################################################
        # Check if boundary source terms given. If given as a jagged array, pad the missing members with zeros
        #####################################################################################################
        if boundary_sources is None:
            boundary_sources = np.zeros((2 * self.dimensionality, self.dimensionality + 2))
        else:
            boundary_sources = self._place_boundary_sources(boundary_sources=boundary_sources, first_order=first_order)
        
        print(f"Computing {'First' if first_order else 'Second'} Order Solution...", flush=True)
        kwargs: dict[str, Any] = {}
        if self.dimensionality  == 1:
            sources = np.zeros_like(self.u) if sources is None else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
            if self.regime == "classical":
                state = PyState(self.u, self.gamma, cfl, x1 = self.x1, coord_system = cython_coordinates)
            else:   
                state = PyStateSR(self.u, self.gamma, cfl, x1 = self.x1, coord_system = cython_coordinates)
                kwargs = {'a': scale_factor, 'adot': scale_factor_derivative}
                if mesh_motion and dens_outer and mom_outer and edens_outer:
                    kwargs['d_outer'] =  dens_outer
                    kwargs['s_outer'] =  mom_outer
                    kwargs['e_outer'] =  edens_outer
                
        elif self.dimensionality  == 2:     
            # ignore the chi term
            sources = np.zeros(self.u[:-1].shape, dtype=float) if sources is None else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)

            if self.regime == "classical":
                state = PyState2D(self.u, self.gamma, cfl=cfl, x1=self.x1, x2=self.x2, coord_system=cython_coordinates)
            else:
                kwargs = {'a': scale_factor, 'adot': scale_factor_derivative, 'quirk_smoothing': quirk_smoothing, 'object_cells': object_cells}
                if mesh_motion and dens_outer and mom_outer and edens_outer:
                    momentum_components    = cast(Sequence[Callable[...,float]], mom_outer)
                    kwargs['d_outer']      = dens_outer
                    kwargs['s1_outer']     = momentum_components[0]
                    kwargs['s2_outer']     = momentum_components[1]
                    kwargs['e_outer']      = edens_outer
                
                state = PyStateSR2D(self.u, self.gamma, cfl=cfl, x1=self.x1, x2=self.x2, coord_system=cython_coordinates)

        else:
            sources = np.zeros(self.u.shape[:-1], dtype=float) if sources is None else np.asarray(sources)
            sources = sources.reshape(sources.shape[0], -1)
                
            if self.regime == "classical":
                # TODO: Implement Newtonian 3D
                pass
                # b = PyState3D(u, self.gamma, cfl=cfl, x1=x1, x2=x2, coord_system=cython_coordinates)
            else:
                state = PyStateSR3D(self.u, self.gamma, cfl=cfl, x1=self.x1, x2=self.x2, x3=self.x3, coord_system=cython_coordinates)
                kwargs = {'object_cells': object_cells}

        self.solution = state.simulate(
            sources            = sources,
            tstart             = self.start_time,
            tend               = tend,
            dlogt              = dlogt,
            plm_theta          = plm_theta,
            engine_duration    = engine_duration,
            chkpt_interval     = chkpt_interval,
            chkpt_idx          = self.chkpt_idx,
            data_directory     = cython_data_directory,
            boundary_conditions= cython_boundary_conditions,
            first_order        = first_order,
            linspace           = linspace,
            hllc               = hllc,
            constant_sources   = constant_sources,
            boundary_sources   = boundary_sources,
            **kwargs)  
        
        if not periodic:
            self._cleanup(first_order)
        return self.solution

class GPUExtNotBuiltWarning(UserWarning):
    pass