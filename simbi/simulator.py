# A Hydro Code Useful for solving MultiD structure problems
# Marcus DuPont
# New York University
# 06/10/2020
import numpy as np
import os
import inspect
import importlib
from itertools import chain, repeat
from .detail import initial_condition as simbi_ic
from .detail import helpers
from .detail.slogger import logger
from .key_types import *


available_regimes = ['classical', "srhd", "srmhd"]
available_coord_systems = [
    'spherical',
    'cartesian',
    'cylindrical',
    'planar_cylindrical',
    'axis_cylindrical']
available_boundary_conditions = ['outflow', 'reflecting', 'inflow', 'periodic']
available_cellspacings = [
    'linear', 
    'log',
    # TODO: implement soon 'log-linear',
    # TODO: implement soon 'linear-log' 
]


class Hydro:
    x1_cell_spacing: StrOrNone = None
    x2_cell_spacing: StrOrNone = None
    x3_cell_spacing: StrOrNone = None
    sources: SequenceOrNone = None
    passive_scalars: SequenceOrNone = None
    scale_factor: CallableOrNone = None
    scale_factor_derivative: CallableOrNone = None
    discontinuity: bool = False
    dens_outer: CallableOrNone = None
    edens_outer: CallableOrNone = None
    mom_outer: Optional[Union[Sequence[Callable[..., Any]],
                              Callable[..., Any]]] = None
    x1: Any = None
    x2: Any = None
    x3: Any = None
    boundary_conditions: list[str]
    coord_system: str
    regime: str
    solution: NDArray[Any]
    geometry: Any
    u: NDArray[Any]
    resolution: Sequence[int]

    def __init__(self, *,
                 gamma: float,
                 initial_state: Union[Sequence[Any], NDArray[Any]],
                 resolution: Union[int, Sequence[int], NDArray[Any]],
                 geometry: Union[Sequence[float], Sequence[Sequence[float]]],
                 coord_system: str = 'cartesian',
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
            raise ValueError(
                f"Invalid coordinate system. Expected one of: {available_coord_systems}. Instead got: {coord_system}")

        if regime not in available_regimes:
            raise ValueError(
                f"Invalid simulation regime. Expected one of: {available_regimes}. Instead got {regime}")

        # Update any static vars with attributes obtained from some setup
        # configuration
        clean_attributes = [x for x in extras.keys() if not x.startswith('__')]
        helpers.for_each(lambda x: setattr(self, x, extras[x]) if x in dir(self) else None, clean_attributes)
        resolution = helpers.get_iterable(resolution)
        resolution = tuple(resolution)
        
        self.geometry   = geometry
        self.resolution = resolution
        self.coord_system = coord_system
        self.regime       = regime
        self.gamma        = gamma
        self.mhd = self.regime in ['srmhd']
        
        tuple_of_tuples: Callable[..., bool] = lambda x: all(isinstance(a, Sequence) for a in x)
        
        if tuple_of_tuples(initial_state): 
            # check if given simple nested sequence to split across the grid
            if all(len(v) == 3 for v in initial_state):
                if self.mhd:
                    raise ValueError("Not enough variables across discontinuity for mhd run")
                self.dimensionality = 1
                self.discontinuity = True
            elif all(len(v) == 4 for v in initial_state):
                if self.mhd:
                    raise ValueError("Not enough variables across discontinuity for mhd run")
                self.dimensionality = 1
                self.discontinuity = True
            elif all(len(v) == 5 for v in initial_state):
                if self.mhd:
                    raise ValueError("Not enough variables across discontinuity for mhd run")
                self.dimensionality = 3
                self.discontinuity = True
            elif all(len(v) == 6 for v in initial_state):
                if self.mhd:
                    raise ValueError("Not enough variables across discontinuity for mhd run")
                self.dimensionality = 2
                self.discontinuity = True
            elif all(len(v) == 8 for v in initial_state):
                if not self.mhd:
                    raise ValueError("Too many variables across discontinuity for non-mhd run")
                self.dimensionality = np.asanyarray(initial_state[0]).ndim
                self.discontinuity = True
            else:
                raise ValueError("State arrays across discontinuity need to have equal length")
        else:
            if all(isinstance(x, (float, int)) for x in initial_state):
                initial_state = tuple(x * np.ones(shape=self.resolution) for x in initial_state)
            self.dimensionality = np.asanyarray(initial_state[0]).ndim

        ngeom = len(self.geometry)
        nres  = len(self.resolution)
        
        # if len(self.geometry) != self.dimensionality:
        #     raise ValueError(f"Detecing a {self.dimensionality}D run, but only {ngeom} geometry tuples")
        
        if len(self.resolution) != self.dimensionality:
            raise ValueError(f"Detecing a {self.dimensionality}D run, but only {nres} resolution args")
        
        initial_state = helpers.pad_jagged_array(initial_state)
        nstates = len(initial_state)
        max_discont = 2 ** self.dimensionality
        self.number_of_non_em_terms = 2 + self.dimensionality if not self.mhd else 5
        max_prims = self.number_of_non_em_terms + 3 * self.mhd
        if nstates < max_prims or (nstates < max_discont and self.discontinuity):
            simbi_ic.construct_the_state(self, initial_state=initial_state)
        else:
            raise ValueError("Initial State contains too many variables")

    @classmethod
    def gen_from_setup(cls, setup: Any) -> Any:
        return cls(**{str(param): getattr(setup, param)
                   for param in dir(setup)})

    def _print_params(self, frame: Any) -> None:
        params = inspect.getargvalues(frame)
        logger.info("=" * 80) 
        logger.info("Simulation Parameters") 
        logger.info("=" * 80) 
        for key, param in params.locals.items():
            if key != 'self':
                if isinstance(param, (float, np.float64)):
                    val_str: Any = f"{param:.3f}"
                elif key == 'sources' and param is not None:
                    val_str = f'user-defined sources terms'
                elif key == 'gsources' and param is not None:
                    val_str = f'user-defined gravity sources'
                elif callable(param):
                    val_str = f"user-defined {key} function"
                elif isinstance(param, tuple):
                    if any(callable(p) for p in param):
                        val_str = f"user-defined {key} function(s)"
                elif isinstance(param, (list, np.ndarray)):
                    if len(param) > 6:
                        val_str = f"user-defined {key} terms"
                    else:
                        if any(isinstance(val, (float, int)) for val in param):
                            val_str = []
                            for val in param:
                                if isinstance(val, list):
                                    val_str += [[float(f'{item:.3f}') for item in val]]
                                else:
                                    val_str += [val]
                        else:
                            val_str = f"{param}"
                else:
                    val_str = str(param)

                my_str = str(key).ljust(30, '.')
                logger.info(f"{my_str} {val_str}") 
        system_dict = {
            'adiabatic_gamma': self.gamma,
            'resolution': self.resolution,
            'geometry': self.geometry,
            'coord_system': self.coord_system,
            'regime': self.regime,
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
                        val_str += '(' + ', '.join('{0:.3f}'.format(t)
                                                   for t in elem) + ')'
            else:
                val_str = str(val)

            logger.info(f"{my_str} {val_str}") 
        logger.info("=" * 80) 

    def _place_boundary_sources(self,
                                boundary_sources: Union[Sequence[Any],
                                                        NDArray[Any]],
                                first_order: bool) -> NDArray[Any]:
        boundary_sources = [np.array([val]).flatten()
                            for val in boundary_sources]
        max_len = np.max([len(a) for a in boundary_sources])
        boundary_sources = np.asanyarray([np.pad(
            a, (0, max_len - len(a)), 'constant', constant_values=0) for a in boundary_sources])
        edges = [0, -1] if first_order else [0, 1, -1, -2]
        view = self.u[:self.dimensionality + 2]

        slices: list[Any]
        if view.ndim == 1:
            slices = [(..., i) for i in edges]
        elif view.ndim == 2:
            slices = [np.s_[:, i, :]
                      for i in edges] + [np.s_[..., i] for i in edges]
        else:
            slices = [np.s_[..., i] for i in edges] + [np.s_[..., i, :] for i in edges] + [np.s_[:, i, ...] for i in edges]

        order = 1 if first_order else 2
        if self.dimensionality == 3:
            source_transform: Any = np.s_[:, None, None]
        elif self.dimensionality == 2:
            source_transform  = np.s_[:, None]
        else:
            source_transform = np.s_[:]

        for boundary in range(self.dimensionality * len(edges)):
            source = boundary_sources[boundary // order]
            if any(val != 0 for val in source):
                view[slices[boundary]] = source[source_transform]
        return boundary_sources

    def _generate_the_grid(
        self, 
        x1_cell_spacing: str,
        x2_cell_spacing: str,
        x3_cell_spacing: str) -> None:
        
        x1_func: Callable[...,Any]
        if x1_cell_spacing == 'log':
            x1_func = np.geomspace
        elif x1_cell_spacing == 'linear':
            x1_func = np.linspace 
        
        x2_func: Callable[...,Any]
        if x2_cell_spacing == 'log':
            x2_func = np.geomspace
        elif x2_cell_spacing == 'linear':
            x2_func = np.linspace
        
        x3_func: Callable[...,Any]
        if x3_cell_spacing == 'log':
            x3_func = np.geomspace
        elif x3_cell_spacing == 'linear':
            x3_func = np.linspace

        if self.dimensionality == 1:
            if self.x1 is None:
                self.x1 = x1_func(*self.geometry[:2], *self.resolution)
        elif self.dimensionality == 2:
            if self.x1 is None:
                self.x1 = x1_func(
                    self.geometry[0][0],
                    self.geometry[0][1],
                    self.resolution[0])
            if self.x2 is None:
                self.x2 = x2_func(
                self.geometry[1][0], self.geometry[1][1], self.resolution[1])
        else:
            if self.x1 is None:
                self.x1 = x1_func(
                    self.geometry[0][0],
                    self.geometry[0][1],
                    self.resolution[0])
            if self.x2 is None:
                self.x2 = x2_func(
                self.geometry[1][0], self.geometry[1][1], self.resolution[1])
            if self.x3 is None:
                self.x3 = x3_func(
                self.geometry[2][0], self.geometry[2][1], self.resolution[2])

        self.x1 = np.asanyarray(self.x1)
        self.x2 = np.asanyarray(self.x2)
        self.x3 = np.asanyarray(self.x3)

    def _check_boundary_conditions(self,
                                   boundary_conditions: Union[Sequence[str],
                                                              str,
                                                              NDArray[numpy_string]]) -> None:
        boundary_conditions = list(helpers.get_iterable(boundary_conditions))
        for bc in boundary_conditions:
            if bc not in available_boundary_conditions:
                raise ValueError(
                    f"Invalid boundary condition. Expected one of: {available_boundary_conditions}. Instead got: {bc}")

        number_of_given_bcs = len(boundary_conditions)
        if number_of_given_bcs != 2 * self.dimensionality:
            if number_of_given_bcs == 1:
                boundary_conditions = list(
                    boundary_conditions) * 2 * self.dimensionality
            elif number_of_given_bcs == (self.dimensionality * 2) // 2:
                boundary_conditions = list(
                    chain.from_iterable(repeat(x, 2) for x in boundary_conditions)
                )
            else:
                raise ValueError(
                    "Please include at a number of boundary conditions equal to at least half the number of cell faces")

        self.boundary_conditions = boundary_conditions

    def simulate(
            self,
            tstart: float = 0.0,
            tend: float = 0.1,
            dlogt: float = 0.0,
            plm_theta: float = 1.5,
            first_order: bool = True,
            x1_cell_spacing: str = 'linear',
            x2_cell_spacing: str = 'linear',
            x3_cell_spacing: str = 'linear',
            cfl: float = 0.4,
            sources: Optional[NDArray[Any]] = None,
            gsources: Optional[NDArray[Any]] = None,
            bsources: Optional[NDArray[Any]] = None,
            passive_scalars: Optional[Union[NDArray[Any], int]] = None,
            solver: str = 'hllc',
            chkpt: Optional[str] = None,
            chkpt_interval: float = 0.1,
            data_directory: str = "data/",
            boundary_conditions: Union[Sequence[str], str] = "outflow",
            engine_duration: float = 10.0,
            compute_mode: str = 'cpu',
            quirk_smoothing: bool = True,
            constant_sources: bool = False,
            scale_factor: Optional[Callable[[float], float]] = None,
            scale_factor_derivative: Optional[Callable[[float], float]] = None,
            dens_outer: Optional[Callable[..., float]] = None,
            mom_outer: Optional[Union[Callable[..., float], Sequence[Callable[..., float]]]] = None,
            edens_outer: Optional[Callable[..., float]] = None,
            object_positions: Optional[Union[Sequence[Any], NDArray[Any]]] = None,
            boundary_sources: Optional[Union[Sequence[Any], NDArray[Any]]] = None) -> None:
        """
        Simulate the Hydro Setup

        Parameters:
            tstart      (flaot):         The start time of the simulation
            tend        (float):         The desired time to end the simulation
            dlogt       (float):         The desired logarithmic spacing in checkpoints
            plm_theta   (float):         The Piecewise Linear Reconstructed slope parameter
            first_order (boolean):       First order RK1 or the RK2 PLM.
            x1_cell_spacing    (str):     Option for a linear or log-spaced mesh on x1 
            x2_cell_spacing    (str):     Option for a linear or log-spaced mesh on x2 
            x3_cell_spacing    (str):     Option for a linear or log-spaced mesh on x3 
            cfl         (float):         The cfl number for min adaptive timestep
            sources     (array_like):    The source terms for the simulations
            passive_scalars  (array_like):    The array of passive passive_scalars
            solver        (str):         Tells the simulation whether to perform HLLC or HLLE
            chkpt       (string):        The path to the checkpoint file to read into the simulation
            chkpt_interval (float):      The interval at which to save the checkpoints
            data_directory (string):     The directory at which to save the checkpoint files
            bounday_condition (string):  The outer conditions at the domain x1 boundaries
            engine_duration (float):     The duration the source terms will last in the simulation
            compute_mode (string):       The compute mode for simulation execution (cpu or gpu)
            quirk_smoothing (bool):       The switch that controls the Quirk (1960) shock smoothing method
            constant_source (bool):      Set to true if wanting the source terms to never die
            scale_factor              (Callable):   The scalar function for moving mesh. Think cosmology
            scale_factor_derivative   (Callable):   The first derivative of the scalar function for moving mesh
            dens_outer     (Callable):   The density to be fed into outer zones if moving mesh
            mom_outer      (Callables):  idem but for momentum density
            edens_outer    (Callable):   idem but for energy density
            object_positions (boolean array_lie): An optional boolean array that masks the immersed boundary
            boundary_source (array_like): An array of conserved quantities at the boundaries of the grid

        Returns:
            solution (array): The hydro solution containing the primitive variables
        """
        self._print_params(inspect.currentframe())
        if x1_cell_spacing not in available_cellspacings:
            raise ValueError(f"cell spacing for x1 should be one of: {available_cellspacings}")
        
        if x2_cell_spacing not in available_cellspacings:
            raise ValueError(f"cell spacing for x2 should be one of: {available_cellspacings}")
        
        if x3_cell_spacing not in available_cellspacings:
            raise ValueError(f"cell spacing for x3 should be one of: {available_cellspacings}")
        
        self.u = np.asanyarray(self.u)
        self.start_time: float = 0.0
        self.chkpt_idx: int = 0
        scale_factor = scale_factor or (lambda t: 1.0)
        scale_factor_derivative = scale_factor_derivative or (lambda t: 0.0)
        self._generate_the_grid(x1_cell_spacing, x2_cell_spacing, x3_cell_spacing)

        mesh_motion = scale_factor_derivative(tstart) / scale_factor(tstart) != 0
        volume_factor: Union[float, NDArray[Any]] = 1.0
        if mesh_motion and self.coord_system != 'cartesian':
            if self.dimensionality == 1:
                volume_factor = helpers.calc_cell_volume1D(
                    x1=self.x1, coord_system=self.coord_system
                )
                volume_factor = helpers.calc_cell_volume1D(x1=self.x1, coord_system=self.coord_system)
            elif self.dimensionality == 2:
                volume_factor = helpers.calc_cell_volume2D(
                    x1=self.x1, x2=self.x2, coord_system=self.coord_system
                )

        self._check_boundary_conditions(boundary_conditions)
        if not chkpt:
            simbi_ic.initializeModel(
                self, first_order, volume_factor, passive_scalars)
        else:
            simbi_ic.load_checkpoint(
                self, chkpt, self.dimensionality, mesh_motion)
        if self.dimensionality == 1 and self.coord_system in [
                'planar_cylindrical', 'axis_cylindrical']:
            self.coord_system = 'cylindrical'

        self.start_time = self.start_time or tstart

        #######################################################################
        # Check if boundary source terms given. If given as a jagged array, pad the missing members with zeros
        #######################################################################
        if boundary_sources is None:
            boundary_sources = np.zeros(
                (2 * self.dimensionality, self.dimensionality + 2))
        else:
            boundary_sources = self._place_boundary_sources(
                boundary_sources=boundary_sources, first_order=first_order)

        for idx, bc in enumerate(self.boundary_conditions):
            if bc == 'inflow' and 0 in [
                    boundary_sources[idx][0], boundary_sources[idx][-1]]:
                self.boundary_conditions[idx] = str('outflow')

        # Convert strings to byte arrays
        cython_data_directory = os.path.join(
            data_directory, '').encode('utf-8')
        cython_coordinates = self.coord_system.encode('utf-8')
        cython_solver = solver.encode('utf-8')
        cython_boundary_conditions: NDArray[numpy_string] = np.array(
            [bc.encode('utf-8') for bc in self.boundary_conditions])

        # Offset the start time from zero if wanting log
        # checkpoints, but with initial time of zero
        if dlogt != 0 and self.start_time == 0:
            self.start_time = 1e-16

        # Check whether the specified path exists or not
        if not os.path.exists(data_directory):
            # Create a new directory because it does not exist
            os.makedirs(data_directory)
            logger.info( 
                f"The data directory provided does not exist. Creating the {data_directory} directory now!") 

        if compute_mode in ['cpu', 'omp']:
            if 'USE_OMP' in os.environ:
                logger.debug("Using OpenMP multithreading")
            else:
                logger.debug("Using STL std::thread multithreading")
        else:
            dim3 = [1, 1, 1]
            for idx, coord in enumerate(['X', 'Y', 'Z']):
                if user_set := f'GPU{coord}BLOCK_SIZE' in os.environ:
                    if idx + 1 <= self.dimensionality:
                        dim3[idx] = int(os.environ[f'GPU{coord}BLOCK_SIZE'])
                else:
                    if self.dimensionality == 1 and coord == 'X':
                        dim3[idx] = 128
                    elif self.dimensionality == 2 and coord in ['X', 'Y']:
                        dim3[idx] = 16
                    elif self.dimensionality == 3 and coord in ['X', 'Y', 'Z']:
                        dim3[idx] = 4
            logger.debug(f"In GPU mode, GPU block dims are: {tuple(dim3)}")
            
        logger.info("") 
        # Loading bar to have chance to check params
        helpers.print_progress()

        # Create boolean masks for object immersed boundaries (impermeable)
        object_cells: NDArray[Any] = np.zeros_like(
            self.u[0], dtype=bool) if object_positions is None else np.asanyarray(
            object_positions, dtype=bool)

        logger.info( 
            f"Computing {'First' if first_order else 'Second'} Order Solution...") 
        kwargs: dict[str, Any] = {}
        
        sources = np.zeros(self.dimensionality + 2) if sources is None else np.asanyarray(sources)
        sources = sources.reshape(sources.shape[0], -1)
        gsources = np.zeros(3) if gsources is None else np.asanyarray(gsources)
        gsources = gsources.reshape(gsources.shape[0], -1)
        bsources = np.zeros(3) if bsources is None else np.asanyarray(bsources)
        bsources = bsources.reshape(bsources.shape[0], -1)
        
        if compute_mode == 'gpu':
            if self.dimensionality == 1:
                if 'GPUXBLOCK_SIZE' not in os.environ:
                    os.environ['GPUXBLOCK_SIZE'] = "128"
            elif self.dimensionality == 2:
                if 'GPUXBLOCK_SIZE' not in os.environ:
                    os.environ['GPUXBLOCK_SIZE'] = "16"
                    
                if 'GPUYBLOCK_SIZE' not in os.environ:
                    os.environ['GPUYBLOCK_SIZE'] = "16" 
            else:
                if 'GPUXBLOCK_SIZE' not in os.environ:
                    os.environ['GPUXBLOCK_SIZE'] = "4"
                    
                if 'GPUYBLOCK_SIZE' not in os.environ:
                    os.environ['GPUYBLOCK_SIZE'] = "4" 
                    
                if 'GPUZBLOCK_SIZE' not in os.environ:
                    os.environ['GPUZBLOCK_SIZE'] = "4"
                
        
        if len(self.resolution) == 1:
            self.nx = self.u[0].shape[0]
            self.ny = 1
            self.nz = 1
        elif len(self.resolution) == 2:
            self.ny, self.nx = self.u[0].shape
            self.nz = 1
        else:
            self.nz, self.ny, self.nx = self.u[0].shape

        init_conditions = {
            'gamma': self.gamma,
            'sources': sources,
            'tstart': self.start_time,
            'tend': tend,
            'cfl': cfl,
            'dlogt': dlogt,
            'plm_theta': plm_theta,
            'engine_duration': engine_duration,
            'chkpt_interval': chkpt_interval,
            'chkpt_idx': self.chkpt_idx,
            'data_directory': cython_data_directory,
            'boundary_conditions': cython_boundary_conditions,
            'first_order': first_order,
            'x1_cell_spacing': x1_cell_spacing.encode('utf-8'),
            'x2_cell_spacing': x2_cell_spacing.encode('utf-8'),
            'x3_cell_spacing': x3_cell_spacing.encode('utf-8'),
            'solver': cython_solver,
            'constant_sources': constant_sources,
            'boundary_sources': boundary_sources,
            'coord_system': cython_coordinates,
            'quirk_smoothing': quirk_smoothing,
            'x1': self.x1,
            'x2': self.x2,
            'x3': self.x3,
            'gsource': gsources,
            'nx': self.nx,
            'ny': self.ny,
            'nz': self.nz,
            'object_cells': object_cells.flatten()
        }
        
        if self.mhd:
            init_conditions['bsources'] = bsources
        
        lambdas: dict[str, Optional[Callable[...,float]]] = {
            'dens_lambda': None,
            'mom1_lambda': None,
            'mom2_lambda': None,
            'mom3_lambda': None,
            'enrg_lambda': None
        }
        if mesh_motion and dens_outer and mom_outer and edens_outer:
            momentum_components = cast(Sequence[Callable[..., float]], mom_outer)
            lambdas['dens_lambda'] = dens_outer
            lambdas['enrg_lambda'] = edens_outer
            if self.dimensionality == 1:
                mom_outer = cast(Callable[..., float], mom_outer)
                lambdas['mom1_lambda'] = mom_outer
            else:
                lambdas['mom1_lambda'] = momentum_components[0]
                lambdas['mom2_lambda'] = momentum_components[1]
                if self.dimensionality == 3:
                    lambdas['mom3_lambda'] = momentum_components[2]

        lib_mode  = 'cpu' if compute_mode in ['cpu', 'omp'] else 'gpu'
        sim_state = getattr(importlib.import_module(f'.{lib_mode}_ext', package='simbi.libs'), 'SimState')
        state_contig = self.u.reshape(self.u.shape[0], -1)
        state = sim_state()
        
        state.run(
            state = state_contig, 
            dim = self.dimensionality,
            regime = self.regime.encode('utf-8'),
            sim_info = init_conditions,
            a = scale_factor,
            adot = scale_factor_derivative,
            **lambdas
        )
