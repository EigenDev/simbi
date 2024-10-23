# A Hydro Code Useful for solving MultiD structure problems
# Marcus DuPont
# New York University
# 06/10/2020
import numpy as np
import os
import inspect
import importlib
import tracemalloc
import simbi.detail as detail
from pathlib import Path
from itertools import chain, repeat
from .detail import initial_condition as simbi_ic
from .detail import helpers
from .detail.slogger import logger
from .key_types import *


available_regimes = ["classical", "srhd", "srmhd"]
available_coord_systems = [
    "spherical",
    "cartesian",
    "cylindrical",
    "planar_cylindrical",
    "axis_cylindrical",
]
available_boundary_conditions = ["outflow", "reflecting", "dynamic", "periodic"]
available_cellspacings = [
    "linear",
    "log",
    # TODO: implement soon 'log-linear',
    # TODO: implement soon 'linear-log'
]


class Hydro:
    x1_cell_spacing: StrOrNone = None
    x2_cell_spacing: StrOrNone = None
    x3_cell_spacing: StrOrNone = None
    passive_scalars: SequenceOrNone = None
    scale_factor: CallableOrNone = None
    scale_factor_derivative: CallableOrNone = None
    discontinuity: bool = False
    bsources: list[CallableOrNone] = [None]
    hsources: list[CallableOrNone] = [None]
    gsources: list[CallableOrNone] = [None]
    x1: Any = []
    x2: Any = []
    x3: Any = []
    boundary_conditions: list[str]
    coord_system: str
    regime: str
    solution: NDArray[Any]
    geometry: Any
    u: NDArray[Any]
    resolution: Sequence[int]
    bfield: NDArray[numpy_float]
    trace_memory: bool = False

    def __init__(
        self,
        *,
        gamma: float,
        initial_state: Union[Sequence[Any], NDArray[Any]],
        resolution: Union[int, Sequence[int], NDArray[Any]],
        geometry: Union[Sequence[float], Sequence[Sequence[float]]],
        coord_system: str = "cartesian",
        regime: str = "classical",
        **extras: Any,
    ) -> None:
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
                f"Invalid coordinate system. Expected one of: {available_coord_systems}. Instead got: {coord_system}"
            )

        if regime not in available_regimes:
            raise ValueError(
                f"Invalid simulation regime. Expected one of: {available_regimes}. Instead got {regime}"
            )

        # Update any static vars with attributes obtained from some setup
        # configuration
        clean_attributes = [x for x in extras.keys() if not x.startswith("__")]
        helpers.for_each(
            lambda x: setattr(self, x, extras[x]) if x in dir(self) else None,
            clean_attributes,
        )
        resolution = helpers.get_iterable(resolution)
        resolution = tuple(resolution)

        self.geometry = geometry
        self.resolution = resolution
        self.coord_system = coord_system
        self.regime = regime
        self.gamma = gamma
        self.mhd = self.regime in ["srmhd", "mhd"]
        self.initial_state = initial_state

        if helpers.tuple_of_tuples(initial_state):
            # check if given simple nested sequence to split across the grid
            if all(len(v) == 3 for v in initial_state):
                if self.mhd:
                    raise ValueError(
                        "Not enough variables across discontinuity for mhd run"
                    )
                self.dimensionality = 1
                self.discontinuity = True
            elif all(len(v) == 4 for v in initial_state):
                if self.mhd:
                    raise ValueError(
                        "Not enough variables across discontinuity for mhd run"
                    )
                self.dimensionality = 1
                self.discontinuity = True
            elif all(len(v) == 5 for v in initial_state):
                if self.mhd:
                    raise ValueError(
                        "Not enough variables across discontinuity for mhd run"
                    )
                self.dimensionality = 3
                self.discontinuity = True
            elif all(len(v) == 6 for v in initial_state):
                if self.mhd:
                    raise ValueError(
                        "Not enough variables across discontinuity for mhd run"
                    )
                self.dimensionality = 2
                self.discontinuity = True
            elif all(len(v) == 8 for v in initial_state):
                if not self.mhd:
                    raise ValueError(
                        "Too many variables across discontinuity for non-mhd run"
                    )
                self.dimensionality = 3
                self.discontinuity = True
            else:
                raise ValueError(
                    "State arrays across discontinuity need to have equal length"
                )
        else:
            if all(isinstance(x, (float, int)) for x in initial_state):
                initial_state = tuple(
                    x * np.ones(shape=self.resolution) for x in initial_state
                )
            self.dimensionality = np.asanyarray(initial_state[0]).ndim

        if not helpers.tuple_of_tuples(self.geometry):
            ngeom = 1
        else:
            ngeom = len(self.geometry)
        nres = len(self.resolution)

        if ngeom != self.dimensionality:
            raise ValueError(
                f"Detecting a {self.dimensionality}D run, but only {ngeom} geometry tuple(s)"
            )

        if len(self.resolution) != self.dimensionality:
            raise ValueError(
                f"Detecting a {self.dimensionality}D run, but only {nres} resolution args"
            )

        initial_state = np.asanyarray(initial_state, dtype=object)
        size = len(initial_state[0])
        if not all(len(x) == size for x in initial_state) and not self.mhd:
            initial_state = helpers.pad_jagged_array(initial_state)

        nstates = len(initial_state)
        max_discont = 2**self.dimensionality
        self.number_of_non_em_terms = 2 + self.dimensionality if not self.mhd else 5
        max_prims = self.number_of_non_em_terms + 3 * self.mhd
        if nstates <= max_prims or (nstates < max_discont and self.discontinuity):
            detail.initial_condition.construct_the_state(
                self, initial_state=initial_state
            )
        else:
            raise ValueError("Initial State contains too many variables")

        if nres < 3:
            self.resolution += (1,) * (3 - self.dimensionality)
        # print("="*80)
        # print("state constructed")
        # snapshot = tracemalloc.take_snapshot()
        # helpers.display_top(snapshot)
        # zzz = input('')

    @classmethod
    def gen_from_setup(cls, setup: Any) -> Any:
        return cls(**{str(param): getattr(setup, param) for param in dir(setup)})

    def _print_params(self, frame: Any) -> None:
        params = inspect.getargvalues(frame)
        logger.info("=" * 80)
        logger.info("Simulation Parameters")
        logger.info("=" * 80)
        for key, param in params.locals.items():
            if key != "self":
                if isinstance(param, (float, np.float64)):
                    val_str: Any = f"{param:.3f}"
                elif key == "bsources" and not all(item is None for item in param):
                    val_str = f"user-defined boundary sources terms"
                elif key == "gsources" and not all(item is None for item in param):
                    val_str = f"user-defined gravity sources"
                elif key == "hsources" and not all(item is None for item in param):
                    val_str = f"user-defined hydro sources"
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
                                    val_str += [[float(f"{item:.3f}") for item in val]]
                                else:
                                    val_str += [val]
                        else:
                            val_str = f"{param}"
                else:
                    val_str = str(param)

                my_str = str(key).ljust(30, ".")
                logger.info(f"{my_str} {val_str}")
        system_dict = {
            "adiabatic_gamma": self.gamma,
            "resolution": self.resolution,
            "geometry": self.geometry,
            "coord_system": self.coord_system,
            "regime": self.regime,
        }

        for key, val in system_dict.items():
            my_str = str(key).ljust(30, ".")
            if isinstance(val, float):
                val_str = f"{val:.2f}"
            elif isinstance(val, tuple):
                val_str = str(val)
                if isinstance(val[0], tuple):
                    val_str = ""
                    for elem in val:
                        val_str += (
                            "(" + ", ".join("{0:.3f}".format(t) for t in elem) + ")"
                        )
            else:
                val_str = str(val)

            logger.info(f"{my_str} {val_str}")
        logger.info("=" * 80)

    def _generate_the_grid(
        self, x1_cell_spacing: str, x2_cell_spacing: str, x3_cell_spacing: str
    ) -> None:
        dim = self.dimensionality
        vfunc: dict[str, Callable[..., Any]] = {
            "log": np.geomspace,
            "linear": np.linspace,
        }

        csp = [x1_cell_spacing, x2_cell_spacing, x3_cell_spacing]
        verts = [
            self.resolution[0] + 1,
            2 if dim < 2 else self.resolution[1] + 1,
            2 if dim < 3 else self.resolution[2] + 1,
        ]

        cgeom = list(self.geometry)

        # ensure geometry is a tuple of tuples
        # for 1D runs
        if not helpers.tuple_of_tuples(cgeom):
            cgeom = [cgeom]

        # if there is a breakpoint in the geometry
        # ignore it for now
        for idx, val in enumerate(cgeom):
            if isinstance(val, tuple):
                cgeom[idx] = val[:2]
            elif len(val) == 3:
                cgeom[idx] = val[:2]

        for didx, dir in zip(range(self.dimensionality), [self.x1, self.x2, self.x3]):
            if len(dir) == 0:
                dir[:] = vfunc[csp[didx]](*cgeom[didx], verts[didx])
            elif dir.shape[0] != verts[didx]:
                raise ValueError(
                    f"x{didx+1} vertices does not match the x{didx+1}-resolution + 1"
                )

        self.x1 = np.asanyarray(self.x1)
        self.x2 = np.asanyarray(self.x2)
        self.x3 = np.asanyarray(self.x3)

    def _check_boundary_conditions(
        self, boundary_conditions: Union[Sequence[str], str, NDArray[numpy_string]]
    ) -> None:
        boundary_conditions = list(helpers.get_iterable(boundary_conditions))
        for bc in boundary_conditions:
            if bc not in available_boundary_conditions:
                raise ValueError(
                    f"Invalid boundary condition. Expected one of: {available_boundary_conditions}. Instead got: {bc}"
                )

        number_of_given_bcs = len(boundary_conditions)
        if number_of_given_bcs != 2 * self.dimensionality:
            if number_of_given_bcs == 1:
                boundary_conditions = (
                    list(boundary_conditions) * 2 * self.dimensionality
                )
            elif number_of_given_bcs == (self.dimensionality * 2) // 2:
                boundary_conditions = list(
                    chain.from_iterable(repeat(x, 2) for x in boundary_conditions)
                )
            else:
                raise ValueError(
                    "Please include at a number of boundary conditions equal to at least half the number of cell faces"
                )
        self.boundary_conditions = boundary_conditions

    def simulate(
        self,
        tstart: float = 0.0,
        tend: float = 0.1,
        dlogt: float = 0.0,
        plm_theta: float = 1.5,
        x1_cell_spacing: str = "linear",
        x2_cell_spacing: str = "linear",
        x3_cell_spacing: str = "linear",
        cfl: float = 0.4,
        passive_scalars: Optional[Union[NDArray[Any], int]] = None,
        solver: str = "hllc",
        chkpt: Optional[str] = None,
        chkpt_interval: float = 0.1,
        data_directory: str = "data/",
        boundary_conditions: Union[Sequence[str], str] = "outflow",
        engine_duration: float = 10.0,
        compute_mode: str = "cpu",
        quirk_smoothing: bool = True,
        constant_sources: bool = False,
        scale_factor: Optional[Callable[[float], float]] = None,
        scale_factor_derivative: Optional[Callable[[float], float]] = None,
        bsources: list[CallableOrNone] = [None],
        hsources: list[CallableOrNone] = [None],
        gsources: list[CallableOrNone] = [None],
        object_positions: Optional[Union[Sequence[Any], NDArray[Any]]] = None,
        spatial_order: str = "plm",
        time_order: str = "rk2",
    ) -> None:
        """
        Simulate the Hydro Setup

        Parameters:
            tstart (float): The start time of the simulation.
            tend (float): The desired time to end the simulation.
            dlogt (float): The desired logarithmic spacing in checkpoints.
            plm_theta (float): The Piecewise Linear Reconstructed slope parameter.
            x1_cell_spacing (str): Option for a linear or log-spaced mesh on x1.
            x2_cell_spacing (str): Option for a linear or log-spaced mesh on x2.
            x3_cell_spacing (str): Option for a linear or log-spaced mesh on x3.
            cfl (float): The CFL number for minimum adaptive timestep.
            passive_scalars (Optional[Union[NDArray[Any], int]]): The array of passive scalars.
            solver (str): The solver to use for the simulation (e.g., "hllc", "hlld").
            chkpt (Optional[str]): The path to the checkpoint file to read into the simulation.
            chkpt_interval (float): The interval at which to save the checkpoints.
            data_directory (str): The directory at which to save the checkpoint files.
            boundary_conditions (Union[Sequence[str], str]): The outer conditions at the domain boundaries.
            engine_duration (float): The duration the source terms will last in the simulation.
            compute_mode (str): The compute mode for simulation execution ("cpu" or "gpu").
            quirk_smoothing (bool): The switch that controls the Quirk (1960) shock smoothing method.
            constant_sources (bool): Set to true if wanting the source terms to never die.
            scale_factor (Optional[Callable[[float], float]]): The scalar function for moving mesh (e.g., in cosmology).
            scale_factor_derivative (Optional[Callable[[float], float]]): The first derivative of the scalar function for moving mesh.
            bsources (list[CallableOrNone]): List of boundary source terms.
            hsources (list[CallableOrNone]): List of hydro source terms.
            gsources (list[CallableOrNone]): List of gravity source terms.
            object_positions (Optional[Union[Sequence[Any], NDArray[Any]]]): An optional boolean array that masks the immersed boundary.
            spatial_order (str): Space order of integration ("pcm" or "plm").
            time_order (str): Time order of integration ("rk1" or "rk2").

        Returns:
            None
        """
        bsources = helpers.as_list(bsources)
        hsources = helpers.as_list(hsources)
        gsources = helpers.as_list(gsources)
        
        if spatial_order not in ["pcm", "plm"]:
            raise ValueError(f"Space order can only be one of {['pcm', 'plm']}")
        if time_order not in ["rk1", "rk2"]:
            raise ValueError(f"Time order can only be one of {['rk1', 'rk2']}")

        self._print_params(inspect.currentframe())
        if x1_cell_spacing not in available_cellspacings:
            raise ValueError(
                f"cell spacing for x1 should be one of: {available_cellspacings}"
            )

        if x2_cell_spacing not in available_cellspacings:
            raise ValueError(
                f"cell spacing for x2 should be one of: {available_cellspacings}"
            )

        if x3_cell_spacing not in available_cellspacings:
            raise ValueError(
                f"cell spacing for x3 should be one of: {available_cellspacings}"
            )

        self.start_time: float = 0.0
        self.chkpt_idx: int = 0
        scale_factor = scale_factor or (lambda t: 1.0)
        scale_factor_derivative = scale_factor_derivative or (lambda t: 0.0)
        self._generate_the_grid(x1_cell_spacing, x2_cell_spacing, x3_cell_spacing)
        mesh_motion = scale_factor_derivative(tstart) / scale_factor(tstart) != 0
        volume_factor: Union[float, NDArray[Any]] = 1.0
        if mesh_motion and self.coord_system != "cartesian":
            if self.dimensionality == 1:
                volume_factor = helpers.calc_cell_volume1D(
                    x1=self.x1, coord_system=self.coord_system
                )
                volume_factor = helpers.calc_cell_volume1D(
                    x1=self.x1, coord_system=self.coord_system
                )
            elif self.dimensionality == 2:
                volume_factor = helpers.calc_cell_volume2D(
                    x1=self.x1, x2=self.x2, coord_system=self.coord_system
                )

        self._check_boundary_conditions(boundary_conditions)
        if not chkpt:
            simbi_ic.initializeModel(
                self, spatial_order == "pcm", volume_factor, passive_scalars
            )
        else:
            simbi_ic.load_checkpoint(self, chkpt, self.dimensionality, mesh_motion)
        if self.dimensionality == 1 and self.coord_system in [
            "planar_cylindrical",
            "axis_cylindrical",
        ]:
            self.coord_system = "cylindrical"

        self.start_time = self.start_time or tstart

        if solver == "hlld" and not self.mhd:
            logger.info(
                "HLLD solver not available for non-MHD runs. Switching to HLLC solver"
            )
            solver = "hllc"

        # Convert strings to byte arrays
        cython_data_directory = os.path.join(data_directory, "").encode("utf-8")
        cython_coordinates = self.coord_system.encode("utf-8")
        cython_solver = solver.encode("utf-8")
        cython_boundary_conditions: NDArray[numpy_string] = np.array(
            [bc.encode("utf-8") for bc in self.boundary_conditions]
        )

        # Offset the start time from zero if wanting log
        # checkpoints, but with initial time of zero
        if dlogt != 0 and self.start_time == 0:
            self.start_time = 1e-16

        # Check whether the specified path exists or not
        data_path: Path = Path(data_directory)
        if not Path.is_dir(data_path):
            # Create a new directory because it does not exist
            Path.mkdir(data_path, parents=True)
            logger.info(
                f"The data directory provided does not exist. Creating the {data_path} directory now!"
            )

        if compute_mode in ["cpu", "omp"]:
            if "USE_OMP" in os.environ:
                logger.debug("Using OpenMP multithreading")
            else:
                logger.debug("Using STL std::thread multithreading")
        else:
            dim3 = [1, 1, 1]
            for idx, coord in enumerate(["X", "Y", "Z"]):
                if user_set := f"GPU{coord}BLOCK_SIZE" in os.environ:
                    if idx + 1 <= self.dimensionality:
                        dim3[idx] = int(os.environ[f"GPU{coord}BLOCK_SIZE"])
                else:
                    if self.dimensionality == 1 and coord == "X":
                        dim3[idx] = 128
                    elif self.dimensionality == 2 and coord in ["X", "Y"]:
                        dim3[idx] = 16
                    elif self.dimensionality == 3 and coord in ["X", "Y", "Z"]:
                        dim3[idx] = 4
            logger.debug(f"In GPU mode, GPU block dims are: {tuple(dim3)}")

        logger.info("")
        # Loading bar to have chance to check params
        if not self.trace_memory:
            helpers.print_progress()

        # Create boolean masks for object immersed boundaries (impermeable)
        object_cells: NDArray[Any] = (
            np.zeros_like(self.u[0], dtype=bool)
            if object_positions is None
            else np.asanyarray(object_positions, dtype=bool)
        )

        print("=" * 80)
        logger.info(
            f"Computing solution using {spatial_order.upper()} in space, {time_order.upper()} in time..."
        )

        if compute_mode == "gpu":
            if self.dimensionality == 1:
                if "GPUXBLOCK_SIZE" not in os.environ:
                    os.environ["GPUXBLOCK_SIZE"] = "128"
            elif self.dimensionality == 2:
                if "GPUXBLOCK_SIZE" not in os.environ:
                    os.environ["GPUXBLOCK_SIZE"] = "16"

                if "GPUYBLOCK_SIZE" not in os.environ:
                    os.environ["GPUYBLOCK_SIZE"] = "16"
            else:
                if "GPUXBLOCK_SIZE" not in os.environ:
                    os.environ["GPUXBLOCK_SIZE"] = "4"

                if "GPUYBLOCK_SIZE" not in os.environ:
                    os.environ["GPUYBLOCK_SIZE"] = "4"

                if "GPUZBLOCK_SIZE" not in os.environ:
                    os.environ["GPUZBLOCK_SIZE"] = "4"

        extra_edges: int = 2 if spatial_order == "pcm" else 4
        nxp = self.resolution[0]
        nyp = self.resolution[1]
        nzp = self.resolution[2]
        self.nx = nxp + extra_edges
        self.ny = nyp + extra_edges * (self.dimensionality > 1)
        self.nz = nzp + extra_edges * (self.dimensionality > 2)

        init_conditions = {
            "gamma": self.gamma,
            "tstart": self.start_time,
            "tend": tend,
            "cfl": cfl,
            "dlogt": dlogt,
            "plm_theta": plm_theta,
            "engine_duration": engine_duration,
            "chkpt_interval": chkpt_interval,
            "chkpt_idx": self.chkpt_idx,
            "data_directory": cython_data_directory,
            "boundary_conditions": cython_boundary_conditions,
            "spatial_order": spatial_order.encode("utf-8"),
            "time_order": time_order.encode("utf-8"),
            "x1_cell_spacing": x1_cell_spacing.encode("utf-8"),
            "x2_cell_spacing": x2_cell_spacing.encode("utf-8"),
            "x3_cell_spacing": x3_cell_spacing.encode("utf-8"),
            "solver": cython_solver,
            "constant_sources": constant_sources,
            "coord_system": cython_coordinates,
            "quirk_smoothing": quirk_smoothing,
            "x1": self.x1,
            "x2": self.x2,
            "x3": self.x3,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "object_cells": object_cells.flat,
            "nxv": nxp + 1,
            "nyv": nyp + 1,
            "nzv": nzp + 1,
            "bfield": [[0], [0], [0]],
        }

        if self.mhd:
            if self.discontinuity:
                b1 = np.zeros(shape=(nzp, nyp, init_conditions["nxv"]))
                b2 = np.zeros(shape=(nzp, init_conditions["nyv"], nxp))
                b3 = np.zeros(shape=(init_conditions["nzv"], nyp, nxp))

                region_one = self.x1 < self.geometry[0][2]
                region_two = np.logical_not(region_one)
                xc = helpers.calc_centroid(self.x1, coord_system=self.coord_system)
                a = xc < self.geometry[0][2]
                b = np.logical_not(a)
                b1[..., region_one] = self.initial_state[0][5]
                b1[..., region_two] = self.initial_state[1][5]
                b2[..., a] = self.initial_state[0][6]
                b2[..., b] = self.initial_state[1][6]
                b3[..., a] = self.initial_state[0][7]
                b3[..., b] = self.initial_state[1][7]
            else:
                b1 = self.initial_state[5]
                b2 = self.initial_state[6]
                b3 = self.initial_state[7]
            init_conditions["bfield"] = [b1.flat, b2.flat, b3.flat]


        lambdas: dict[str, list[Optional[Callable[..., float]]]] = {
            "boundary_sources": bsources,
            "hydro_sources": hsources,
            "gravity_sources": gsources,
        }
        for name, source in lambdas.items():
            ngiven = len(source)
            if not all(item is None for item in source):
                nterms: int = self.dimensionality + 3 * ((source != gsources) + self.mhd)
                if ngiven < nterms:
                    if ngiven == nterms - 1:
                        if self.dimensionality == 1:
                            # define null source terms for scalar concentration
                            source += [lambda x,t: 0.0]
                        elif self.dimensionality == 2:
                            source += [lambda x,y,t: 0.0]
                        else:
                            source += [lambda x,y,z,t: 0.0]
                    else:
                        raise ValueError(
                            f"Number of {name} terms should be equal to the number {nterms}"
                        )

        lib_mode = "cpu" if compute_mode in ["cpu", "omp"] else "gpu"
        sim_state = getattr(
            importlib.import_module(f".{lib_mode}_ext", package="simbi.libs"),
            "SimState",
        )

        if self.trace_memory:
            print(r"-*" * 40)
            snapshot = tracemalloc.take_snapshot()
            helpers.display_top(snapshot)
            tracemalloc.stop()
            print(r"-*" * 40)
            helpers.print_progress()

        state_contig = self.u.reshape(self.u.shape[0], -1)

        sim_state().run(
            state=state_contig,
            dim=self.dimensionality,
            regime=self.regime.encode("utf-8"),
            sim_info=init_conditions,
            a=scale_factor,
            adot=scale_factor_derivative,
            **lambdas,
        )
