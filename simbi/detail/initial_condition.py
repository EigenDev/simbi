import numpy as np
import numpy.typing as npt
from ..key_types import *
from .mem import release_memory
from . import helpers
from .slogger import logger
from ..tools.utility import read_file
from itertools import product, permutations
from typing import Any, Sequence, Tuple, List, Optional
from dataclasses import dataclass

# alias nested array types
nested_array = NDArray[numpy_float] | list[NDArray[numpy_float]] | list[float]

@dataclass
class ModelState:
    nvars: int
    u: NDArray[numpy_float]
    bfield: Optional[List[NDArray[numpy_float]]] = None


@dataclass
class GeometryConfig:
    geometry: Sequence[Tuple[float, ...]]
    resolution: Sequence[int]
    dimensionality: int

    def _get_dimension_breaks(self, dim: int) -> List[float]:
        """Extract all breakpoints for a given dimension"""
        geom = self.geometry[dim]
        return list(geom[2:]) if len(geom) > 2 else []

    def _breaks_to_indices(self, breaks: List[float], dim: int) -> List[int]:
        """Convert physical breakpoints to grid indices"""
        if not breaks:
            return [0, self.resolution[dim]]
            
        spacing = (self.geometry[dim][1] - self.geometry[dim][0]) / self.resolution[dim]
        indices = [0]  # Start point
        indices.extend(round(abs(x - self.geometry[dim][0]) / spacing) for x in breaks)
        indices.append(self.resolution[dim])  # End point
        return sorted(indices)

    def calculate_partitions(self) -> List[Tuple[slice, ...]]:
        """Generate all partition slices for N dimensions"""
        # Get breaks for each dimension
        slices_per_dim = []
        for dim in range(self.dimensionality):
            breaks = self._get_dimension_breaks(dim)
            indices = self._breaks_to_indices(breaks, dim)
            # Create slices between consecutive indices
            dim_slices = [slice(start, end) for start, end in zip(indices[:-1], indices[1:])]
            slices_per_dim.append(dim_slices)
        

        # Generate all combinations
        # we invert the order of slices_per_dim to match the order of dimensions
        return list(product(*slices_per_dim[::-1]))

@dataclass
class Partition:
    indices: Tuple[slice, ...]
    initial_state: NDArray[numpy_float]
    
    
def check_valid_state(x: Any, name: str) -> None:
    if np.isnan(np.sum(x)):
        raise ValueError(f"Initial state: {name} contains NaNs")

def elemental_multiply(
    a: nested_array,
    b: nested_array
) -> Any:
    if isinstance(a, list):
        return np.array([a[i] * b[i] for i in range(len(a))])
    return a * b

def dot_product(
    a: nested_array,
    b: nested_array,
) -> Any:
    if isinstance(a, list):
        return np.sum([a[i] * b[i] for i in range(len(a))], axis=0)
    return np.sum(a * b, axis=0)

def calc_lorentz_factor(velocity: nested_array, regime: str) -> FloatOrArray:
    vsquared = dot_product(velocity, velocity)
    if regime != "classical" and np.any(vsquared >= 1.0):
        raise ValueError("Lorentz factor is not real. Velocity exceeds speed of light.")
    return 1.0 if regime == "classical" else (1.0 - vsquared) ** (-0.5)


def calc_spec_enthalpy(
    gamma: float,
    rho: FloatOrArray,
    pressure: FloatOrArray,
    regime: str,
) -> FloatOrArray:
    return (
        1.0 if regime == "classical" else 1.0 + gamma * pressure / (rho * (gamma - 1.0))
    )


def calc_labframe_density(
    rho: FloatOrArray, velocity: nested_array, regime: str
) -> FloatOrArray:
    return rho * calc_lorentz_factor(velocity, regime)


def calc_labframe_momentum(
    gamma: float,
    rho: FloatOrArray,
    velocity: nested_array,
    pressure: FloatOrArray,
    bfields: nested_array,
    regime: str,
) -> NDArray[numpy_float]:
    vdb = dot_product(velocity, bfields) if np.any(bfields) else 0.0
    bsq = dot_product(bfields, bfields) if np.any(bfields) else 0.0
    vdb_bvec = (
        np.array([bn * vdb for bn in bfields])
        if np.any(bfields)
        else [0.0] * len(velocity)
    )

    enthalpy = calc_spec_enthalpy(gamma, rho, pressure, regime)
    lorentz = calc_lorentz_factor(velocity, regime)
    return np.array(
        [
            (rho * lorentz**2 * enthalpy + bsq) * velocity[i] - vdb_bvec[i]
            for i in range(len(velocity))
        ]
    )


def calc_labframe_energy(
    gamma: float,
    rho: FloatOrArray,
    pressure: FloatOrArray,
    velocity: nested_array,
    bfields: nested_array,
    regime: str,
) -> FloatOrArray:
    res: FloatOrArray
    bsq = dot_product(bfields, bfields) if np.any(bfields) else 0.0
    vdb = dot_product(velocity, bfields) if np.any(bfields) else 0.0
    vsq = dot_product(velocity, velocity)
    lorentz = calc_lorentz_factor(velocity, regime)
    enthalpy = calc_spec_enthalpy(gamma, rho, pressure, regime)

    if regime == "classical":
        res = pressure / (gamma - 1.0) + 0.5 * rho * vsq + 0.5 * bsq
    else:
        res = (
            rho * lorentz**2 * enthalpy
            - pressure
            - rho * lorentz
            + 0.5 * bsq
            + 0.5 * (bsq * vsq - vdb**2)
        )

    return res


def flatten_fully(x: Any) -> Any:
    while any(dim == 1 for dim in x.shape):
        x = np.vstack(x)
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flat
    return np.asanyarray(x)


def load_checkpoint(model: Any, filename: str) -> None:
    print(f"Loading from checkpoint: {filename}...", flush=True)
    setup: dict[str, Any] = {}
    volume_factor: Union[float, NDArray[numpy_float]] = 1.0
    fields, setup, mesh = read_file(filename, return_staggered_field=model.mhd)
    dim: int = setup["dimensions"]

    vel = np.array([fields[f"v{i}"] for i in range(1, dim + 1)])
    bfields = (
        np.array([fields[f"b{i}"] for i in range(1, dim + 1)])
        if setup["regime"] == "srmhd"
        else np.array([])
    )

    dens = calc_labframe_density(fields["rho"], vel, setup["regime"])
    mom = calc_labframe_momentum(
        setup["adiabatic_gamma"],
        fields["rho"],
        vel,
        fields["p"],
        bfields,
        setup["regime"],
    )
    energy = calc_labframe_energy(
        setup["adiabatic_gamma"],
        fields["rho"],
        fields["p"],
        vel,
        bfields,
        setup["regime"],
    )

    check_valid_state(dens, "density")
    check_valid_state(mom, "momentum")
    check_valid_state(energy, "energy")

    model.start_time = setup["time"]
    model.x1 = mesh["x1v"]
    if dim > 1:
        model.x2 = mesh["x2v"]
    if dim > 2:
        model.x3 = mesh["x3v"]

    if setup["mesh_motion"]:
        if dim == 1 and setup["coord_system"] != "cartesian":
            volume_factor = helpers.calc_cell_volume1D(
                x1=mesh["x1"], coord_system=setup["coord_system"]
            )
        elif dim == 2:
            volume_factor = helpers.calc_cell_volume2D(
                x1=mesh["x1"], x2=mesh["x2"], coord_system=setup["coord_system"]
            )
        elif dim == 3:
            volume_factor = helpers.calc_cell_volume3D(
                x1=mesh["x1"],
                x2=mesh["x2"],
                x3=mesh["x3"],
                coord_system=setup["coord_system"],
            )

    model.u = np.array([dens, *mom, energy, *bfields, dens * fields["chi"]])

    padwith = (setup["spatial_order"] != "pcm") + 1
    npad = ((0, 0),) + ((padwith, padwith),) * dim
    model.u = np.pad(model.u * volume_factor, npad, "edge")
    model.checkpoint_idx = setup["checkpoint_idx"]

    if model.mhd:
        model.bfield = [fields["b1stag"], fields["b2stag"], fields["b3stag"]]


@release_memory
def initializeModel(
    model: Any,
    spatial_order: str,
    volume_factor: Union[float, NDArray[Any]],
    passive_scalars: Union[npt.NDArray[Any], Any],
) -> None:
    model.u = np.insert(model.u, model.u.shape[0], 0.0, axis=0)
    if passive_scalars is not None:
        model.u[-1, ...] = passive_scalars * model.u[0]

    # npad is a tuple of (n_before, n_after) for each dimension
    print("Initializing Model...", flush=True)
    padwith = (spatial_order != "pcm") + 1
    npad = ((0, 0),) + ((padwith, padwith),) * model.dimensionality
    model.u = np.pad(model.u * volume_factor, npad, "edge")


def calculate_break_points(
    geometry: Sequence[Tuple[float, ...]], resolution: Sequence[int], ndims: int
) -> List[Tuple[None, int]]:
    """Calculate partition indices from break points"""
    break_points = [val[2] for val in geometry if len(val) == 3]
    if len(break_points) > ndims:
        raise ValueError("Too many break points for dimension")

    spacings = [
        (geometry[idx][1] - geometry[idx][0]) / resolution[idx]
        for idx in range(len(geometry))
    ]

    return [
        (None, round(abs(break_points[idx] - geometry[idx][0]) / spacings[idx]))
        for idx in range(len(break_points))
    ]


def initialize_partition_state(
    partition: NDArray[numpy_float],
    state: NDArray[numpy_float] | list[float],
    regime: str,
    gamma: float,
    ndims: int,
) -> None:
    """Initialize state variables for a partition"""
    n_non_em = len(state) - 3 if "mhd" in regime else len(state)
    rho, *velocity, pressure = state[:n_non_em]
    mean_bfields = list(state[n_non_em:])

    # Calculate lab frame quantities
    dens = calc_labframe_density(rho, velocity, regime)
    mom = calc_labframe_momentum(gamma, rho, velocity, pressure, mean_bfields, regime)
    energy = calc_labframe_energy(gamma, rho, pressure, velocity, mean_bfields, regime)

    # Validate
    for name, val in [("density", dens), ("momentum", mom), ("energy", energy)]:
        check_valid_state(val, name)

    # Set values
    state_vector = np.array([dens, *mom, energy, *mean_bfields])
    if ndims == 1:
        partition[...] = state_vector[:, None]
    else:
        partition[...] = (partition[...].transpose() + state_vector).transpose()


def initialize_mhd_fields(
    u: NDArray[numpy_float],
    partitions: List[Partition],
    initial_states: Sequence[Sequence[float]],
) -> Tuple[NDArray[numpy_float], NDArray[numpy_float], NDArray[numpy_float]]:
    """Initialize staggered MHD fields"""
    b1 = np.zeros_like(u[0])
    b2 = np.zeros_like(u[0])
    b3 = np.zeros_like(u[0])

    for part, state in zip(partitions, initial_states):
        mean_bfields = state[-3:]  # Last 3 components are B-fields
        b1[part.indices] = mean_bfields[0]
        b2[part.indices] = mean_bfields[1]
        b3[part.indices] = mean_bfields[2]
    
    # Pad the staggered fields along perpendicular and parallel directions
    b1 = np.pad(b1, ((1, 1), (1, 1), (0, 1)), "edge")
    b2 = np.pad(b2, ((1, 1), (0, 1), (1, 1)), "edge")
    b3 = np.pad(b3, ((0, 1), (1, 1), (1, 1)), "edge")
    
    return b1, b2, b3


def initialize_discontinuous_problem(model: Any) -> None:
    """Main initialization function"""
    if not model.discontinuity:
        return

    logger.info(f"Initializing {model.dimensionality}D Discontinuity...")

    # Calculate partitions
    geom_config = GeometryConfig(model.geometry, model.resolution, model.dimensionality)
    partition_slices = geom_config.calculate_partitions()

    # Create partition objects
    partitions = [
        Partition(indices=slices, initial_state=model.u[(..., *slices)])
        for slices in partition_slices
    ]

    # Initialize each partition
    for part, state in zip(partitions, model.initial_state):
        initialize_partition_state(
            part.initial_state, state, model.regime, model.gamma, model.dimensionality
        )

    # Handle MHD fields if needed
    if model.mhd:
        model.bfield = initialize_mhd_fields(model.u, partitions, model.initial_state)

@release_memory
def construct_the_state(
    model: Any, initial_state: list[NDArray[numpy_float]] | list[list[float]]
) -> None:
    """Initialize model state for continuous or discontinuous problems"""

    # Initialize model state
    model.nvars = 3 + model.dimensionality if not model.mhd else 9
    model.u = np.zeros((model.nvars - 1, *np.asanyarray(model.resolution).flat[::-1]))

    if model.discontinuity:
        # initial_state = cast(list[list[float]], initial_state)
        # _handle_discontinuous_state(model, initial_state)
        initialize_discontinuous_problem(model)
    else:
        initial_state = cast(list[NDArray[numpy_float]], initial_state)
        initialize_continuous_state(model, initial_state)


def pad_mhd_fields(bfields: list[NDArray[numpy_float]]) -> List[NDArray[numpy_float]]:
    # pad the mhd fields along perpendicular directions
    b1, b2, b3 = bfields
    b1 = np.pad(b1, ((1, 1), (1, 1), (0, 0)), "edge")
    b2 = np.pad(b2, ((1, 1), (0, 0), (1, 1)), "edge")
    b3 = np.pad(b3, ((0, 0), (1, 1), (1, 1)), "edge")
    return [b1, b2, b3]


def pad_staggered_fields(
    bfields: list[NDArray[numpy_float]],
) -> List[NDArray[numpy_float]]:
    # pad the staggered fields along parallel directions
    b1, b2, b3 = bfields
    b1 = np.pad(b1, ((1, 1), (1, 1), (0, 0)), "edge")
    b2 = np.pad(b2, ((1, 1), (0, 0), (1, 1)), "edge")
    b3 = np.pad(b3, ((0, 0), (1, 1), (1, 1)), "edge")
    return [b1, b2, b3]


def calculate_mean_bfields(bfields: list[NDArray[numpy_float]]) -> list[NDArray[Any]]:
    # calculate mean B-fields from staggered fields
    b1, b2, b3 = bfields
    return [
        0.5 * (b1[..., :-1] + b1[..., 1:]),
        0.5 * (b2[:, :-1, :] + b2[:, 1:, :]),
        0.5 * (b3[:-1, ...] + b3[1:, ...]),
    ]


def update_mhd_fields(
    bfields: list[NDArray[numpy_float]],
    partition_inds: Tuple[slice, ...],
    mean_bfields: list[NDArray[numpy_float]] | list[float],
) -> None:
    b1, b2, b3 = bfields
    b1[partition_inds] = mean_bfields[0]
    b2[partition_inds] = mean_bfields[1]
    b3[partition_inds] = mean_bfields[2]


def calculate_state_vector(
    gamma: float,
    rho: FloatOrArray,
    velocity: list[NDArray[numpy_float]],
    pressure: FloatOrArray,
    mean_bfields: List[NDArray[numpy_float]],
    regime: str,
) -> NDArray[numpy_float]:
    dens = calc_labframe_density(rho, velocity, regime)
    mom = calc_labframe_momentum(gamma, rho, velocity, pressure, mean_bfields, regime)
    energy = calc_labframe_energy(gamma, rho, pressure, velocity, mean_bfields, regime)
    return np.array([dens, *mom, energy, *mean_bfields])


def initialize_continuous_state(
    model: Any, initial_state: list[NDArray[numpy_float]]
) -> None:
    """Handle initialization of continuous problems"""
    rho, *velocity, pressure = initial_state[: model.number_of_non_em_terms]

    if model.mhd:
        bfields_stag = initial_state[model.number_of_non_em_terms :]
        mean_bfields = calculate_mean_bfields(bfields_stag)
        model.bfield = pad_staggered_fields(bfields_stag)
    else:
        mean_bfields = []

    state_vector = calculate_state_vector(
        model.gamma, rho, velocity, pressure, mean_bfields, model.regime
    )
    model.u[...] = state_vector
