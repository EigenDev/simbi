import numpy as np
import numpy.typing as npt
from .calculations import (
    calc_labframe_density,
    calc_labframe_momentum,
    calc_lorentz_factor,
    calc_spec_enthalpy,
    calc_labframe_energy,
)
from ..detail.mem import release_memory
from ..functional import helpers
from ..io.logging import logger
from ..tools.utility import read_file
from itertools import product, permutations
from typing import Any, Sequence, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ModelState:
    nvars: int
    u: NDArray[np.floating[Any]]
    bfield: Optional[List[NDArray[np.floating[Any]]]] = None


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
            dim_slices = [
                slice(start, end) for start, end in zip(indices[:-1], indices[1:])
            ]
            slices_per_dim.append(dim_slices)

        # Generate all combinations
        # we invert the order of slices_per_dim to match the order of dimensions
        return list(product(*slices_per_dim[::-1]))


@dataclass
class Partition:
    indices: Tuple[slice, ...]
    initial_primitive_state: NDArray[np.floating[Any]]


def check_valid_state(x: Any, name: str) -> None:
    if np.isnan(np.sum(x)):
        raise ValueError(f"Initial state: {name} contains NaNs")


def flatten_fully(x: Any) -> Any:
    while any(dim == 1 for dim in x.shape):
        x = np.vstack(x)
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flat
    return np.asanyarray(x)


def load_checkpoint(model: Any, filename: str) -> None:
    print(f"Loading from checkpoint: {filename}...", flush=True)
    setup: dict[str, Any] = {}
    volume_factor: Union[float, NDArray[np.floating[Any]]] = 1.0
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
        setup["adiabatic_index"],
        fields["rho"],
        vel,
        fields["p"],
        bfields,
        setup["regime"],
    )
    energy = calc_labframe_energy(
        setup["adiabatic_index"],
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
    partition: NDArray[np.floating[Any]],
    state: NDArray[np.floating[Any]] | list[float],
    regime: str,
    adiabatic_index: float,
    ndims: int,
) -> None:
    """Initialize state variables for a partition"""
    n_non_em = len(state) - 3 if "mhd" in regime else len(state)
    rho, *velocity, pressure = state[:n_non_em]
    mean_bfields = list(state[n_non_em:])

    # Calculate lab frame quantities
    dens = calc_labframe_density(rho, velocity, regime)
    mom = calc_labframe_momentum(
        adiabatic_index, rho, velocity, pressure, mean_bfields, regime
    )
    energy = calc_labframe_energy(
        adiabatic_index, rho, pressure, velocity, mean_bfields, regime
    )

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
    u: NDArray[np.floating[Any]],
    partitions: List[Partition],
    initial_primitive_states: Sequence[Sequence[float]],
) -> Tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
]:
    """Initialize staggered MHD fields"""
    b1 = np.zeros_like(u[0])
    b2 = np.zeros_like(u[0])
    b3 = np.zeros_like(u[0])

    for part, state in zip(partitions, initial_primitive_states):
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
        Partition(indices=slices, initial_primitive_state=model.u[(..., *slices)])
        for slices in partition_slices
    ]

    # Initialize each partition
    for part, state in zip(partitions, model.initial_primitive_state):
        initialize_partition_state(
            part.initial_primitive_state,
            state,
            model.regime,
            model.adiabatic_index,
            model.dimensionality,
        )

    # Handle MHD fields if needed
    if model.mhd:
        model.bfield = initialize_mhd_fields(
            model.u, partitions, model.initial_primitive_state
        )


@release_memory
def construct_the_state(
    model: Any,
    initial_primitive_state: Sequence[NDArray[np.floating[Any]]] | list[list[float]],
) -> None:
    """Initialize model state for continuous or discontinuous problems"""

    # Initialize model state
    model.nvars = 3 + model.dimensionality if not model.mhd else 9
    model.u = np.zeros((model.nvars - 1, *np.asanyarray(model.resolution).flat[::-1]))

    if model.discontinuity:
        initialize_discontinuous_problem(model)
    else:
        initial_primitive_state = cast(
            Sequence[NDArray[np.floating[Any]]], initial_primitive_state
        )
        initialize_continuous_state(model, initial_primitive_state)


def pad_mhd_fields(
    bfields: Sequence[NDArray[np.floating[Any]]],
) -> List[NDArray[np.floating[Any]]]:
    # pad the mhd fields along perpendicular directions
    b1, b2, b3 = bfields
    b1 = np.pad(b1, ((1, 1), (1, 1), (0, 0)), "edge")
    b2 = np.pad(b2, ((1, 1), (0, 0), (1, 1)), "edge")
    b3 = np.pad(b3, ((0, 0), (1, 1), (1, 1)), "edge")
    return [b1, b2, b3]


def pad_staggered_fields(
    bfields: Sequence[NDArray[np.floating[Any]]],
) -> List[NDArray[np.floating[Any]]]:
    # pad the staggered fields along parallel directions
    b1, b2, b3 = bfields
    b1 = np.pad(b1, ((1, 1), (1, 1), (0, 0)), "edge")
    b2 = np.pad(b2, ((1, 1), (0, 0), (1, 1)), "edge")
    b3 = np.pad(b3, ((0, 0), (1, 1), (1, 1)), "edge")
    return [b1, b2, b3]


def calculate_mean_bfields(
    bfields: Sequence[NDArray[np.floating[Any]]],
) -> Sequence[NDArray[Any]]:
    # calculate mean B-fields from staggered fields
    b1, b2, b3 = bfields
    return [
        0.5 * (b1[..., :-1] + b1[..., 1:]),
        0.5 * (b2[:, :-1, :] + b2[:, 1:, :]),
        0.5 * (b3[:-1, ...] + b3[1:, ...]),
    ]


def update_mhd_fields(
    bfields: Sequence[NDArray[np.floating[Any]]],
    partition_inds: Tuple[slice, ...],
    mean_bfields: Sequence[NDArray[np.floating[Any]]] | list[float],
) -> None:
    b1, b2, b3 = bfields
    b1[partition_inds] = mean_bfields[0]
    b2[partition_inds] = mean_bfields[1]
    b3[partition_inds] = mean_bfields[2]


def calculate_state_vector(
    adiabatic_index: float,
    rho: NDArray[np.floating[Any]],
    velocity: Sequence[NDArray[np.floating[Any]]],
    pressure: NDArray[np.floating[Any]],
    mean_bfields: List[NDArray[np.floating[Any]]],
    regime: str,
) -> NDArray[np.floating[Any]]:
    dens = calc_labframe_density(rho, velocity, regime)
    mom = calc_labframe_momentum(
        adiabatic_index, rho, velocity, pressure, mean_bfields, regime
    )
    energy = calc_labframe_energy(
        adiabatic_index, rho, pressure, velocity, mean_bfields, regime
    )
    return np.array([dens, *mom, energy, *mean_bfields])


def initialize_continuous_state(
    model: Any, initial_primitive_state: Sequence[NDArray[np.floating[Any]]]
) -> None:
    """Handle initialization of continuous problems"""
    rho, *velocity, pressure = initial_primitive_state[: model.number_of_non_em_terms]

    if model.mhd:
        bfields_stag = initial_primitive_state[model.number_of_non_em_terms :]
        mean_bfields = calculate_mean_bfields(bfields_stag)
        model.bfield = pad_staggered_fields(bfields_stag)
    else:
        mean_bfields = []

    state_vector = calculate_state_vector(
        model.adiabatic_index, rho, velocity, pressure, mean_bfields, model.regime
    )
    model.u[...] = state_vector
