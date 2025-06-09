"""
Streamlined simulation state initialization.

This module provides efficient functions for creating and initializing simulation states.
"""

from dataclasses import dataclass
from typing import Optional, cast
import numpy as np
from numpy.typing import NDArray

from ...functional import Maybe
from ..config.base_config import SimbiBaseConfig
from ..types.typing import InitialStateType, GasStateFunction, MHDStateGenerators
import itertools


@dataclass
class SimulationState:
    """Container for simulation state data."""

    primitive_state: NDArray[np.floating]  # (nvars, nx, ny, nz)
    conserved_state: NDArray[np.floating]  # (nvars, nx, ny, nz)
    config: SimbiBaseConfig
    staggered_bfields: Optional[list[NDArray[np.floating]]] = None


def is_mhd_generator(gen: InitialStateType) -> bool:
    """Check if generator is for MHD simulation."""
    return not callable(gen) and len(gen) == 4


def primitive_to_conserved(
    primitive: NDArray[np.floating],
    config: SimbiBaseConfig,
    staggered_bfields: Optional[list[NDArray[np.floating]]] = None,
) -> NDArray[np.floating]:
    """
    Convert primitive variables to conserved variables.

    This is a streamlined conversion function that handles both hydro and MHD.
    """
    # Extract key variables from config
    adiabatic_index = config.adiabatic_index
    regime = config.regime.value
    is_mhd = config.is_mhd
    ndim = config.dimensionality

    # Create output array same shape as input
    conserved = np.zeros_like(primitive)

    # Handle special case for isothermal EOS
    isothermal = abs(adiabatic_index - 1.0) < 1e-10

    # Extract primitive variables
    # Format is: [density, vel_x, vel_y, vel_z, pressure, ...]
    pidx = (
        4 if is_mhd else config.dimensionality + 1
    )  # Pressure index in primitive array
    rho, *velocities, pressure = primitive[0], *primitive[1:pidx], primitive[pidx]

    # Calculate square of velocity
    vsq = velocities[0] ** 2
    if config.dimensionality > 1:
        vsq += velocities[1] ** 2
    if config.dimensionality > 2 or is_mhd:
        vsq += velocities[2] ** 2

    lorentz: float | NDArray[np.floating] = 1.0
    # Calculate Lorentz factor for relativistic simulations
    if "sr" in regime:
        if np.any(vsq >= 1.0):
            raise ValueError("Velocity exceeds speed of light")
        lorentz = 1.0 / np.sqrt(1.0 - vsq)

    # Conserved density (D)
    conserved[0] = rho * lorentz

    # Conserved momentum (m = ρh\gamma^2v for relativistic, \rho v for classical)
    if "sr" in regime:
        # Relativistic enthalpy
        h = 1.0 + adiabatic_index * pressure / ((adiabatic_index - 1.0) * rho)
        for i in range(ndim):
            conserved[i + 1] = rho * h * lorentz**2 * velocities[i]
    else:
        # Classical momentum
        for i in range(ndim):
            conserved[i + 1] = rho * velocities[i]

    # Energy equation
    energy_index = config.dimensionality + 1
    if isothermal:
        # For isothermal flows, we store sound speed squared in energy slot
        conserved[energy_index] = pressure / rho
    elif "sr" in regime:
        # Relativistic energy
        h = 1.0 + adiabatic_index * pressure / ((adiabatic_index - 1.0) * rho)
        conserved[energy_index] = rho * h * lorentz**2 - pressure - rho * lorentz
    else:
        # Classical energy
        conserved[energy_index] = pressure / (adiabatic_index - 1.0) + 0.5 * rho * vsq

    # Handle magnetic fields for MHD
    if is_mhd and staggered_bfields:
        # Calculate cell-centered magnetic fields from staggered fields
        b_mean = get_cell_centered_bfields(staggered_bfields)

        # Store magnetic fields in conserved array
        conserved[5] = b_mean[0]
        conserved[6] = b_mean[1]
        conserved[7] = b_mean[2]

        # Adjust energy and momentum for magnetic contribution
        b_squared = b_mean[0] ** 2 + b_mean[1] ** 2 + b_mean[2] ** 2
        if "sr" in regime:
            # SRMHD energy
            v_dot_b = (
                velocities[0] * b_mean[0]
                + velocities[1] * b_mean[1]
                + velocities[2] * b_mean[2]
            )
            conserved[energy_index] += 0.5 * (b_squared + vsq * b_squared - v_dot_b**2)
            for i in range(3):
                conserved[i + 1] += b_squared * velocities[i] - v_dot_b * b_mean[i]
        else:
            # MHD energy
            conserved[energy_index] += 0.5 * b_squared

    # Passive scalar
    conserved[-1] = conserved[0] * primitive[-1]

    return conserved


def get_cell_centered_bfields(
    staggered_bfields: list[NDArray[np.floating]],
) -> list[NDArray[np.floating]]:
    """
    Convert staggered magnetic fields to cell-centered fields.

    Args:
        staggered_bfields: List of staggered B-field components

    Returns:
        List of cell-centered B-field components
    """
    if not staggered_bfields or len(staggered_bfields) != 3:
        return []

    # Simple averaging of adjacent face values
    b1, b2, b3 = staggered_bfields

    # Handle edges based on shapes
    b1_centered = 0.5 * (b1[..., :-1] + b1[..., 1:])
    b2_centered = 0.5 * (b2[:, :-1, :] + b2[:, 1:, :])
    b3_centered = 0.5 * (b3[:-1, ...] + b3[1:, ...])

    return [b1_centered, b2_centered, b3_centered]


def pad_staggered_fields(
    staggered_bfields: list[NDArray[np.floating]],
) -> list[NDArray[np.floating]]:
    """Pad staggered magnetic fields along appropriate directions."""
    if not staggered_bfields:
        return []

    b1, b2, b3 = staggered_bfields

    # Pad each field appropriately
    # Note: B1 is staggered in x, B2 in y, B3 in z
    b1_padded = np.pad(b1, ((1, 1), (1, 1), (0, 0)), "edge")
    b2_padded = np.pad(b2, ((1, 1), (0, 0), (1, 1)), "edge")
    b3_padded = np.pad(b3, ((0, 0), (1, 1), (1, 1)), "edge")

    return [b1_padded, b2_padded, b3_padded]


def initialize_state(config: SimbiBaseConfig) -> SimulationState:
    """
    Initialize simulation state from configuration.

    This function:
    i). Determines grid dimensions
    ii). Initializes arrays for primitive variables
    iii). Populates arrays using generator functions from config
    iv). Converts primitive variables to conserved variables
    v). Returns a complete SimulationState object

    Args:
        config: The simulation configuration

    Returns:
        Complete simulation state with both primitive and conserved variables
    """
    from ...functional import to_iterable

    # Determine grid dimensions from resolution
    if isinstance(config.resolution, int):
        nx, ny, nz = config.resolution, 1, 1
        resolution = [nx, ny, nz]
    else:
        resolution = list(config.resolution)
        nx = resolution[0]
        ny = resolution[1] if len(resolution) > 1 else 1
        nz = resolution[2] if len(resolution) > 2 else 1

    # Adjust for MHD (always 3D)
    if config.is_mhd:
        ny = max(ny, 1)
        nz = max(nz, 1)
        active_resolution = [nx, ny, nz]
    else:
        active_resolution = list(r for r in resolution if r > 1)

    # Determine ghost cell padding
    pad_width = 1 + (config.spatial_order.value == "plm")

    # Number of variables
    nvars = config.nvars

    # Create array for primitive variables with ghost cells
    padded_shape = (nvars,) + tuple(
        r + 2 * pad_width for r in to_iterable(active_resolution)[::-1]
    )
    primitive = np.zeros(padded_shape, dtype=np.float64)

    # Get appropriate generator(s)
    generator = config.initial_primitive_state()
    staggered_bfields: list[NDArray[np.floating]] = []

    # Handle MHD generators
    if is_mhd_generator(generator):
        # Unpack MHD generators
        gen_tuple = cast(MHDStateGenerators, generator)
        gas_gen, b1_gen, b2_gen, b3_gen = gen_tuple

        # Initialize staggered magnetic field arrays
        b1_shape = (nz, ny, nx + 1)
        b2_shape = (nz, ny + 1, nx)
        b3_shape = (nz + 1, ny, nx)

        # Use generators to fill magnetic field arrays
        staggered_bfields = [
            np.fromiter(b1_gen(), dtype=float).reshape(b1_shape),
            np.fromiter(b2_gen(), dtype=float).reshape(b2_shape),
            np.fromiter(b3_gen(), dtype=float).reshape(b3_shape),
        ]

        # Get gas state generator
        gas_state = gas_gen()
    else:
        # Pure hydro case - just get the gas state generator
        gas_state = cast(GasStateFunction, generator)()

    # Peek at first value to determine number of components
    values_iter, gas_iter = itertools.tee(gas_state)
    first_values = next(values_iter)
    n_components = len(first_values)

    interior = (slice(pad_width, -pad_width),) * len(active_resolution)
    interior_shape = primitive[:n_components, *interior].shape
    primitive[:n_components, *interior] = np.fromiter(
        gas_iter, dtype=(float, n_components)
    ).T.reshape(interior_shape)

    conserved = np.zeros_like(primitive)
    # Convert primitive to conserved variables
    conserved[:nvars, *interior] = primitive_to_conserved(
        primitive[:nvars, *interior], config, staggered_bfields
    )

    # fill the ghost cells at the edges with values from the interior
    ndim = len(active_resolution)
    for dim in range(ndim):
        # create slices for each dimension
        for j in range(pad_width):
            # Fill left boundary ghost cells
            left_ghost_slices = [slice(None)] * (ndim + 1)  # +1 for variables dimension
            left_ghost_slices[dim + 1] = slice(j, j + 1)

            left_interior_slices = [slice(None)] * (ndim + 1)
            left_interior_slices[dim + 1] = slice(pad_width, pad_width + 1)

            conserved[tuple(left_ghost_slices)] = conserved[tuple(left_interior_slices)]
            primitive[tuple(left_ghost_slices)] = primitive[tuple(left_interior_slices)]

            # Fill right boundary ghost cells
            right_ghost_slices = [slice(None)] * (ndim + 1)
            right_ghost_slices[dim + 1] = slice(-j - 1, -j if j > 0 else None)

            right_interior_slices = [slice(None)] * (ndim + 1)
            right_interior_slices[dim + 1] = slice(-pad_width - 1, -pad_width)

            conserved[tuple(right_ghost_slices)] = conserved[
                tuple(right_interior_slices)
            ]
            primitive[tuple(right_ghost_slices)] = primitive[
                tuple(right_interior_slices)
            ]

    if config.is_mhd:
        # Pad staggered fields
        staggered_bfields = pad_staggered_fields(staggered_bfields)

    # Create and return simulation state
    return SimulationState(
        primitive_state=primitive,
        conserved_state=conserved,
        staggered_bfields=staggered_bfields,
        config=config,
    )


def load_or_initialize_state(
    config: SimbiBaseConfig,
) -> Maybe[SimulationState]:
    """
    Load state from checkpoint if specified or initialize from scratch.

    Args:
        config: Simulation configuration

    Returns:
        SimulationState loaded from checkpoint or initialized from config
    """
    from ..io.checkpoint import load_checkpoint_to_state

    # Try to load from checkpoint if specified
    if config.checkpoint_file:
        checkpoint_state = load_checkpoint_to_state(config)
        return checkpoint_state

    # Initialize from config and wrap in Maybe for consistency
    return Maybe.of(initialize_state(config))
