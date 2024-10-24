# Module to config the initial condition for the SIMBI
# hydro setup. From here on, I will fragment the code
# to try and reduce the confusion between functions

import numpy as np
import h5py
import numpy.typing as npt
from ..key_types import *
from .mem import release_memory
from . import helpers
from .slogger import logger
from ..tools.utility import read_file
from itertools import product, permutations
from multiprocessing import Process
from typing import Any, Callable


def dot_product(a: NDArray[Any], b: NDArray[Any]) -> Any:
    return np.sum([x * y for x, y in zip(a, b)], axis=0)


def calc_lorentz_factor(velocity: NDArray[Any], regime: str) -> FloatOrArray:
    vsquared = dot_product(velocity, velocity)
    return 1.0 if regime == "classical" else (1.0 - vsquared) ** (-0.5)


def calc_spec_enthalpy(
    gamma: float,
    rho: FloatOrArray,
    pressure: FloatOrArray,
    regime: str,
) -> FloatOrArray:
    return (1.0 if regime == "classical" else 1.0 +
            gamma * pressure / (rho * (gamma - 1.0)))


def calc_labframe_densiity(
        rho: FloatOrArray,
        velocity: NDArray[numpy_float],
        regime: str) -> FloatOrArray:
    return rho * calc_lorentz_factor(velocity, regime)


def calc_labframe_momentum(
    gamma: float,
    rho: FloatOrArray,
    velocity: NDArray[numpy_float],
    pressure: FloatOrArray,
    bfields: NDArray[numpy_float],
    regime: str
) -> NDArray[Any]:
    if len(bfields) == 0:
        vdb: FloatOrArray = 0.0
        bsq: FloatOrArray = 0.0
        vdb_bvec: FloatOrArray = 0.0
    else:
        vdb = dot_product(velocity, bfields)
        bsq = dot_product(bfields, bfields)
        vdb_bvec = vdb * bfields

    enthalpy = calc_spec_enthalpy(gamma, rho, pressure, regime)
    lorentz = calc_lorentz_factor(velocity, regime)

    return (rho * lorentz * lorentz * enthalpy + bsq) * velocity - vdb_bvec


def calc_labframe_energy(
    gamma: float,
    rho: FloatOrArray,
    pressure: FloatOrArray,
    velocity: NDArray[numpy_float],
    bfields: NDArray[numpy_float],
    regime: str,
) -> FloatOrArray:
    if len(bfields) == 0:
        bsq: FloatOrArray = 0.0
        vdb: FloatOrArray = 0.0
    else:
        bsq = dot_product(bfields, bfields)
        vdb = dot_product(velocity, bfields)

    vsq: FloatOrArray = dot_product(velocity, velocity)
    lorentz = calc_lorentz_factor(velocity, regime)
    enthalpy = calc_spec_enthalpy(gamma, rho, pressure, regime)
    return (
        pressure / (gamma - 1.0) + 0.5 * rho * vsq + 0.5 * bsq
        if regime == "classical"
        else (
            rho * lorentz * lorentz * enthalpy
            - pressure
            - rho * lorentz
            + 0.5 * bsq
            + 0.5 * (bsq * vsq - vdb**2)
        )
    )


def flatten_fully(x: Any) -> Any:
    if any(dim == 1 for dim in x.shape):
        x = np.vstack(x)
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flat
        return flatten_fully(x)
    else:
        return np.asanyarray(x)


def load_checkpoint(model: Any, filename: str) -> None:
    print(f"Loading from checkpoint: {filename}...", flush=True)
    setup: dict[str, Any] = {}
    volume_factor: Union[float, NDArray[numpy_float]] = 1.0
    fields, setup, mesh = read_file(filename)
    dim: int = setup["dimensions"]
    
    vel = np.array([fields[f"v{i}"] for i in range(1, dim + 1)])
    if setup["regime"] == "srmhd":
        bfields = np.array([fields[f"b{i}"] for i in range(1, dim + 1)])
    else:
        bfields = np.array([])
    dens = calc_labframe_densiity(
        fields["rho"], vel, setup["regime"])
    mom = calc_labframe_momentum(
        setup["ad_gamma"],
        fields["rho"],
        vel,
        fields["p"],
        bfields,
        setup["regime"],
    )
    energy = calc_labframe_energy(
        setup["ad_gamma"],
        fields["rho"],
        fields["p"],
        vel,
        bfields,
        setup["regime"],
    )
    
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
                x1=mesh["x1"], x2=mesh["x2"], coord_system=setup["coord_system"])
        elif dim == 3:
            volume_factor = helpers.calc_cell_volume3D(
                x1=mesh["x1"],
                x2=mesh["x2"],
                x3=mesh["x3"],
                coord_system=setup["coord_system"],
            )
            
    model.u = np.array([dens, *mom, energy, *bfields, dens * fields["chi"]])

    padwith = (setup["spatial_order"] != "pcm") + 1
    npad = ((0, 0),) + tuple(
        tuple(val)
        for val in [[(padwith), (padwith)]]
        * dim
    )
    model.u = np.pad(model.u * volume_factor, npad, "edge")
    model.chkpt_idx = setup["chkpt_idx"]

@release_memory
def initializeModel(
    model: Any,
    first_order: bool,
    volume_factor: Union[float, NDArray[Any]],
    passive_scalars: Union[npt.NDArray[Any], Any],
) -> None:
    model.u = np.insert(model.u, model.u.shape[0], 0.0, axis=0)
    if passive_scalars is not None:
        model.u[-1, ...] = passive_scalars * model.u[0]

    # npad is a tuple of (n_before, n_after) for each dimension
    npad = ((0, 0),) + tuple(
        tuple(val)
        for val in [[((first_order ^ 1) + 1), ((first_order ^ 1) + 1)]]
        * model.dimensionality
    )
    model.u = np.pad(model.u * volume_factor, npad, "edge")


# @release_memory
def construct_the_state(
        model: Any,
        initial_state: NDArray[numpy_float]) -> None:
    if not model.mhd:
        model.nvars = 3 + model.dimensionality
    else:
        model.nvars = 9

    # Initialize conserved u-array and flux arrays
    model.u = np.zeros(
        shape=(model.nvars - 1, *np.asanyarray(model.resolution).flat[::-1])
    )
    # model.u = initial_state

    if model.discontinuity:
        logger.info(
            f"Initializing Problem With a {str(model.dimensionality)}D Discontinuity..."
        )

        if len(
                model.geometry) == 3 and isinstance(
                model.geometry[0], (int, float)):
            geom_tuple: Any = (model.geometry,)
        else:
            geom_tuple = model.geometry

        break_points = [val[2] for val in geom_tuple if len(val) == 3]
        if len(break_points) > model.dimensionality:
            raise ValueError(
                "Number of break points must be less than or equal to the number of dimensions"
            )

        # \vector{dx}
        spacings = [
            (geom_tuple[idx][1] - geom_tuple[idx][0]) / model.resolution[idx]
            for idx in range(len(geom_tuple))
        ]

        pieces = [
            (None, round(abs(break_points[idx] - geom_tuple[idx][0]) / spacings[idx]))
            for idx in range(len(break_points))
        ]

        # partition the grid based on user-defined partition coordinates
        partition_inds: list[Any] = list(
            product(*[permutations(x) for x in pieces]))
        partition_inds = [tuple([slice(*y) for y in x])
                          for x in partition_inds]
        partitions = [model.u[(..., *sector)] for sector in partition_inds]

        for idx, part in enumerate(partitions):
            state = initial_state[idx]
            rho, *velocity, pressure = state[: model.number_of_non_em_terms]
            velocity = np.asanyarray(velocity)
            # check if there are any bfields
            mean_bfields = state[model.number_of_non_em_terms:]

            dens = calc_labframe_densiity(rho, velocity, model.regime)
            mom = calc_labframe_momentum(
                model.gamma,
                rho,
                velocity,
                pressure,
                mean_bfields,
                model.regime)
            energy = calc_labframe_energy(
                model.gamma,
                rho,
                pressure,
                velocity,
                mean_bfields,
                model.regime,
            )

            if model.dimensionality == 1:
                part[...] = np.array(
                    [dens, *mom, energy, *mean_bfields])[:, None]
            else:
                part[...] = (
                    part[...].transpose()
                    + np.array([dens, *mom, energy, *mean_bfields])
                ).transpose()
    else:
        rho, * \
            velocity, pressure = initial_state[: model.number_of_non_em_terms]
        velocity = np.asanyarray(velocity)
        # check if there are any bfields
        bfields_stag = initial_state[model.number_of_non_em_terms:]
        if len(bfields_stag) != 0:
            mean_bfields = np.array(
                [
                    0.5 * (bfields_stag[0][..., 1:] + bfields_stag[0][..., :-1]),
                    0.5 * (bfields_stag[1][:, 1:] + bfields_stag[1][:, :-1]),
                    0.5 * (bfields_stag[2][1:] + bfields_stag[2][:-1]),
                ]
            )
        else:
            mean_bfields = []

        dens = calc_labframe_densiity(rho, velocity, model.regime)
        mom = calc_labframe_momentum(
            model.gamma,
            rho,
            velocity,
            pressure,
            mean_bfields,
            model.regime
        )
        energy = calc_labframe_energy(
            model.gamma,
            rho,
            pressure,
            velocity,
            mean_bfields,
            model.regime,
        )

        model.u[...] = np.array(
            [
                rho,
                *mom,
                energy,
                *mean_bfields,
            ]
        )
