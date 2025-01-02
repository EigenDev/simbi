import numpy as np
import numpy.typing as npt
from ..key_types import *
from .mem import release_memory
from . import helpers
from .slogger import logger
from ..tools.utility import read_file
from itertools import product, permutations
from typing import Any


def check_valid_state(x: Any, name: str) -> None:
    if np.isnan(np.sum(x)):
        raise ValueError(f"Initial state: {name} contains NaNs")


def dot_product(a: NDArray[Any], b: NDArray[Any]) -> Any:
    return np.sum(a * b, axis=0)


def calc_lorentz_factor(velocity: NDArray[Any], regime: str) -> FloatOrArray:
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
    rho: FloatOrArray, velocity: NDArray[numpy_float], regime: str
) -> FloatOrArray:
    return rho * calc_lorentz_factor(velocity, regime)


def calc_labframe_momentum(
    gamma: float,
    rho: FloatOrArray,
    velocity: NDArray[numpy_float],
    pressure: FloatOrArray,
    bfields: NDArray[numpy_float],
    regime: str,
) -> NDArray[numpy_float]:
    res: NDArray[numpy_float]
    vdb = dot_product(velocity, bfields) if np.any(bfields) else 0.0
    bsq = dot_product(bfields, bfields) if np.any(bfields) else 0.0
    vdb_bvec = vdb * bfields if np.any(bfields) else 0.0

    enthalpy = calc_spec_enthalpy(gamma, rho, pressure, regime)
    lorentz = calc_lorentz_factor(velocity, regime)
    res = (rho * lorentz**2 * enthalpy + bsq) * velocity - vdb_bvec
    return res


def calc_labframe_energy(
    gamma: float,
    rho: FloatOrArray,
    pressure: FloatOrArray,
    velocity: NDArray[numpy_float],
    bfields: NDArray[numpy_float],
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
        setup["adiabatic_gamma"], fields["rho"], vel, fields["p"], bfields, setup["regime"]
    )
    energy = calc_labframe_energy(
        setup["adiabatic_gamma"], fields["rho"], fields["p"], vel, bfields, setup["regime"]
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
    model.chkpt_idx = setup["chkpt_idx"]
    
    if model.mhd:
        model.bfield = [
            fields["b1stag"],
            fields["b2stag"],
            fields["b3stag"]
        ]


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


@release_memory
def construct_the_state(model: Any, initial_state: NDArray[numpy_float]) -> None:
    model.nvars = 3 + model.dimensionality if not model.mhd else 9

    model.u = np.zeros((model.nvars - 1, *np.asanyarray(model.resolution).flat[::-1]))

    
    if model.discontinuity:
        logger.info(
            f"Initializing Problem With a {
                str(model.dimensionality)}D Discontinuity..."
        )

        geom_tuple = (
            (model.geometry,)
            if len(model.geometry) == 3 and isinstance(model.geometry[0], (int, float))
            else model.geometry
        )

        break_points = [val[2] for val in geom_tuple if len(val) == 3]
        if len(break_points) > model.dimensionality:
            raise ValueError(
                "Number of break points must be less than or equal to the number of dimensions"
            )

        spacings = [
            (geom_tuple[idx][1] - geom_tuple[idx][0]) / model.resolution[idx]
            for idx in range(len(geom_tuple))
        ]

        pieces = [
            (None, round(abs(break_points[idx] - geom_tuple[idx][0]) / spacings[idx]))
            for idx in range(len(break_points))
        ]

        # partition the grid based on user-defined partition coordinates
        partition_inds: list[Any] = list(product(*[permutations(x) for x in pieces]))
        partition_inds = [tuple([slice(*y) for y in x]) for x in partition_inds]
        partitions = [model.u[(..., *sector)] for sector in partition_inds]

        for idx, part in enumerate(partitions):
            state = initial_state[idx]
            rho, *velocity, pressure = state[: model.number_of_non_em_terms]
            velocity = np.asanyarray(velocity)
            mean_bfields = state[model.number_of_non_em_terms :]

            dens = calc_labframe_density(rho, velocity, model.regime)
            mom = calc_labframe_momentum(
                model.gamma, rho, velocity, pressure, mean_bfields, model.regime
            )
            energy = calc_labframe_energy(
                model.gamma, rho, pressure, velocity, mean_bfields, model.regime
            )

            check_valid_state(dens, "density")
            check_valid_state(mom, "momentum")
            check_valid_state(energy, "energy")

            if model.dimensionality == 1:
                part[...] = np.array([dens, *mom, energy, *mean_bfields])[:, None]
            else:
                part[...] = (
                    part[...].transpose()
                    + np.array([dens, *mom, energy, *mean_bfields])
                ).transpose()
         
         # if it's an MHD run, load the staggered fields
        if model.mhd:
            nxp, nyp, nzp = model.resolution
            b1 = np.zeros(shape=(nzp + 0, nyp + 0, nxp + 1))
            b2 = np.zeros(shape=(nzp + 0, nyp + 1, nxp + 0))
            b3 = np.zeros(shape=(nzp + 1, nyp + 0, nxp + 0))

            region_one = model.x1 < model.geometry[0][2]
            region_two = np.logical_not(region_one)
            xc = helpers.calc_centroid(
                model.x1, coord_system=model.coord_system)
            a = xc < model.geometry[0][2]
            b = np.logical_not(a)
            b1[..., region_one] = model.initial_state[0][5]
            b1[..., region_two] = model.initial_state[1][5]
            b2[..., a] = model.initial_state[0][6]
            b2[..., b] = model.initial_state[1][6]
            b3[..., a] = model.initial_state[0][7]
            b3[..., b] = model.initial_state[1][7]
            model.bfield = [
                np.pad(b1, ((1, 1), (1, 1), (0, 0))),
                np.pad(b2, ((1, 1), (0, 0), (1, 1))),
                np.pad(b3, ((1, 1), (1, 1), (1, 1)))
            ]
    else:
        rho, *velocity, pressure = initial_state[: model.number_of_non_em_terms]
        velocity = np.asanyarray(velocity)
        bfields_stag = initial_state[model.number_of_non_em_terms :]
        mean_bfields = (
            np.array(
                [
                    0.5 * (bfields_stag[0][..., 1:] + bfields_stag[0][..., :-1]),
                    0.5 * (bfields_stag[1][:, 1:] + bfields_stag[1][:, :-1]),
                    0.5 * (bfields_stag[2][1:] + bfields_stag[2][:-1]),
                ], dtype=object
            )
            if bfields_stag.size
            else []
        )

        dens = calc_labframe_density(rho, velocity, model.regime)
        mom = calc_labframe_momentum(
            model.gamma, rho, velocity, pressure, mean_bfields, model.regime
        )
        energy = calc_labframe_energy(
            model.gamma, rho, pressure, velocity, mean_bfields, model.regime
        )

        check_valid_state(dens, "density")
        check_valid_state(mom, "momentum")
        check_valid_state(energy, "energy")
        model.u[...] = np.array([dens, *mom, energy, *mean_bfields])
        
        if model.mhd:
            model.bfield = [
                np.pad(bfields_stag[0], ((1, 1), (1, 1), (0, 0))),
                np.pad(bfields_stag[1], ((1, 1), (0, 0), (1, 1))),
                np.pad(bfields_stag[2], ((0, 0), (1, 1), (1, 1)))
            ]

