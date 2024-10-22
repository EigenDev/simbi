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
from itertools import product, permutations
from multiprocessing import Process
from typing import Any, Callable


def dot_product(a: NDArray[Any], b: NDArray[Any]) -> Any:
    return np.sum([x * y for x, y in zip(a, b)], axis=0)


def calc_lorentz_factor(vsquared: NDArray[Any], regime: str) -> FloatOrArray:
    return 1.0 if regime == "classical" else (1.0 - np.asanyarray(vsquared)) ** (-0.5)


def calc_spec_enthalpy(
    rho: FloatOrArray,
    pressure: FloatOrArray,
    gamma: float,
    regime: str,
) -> FloatOrArray:
    return (
        1.0 if regime == "classical" else 1.0 + gamma * pressure / (rho * (gamma - 1.0))
    )


def calc_labframe_densiity(rho: FloatOrArray, lorentz: FloatOrArray) -> FloatOrArray:
    return rho * lorentz


def calc_labframe_momentum(
    rho: FloatOrArray,
    lorentz: FloatOrArray,
    enthalpy: FloatOrArray,
    velocity: NDArray[numpy_float],
    bfields: NDArray[numpy_float],
) -> NDArray[Any]:
    if len(bfields) == 0:
        bvec: FloatOrArray = 0.0
        vdb: FloatOrArray = 0.0
        bsq: FloatOrArray = 0.0
        vdb_bvec: FloatOrArray = 0.0
    else:
        bvec = bfields
        vdb = dot_product(velocity, bfields)
        bsq = dot_product(bfields, bfields)
        vdb_bvec = vdb * bfields

    return (rho * lorentz * lorentz * enthalpy + bsq) * velocity - vdb_bvec


def calc_labframe_energy(
    gamma: float,
    rho: FloatOrArray,
    lorentz: FloatOrArray,
    enthalpy: FloatOrArray,
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


def load_checkpoint(model: Any, filename: str, dim: int, mesh_motion: bool) -> None:
    print(f"Loading from checkpoint: {filename}...", flush=True)
    setup: dict[str, Any] = {}
    volume_factor: Union[float, NDArray[numpy_float]] = 1.0
    with h5py.File(filename, "r") as hf:
        ds = hf.get("sim_info").attrs
        nx = ds["nx"] or 1
        ny = ds["ny"] if "ny" in ds.keys() else 1
        nz = ds["nz"] if "nz" in ds.keys() else 1
        try:
            ndim = ds["dimensions"]
        except KeyError:
            ndim = 1 + (ny > 1) + (nz > 1)

        setup["ad_gamma"] = ds["adiabatic_gamma"]
        setup["regime"] = ds["regime"].decode("utf-8")
        setup["coord_system"] = ds["geometry"].decode("utf-8")
        setup["mesh_motion"] = ds["mesh_motion"]

        # ------------------------
        # Generate Mesh
        # ------------------------
        arr_select: Callable[..., function] = lambda x: (
            np.linspace if x == b"linear" else np.geomspace
        )
        funcs = [
            arr_select(val)
            for val, _ in zip(
                [ds["x1_cell_spacing"], ds["x2_cell_spacing"], ds["x3_cell_spacing"]],
                range(ndim),
            )
        ]
        mesh = {f"x{i+1}": hf.get(f"x{i+1}")[:] for i in range(ndim)}

        if ds["x1max"] > mesh["x1"][-1]:
            mesh["x1"] = funcs[0](
                ds["x1min"], ds["x1max"], ds["xactive_zones"]
            )  # type: ignore

        if setup["mesh_motion"]:
            if ndim == 1 and setup["coord_system"] != "cartesian":
                volume_factor = helpers.calc_cell_volume1D(
                    x1=mesh["x1"], coord_system=setup["coord_system"]
                )
            elif ndim == 2:
                volume_factor = helpers.calc_cell_volume2D(
                    x1=mesh["x1"], x2=mesh["x2"], coord_system=setup["coord_system"]
                )
            elif ndim == 3:
                volume_factor = helpers.calc_cell_volume3D(
                    x1=mesh["x1"],
                    x2=mesh["x2"],
                    x3=mesh["x3"],
                    coord_system=setup["coord_system"],
                )

            if setup["coord_system"] != "cartesian":
                npad = tuple(
                    tuple(val)
                    for val in [
                        [((ds["first_order"] ^ 1) + 1), ((ds["first_order"] ^ 1) + 1)]
                    ]
                    * ndim
                )
                volume_factor = np.pad(volume_factor, npad, "edge")

        rho = hf.get("rho")[:]
        v = [(hf.get(f"v{dim}") or hf.get(f"v"))[:] for dim in range(1, ndim + 1)]
        p = hf.get("p")[:]
        chi = (hf.get("chi") or np.zeros_like(rho))[:]
        rho = flatten_fully(rho.reshape(nz, ny, nx))
        v = [flatten_fully(vel.reshape(nz, ny, nx)) for vel in v]
        p = flatten_fully(p.reshape(nz, ny, nx))
        chi = flatten_fully(chi.reshape(nz, ny, nx))

        # -------------------------------
        # Load Fields
        # -------------------------------
        vsqr = np.sum(vel * vel for vel in v)  # type: ignore
        if setup["regime"] in ["srhd", "srmhd"]:
            try:
                if ds["using_gamma_beta"]:
                    W = (1 + vsqr) ** 0.5
                    v = [vel / W for vel in v]
                    vsqr /= W**2
                else:
                    W = (1 - vsqr) ** (-0.5)
            except KeyError:
                W = (1 - vsqr) ** (-0.5)
        else:
            W = 1

        if setup["regime"] == "srhd":
            h = 1.0 + setup["ad_gamma"] * p / (rho * (setup["ad_gamma"] - 1.0))
            e = rho * W * W * h - p - rho * W
        else:
            h = 1.0
            e = p / (setup["ad_gamma"] - 1.0) + 0.5 * rho * vsqr

        momentum = np.asarray([rho * W * W * h * vel for vel in v])
        model.start_time = ds["current_time"]
        model.x1 = mesh["x1"]
        if ndim >= 2:
            model.x2 = mesh["x2"]
        if ndim >= 3:
            model.x3 = mesh["x3"]

        model.u = np.array([rho * W, *momentum, e, rho * W * chi]) * volume_factor

        model.chkpt_idx = ds["chkpt_idx"]


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
def construct_the_state(model: Any, initial_state: NDArray[numpy_float]) -> None:
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

        if len(model.geometry) == 3 and isinstance(model.geometry[0], (int, float)):
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
        partition_inds: list[Any] = list(product(*[permutations(x) for x in pieces]))
        partition_inds = [tuple([slice(*y) for y in x]) for x in partition_inds]
        partitions = [model.u[(..., *sector)] for sector in partition_inds]

        for idx, part in enumerate(partitions):
            state = initial_state[idx]
            rho, *velocity, pressure = state[: model.number_of_non_em_terms]
            velocity = np.asanyarray(velocity)
            # check if there are any bfields
            mean_bfields = state[model.number_of_non_em_terms :]

            vsqr = dot_product(velocity, velocity)
            lorentz_factor = calc_lorentz_factor(vsqr, model.regime)
            enthalpy = calc_spec_enthalpy(rho, pressure, model.gamma, model.regime)

            energy = calc_labframe_energy(
                model.gamma,
                rho,
                lorentz_factor,
                enthalpy,
                pressure,
                velocity,
                mean_bfields,
                model.regime,
            )

            dens = calc_labframe_densiity(rho, lorentz_factor)
            mom = calc_labframe_momentum(
                rho, lorentz_factor, enthalpy, velocity, mean_bfields
            )

            if model.dimensionality == 1:
                part[...] = np.array([dens, *mom, energy, *mean_bfields])[:, None]
            else:
                part[...] = (
                    part[...].transpose()
                    + np.array([dens, *mom, energy, *mean_bfields])
                ).transpose()
    else:
        rho, *velocity, pressure = initial_state[: model.number_of_non_em_terms]
        velocity = np.asanyarray(velocity)
        # check if there are any bfields
        bfields_stag = initial_state[model.number_of_non_em_terms :]
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

        vsqr = dot_product(velocity, velocity)
        lorentz_factor = calc_lorentz_factor(vsqr, model.regime)
        enthalpy = calc_spec_enthalpy(rho, pressure, model.gamma, model.regime)

        model.init_density = calc_labframe_densiity(rho, lorentz_factor)
        model.init_momentum = calc_labframe_momentum(
            rho, lorentz_factor, enthalpy, velocity, mean_bfields
        )
        model.init_energy = calc_labframe_energy(
            model.gamma,
            rho,
            lorentz_factor,
            enthalpy,
            pressure,
            velocity,
            mean_bfields,
            model.regime,
        )

        model.u[...] = np.array(
            [
                model.init_density,
                *model.init_momentum,
                model.init_energy,
                *mean_bfields,
            ]
        )
