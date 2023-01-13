import numpy as np
import sys
from time import sleep
from .key_types import *
from typing import TextIO, Generator
from numbers import Number

generic_numpy_array = np.ndarray[Any, Any]


def calc_cell_volume1D(*, x1: generic_numpy_array) -> generic_numpy_array:
    x1vertices = np.sqrt(x1[1:] * x1[:-1])
    x1vertices = np.insert(x1vertices,  0, x1[0])
    x1vertices = np.insert(x1vertices, x1.shape, x1[-1])
    x1mean = 0.75 * (x1vertices[1:]**4 - x1vertices[:-1]
                     ** 4) / (x1vertices[1:]**3 - x1vertices[:-1]**3)
    dx1 = x1vertices[1:] - x1vertices[:-1]
    return cast(generic_numpy_array, x1mean * x1mean * dx1)


def calc_cell_volume2D(*, x1: generic_numpy_array, x2: generic_numpy_array, coord_system: str = 'spherical') -> generic_numpy_array:
    if coord_system == 'spherical':
        if x1.ndim == 1 and x2.ndim == 1:
            rr, thetta = np.meshgrid(x1, x2)
        else:
            rr, thetta = x1, x2
        tvertices = 0.5 * (thetta[1:] + thetta[:-1])
        tvertices = np.insert(tvertices, 0, thetta[0], axis=0)
        tvertices = np.insert(
            tvertices, tvertices.shape[0], thetta[-1], axis=0)
        dcos = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])

        rvertices = np.sqrt(rr[:, 1:] * rr[:, :-1])
        rvertices = np.insert(rvertices,  0, rr[:, 0], axis=1)
        rvertices = np.insert(rvertices, rvertices.shape[1], rr[:, -1], axis=1)
        return cast(generic_numpy_array, 2.0 * np.pi * (1./3.) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3) * dcos)
    else:
        if x1.ndim == 1 and x2.ndim == 1:
            rr, zz = np.meshgrid(x1, x2)
        else:
            rr, zz = x1, x2

        zvertices = 0.5 * (zz[1:] + zz[:-1])
        zvertices = np.insert(zvertices, 0, zz[0], axis=0)
        zvertices = np.insert(zvertices, zvertices.shape[0], zz[-1], axis=0)
        dz = zvertices[1:] - zvertices[:-1]

        rvertices = 0.5 * (rr[:, 1:] + rr[:, :-1])
        rvertices = np.insert(rvertices,  0, rr[:, 0], axis=1)
        rvertices = np.insert(rvertices, rvertices.shape[1], rr[:, -1], axis=1)
        rmean = (2.0 / 3.0) * (rvertices[:, 1:]**3 - rvertices[:, :-1]
                               ** 3) / (rvertices[:, 1:]**2 - rvertices[:, :-1]**2)
        return cast(generic_numpy_array, rmean * (rvertices[:, 1:] - rvertices[:, :-1]) * dz)


def calc_cell_volume3D(*, r: generic_numpy_array, theta: generic_numpy_array, phi: generic_numpy_array) -> generic_numpy_array:
    if r.ndim == 1 and theta.ndim == 1 and phi.ndim == 1:
        thetta, phii, rr = np.meshgrid(theta, phi, r)
    else:
        rr, thetta, phii = r, theta, phi

    pvertices = 0.5 * (phii[1:] + phii[:-1])
    pvertices = np.insert(pvertices, 0, phii[0], axis=0)
    pvertices = np.insert(pvertices, pvertices.shape[0], phii[-1], axis=0)
    dphi = pvertices[1:] - pvertices[:-1]

    tvertices = 0.5 * (thetta[:, 1:] + thetta[:, :-1])
    tvertices = np.insert(tvertices, 0, thetta[0], axis=1)
    tvertices = np.insert(tvertices, tvertices.shape[1], thetta[:, -1], axis=1)
    dcos = np.cos(tvertices[:, :-1]) - np.cos(tvertices[:, 1:])

    rvertices = np.sqrt(rr[:, :,  1:] * rr[:, :, :-1])
    rvertices = np.insert(rvertices,  0, rr[:, :, 0], axis=2)
    rvertices = np.insert(rvertices, rvertices.shape[2], rr[:, :, -1], axis=2)
    return cast(generic_numpy_array, (1./3.) * (rvertices[:, :, 1:]**3 - rvertices[:, :, :-1]**3) * dcos * dphi)


def compute_num_polar_zones(*,
                            rmin: Optional[float] = None,
                            rmax: Optional[float] = None,
                            nr:   Optional[int] = None,
                            zpd:  Optional[int] = None,
                            theta_bounds: tuple[float, float] = (0.0, np.pi)) -> int:
    # Convert the values if None
    rmin = rmin or 1.0
    rmax = rmax or 1.0
    nr = nr or 1
    if zpd is not None:
        return round((theta_bounds[1] - theta_bounds[0]) * zpd / np.log(10))
    elif None not in (rmin, rmax, nr):
        dlogr: float = np.log(rmax / rmin) / nr
        return round(1 + (theta_bounds[1] - theta_bounds[0]) / dlogr)
    else:
        raise ValueError(
            "Please either specify zones per decade or rmin, rmax, and nr")


def calc_dlogt(tmin: float, tmax: float, ncheckpoints: int) -> float:
    if tmin == 0:
        return cast(float, np.log10(tmax / 1e-10) / (ncheckpoints - 1))
    return cast(float, np.log10(tmax / tmin) / (ncheckpoints - 1))


def progressbar(it: range, prefix: str = "", size: int = 100, out: TextIO = sys.stdout) -> Generator[int, None, None]:
    count = len(it)

    def show(j: int) -> None:
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}",
              end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def print_progress() -> None:
    for i in progressbar(range(100), "Loading: ", 60):
        sleep(0.01)


def pad_jagged_array(arr: Union[NDArray[Any] | Sequence[Any]]) -> NDArray[Any]:
    arr       = [np.array(val) if isinstance(val, (Sequence, np.ndarray)) else np.array([val]) for val in arr]
    max_dim   = max(a.ndim for a in arr)
    max_size  = np.max([a.size for a in arr if a.ndim == max_dim])
    max_shape = [a.shape for a in arr if a.size == max_size][0]
    arr       = np.array([a if a.shape == max_shape else np.ones(max_shape) * a for a in arr])
    return arr 