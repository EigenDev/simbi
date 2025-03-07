import numpy as np
import math
import sys
import linecache
import os
import tracemalloc
from numpy.typing import NDArray
from typing import Any, Callable, Generator, Optional, Sequence, Union, cast
from ..io.logging import logger
from time import sleep
from typing import TextIO, Generator


__all__ = [
    "calc_centroid",
    "calc_vertices",
    "calc_cell_volume",
    "to_iterable",
    "compute_num_polar_zones",
    "calc_dlogt",
    "print_progress",
    "progressbar",
    "find_nearest",
    "display_top",
    "tuple_of_tuples",
    "expand_axis_if_needed",
    "order_of_mag",
    "to_tuple_of_tuples",
]


def as_list(x: Any) -> list[Any]:
    if isinstance(x, (Sequence, list, np.ndarray)):
        return list(x)
    else:
        return [x]


def calc_any_mean(arr: NDArray[Any], cellspacing: str) -> Any:
    if cellspacing == "linear":
        return 0.5 * (arr[1:] + arr[:-1])
    else:
        return np.sqrt(arr[1:] * arr[:-1])


def calc_centroid(arr: NDArray[Any], coord_system: str = "spherical") -> NDArray[Any]:
    if coord_system == "spherical":
        return np.asanyarray(
            0.75
            * (arr[..., 1:] ** 4 - arr[..., :-1] ** 4)
            / (arr[..., 1:] ** 3 - arr[..., :-1] ** 3)
        )
    elif coord_system == "cylindrical":
        return np.asanyarray(
            (2.0 / 3.0)
            * (arr[..., 1:] ** 3 - arr[..., :-1] ** 3)
            / (arr[..., 1:] ** 2 - arr[..., :-1] ** 2)
        )
    else:
        return np.asanyarray(0.5 * (arr[..., 1:] + arr[..., :-1]))


def calc_vertices(
    *, arr: NDArray[Any], direction: int, cell_spacing: str = "linear"
) -> Any:
    if direction not in [1, 2, 3]:
        raise ValueError("Direction must be either 1, 2, or 3")
    dims = arr.ndim
    padding: Any = [[0, 0]] * dims
    padding[-direction] = [1, 1]
    padding = tuple([tuple(tup) for tup in padding])

    tmp: NDArray[Any] = np.pad(arr, padding, "edge")
    if dims == 1:
        if cell_spacing == "linear":
            return np.asarray(0.5 * (tmp[1:] + tmp[:-1]))
        else:
            return np.sqrt(tmp[1:] * tmp[:-1])
    elif dims == 2:
        if direction == 2:
            if cell_spacing == "linear":
                return np.asarray(0.5 * (tmp[1:] + tmp[:-1]))
            else:
                return np.sqrt(tmp[1:] * tmp[:-1])
        else:
            if cell_spacing == "linear":
                return np.asarray(0.5 * (tmp[:, 1:] + tmp[:, :-1]))
            else:
                return np.sqrt(tmp[:, 1:] * tmp[:, :-1])
    else:
        if direction == 3:
            if cell_spacing == "linear":
                return np.asarray(0.5 * (tmp[1:] + tmp[:-1]))
            else:
                return np.sqrt(tmp[1:] * tmp[:-1])
        elif direction == 2:
            if cell_spacing == "linear":
                return np.asarray(0.5 * (tmp[:, 1:] + tmp[:, :-1]))
            else:
                return np.sqrt(tmp[:, 1:] * tmp[:, :-1])
        else:
            if cell_spacing == "linear":
                return np.asarray(0.5 * (tmp[..., 1:] + tmp[..., :-1]))
            else:
                return np.sqrt(tmp[..., 1:] * tmp[..., :-1])


def calc_domega(*, x2: NDArray[Any], x3: NDArray[Any] | None = None) -> NDArray[Any]:
    x2v = calc_vertices(arr=x2, direction=1)
    dcos = np.cos(x2v[:-1]) - np.cos(x2v[1:])
    if x3:
        x3v = calc_vertices(arr=x3, direction=1)
        return np.asanyarray(dcos * (x3v[1:] - x3v[:-1]))

    return np.asanyarray(2.0 * np.pi * dcos)


def get_vertices(
    coords: NDArray[np.floating[Any]], is_radial: bool = False, axis: int = -1
) -> NDArray[np.floating[Any]]:
    """Calculate vertices from cell centers or return input if already vertices"""
    vertices: NDArray[np.floating[Any]]
    if is_radial:
        vertices = np.sqrt(coords[..., 1:] * coords[..., :-1])
    else:
        vertices = 0.5 * (coords[..., 1:] + coords[..., :-1])

    # Add boundary vertices
    vertices = np.insert(vertices, 0, coords[..., 0], axis=axis)
    vertices = np.insert(vertices, vertices.shape[axis], coords[..., -1], axis=axis)
    return vertices


def calc_volume_1d(
    x1v: NDArray[np.floating[Any]], coord_system: str
) -> NDArray[np.floating[Any]]:
    """Calculate 1D cell volumes from vertices"""
    dx1: NDArray[np.floating[Any]] = x1v[1:] - x1v[:-1]
    vol: NDArray[np.floating[Any]]

    if coord_system in ["spherical"]:
        x1mean: NDArray[np.floating[Any]] = np.sqrt(x1v[1:] * x1v[:-1])
        vol = 4.0 * np.pi * x1mean * x1mean * dx1
        return vol
    elif coord_system in ["cylindrical"]:
        x1mean = np.sqrt(x1v[1:] * x1v[:-1])
        vol = 2.0 * np.pi * x1mean * dx1
        return vol

    return dx1**3


def calc_volume_2d(
    x1v: NDArray[np.floating[Any]], x2v: NDArray[np.floating[Any]], coord_system: str
) -> NDArray[np.floating[Any]]:
    """Calculate 2D cell volumes from vertices

    Args:
        x1v: (n1+1,) vertex coordinates along first dimension
        x2v: (n2+1,) vertex coordinates along second dimension

    Returns:
        (n2, n1) array of cell volumes
    """
    # Create meshgrids from vertices
    x1vv: NDArray[np.floating[Any]]
    x1vv, _ = np.meshgrid(x1v, x2v, indexing="ij")

    if coord_system == "spherical":
        # dr³ has shape (n1, n2)
        dr3: NDArray[np.floating[Any]] = (x1vv[1:] ** 3 - x1vv[:-1] ** 3) / 3.0
        # dcos has shape (n2,)
        dcos: NDArray[np.floating[Any]] = np.cos(x2v[:-1]) - np.cos(x2v[1:])
        # Broadcast for multiplication
        return (2.0 * np.pi * dr3[:, :-1] * dcos[None, :]).T

    elif coord_system == "cartesian":
        dx: NDArray[np.floating[Any]] = x1v[1:] - x1v[:-1]  # shape (n1,)
        dy: NDArray[np.floating[Any]] = x2v[1:] - x2v[:-1]  # shape (n2,)
        # Use outer product for correct broadcasting
        return np.outer(dy, dx)

    elif coord_system == "cylindrical":
        dr2: NDArray[np.floating[Any]] = (
            x1v[1:] ** 2 - x1v[:-1] ** 2
        ) / 2.0  # shape (n1,)
        dphi: NDArray[np.floating[Any]] = x2v[1:] - x2v[:-1]  # shape (n2,)
        # Use outer product for correct broadcasting
        return np.outer(dphi, dr2)

    raise ValueError(f"Unsupported coordinate system: {coord_system}")


def calc_volume_3d(
    x1v: NDArray[np.floating[Any]],
    x2v: NDArray[np.floating[Any]],
    x3v: NDArray[np.floating[Any]],
    coord_system: str,
) -> NDArray[np.floating[Any]]:
    """Calculate 3D cell volumes from vertices"""
    dr3: NDArray[np.floating[Any]]
    dcos: NDArray[np.floating[Any]]
    dphi: NDArray[np.floating[Any]]
    dx: NDArray[np.floating[Any]]
    dy: NDArray[np.floating[Any]]
    dz: NDArray[np.floating[Any]]
    dr2: NDArray[np.floating[Any]]
    vol: NDArray[np.floating[Any]]

    if coord_system == "spherical":
        dr3 = (x1v[..., 1:] ** 3 - x1v[..., :-1] ** 3) / 3.0
        dcos = np.cos(x2v[:-1]) - np.cos(x2v[1:])
        dphi = x3v[1:] - x3v[:-1]
        vol = 0.5 * dr3 * dcos * dphi
        return vol
    elif coord_system == "cartesian":
        dx = x1v[..., 1:] - x1v[..., :-1]
        dy = x2v[1:] - x2v[:-1]
        dz = x3v[1:] - x3v[:-1]
        vol = dx * dy * dz
        return vol
    elif coord_system == "cylindrical":
        dr2 = (x1v[..., 1:] ** 2 - x1v[..., :-1] ** 2) / 2.0
        dphi = x2v[:, 1:] - x2v[:, :-1]
        dz = x3v[1:] - x3v[:-1]
        vol = dr2 * dphi * dz
        return vol
    else:
        raise ValueError(f"Unsupported coordinate system: {coord_system}")


def calc_cell_volume(
    coords: Sequence[NDArray[np.floating[Any]]],
    coord_system: str = "spherical",
    vertices: bool = False,
) -> NDArray[np.floating[Any]]:
    """Calculate cell volumes for 1D, 2D, or 3D grids"""
    ndim: int = len(coords)

    # Convert to vertices if needed
    if not vertices:
        is_radial: bool = coord_system in ["spherical", "cylindrical"]
        coords = [get_vertices(c, is_radial and i == 0) for i, c in enumerate(coords)]

    if ndim == 1:
        return calc_volume_1d(coords[0], coord_system)
    elif ndim == 2:
        return calc_volume_2d(coords[0], coords[1], coord_system)
    else:
        return calc_volume_3d(coords[0], coords[1], coords[2], coord_system)


def compute_num_polar_zones(
    *,
    rmin: Optional[Any] = None,
    rmax: Optional[Any] = None,
    nr: Optional[int] = None,
    zpd: Optional[int] = None,
    theta_bounds: tuple[float, float] = (0.0, np.pi),
) -> int:
    # Convert the values if None
    rmin = rmin or 1.0
    rmax = rmax or 1.0
    nr = nr or 1
    if zpd is not None:
        return int(round((theta_bounds[1] - theta_bounds[0]) * zpd / np.log(10)))
    elif None not in (rmin, rmax, nr):
        dlogr: float = np.log(rmax / rmin) / nr
        return int(round(1 + (theta_bounds[1] - theta_bounds[0]) / dlogr))
    else:
        raise ValueError("Please either specify zones per decade or rmin, rmax, and nr")


def calc_dlogt(tmin: float, tmax: float, ncheckpoints: int) -> float:
    if tmin == 0:
        return cast(float, np.log10(tmax / 1e-10) / (ncheckpoints - 1))
    return cast(float, np.log10(tmax / tmin) / (ncheckpoints - 1))


def progressbar(
    it: range, prefix: str = "", size: int = 100, out: TextIO = sys.stdout
) -> Generator[int, None, None]:
    count = len(it)

    def show(j: int) -> None:
        x = int(size * j / count)
        print(
            f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def print_progress() -> None:
    try:
        from rich.progress import track

        for _ in track(range(150), description="Loading..."):
            sleep(0.01)
    except ImportError:
        for _ in progressbar(range(150), "Loading: ", 60):
            sleep(0.01)


def find_nearest(arr: NDArray[Any], val: Any) -> Any:
    if arr.ndim > 1:
        ids = np.argmin(np.abs(arr - val), axis=1)
        return ids
    else:
        idx = np.argmin(np.abs(arr - val))
        return idx, arr[idx]


def to_iterable(x: Any, func: Callable[..., Sequence[Any]] = list) -> Sequence[Any]:
    if isinstance(x, (Sequence, np.ndarray)) and not isinstance(x, str):
        return func(x)
    else:
        return func((x,))


def to_tuple_of_tuples(x: Any) -> tuple[tuple[Any, ...], ...]:
    if not tuple_of_tuples(x):
        return (x,)
    return tuple(y for y in x)


def display_top(
    snapshot: tracemalloc.Snapshot, key_type: str = "lineno", limit: int = 3
) -> None:
    def format_size(size: float) -> str:
        if size >= 1e9:
            return f"{size / 1e9:.2f} GB"
        elif size >= 1e6:
            return f"{size / 1e6:.2f} MB"
        elif size >= 1e3:
            return f"{size / 1e3:.2f} KB"
        else:
            return f"{size:.2f} B"

    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    logger.info(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        logger.info(f"#{index}: {filename}:{frame.lineno}: {format_size(stat.size)}")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            logger.info(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logger.info(f"{len(other)} other: {format_size(size)}")
    total = sum(stat.size for stat in top_stats)
    logger.info(f"Total allocated size: {format_size(total)}")


def tuple_of_tuples(x: Any) -> bool:
    return all(isinstance(a, tuple) for a in x)


def expand_axis_if_needed(
    arr: NDArray[Any], axis: Optional[tuple[int]]
) -> NDArray[Any]:
    if axis is None:
        return arr
    return np.expand_dims(arr, axis=axis)


def order_of_mag(val: float) -> int:
    if val == 0:
        return 0
    return int(math.floor(math.log10(val)))
