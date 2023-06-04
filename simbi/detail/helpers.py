import numpy as np
import sys
from time import sleep
from ..key_types import *
from typing import TextIO, Generator


__all__ = [
    'calc_centroid',
    'calc_vertices',
    'calc_cell_volume1D',
    'calc_cell_volume2D',
    'calc_cell_volume3D',
    'for_each',
    'get_iterable',
    'compute_num_polar_zones',
    'calc_dlogt',
    'print_progress',
    'progressbar',
    'find_nearest',
    'pad_jagged_array',
]
generic_numpy_array = NDArray[Any]
def calc_centroid(arr: generic_numpy_array, coord_system: str = 'spherical') -> generic_numpy_array:
    if coord_system  == 'spherical':
        return np.asanyarray(0.75 * (arr[...,1:]**4 - arr[...,:-1] ** 4) / (arr[...,1:]**3 - arr[...,:-1]**3))
    elif coord_system == 'cylindrical':
        return np.asanyarray((2.0 / 3.0) * (arr[...,1:]**3 - arr[...,:-1] ** 3) / (arr[...,1:]**2 - arr[...,:-1]**2))
    else:
        return np.asanyarray(0.5 * (arr[...,1:] - arr[...,:-1]))
    

def calc_vertices(*, arr: generic_numpy_array, direction: int, cell_spacing: str = 'linear') -> Any:
    if direction not in [1,2,3]:
        raise ValueError("Direction must be either 1, 2, or 3")
    dims = arr.ndim
    padding: Any = [[0,0]] * dims
    padding[-direction] = [1,1]
    padding = tuple([tuple(tup) for tup in padding])
    
    tmp: generic_numpy_array = np.pad(arr, padding, 'edge')
    if dims == 1:
        if cell_spacing == 'linear':
            return np.asarray(0.5 * (tmp[1:] + tmp[:-1]))
        else:
            return np.sqrt(tmp[1:] * tmp[:-1])
    elif dims == 2:
        if direction == 2:
            if cell_spacing == 'linear':
                return np.asarray(0.5 * (tmp[1:] + tmp[:-1]))
            else:
                return np.sqrt(tmp[1:] * tmp[:-1])
        else:
            if cell_spacing == 'linear':
                return np.asarray(0.5 * (tmp[:, 1:] + tmp[:, :-1]))
            else:
                return np.sqrt(tmp[:, 1:] * tmp[:, :-1])
    else:
        if direction == 3:
            if cell_spacing == 'linear':
                return np.asarray(0.5 * (tmp[1:] + tmp[:-1]))
            else:
                return np.sqrt(tmp[1:] * tmp[:-1])
        elif direction == 2:
            if cell_spacing == 'linear':
                return np.asarray(0.5 * (tmp[:, 1:] + tmp[:, :-1]))
            else:
                return np.sqrt(tmp[:, 1:] * tmp[:, :-1])
        else:
            if cell_spacing == 'linear':
                return np.asarray(0.5 * (tmp[..., 1:] + tmp[...,:-1]))
            else:
                return np.sqrt(tmp[...,1:] * tmp[...,:-1])
    
            
def calc_cell_volume1D(*, x1: generic_numpy_array, coord_system: str = 'spherical') -> generic_numpy_array:
    if coord_system in ['spherical', 'cylindrical']:
        x1vertices = np.sqrt(x1[1:] * x1[:-1])
    else:
        x1vertices = 0.5 * (x1[1:] + x1[:-1])
        
    x1vertices = np.insert(x1vertices,  0, x1[0])
    x1vertices = np.insert(x1vertices, x1.shape, x1[-1])
    dx1 = x1vertices[1:] - x1vertices[:-1]
    if coord_system in ['spherical', 'cylindrical']:
        x1mean = calc_centroid(x1vertices, coord_system)
        return np.asanyarray(4.0 * np.pi * x1mean * x1mean * dx1)
    elif coord_system == 'cartesian':
        return np.asanyarray(dx1 ** 3) 
    else:
        raise ValueError("The coordinate system given is not avaiable at this time")

def calc_domega(*, x2: generic_numpy_array, x3: generic_numpy_array | None = None) -> generic_numpy_array:
    x2v = calc_vertices(arr=x2, direction=1)    
    dcos = np.cos(x2v[:-1]) - np.cos(x2v[1:])
    if x3:
        x3v = calc_vertices(arr=x3, direction=1)
        return np.asanyarray(dcos * (x3v[1:] - x3v[:-1]))
    
    return np.asanyarray(2.0 * np.pi * dcos)
    
def calc_cell_volume2D(*, x1: generic_numpy_array, x2: generic_numpy_array, coord_system: str = 'spherical') -> generic_numpy_array:
    if x1.ndim == 1 and x2.ndim == 1:
        xx1, xx2 = np.meshgrid(x1, x2)
    else:
        xx1, xx2 = x1, x2

    x2vertices = 0.5 * (xx2[1:] + xx2[:-1])
    x2vertices = np.insert(x2vertices, 0, xx2[0], axis=0)
    x2vertices = np.insert(x2vertices, x2vertices.shape[0], xx2[-1], axis=0)
    if coord_system == 'spherical':
        x1vertices = np.sqrt(xx1[...,  1:] * xx1[..., :-1])
        x1vertices = np.insert(x1vertices,  0, xx1[..., 0], axis=1)
        x1vertices = np.insert(x1vertices, x1vertices.shape[1], xx1[..., -1],axis=1)
        
        dcos = np.asanyarray(np.cos(x2vertices[:-1]) - np.cos(x2vertices[1:]))
        return np.asanyarray((1./3.) * (x1vertices[..., 1:]**3 - x1vertices[..., :-1]**3) * dcos * 2.0 * np.pi)
    elif coord_system == 'axis_cylindrical':
        x1vertices = 0.5 * (xx1[...,  1:] + xx1[..., :-1])
        x1vertices = np.insert(x1vertices,  0, xx1[..., 0],axis=1)
        x1vertices = np.insert(x1vertices, x1vertices.shape[1], xx1[..., -1],axis=1)
        
        dz   = x2vertices[1:] - x2vertices[:-1]
        dr   = x1vertices[:, 1:] - x1vertices[:, :-1]
        rmean = (2.0 / 3.0) * (x1vertices[...,1:]**3 - x1vertices[...,:-1]**3) / (x1vertices[...,1:]**2 - x1vertices[...,:-1]**2)
        return np.asanyarray(2.0 * np.pi * rmean * dr * dz)
    elif coord_system == 'planar_cylindrical':
        x1vertices = np.sqrt(xx1[...,  1:] * xx1[..., :-1])
        x1vertices = np.insert(x1vertices,  0, xx1[..., 0],axis=1)
        x1vertices = np.insert(x1vertices, x1vertices.shape[1], xx1[..., -1],axis=1)
        
        dphi = x2vertices[1:] - x2vertices[:-1]
        dr   = x1vertices[:, 1:] - x1vertices[:, :-1]
        rmean = (2.0 / 3.0) * (x1vertices[...,1:]**3 - x1vertices[...,:-1]**3) / (x1vertices[...,1:]**2 - x1vertices[...,:-1]**2)
        return np.asanyarray(rmean * dr * dphi)
    elif coord_system == 'cartesian':
        x1vertices = 0.5 * (xx1[...,  1:] + xx1[..., :-1])
        x1vertices = np.insert(x1vertices,  0, xx1[..., 0],axis=1)
        x1vertices = np.insert(x1vertices, x1vertices.shape[1], xx1[..., -1],axis=1)
        dy   = x2vertices[1:] - x2vertices[:-1]
        dx   = x1vertices[:, 1:] - x1vertices[:, :-1]
        return np.asanyarray(dx * dy)
    else:
        raise ValueError("The coordinate system given is not avaiable at this time")


def calc_cell_volume3D(*, x1: generic_numpy_array, x2: generic_numpy_array, x3: generic_numpy_array, coord_system: str = 'spherical') -> generic_numpy_array:
    if x1.ndim == 1 and x2.ndim == 1 and x3.ndim == 1:
        xx2, xx3, xx1 = np.meshgrid(x2, x3, x1)
    else:
        xx1, xx2, xx3 = x1, x2, x3

    x3vertices = 0.5 * (xx3[1:] + xx3[:-1])
    x3vertices = np.insert(x3vertices, 0, xx3[0], axis=0)
    x3vertices = np.insert(x3vertices, x3vertices.shape[0], xx3[-1], axis=0)
    
    x2vertices = 0.5 * (xx2[:, 1:] + xx2[:, :-1])
    x2vertices = np.insert(x2vertices, 0, xx2[0], axis=1)
    x2vertices = np.insert(x2vertices, x2vertices.shape[1], xx2[:, -1], axis=1)
    if coord_system == 'spherical':
        x1vertices = np.sqrt(xx1[...,  1:] * xx1[..., :-1])
        x1vertices = np.insert(x1vertices,  0, xx1[:, :, 0], axis=2)
        x1vertices = np.insert(x1vertices, x1vertices.shape[2], xx1[:, :, -1], axis=2)
        
        dphi = x3vertices[1:] - x3vertices[:-1]
        dcos = np.cos(x2vertices[:, :-1]) - np.cos(x2vertices[:, 1:])
        return np.asanyarray((1./3.) * (x1vertices[..., 1:]**3 - x1vertices[..., :-1]**3) * dcos * dphi)
    elif coord_system == 'cylindrical':
        x1vertices = 0.5 * (xx1[...,  1:] + xx1[..., :-1])
        x1vertices = np.insert(x1vertices,  0, xx1[..., 0], axis=2)
        x1vertices = np.insert(x1vertices, x1vertices.shape[2], xx1[..., -1], axis=2)
        dz   = x3vertices[1:] - x3vertices[:-1]
        dphi = x2vertices[:, 1:] - x2vertices[:, :-1]
        return np.asanyarray(0.5 * (x1vertices[...,1:]**2 - x1vertices[...,:-1]**2) * dphi * dz)
    elif coord_system == 'cartesian':
        x1vertices = 0.5 * (xx1[...,  1:] + xx1[..., :-1])
        x1vertices = np.insert(x1vertices,  0, xx1[..., 0], axis=2)
        x1vertices = np.insert(x1vertices, x1vertices.shape[2], xx1[..., -1], axis=2)
        dx   = x1vertices[...,1:] - x1vertices[...,:-1]
        dz   = x3vertices[1:] - x3vertices[:-1]
        dy   = x2vertices[:, 1:] - x2vertices[:, :-1]
        
        return np.asanyarray(dx * dy * dz)
    else:
        raise ValueError("The coordinate system given is not avaiable at this time")
        


def compute_num_polar_zones(*,
                            rmin: Optional[Any] = None,
                            rmax: Optional[Any] = None,
                            nr:   Optional[int] = None,
                            zpd:  Optional[int] = None,
                            theta_bounds: tuple[float, float] = (0.0, np.pi)) -> int:
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
    try:
        from rich.progress import track
        for _ in track(range(100), description="Loading..."):
            sleep(0.01)
    except ImportError:
        for _ in progressbar(range(100), "Loading: ", 60):
            sleep(0.01)


def pad_jagged_array(arr: Union[NDArray[Any], Sequence[Any]]) -> NDArray[Any]:
    arr       = [np.array(val) if isinstance(val, (Sequence, np.ndarray)) else np.array([val]) for val in arr]
    max_dim   = max(a.ndim for a in arr)
    max_size  = np.max([a.size for a in arr if a.ndim == max_dim])
    max_shape = [a.shape for a in arr if a.size == max_size][0]
    arr       = np.array([a if a.shape == max_shape else np.ones(max_shape) * a for a in arr])
    return arr 

def find_nearest(arr: NDArray[Any], val: Any) -> Any:
    if arr.ndim > 1:
        ids = np.argmin(np.abs(arr - val), axis=1)
        return ids
    else:
        idx = np.argmin(np.abs(arr - val))
        return idx, arr[idx]
    
def for_each(func: Callable[..., Any], x: Any) -> None:
    for i in x:
        func(i)
        
def get_iterable(x: Any) -> Sequence[Any]:
    if isinstance(x,  Sequence) and not isinstance(x, str):
        return x
    else:
        return (x,)