from ..detail.helpers import (
    get_iterable,
    calc_cell_volume1D,
    calc_cell_volume2D,
    calc_cell_volume3D,
    calc_domega,
    find_nearest,
    calc_any_mean,
)

def calc_cell_volumes(mesh: dict['str'], ndim):
    """Calculate cell volumes based on dimension"""
    if ndim == 1:
        return calc_cell_volume1D(mesh['x1v'])
    elif ndim == 2:
        return calc_cell_volume2D(mesh['x1v'], mesh['x2v'])
    elif ndim == 3:
        return calc_cell_volume3D(mesh['x1v'], mesh['x2v'], mesh['x3v'])
    else:
        raise ValueError("ndim must be 1, 2, or 3")