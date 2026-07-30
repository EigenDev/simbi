# =============================================================================
# test_derived_field_broadcast.py
#
# derived fields that combine 1d coordinate arrays with storage-ordered
# (x3, x2, x1) field arrays must reshape each coordinate to broadcast along its
# own storage axis. a bare 1d array broadcasts against the last axis only:
# correct for x1, but it transposes x2/x3 against the data — silent on a square
# grid, a shape error on a non-square one.
# =============================================================================

from types import SimpleNamespace

import numpy as np

from simbi.reader.computation import _broadcast_cell_centers


def _mesh(nx: int, ny: int | None = None, nz: int | None = None) -> SimpleNamespace:
    m = SimpleNamespace()
    m.x1v = np.linspace(0.0, 1.0, nx + 1)
    if ny is not None:
        m.x2v = np.linspace(0.0, 2.0, ny + 1)
    if nz is not None:
        m.x3v = np.linspace(0.0, 3.0, nz + 1)
    return m


def test_2d_coords_carry_singleton_storage_axes() -> None:
    nx, ny = 5, 3
    x, y = _broadcast_cell_centers(_mesh(nx, ny), 2)
    assert x.shape == (1, nx)
    assert y.shape == (ny, 1)


def test_2d_coords_broadcast_against_nonsquare_field() -> None:
    # a bare (ny,) coordinate against a (ny, nx) field raises when nx != ny;
    # the storage-shaped coordinates must broadcast cleanly.
    nx, ny = 5, 3
    x, y = _broadcast_cell_centers(_mesh(nx, ny), 2)
    field = np.zeros((ny, nx))
    assert (x * field).shape == (ny, nx)
    assert (y * field).shape == (ny, nx)


def test_3d_coords_carry_singleton_storage_axes() -> None:
    nx, ny, nz = 5, 3, 2
    x, y, z = _broadcast_cell_centers(_mesh(nx, ny, nz), 3)
    assert x.shape == (1, 1, nx)
    assert y.shape == (1, ny, 1)
    assert z.shape == (nz, 1, 1)
    field = np.zeros((nz, ny, nx))
    for coord in (x, y, z):
        assert (coord * field).shape == (nz, ny, nx)


def test_uniform_x_flow_radial_velocity_signs_with_x() -> None:
    # replicate mass_flux's core: radial velocity of a uniform +x flow is +x/r,
    # so its sign must follow x at every cell — regardless of the row (y). the
    # pre-fix bare-coordinate version raised on this non-square grid; a square-
    # grid version would have transposed the sign map onto y silently.
    nx, ny = 5, 3
    x, y = _broadcast_cell_centers(_mesh(nx, ny), 2)
    vx = np.ones((ny, nx))
    vy = np.zeros((ny, nx))
    r = np.sqrt(x**2 + y**2)
    vr = (x * vx + y * vy) / (r + np.finfo(float).tiny)
    assert vr.shape == (ny, nx)
    expected_sign = np.sign(np.broadcast_to(x, (ny, nx)))
    assert np.array_equal(np.sign(vr), expected_sign)


def test_1d_returns_bare_coordinate() -> None:
    (x,) = _broadcast_cell_centers(_mesh(4), 1)
    assert x.shape == (4,)
