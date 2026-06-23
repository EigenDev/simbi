# =============================================================================
# test_ot_bfield_symmetry.py
#
# regression: the staggered orszag-tang B-field IC must sample its TRANSVERSE
# coordinate at the cell center, not the cell edge. Bx = -b0 sin(2 pi y) lives on
# the x-face; its y must be (jj+0.5)*dy so the discrete field is antisymmetric
# about the domain center (exact 180-degree point symmetry) and aligned with the
# cell-centered velocity + the rust reference. edge sampling (jj*dy) put Bx[0]=0
# and broke the symmetry.
# =============================================================================
import math

import numpy as np
import pytest

from simbi_configs.examples.imhd_orszag_tang import IsothermalOrszagTang
from simbi_configs.examples.nmhd_orszag_tang import NewtonianOrszagTang
from simbi_configs.examples.orszag_tang import OrszagTang

_N = 8  # small even grid; bounds default to [0,1]^2 so dy = 1/N


@pytest.mark.parametrize("cls", [OrszagTang, NewtonianOrszagTang, IsothermalOrszagTang])
def test_bx_transverse_is_cell_centered_and_symmetric(cls) -> None:
    prob = cls(resolution=(_N, _N, 1))
    _, bx_gen, _by, _bz = prob.initial_primitive_state()
    b0 = prob.b0
    dy = 1.0 / _N

    # bx is emitted in (kk, jj, ii) order, ii fastest; shape (nk=1, nj=N, ni+1=N+1).
    bx = np.array(list(bx_gen())).reshape(_N, _N + 1)

    # bx depends only on y (the transverse axis) -> constant along x (each row).
    assert np.allclose(bx, bx[:, :1]), "Bx must be constant along x (it depends only on y)"

    col = bx[:, 0]
    # cell-CENTER sampling: Bx[jj] = -b0 sin(2 pi (jj+0.5) dy), NOT -b0 sin(2 pi jj dy).
    expected = np.array(
        [-b0 * math.sin(2.0 * math.pi * (jj + 0.5) * dy) for jj in range(_N)]
    )
    assert np.allclose(col, expected, atol=1e-12), "Bx not sampled at the cell center"
    # the edge value would be exactly zero at jj=0; the center value is not.
    assert abs(col[0]) > 1e-6, "Bx[0]==0 indicates edge (jj*dy) sampling regression"
    # exact 180-degree point symmetry: Bx(y_j) == -Bx(y_{N-1-j}).
    assert np.max(np.abs(col + col[::-1])) < 1e-12, "Bx is not antisymmetric about y=0.5"


@pytest.mark.parametrize("cls", [OrszagTang, NewtonianOrszagTang, IsothermalOrszagTang])
def test_by_transverse_is_cell_centered_and_symmetric(cls) -> None:
    prob = cls(resolution=(_N, _N, 1))
    _, _bx, by_gen, _bz = prob.initial_primitive_state()
    b0 = prob.b0
    dx = 1.0 / _N

    # by shape (nk=1, nj+1=N+1, ni=N); depends only on x -> constant along y.
    by = np.array(list(by_gen())).reshape(_N + 1, _N)
    assert np.allclose(by, by[:1, :]), "By must be constant along y (it depends only on x)"

    row = by[0, :]
    expected = np.array(
        [b0 * math.sin(4.0 * math.pi * (ii + 0.5) * dx) for ii in range(_N)]
    )
    assert np.allclose(row, expected, atol=1e-12), "By not sampled at the cell center"
