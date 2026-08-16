# =============================================================================
# test_field_loop_flat_uct.py
#
# the flat (Minkowski, cartesian) constrained-transport gate, covering the flat cartesian 2.5D
# UCT/contact kernels (nmhd here) that no GR fixture reaches. the UCT failure modes it pins are
# anti-diffusive HLLD, advective upwind-pairing, and transverse reconstruction.
#
# vehicle: the Gardiner & Stone (2005) magnetic field-loop advection — a weak (passive, beta >> 1)
# loop advected diagonally across a periodic box. two properties must hold for every ct_method/solver:
#   - div(B) stays at machine zero (the CT invariant), and
#   - the run is stable — the loop advects without amplifying. a broken EMF blows the loop up: the
#     upwind-pairing failure drives gas pressure 1 -> 29. the velocity is supersonic
#     (|v| = sqrt5, the paper's v=(2,1)) by design — the EMF-pairing failures are invisible
#     subsonic (Orszag-Tang) and only fire supersonically.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi.types import CtMethod, Solver

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

from simbi_configs.examples.newtonian.field_loop import FieldLoop

_NX, _NY = 32, 16


def _load(path: str):
    base = "level_0/partition_0/hydro/"
    with h5py.File(path) as h:
        b1 = h[base + "magnetic/B1/data"][:]  # x-faces: (ny, nx+1)
        b2 = h[base + "magnetic/B2/data"][:]  # y-faces: (ny+1, nx)
        g = h[base + "primitives"]
        shp = g["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (_NY, _NX))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (_NY, _NX)))
        rho = g["rho"][sl]
        pre = g["pre"][sl]
    return b1, b2, rho, pre


def _div_b_max(b1, b2, dx, dy) -> float:
    # discrete cartesian divergence per cell from the staggered faces: it is identically zero for a
    # constrained-transport update, so any nonzero value is a CT bug. normalized by |B|/dx.
    d = (b1[:, 1:] - b1[:, :-1]) / dx + (b2[1:, :] - b2[:-1, :]) / dy
    scale = max(float(np.abs(b1).max()), float(np.abs(b2).max())) / min(dx, dy) + 1e-30
    return float(np.abs(d).max()) / scale


@needs_backend
@pytest.mark.parametrize(
    "ct,solver",
    [
        (CtMethod.CONTACT, Solver.HLLD),
        (CtMethod.UCT, Solver.HLLD),
        (CtMethod.UCT, Solver.HLLE),
    ],
)
def test_flat_field_loop_preserves_divergence_and_is_stable(ct, solver) -> None:
    d = tempfile.mkdtemp() + "/"
    p = FieldLoop.from_cli([])
    p.resolution = (_NX, _NY, 1)
    p.ct_method = ct
    p.solver = solver
    p.end_time = 1.0  # supersonic advection: ~1 box-crossing, enough for a broken EMF to blow up
    p.data_directory = d
    p.checkpoint_interval = 100.0
    runner.run(p, compute_mode="cpu", max_steps=400)

    tag = f"ct={ct}, solver={solver}"
    assert not glob.glob(os.path.join(d, "*crashed*.h5")), f"field loop crashed at {tag}"
    (xlo, xhi), (ylo, yhi) = p.bounds[0], p.bounds[1]
    dx, dy = (xhi - xlo) / _NX, (yhi - ylo) / _NY

    ib1, ib2, irho, ipre = _load(sorted(glob.glob(os.path.join(d, "*000_000*.h5")))[0])
    fb1, fb2, frho, fpre = _load(glob.glob(os.path.join(d, "*final*.h5"))[0])

    # CT invariant: div(B) at machine zero, both at t=0 and after the advection.
    assert _div_b_max(ib1, ib2, dx, dy) < 1e-10, f"initial div(B) nonzero at {tag}"
    assert _div_b_max(fb1, fb2, dx, dy) < 1e-10, f"div(B) broke to nonzero at {tag}"

    # stability: no NaN, pressure stays positive and does not blow up (a broken supersonic EMF
    # amplifies the loop and heats the gas, measured pre 1 -> 29), and the field does not amplify.
    assert not np.isnan(frho).any() and not np.isnan(fpre).any(), f"NaN at {tag}"
    assert fpre.min() > 0.0, f"pressure went non-positive at {tag}"
    assert fpre.max() < 3.0 * float(ipre.max()), (
        f"pressure blew up at {tag}: {fpre.max():.3e} vs initial {float(ipre.max()):.3e} "
        f"(supersonic EMF amplification)"
    )
    b_ref = max(float(np.abs(ib1).max()), float(np.abs(ib2).max()))
    b_fin = max(float(np.abs(fb1).max()), float(np.abs(fb2).max()))
    assert b_fin < 3.0 * b_ref, f"field amplified at {tag}: {b_fin:.3e} vs initial {b_ref:.3e}"
