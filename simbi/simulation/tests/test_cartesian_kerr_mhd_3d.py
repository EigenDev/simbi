# =============================================================================
# test_cartesian_kerr_mhd_3d.py
#
# GRMHD on the FULL 3D cartesian SPINNING-KERR chart. the seeded field is the
# vertical densitized-divergence-free B_z = B0 / w with w = sqrt(gamma) =
# sqrt(1 + 2H |l|^2) at the face (the spinning rank-1 metric's own weight), so
# d_i(sqrt(gamma) B^i) = 0 on the staggered mesh to machine precision. gates:
# - the constrained transport preserves the DENSITIZED divergence at spin:
#   recomputed from the evolved staggered field with the KERNEL's clamped
#   weights, it stays at machine zero;
# - the spinning metric and the vertical field are invariant under the quarter
#   turn about the spin axis, so the accreting gas stays rot90-symmetric to
#   roundoff (the coordinate-role gate for the kerr CT chain);
# - a = 0 reduces to the kerr_schild chart outside the frozen clamped core
#   (the two metrics continue the core differently), decaying outward.
# =============================================================================
import glob
import os
import tempfile
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest

from simbi import ProblemParam, SimbiProblem
from simbi.simulation import runner
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

RES = 32
RHO0 = 1.0
PRE0 = 0.1
BZ0 = 0.05
MASS = 1.0
SPIN = 0.9
L = 5.0


def _w(x: float, y: float, z: float, a: float) -> float:
    # the kernel's sqrt(gamma) = sqrt(1 + 2H |l|^2) with the CLAMPED (r >= M/2)
    # oblate-spheroidal kerr-schild radius — the exact face weight the CT divides by.
    rr2 = x * x + y * y + z * z
    d = 0.5 * (rr2 - a * a)
    r = max(np.sqrt(max(d + np.sqrt(d * d + (a * z) ** 2), 0.0)), 0.5 * MASS)
    rr = r * r
    az = a * z
    two_h = 2.0 * MASS * rr * r / (rr * rr + az * az)
    den = 1.0 / (rr + a * a)
    lx = (r * x + a * y) * den
    ly = (r * y - a * x) * den
    lz = z / r
    return float(np.sqrt(1.0 + two_h * (lx * lx + ly * ly + lz * lz)))


class _CartesianKerrMhd3D(SimbiProblem):
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0)]
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.KERR)]
    schwarzschild_mass: Annotated[float, ProblemParam(MASS)]
    kerr_spin: Annotated[float, ProblemParam(SPIN)]
    regime: Annotated[Regime, ProblemParam(Regime.RMHD)]
    resolution: Annotated[tuple[int, int, int], ProblemParam((RES, RES, RES))]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-L, L), (-L, L), (-L, L)]),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN)]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE)]
    boundary_conditions: Annotated[
        BoundaryCondition, ProblemParam(BoundaryCondition.OUTFLOW)
    ]
    cfl_number: Annotated[float, ProblemParam(0.3)]
    end_time: Annotated[float, ProblemParam(1.0, checkpoint_safe=True)]
    checkpoint_interval: Annotated[
        float, ProblemParam(1.0e30, cli=True, checkpoint_safe=True)
    ]
    data_directory: Annotated[
        Path, ProblemParam(Path("data/_test"), cli=True, checkpoint_safe=True)
    ]

    def initial_primitive_state(self) -> InitialStateType:
        nx, ny, nz = self.resolution
        d = 2.0 * L / RES
        xc = lambda i: -L + (i + 0.5) * d
        zf = lambda k: -L + k * d
        a = self.kerr_spin

        def gas_state() -> GasStateGenerator:
            for _ in range(nx * ny * nz):
                yield (RHO0, 0.0, 0.0, 0.0, PRE0)

        def b_field(bn: str):
            def gen():
                if bn == "b1":
                    for _ in range((nx + 1) * ny * nz):
                        yield 0.0
                elif bn == "b2":
                    for _ in range(nx * (ny + 1) * nz):
                        yield 0.0
                else:
                    for k in range(nz + 1):
                        for j in range(ny):
                            for i in range(nx):
                                yield BZ0 / _w(xc(i), xc(j), zf(k), a)

            return gen

        return (gas_state, b_field("b1"), b_field("b2"), b_field("b3"))


def _run(spin: float, spacetime: Spacetime = Spacetime.KERR) -> dict[str, np.ndarray]:
    d = tempfile.mkdtemp() + "/"
    p = _CartesianKerrMhd3D(data_directory=Path(d))
    p.kerr_spin = spin
    p.spacetime = spacetime
    runner.run(p, compute_mode="cpu")
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"3d cartesian kerr GRMHD run (a={spin}) crashed"
    out = {}
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        for nm in ("rho", "pre", "b1", "b2"):
            out[nm] = prims[nm][sl, sl, sl]
        mag = h["level_0/partition_0/hydro/magnetic"]
        for k in (1, 2, 3):
            out[f"B{k}"] = mag[f"B{k}"]["data"][:]
    return out


@needs_backend
def test_cartesian_kerr_mhd_3d_divb_and_quarter_turn() -> None:
    f = _run(SPIN)
    rho = f["rho"]
    assert np.isfinite(rho).all(), "non-finite state"
    assert rho.min() > 0.0, "density went non-positive"
    assert rho.max() > 1.05 * RHO0, f"no accretion (max rho {rho.max():.3f})"
    assert np.abs(f["b1"]).max() > 1e-6, "B_x never developed; the CT never acted"

    # the quarter turn about the spin axis is exact at ANY spin for the scalar
    # density (the vertical seed and the metric share the symmetry).
    err = np.abs(rho - np.rot90(rho, 1, (1, 2))).max()
    assert err < 1e-10, f"quarter-turn symmetry broken at spin: {err:e}"

    # the defining CT invariant at spin: the densitized divergence recomputed from
    # the EVOLVED staggered faces with the kernel's clamped kerr weights stays at
    # machine zero relative to the field scale.
    dd = 2.0 * L / RES
    xs = (np.arange(RES) + 0.5) * dd - L
    xs_f = np.arange(RES + 1) * dd - L
    b1, b2, b3 = f["B1"], f["B2"], f["B3"]
    assert b1.shape == (RES, RES, RES + 1), f"B1 shape {b1.shape}"

    def wgrid(xg, yg, zg):
        out = np.empty_like(xg)
        it = np.nditer([xg, yg, zg], flags=["multi_index"])
        for xv, yv, zv in it:
            out[it.multi_index] = _w(float(xv), float(yv), float(zv), SPIN)
        return out

    zc, yc, xf = np.meshgrid(xs, xs, xs_f, indexing="ij")
    w1 = wgrid(xf, yc, zc)
    zc, yf, xc2 = np.meshgrid(xs, xs_f, xs, indexing="ij")
    w2 = wgrid(xc2, yf, zc)
    zff, yc3, xc3 = np.meshgrid(xs_f, xs, xs, indexing="ij")
    w3 = wgrid(xc3, yc3, zff)
    div = (
        (w1[:, :, 1:] * b1[:, :, 1:] - w1[:, :, :-1] * b1[:, :, :-1])
        + (w2[:, 1:, :] * b2[:, 1:, :] - w2[:, :-1, :] * b2[:, :-1, :])
        + (w3[1:, :, :] * b3[1:, :, :] - w3[:-1, :, :] * b3[:-1, :, :])
    ) / dd
    dmax = float(np.abs(div).max())
    bmax = max(float(np.abs(b1).max()), float(np.abs(b3).max()), 1e-300)
    assert dmax < 1e-12 * max(1.0, bmax / 1e-3), (
        f"densitized div(B) not preserved at spin: {dmax:e}"
    )


@needs_backend
def test_cartesian_kerr_mhd_a0_matches_kerr_schild_exterior() -> None:
    a = _run(0.0)
    b = _run(0.0, spacetime=Spacetime.KERR_SCHILD)
    assert a["rho"].max() > 1.05 * RHO0, "no accretion; the comparison is vacuous"
    dd = 2.0 * L / RES
    xs = (np.arange(RES) + 0.5) * dd - L
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    # the a = 0 kerr metric equals the kerr_schild chart outside the clamped core
    # (the frozen cores are DIFFERENT consistent continuations); the stiff
    # magnetized c2p amplifies the core difference more than the hydro chain, so
    # the exterior band carries a looser bound than the hydro gate, still decaying.
    mid = (r > 0.75) & (r <= 2.0)
    ext = r > 2.0
    d_mid = np.abs(a["rho"] - b["rho"])[mid].max()
    d_ext = np.abs(a["rho"] - b["rho"])[ext].max()
    assert d_mid < 5e-2, f"a=0 kerr vs kerr_schild strong-field mismatch: {d_mid:e}"
    assert d_ext < 1e-3, f"a=0 kerr vs kerr_schild exterior mismatch: {d_ext:e}"
    assert d_ext < 0.5 * max(d_mid, 1e-300), (
        f"core disagreement does not decay outward (mid {d_mid:e}, ext {d_ext:e})"
    )
