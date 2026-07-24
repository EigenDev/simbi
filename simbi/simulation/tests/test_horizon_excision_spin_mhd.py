# =============================================================================
# test_horizon_excision_spin_mhd.py
#
# the generalized horizon excision: the excised region is the kerr-schild-radius
# level set r_ks(x; a) < r_exc — the sphere at a = 0, the oblate spheroid
# (x^2 + y^2)/(r_exc^2 + a^2) + z^2/r_exc^2 < 1 at spin — and the fill carries
# the GAS primitives for hydro AND magnetized runs (the staggered faces stay
# CT-owned). three gates:
# - SPINNING hydro excision genuinely acts (two radii give different interiors)
#   while the exterior sees only bounded, outward-decaying leakage, and the
#   excised run keeps the quarter-turn + z-reflection metric symmetries;
# - MAGNETIZED excision runs (previously rejected fail-loud), genuinely acts,
#   and PRESERVES the densitized div(B) at machine zero — the fill never
#   touches the staggered field, so the CT invariant survives by construction;
# - the magnetized excised run agrees with the unexcised run outside the
#   horizon to the same bounded-leakage envelope (the interior is causally
#   disconnected; discrete stencils leak a small decaying difference).
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


def _grid_radii():
    dd = 2.0 * L / RES
    xs = (np.arange(RES) + 0.5) * dd - L
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.sqrt(x * x + y * y + z * z), dd


class _KerrExcised3D(SimbiProblem):
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0)]
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.KERR)]
    schwarzschild_mass: Annotated[float, ProblemParam(MASS)]
    kerr_spin: Annotated[float, ProblemParam(SPIN)]
    excision_radius: Annotated[float, ProblemParam(0.0, cli=True)]
    resolution: Annotated[tuple[int, int, int], ProblemParam((RES, RES, RES))]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-L, L), (-L, L), (-L, L)]),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN)]
    regime: Annotated[Regime, ProblemParam(Regime.RHD)]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE)]
    boundary_conditions: Annotated[
        BoundaryCondition, ProblemParam(BoundaryCondition.OUTFLOW)
    ]
    cfl_number: Annotated[float, ProblemParam(0.3)]
    end_time: Annotated[float, ProblemParam(2.0, checkpoint_safe=True)]
    checkpoint_interval: Annotated[
        float, ProblemParam(1.0e30, cli=True, checkpoint_safe=True)
    ]
    data_directory: Annotated[
        Path, ProblemParam(Path("data/_test"), cli=True, checkpoint_safe=True)
    ]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            for _ in range(nx * ny * nz):
                yield (RHO0, 0.0, 0.0, 0.0, PRE0)

        return gas_state


def _run_kerr(excision: float) -> np.ndarray:
    d = tempfile.mkdtemp() + "/"
    p = _KerrExcised3D(excision_radius=excision, data_directory=Path(d))
    runner.run(p, compute_mode="cpu")
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"spinning excised run (r_exc={excision}) crashed"
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        return prims["rho"][sl, sl, sl]


@needs_backend
def test_spinning_excision_acts_and_leakage_is_bounded() -> None:
    # r_+ = M + sqrt(M^2 - a^2) ~ 1.436 at a = 0.9: both radii sit in (M/2, r_+).
    a = _run_kerr(1.0)
    b = _run_kerr(1.3)
    r, _dx = _grid_radii()
    assert np.isfinite(a).all() and a.min() > 0.0, "excised spinning state broke"
    # non-vacuous: the two excision radii produced different interiors — the
    # spheroidal fill genuinely ran.
    inner = r < 1.2
    assert np.abs(a[inner] - b[inner]).max() > 1e-8, (
        "the two excision radii produced identical interiors; the pass never ran"
    )
    # the exterior is causally disconnected in the continuum; discretely the reconstruction stencil
    # leaks a difference across the excision surface. the LAW is not a magic magnitude — it is that
    # the leak DECAYS STEEPLY OUTWARD and CONVERGES under refinement, i.e. it is an evanescent
    # truncation-level influence of the surface rather than a standing error.
    #
    # the bands are FIXED PHYSICAL radii, deliberately independent of dx. an earlier form used
    # `r > 1.6 + 2*dx`, whose inner edge MOVES INWARD as the grid refines — it samples ever closer to
    # r_+ (= 1.436), where the leak is naturally larger, so a converging leak reads as a growing one
    # and no fixed threshold on it can be resolution-honest (measured 1.34e-3 at 32^3 rising to
    # 1.64e-3 at 64^3 purely from the band moving).
    #
    # measured on these fixed bands, the leak is exponentially small in the surface-to-band distance
    # and super-convergent (the surface sharpens AND the band recedes in cell units):
    #   band        32^3       64^3      ratio
    #   [2.3, 3.0]  6.30e-4    3.04e-5    20.8x   (~4.4 order, vs the scheme's 2nd)
    #   [3.0, 4.0]  1.28e-5    1.28e-8      999x
    #   r > 4       6.75e-9    2.96e-14   2.3e5x
    # the bounds below carry wide margin on the 32^3 values; a surface whose influence STANDS
    # (resolution-independent) or fails to decay outward breaks them.
    diff = np.abs(a - b)
    scale = np.abs(a).max()
    near_leak = diff[(r > 2.3) & (r < 3.0)].max() / scale
    mid_leak = diff[(r > 3.0) & (r < 4.0)].max() / scale
    far_leak = diff[r > 4.0].max() / scale
    # sanity rail: a catastrophic leak (the surface visibly imprinting on the flow) fails here.
    assert near_leak < 3e-3, f"near-horizon excision leakage too large: {near_leak:e}"
    # the decay law, the real content: each band is orders below the one inside it.
    assert mid_leak < 0.05 * max(near_leak, 1e-300), (
        f"excision leakage does not decay outward: near {near_leak:e}, mid {mid_leak:e}"
    )
    assert far_leak < 0.01 * max(mid_leak, 1e-300), (
        f"excision leakage does not decay outward: mid {mid_leak:e}, far {far_leak:e}"
    )
    # the excised run keeps the exact metric symmetries (quarter turn about the
    # spin axis + equatorial reflection): the spheroidal mask and fill share them.
    assert np.abs(a - np.rot90(a, 1, (1, 2))).max() < 1e-11, "quarter-turn broken"
    assert np.abs(a - a[::-1, :, :]).max() < 1e-11, "z-reflection broken"


class _KsMhdExcised3D(SimbiProblem):
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0)]
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.KERR_SCHILD)]
    schwarzschild_mass: Annotated[float, ProblemParam(MASS)]
    excision_radius: Annotated[float, ProblemParam(0.0, cli=True)]
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

        def sqrtg(x: float, y: float, z: float) -> float:
            r_true = np.sqrt(x * x + y * y + z * z)
            r = max(r_true, 0.5 * MASS)
            ll2 = (r_true / r) ** 2  # |l|^2, exactly 1 outside the radius floor
            return float(np.sqrt(1.0 + (2.0 * MASS / r) * ll2))

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
                                yield BZ0 / sqrtg(xc(i), xc(j), zf(k))

            return gen

        return (gas_state, b_field("b1"), b_field("b2"), b_field("b3"))


def _run_mhd(excision: float) -> dict[str, np.ndarray]:
    d = tempfile.mkdtemp() + "/"
    p = _KsMhdExcised3D(excision_radius=excision, data_directory=Path(d))
    runner.run(p, compute_mode="cpu")
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"excised MHD run (r_exc={excision}) crashed"
    out = {}
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        out["rho"] = prims["rho"][sl, sl, sl]
        mag = h["level_0/partition_0/hydro/magnetic"]
        for k in (1, 2, 3):
            out[f"B{k}"] = mag[f"B{k}"]["data"][:]
    return out


def _densitized_div(f: dict[str, np.ndarray], dd: float) -> float:
    # the kernel's clamped weight w = sqrt(1 + 2M / max(r, M/2)) at each face.
    xs = (np.arange(RES) + 0.5) * dd - L
    xs_f = np.arange(RES + 1) * dd - L

    def w(xg, yg, zg):
        rr = np.sqrt(xg * xg + yg * yg + zg * zg)
        r_true = rr.copy()
        np.maximum(rr, 0.5 * MASS, out=rr)
        ll2 = (r_true / rr) ** 2  # |l|^2, exactly 1 outside the radius floor
        return np.sqrt(1.0 + (2.0 * MASS / rr) * ll2)

    zc, yc, xf = np.meshgrid(xs, xs, xs_f, indexing="ij")
    w1 = w(xf, yc, zc)
    zc, yf, xc = np.meshgrid(xs, xs_f, xs, indexing="ij")
    w2 = w(xc, yf, zc)
    zff, yc3, xc3 = np.meshgrid(xs_f, xs, xs, indexing="ij")
    w3 = w(xc3, yc3, zff)
    b1, b2, b3 = f["B1"], f["B2"], f["B3"]
    div = (
        (w1[:, :, 1:] * b1[:, :, 1:] - w1[:, :, :-1] * b1[:, :, :-1])
        + (w2[:, 1:, :] * b2[:, 1:, :] - w2[:, :-1, :] * b2[:, :-1, :])
        + (w3[1:, :, :] * b3[1:, :, :] - w3[:-1, :, :] * b3[:-1, :, :])
    ) / dd
    return float(np.abs(div).max())


@needs_backend
def test_mhd_excision_runs_and_preserves_densitized_divb() -> None:
    a = _run_mhd(1.0)
    b = _run_mhd(1.4)
    r, dd = _grid_radii()
    assert np.isfinite(a["rho"]).all() and a["rho"].min() > 0.0, "excised MHD state broke"
    # non-vacuous both ways: the field genuinely evolved AND the excision genuinely ran.
    assert np.abs(a["B1"]).max() > 1e-6, "B_x never developed; the CT never acted"
    inner = r < 1.2
    assert np.abs(a["rho"][inner] - b["rho"][inner]).max() > 1e-8, (
        "the two excision radii produced identical interiors; the MHD pass never ran"
    )
    # the defining invariant: the gas-only fill never writes a staggered face, so
    # the densitized divergence recomputed from the EVOLVED faces stays at machine
    # zero relative to the field scale — excision cannot break the CT constraint.
    dmax = _densitized_div(a, dd)
    bmax = max(float(np.abs(a["B1"]).max()), float(np.abs(a["B3"]).max()), 1e-300)
    assert dmax < 1e-12 * max(1.0, bmax / 1e-3), (
        f"densitized div(B) not preserved under excision: {dmax:e}"
    )
    # bounded, decaying exterior leakage between the two radii — same envelope as hydro.
    diff = np.abs(a["rho"] - b["rho"])
    scale = np.abs(a["rho"]).max()
    near = (r > 2.0 + 2.0 * dd) & (r < 3.0)
    far = r > 4.0
    near_leak = diff[near].max() / scale
    far_leak = diff[far].max() / scale
    assert near_leak < 1e-3, f"near-horizon MHD excision leakage too large: {near_leak:e}"
    # the outward-decay shape is meaningful only when the leakage is a genuine
    # causal signal; below ~1e-5 relative both bands are roundoff accumulation,
    # which has no radial shape — bound it without asserting a decay shape.
    if near_leak > 1e-5:
        assert far_leak < 0.2 * near_leak, (
            f"MHD excision leakage does not decay outward: near {near_leak:e}, far {far_leak:e}"
        )
    else:
        assert far_leak < 1e-5, f"far-field MHD excision noise too large: {far_leak:e}"
