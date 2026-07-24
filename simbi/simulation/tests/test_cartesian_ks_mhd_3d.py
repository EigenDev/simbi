# =============================================================================
# test_cartesian_ks_mhd_3d.py
#
# GRMHD on the FULL 3D cartesian kerr-schild chart. the seeded field is
# B_z = B0 / sqrt(gamma), the exactly densitized-divergence-free vertical
# field (sqrt(gamma) B^z = B0 is constant, so d_i(sqrt(gamma) B^i) = 0 on the
# staggered mesh to machine precision). two invariants:
# - the a = 0 metric and the vertical field are symmetric under x <-> y, so
#   the accreting state must stay transpose-symmetric over that pair to
#   roundoff (the 3d GR CT chain's coordinate-role gate);
# - the constrained transport preserves the DENSITIZED divergence: recomputed
#   from the evolved staggered field, it stays at machine zero.
#
# the run EXCISES the singular core (r_ks < 0.7 r_+ = 1.4 M, r_+ = 2M): the
# region inside the horizon is causally disconnected and its metric is the
# clamped fiction the r >= M/2 floor fabricates, so its state is numerical
# padding, not flow. the stiff magnetized c2p amplifies ULP-level differences
# on x <-> y-mirrored inputs there into O(1e-4) chatter that is meaningless and
# would otherwise mask the exterior symmetry the gate exists to verify. excising
# it donor-fills those cells and leaves the physical exterior — the only region
# an observer sees — to carry the invariants at roundoff.
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
from simbi.types import BoundaryCondition, CoordSystem, CtMethod, Regime, Solver, Spacetime
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
L = 5.0


def _sqrtg(x: float, y: float, z: float) -> float:
    # the KERNEL's sqrt(gamma), including its r >= M/2 clamp: the seeded densitized
    # field must satisfy the constraint in the same weights the CT curl divides by.
    r_true = np.sqrt(x * x + y * y + z * z)
    r = max(r_true, 0.5 * MASS)
    ll2 = (r_true / r) ** 2  # |l|^2, exactly 1 outside the radius floor
    return float(np.sqrt(1.0 + (2.0 * MASS / r) * ll2))


class _CartesianKsMhd3D(SimbiProblem):
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0)]
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.KERR_SCHILD)]
    schwarzschild_mass: Annotated[float, ProblemParam(MASS)]
    # excision of the singular core at 0.7 r_+ (r_+ = 2M) is OPT-IN per test: the
    # full-evolution symmetry gate excises it (the sub-horizon interior is causally
    # disconnected fiction whose grown c2p chatter would mask the exterior it checks),
    # while the two-step early-stencil gate leaves it in place (its whole point is the
    # raw CT-stencil symmetry BEFORE any core contamination, so it must not be filled).
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
                                yield BZ0 / _sqrtg(xc(i), xc(j), zf(k))

            return gen

        return (gas_state, b_field("b1"), b_field("b2"), b_field("b3"))


@needs_backend
def test_cartesian_ks_mhd_3d_symmetry_and_densitized_divb() -> None:
    d = tempfile.mkdtemp() + "/"
    # excise the causally disconnected singular core (r_ks < 0.7 r_+ = 1.4 M): over a
    # full evolution its stiff-c2p chatter grows into O(1e-4) noise that would mask the
    # exterior x <-> y symmetry this gate exists to verify.
    p = _CartesianKsMhd3D(excision_radius=1.4, data_directory=Path(d))
    runner.run(p, compute_mode="cpu")
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, "3d cartesian KS GRMHD run crashed"
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        rho = prims["rho"][sl, sl, sl]
        b1c = prims["b1"][sl, sl, sl]
        b2c = prims["b2"][sl, sl, sl]
        # the staggered faces live under hydro/magnetic/Bn (interior + 1 on the own axis).
        mag = h["level_0/partition_0/hydro/magnetic"]
        bf = [mag[f"B{k}"]["data"][:] for k in (1, 2, 3)]
    assert np.isfinite(rho).all(), "non-finite density"
    assert rho.min() > 0.0, "density went non-positive"
    assert rho.max() > 1.05 * RHO0, f"no accretion (max rho {rho.max():.3f})"

    dd = 2.0 * L / RES
    xs = (np.arange(RES) + 0.5) * dd - L
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    # x <-> y transpose symmetry (storage [k, j, i]). the gas-side chain holds it to
    # roundoff (proven with B = 0); with the field on, the UNEXCISED clamped core is
    # numerically noisy under the stiff magnetized c2p (tolerance-level chatter on
    # ULP-mirrored inputs) and the noise spreads — bound it, tightly OUTSIDE the
    # horizon and loosely inside. an excised run owns those cells instead.
    asym = np.abs(rho - np.transpose(rho, (0, 2, 1)))
    assert asym.max() < 1e-4, f"gross x<->y asymmetry: {asym.max():e}"
    ext = r > 2.0
    assert asym[ext].max() < 2e-5, f"exterior x<->y asymmetry: {asym[ext].max():e}"
    # the FIELD mirror bound at the same radial split: B_x(x, y) <-> B_y(y, x) (the
    # x <-> y reflection composed with B -> -B, an exact MHD symmetry of this data).
    berr = np.abs(b1c - np.transpose(b2c, (0, 2, 1)))
    assert berr[ext].max() < 1e-3, f"exterior B mirror error: {berr[ext].max():e}"
    # the field genuinely evolved (non-vacuous).
    assert np.abs(b1c).max() > 1e-6, "B_x never developed; the CT never acted"

    # the DENSITIZED divergence d_i(sqrt(gamma) B^i), recomputed from the evolved
    # staggered faces, stays at machine zero relative to the field scale — the 3d GR
    # constrained transport's defining invariant.
    # the checkpointed face arrays are INTERIOR-only, [k, j, i]-ordered, +1 on the
    # own axis (B1 staggered on x = the last index, B3 on z = the first).
    b1, b2, b3 = bf[0], bf[1], bf[2]
    assert b1.shape == (RES, RES, RES + 1), f"B1 shape {b1.shape}"
    assert b2.shape == (RES, RES + 1, RES), f"B2 shape {b2.shape}"
    assert b3.shape == (RES + 1, RES, RES), f"B3 shape {b3.shape}"
    xs_f = np.arange(RES + 1) * dd - L

    def w(xg, yg, zg):
        # the kernel's sqrt(gamma), INCLUDING its r >= M/2 clamp (the frozen singular
        # core): the invariant the curl preserves is div(w B) with the kernel's own w.
        rr = np.sqrt(xg * xg + yg * yg + zg * zg)
        r_true = rr.copy()
        np.maximum(rr, 0.5 * MASS, out=rr)
        ll2 = (r_true / rr) ** 2  # |l|^2, exactly 1 outside the radius floor
        return np.sqrt(1.0 + (2.0 * MASS / rr) * ll2)

    zc, yc2, xf = np.meshgrid(xs, xs, xs_f, indexing="ij")
    w1 = w(xf, yc2, zc)
    zc, yf, xc2 = np.meshgrid(xs, xs_f, xs, indexing="ij")
    w2 = w(xc2, yf, zc)
    zff, yc3, xc3 = np.meshgrid(xs_f, xs, xs, indexing="ij")
    w3 = w(xc3, yc3, zff)
    div = (
        (w1[:, :, 1:] * b1[:, :, 1:] - w1[:, :, :-1] * b1[:, :, :-1])
        + (w2[:, 1:, :] * b2[:, 1:, :] - w2[:, :-1, :] * b2[:, :-1, :])
        + (w3[1:, :, :] * b3[1:, :, :] - w3[:-1, :, :] * b3[:-1, :, :])
    ) / dd
    dmax = np.abs(div).max()
    bmax = max(abs(float(np.abs(b1).max())), abs(float(np.abs(b3).max())), 1e-300)
    assert dmax < 1e-12 * max(1.0, bmax / 1e-3), f"densitized div(B) not preserved: {dmax:e}"


@needs_backend
def test_cartesian_ks_mhd_3d_uct_preserves_divb_and_mirror() -> None:
    # the FULL-3D UCT-HLL corner EMF (three edge orientations, materialized wave
    # speeds): the same two defining invariants as the contact CT — the recomputed
    # densitized divergence stays at machine zero from the EVOLVED staggered field,
    # and the early-time field mirror symmetry holds to roundoff (the sharp
    # coordinate-role gate over the per-edge slot bindings).
    d = tempfile.mkdtemp() + "/"
    p = _CartesianKsMhd3D(data_directory=Path(d))
    p.ct_method = CtMethod.UCT
    runner.run(p, compute_mode="cpu", max_steps=2)
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, "3d UCT GRMHD run crashed"
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        b1 = prims["b1"][sl, sl, sl]
        b2 = prims["b2"][sl, sl, sl]
        mag = h["level_0/partition_0/hydro/magnetic"]
        bf = [mag[f"B{k}"]["data"][:] for k in (1, 2, 3)]
    assert np.abs(b1).max() > 1e-5, "the field never moved under UCT; vacuous"
    berr = np.abs(b1 - np.transpose(b2, (0, 2, 1))).max()
    assert berr < 1e-14, f"UCT early B mirror symmetry broken: {berr:e}"

    dd = 2.0 * L / RES
    xs = (np.arange(RES) + 0.5) * dd - L
    xs_f = np.arange(RES + 1) * dd - L

    def w(xg, yg, zg):
        rr = np.sqrt(xg * xg + yg * yg + zg * zg)
        r_true = rr.copy()
        np.maximum(rr, 0.5 * MASS, out=rr)
        ll2 = (r_true / rr) ** 2  # |l|^2, exactly 1 outside the radius floor
        return np.sqrt(1.0 + (2.0 * MASS / rr) * ll2)

    zc, yc2, xf = np.meshgrid(xs, xs, xs_f, indexing="ij")
    w1 = w(xf, yc2, zc)
    zc, yf, xc2 = np.meshgrid(xs, xs_f, xs, indexing="ij")
    w2 = w(xc2, yf, zc)
    zff, yc3, xc3 = np.meshgrid(xs_f, xs, xs, indexing="ij")
    w3 = w(xc3, yc3, zff)
    b1f, b2f, b3f = bf[0], bf[1], bf[2]
    div = (
        (w1[:, :, 1:] * b1f[:, :, 1:] - w1[:, :, :-1] * b1f[:, :, :-1])
        + (w2[:, 1:, :] * b2f[:, 1:, :] - w2[:, :-1, :] * b2f[:, :-1, :])
        + (w3[1:, :, :] * b3f[1:, :, :] - w3[:-1, :, :] * b3f[:-1, :, :])
    ) / dd
    dmax = float(np.abs(div).max())
    bmax = max(float(np.abs(b1f).max()), float(np.abs(b3f).max()), 1e-300)
    assert dmax < 1e-12 * max(1.0, bmax / 1e-3), (
        f"UCT densitized div(B) not preserved: {dmax:e}"
    )


@needs_backend
def test_cartesian_ks_mhd_3d_stencils_are_mirror_exact_early() -> None:
    # before the clamped singular core contaminates the flow (its iterative-c2p chatter
    # is tolerance-level and takes several steps to grow), the discrete CT stencils must
    # hold the field mirror symmetry to roundoff — the sharp coordinate-role gate for the
    # 3d GR corner-EMF and curl.
    #
    # UCT, not the contact scheme: this initial data is uniform gas falling radially onto
    # the hole, so v_x = v_r x/r is identically zero on the plane x = 0 and the contact
    # upwind selector — which reads the sign of the normal mass flux — has no defined
    # direction on that entire face plane. UCT selects on wave speeds, which stay bounded
    # away from zero, so it is well posed here. the contact scheme's failure on exactly this
    # configuration is gated by
    # `test_ct_contact_upwinding_is_ill_posed_on_a_flow_symmetry_plane` below.
    d = tempfile.mkdtemp() + "/"
    p = _CartesianKsMhd3D(data_directory=Path(d))
    p.ct_method = CtMethod.UCT
    runner.run(p, compute_mode="cpu", max_steps=2)
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        b1 = prims["b1"][sl, sl, sl]
        b2 = prims["b2"][sl, sl, sl]
        rho = prims["rho"][sl, sl, sl]
    assert np.abs(b1).max() > 1e-5, "the field never moved in two steps; vacuous"
    berr = np.abs(b1 - np.transpose(b2, (0, 2, 1))).max()
    assert berr < 1e-14, f"early B mirror symmetry broken: {berr:e} (stencil asymmetry)"
    rerr = np.abs(rho - np.transpose(rho, (0, 2, 1))).max()
    assert rerr < 1e-13, f"early rho transpose symmetry broken: {rerr:e}"


@needs_backend
def test_ct_contact_upwinding_is_ill_posed_on_a_flow_symmetry_plane() -> None:
    # the contact CT scheme selects the electromotive-force derivative by the SIGN of the
    # normal mass flux (Gardiner & Stone 2005, eq. 51). that flux vanishes identically on a
    # symmetry plane of the flow — here uniform gas falls radially onto the hole, so
    # v_x = v_r x/r is zero on x = 0 — leaving the selector with no defined direction across
    # that whole face plane, and the sign is then set by roundoff. the resulting
    # electromotive-force error is fed back through the next step's fluxes and grows.
    #
    # UCT selects on wave speeds instead, which carry the fast magnetosonic speed and never
    # approach zero, so it has a well-defined direction everywhere and holds the symmetry.
    #
    # this configuration is EXACTLY mirror symmetric under exchanging x and y: the metric is
    # spherically symmetric (zero spin), the gas is uniform and at rest, and the field is
    # purely vertical with a magnitude depending only on radius. so B_x(x, y, z) must equal
    # B_y(y, x, z) for all time, and any departure is numerical.
    #
    # the gate is RELATIVE, comparing the two schemes on identical data rather than pinning a
    # measured number: UCT must hold the symmetry, and the contact scheme must violate it by
    # orders of magnitude. if a future contact formulation becomes well posed here this test
    # FAILS, which is the signal to delete it and the UCT pin in the mirror test above.
    def run_with(ct: CtMethod) -> float:
        d = tempfile.mkdtemp() + "/"
        p = _CartesianKsMhd3D(data_directory=Path(d))
        p.ct_method = ct
        runner.run(p, compute_mode="cpu", max_steps=2)
        finals = glob.glob(os.path.join(d, "*final*.h5"))
        assert finals, f"3d cartesian KS GRMHD run crashed under {ct}"
        with h5py.File(finals[0], "r") as h:
            prims = h["level_0/partition_0/hydro/primitives"]
            halo = (prims["rho"].shape[0] - RES) // 2
            sl = slice(halo, halo + RES)
            b1 = prims["b1"][sl, sl, sl]
            b2 = prims["b2"][sl, sl, sl]
        assert np.abs(b1).max() > 1e-5, f"the field never moved under {ct}; vacuous"
        return float(np.abs(b1 - np.transpose(b2, (0, 2, 1))).max())

    uct = run_with(CtMethod.UCT)
    contact = run_with(CtMethod.CONTACT)

    assert uct < 1e-14, (
        f"UCT broke the x<->y mirror symmetry of an exactly symmetric configuration: {uct:e}"
    )
    assert contact > 100.0 * uct, (
        f"the contact CT scheme held the mirror symmetry ({contact:e} vs UCT {uct:e}) — its "
        "upwind selector is no longer ill posed where the normal mass flux vanishes, so this "
        "test and the UCT pin in the mirror-exactness test above are both obsolete"
    )
