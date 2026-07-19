# =============================================================================
# test_cartesian_kerr_bh_3d.py
#
# spinning-kerr GRHD on the FULL 3D cartesian kerr-schild chart (spin about z):
# gamma_ij = delta_ij + 2H l_i l_j with the oblate-spheroidal radius and the
# swirl components of l carrying the frame dragging. three oracle-free gates:
# - a = 0 reduces the metric EXACTLY to the schwarzschild cartesian KS chart,
#   so the spin-zero kerr run must match the kerr_schild run OUTSIDE the
#   frozen core: the r >= M/2 clamp fills the unphysical center with two
#   different (both consistent) metric continuations, so the cores differ
#   structurally while every physical cell agrees to amplified roundoff,
#   decaying outward (the same bounded-leakage shape the excision gate pins);
# - the metric at any spin is invariant under the QUARTER TURN about z,
#   (x, y) -> (-y, x), and under z -> -z, so a symmetric initial state must
#   evolve symmetrically under both to roundoff — the coordinate-role gate
#   for the swirl of l (a transpose-style bug breaks these exactly);
# - the x <-> y transpose maps a -> -a exactly (reflection flips the spin
#   axis sense), so the +a run transposed must equal the -a run to roundoff;
#   the dragging swirl is nonzero and flips sign with a, while the a = 0 run
#   stays swirl-free (non-vacuous: the spin genuinely acts). the SIGN of the
#   coordinate swirl is chart-dependent (ingoing-KS phi differs from the
#   boyer-lindquist azimuth by a radial offset, so infalling S_phi = 0 gas
#   has v^phi_KS = omega + (dphi_KS/dr) v^r with the infall term dominant
#   near the hole) — only the antisymmetry in a is gated.
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

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

RES = 32
RHO0 = 1.0
PRE0 = 0.1
MASS = 1.0
L = 5.0


class _CartesianKerrBH3D(SimbiProblem):
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0)]
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.KERR, cli=True)]
    schwarzschild_mass: Annotated[float, ProblemParam(MASS)]
    kerr_spin: Annotated[float, ProblemParam(0.9, cli=True)]
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

    def initial_primitive_state(self):
        def gas_state():
            nx, ny, nz = self.resolution
            for _ in range(nx * ny * nz):
                yield (RHO0, 0.0, 0.0, 0.0, PRE0)

        return gas_state


def _run(spacetime: Spacetime, spin: float) -> dict[str, np.ndarray]:
    d = tempfile.mkdtemp() + "/"
    p = _CartesianKerrBH3D(data_directory=Path(d))
    p.spacetime = spacetime
    p.kerr_spin = spin
    runner.run(p, compute_mode="cpu")
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"3d cartesian kerr run (a={spin}) crashed"
    out = {}
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        for nm in ("rho", "pre", "v1", "v2", "v3"):
            out[nm] = prims[nm][sl, sl, sl]
    return out


def _vphi_moment(f: dict[str, np.ndarray]) -> float:
    # the density-weighted azimuthal swirl (x v_y - y v_x) rho summed over the
    # near-hole exterior 1.5 < r < 3 — antisymmetric in the spin a; the sign is
    # chart-dependent (ingoing-KS azimuth), so only |.| and the a-antisymmetry gate.
    dd = 2.0 * L / RES
    xs = (np.arange(RES) + 0.5) * dd - L
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    band = (r > 1.5) & (r < 3.0)
    swirl = (x * f["v2"] - y * f["v1"]) * f["rho"]
    return float(swirl[band].sum())


@needs_backend
def test_cartesian_kerr_a0_matches_kerr_schild() -> None:
    a = _run(Spacetime.KERR, 0.0)
    b = _run(Spacetime.KERR_SCHILD, 0.0)
    # non-vacuous: the infall genuinely developed (a pair of identically-floored
    # states would also "agree").
    assert a["rho"].max() > 1.05 * RHO0, "no accretion developed in the a = 0 kerr run"
    # the a = 0 kerr metric is ALGEBRAICALLY the schwarzschild cartesian KS chart
    # (2H = 2M/r, l = x/r) EVERYWHERE the oblate-spheroidal radius is unclamped;
    # inside the r < M/2 guard the two metrics continue the frozen core differently
    # (both consistent, both unphysical — excision owns those cells), so the gate
    # bounds the physical region: tight outside the horizon, looser in the strong
    # field just outside the clamp, silent inside it. the bound DECAYS outward.
    dd = 2.0 * L / RES
    xs = (np.arange(RES) + 0.5) * dd - L
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    mid = (r > 0.75) & (r <= 2.0)
    ext = r > 2.0
    for nm in ("rho", "pre", "v1", "v2", "v3"):
        d_mid = np.abs(a[nm] - b[nm])[mid].max()
        d_ext = np.abs(a[nm] - b[nm])[ext].max()
        assert d_mid < 1e-2, f"a=0 kerr vs kerr_schild strong-field mismatch in {nm}: {d_mid:e}"
        assert d_ext < 1e-4, f"a=0 kerr vs kerr_schild exterior mismatch in {nm}: {d_ext:e}"
        assert d_ext < 0.5 * max(d_mid, 1e-300), (
            f"{nm}: core disagreement does not decay outward (mid {d_mid:e}, ext {d_ext:e})"
        )


@needs_backend
def test_cartesian_kerr_symmetries_and_frame_dragging() -> None:
    f = _run(Spacetime.KERR, 0.9)
    rho = f["rho"]
    assert np.isfinite(rho).all(), "non-finite state at a = 0.9"
    assert rho.min() > 0.0, "density went non-positive"
    assert rho.max() > 1.05 * RHO0, f"no accretion developed (max rho {rho.max():.3f})"

    # the quarter turn about the spin axis, (x, y) -> (-y, x): storage is [k, j, i],
    # so the rotated field is rot90 in the (j, i) plane. exact metric symmetry at
    # ANY spin — a swirl coordinate-role bug breaks it exactly.
    rot = np.rot90(rho, k=1, axes=(1, 2))
    err = np.abs(rho - rot).max()
    assert err < 1e-11, f"quarter-turn symmetry about the spin axis broken: {err:e}"

    # the equatorial reflection z -> -z (spin about z is reflection-even).
    err = np.abs(rho - rho[::-1, :, :]).max()
    assert err < 1e-11, f"z -> -z reflection symmetry broken: {err:e}"

    # the x <-> y transpose is a reflection, so it maps the spin a -> -a while
    # fixing everything else: the +a run transposed must equal the -a run to
    # roundoff (a sharp equivalence at FULL spin, storage [k, j, i]).
    g = _run(Spacetime.KERR, -0.9)
    err = np.abs(np.transpose(f["rho"], (0, 2, 1)) - g["rho"]).max()
    assert err < 1e-11, f"transpose(+a) != run(-a): {err:e}"

    # frame dragging is real: the near-hole swirl moment dwarfs the a = 0 run's
    # roundoff floor and FLIPS SIGN with a (the coordinate sense is chart-dependent;
    # the antisymmetry in a is the invariant).
    swirl = _vphi_moment(f)
    swirl_m = _vphi_moment(g)
    f0 = _run(Spacetime.KERR, 0.0)
    swirl0 = abs(_vphi_moment(f0))
    assert abs(swirl) > 100.0 * max(swirl0, 1e-300), (
        f"spin swirl {swirl:e} does not dominate the a = 0 floor {swirl0:e}"
    )
    anti = abs(swirl + swirl_m) / abs(swirl)
    assert anti < 1e-9, f"swirl moment not antisymmetric in a: {swirl:e} vs {swirl_m:e}"
