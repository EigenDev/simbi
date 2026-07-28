# =============================================================================
# test_cartesian_ks_bh_3d.py
#
# GRHD on the FULL 3D cartesian kerr-schild chart. the a = 0 metric
# gamma_ij = delta_ij + 2M x_i x_j / r^3 and shift beta^i = 2M x_i/(r^2(r+2M))
# are exactly symmetric under ANY coordinate permutation, so a symmetric
# initial state in a cube must evolve symmetrically under every axis transpose
# to roundoff — the oracle-free correctness gate for the 3d chart-generic GR
# chain (metric-aware flux with the shift on every sweep, metric-aware c2p,
# covariant geodesic source, light-cone CFL). any coordinate-role bug breaks a
# transpose exactly. the excision variant pins that varying the excision radius
# changes the interior (the pass genuinely runs) while the exterior sees only a
# bounded, outward-decaying leakage — the reconstruction stencil reaches across
# the horizon, so exact discrete independence is impossible; smallness + decay
# is the honest invariant.
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

RES = 48
RHO0 = 1.0
PRE0 = 0.1


class _CartesianKsBH3D(SimbiProblem):
    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0)]
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime.SCHWARZSCHILD_KS)]
    schwarzschild_mass: Annotated[float, ProblemParam(1.0)]
    excision_radius: Annotated[float, ProblemParam(0.0, cli=True)]
    resolution: Annotated[tuple[int, int, int], ProblemParam((RES, RES, RES))]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]),
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


def _run(excision: float) -> dict[str, np.ndarray]:
    d = tempfile.mkdtemp() + "/"
    p = _CartesianKsBH3D(excision_radius=excision, data_directory=Path(d))
    runner.run(p, compute_mode="cpu", max_steps=400)
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, f"3d cartesian KS run (r_exc={excision}) crashed"
    out = {}
    with h5py.File(finals[0], "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        halo = (prims["rho"].shape[0] - RES) // 2
        sl = slice(halo, halo + RES)
        for nm in ("rho", "pre", "v1", "v2", "v3"):
            out[nm] = prims[nm][sl, sl, sl]
    return out


@needs_backend
def test_cartesian_ks_bh_3d_is_permutation_symmetric() -> None:
    f = _run(0.0)
    rho, pre = f["rho"], f["pre"]
    assert np.isfinite(rho).all() and np.isfinite(pre).all(), "non-finite state"
    assert rho.min() > 0.0 and pre.min() > 0.0, "state went non-positive"
    # the infall genuinely developed (non-vacuous): density piles up near the hole.
    assert rho.max() > 1.05 * RHO0, f"no accretion developed (max rho {rho.max():.3f})"
    # storage is [k, j, i] = (z, y, x); the metric is symmetric under any coordinate
    # permutation, so rho must equal its transpose over every axis pair to roundoff.
    for axes, name in [((0, 2, 1), "x<->y"), ((2, 1, 0), "x<->z"), ((1, 0, 2), "y<->z")]:
        err = np.abs(rho - np.transpose(rho, axes)).max()
        assert err < 1e-11, f"{name} transpose symmetry broken: {err:e}"


@needs_backend
def test_cartesian_ks_bh_3d_exterior_excision_leakage_is_bounded_and_decays() -> None:
    a = _run(1.0)
    b = _run(1.4)
    n = RES
    dx = 10.0 / n
    xs = (np.arange(n) + 0.5) * dx - 5.0
    z, y, x = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    # the excision genuinely acted (non-vacuous): the two interiors differ. this
    # assert is what exposed the excision phase being silently absent from the
    # hierarchy step (the raw single-grid loop had it; the python path did not).
    inner = r < 1.2
    assert np.abs(a["rho"][inner] - b["rho"][inner]).max() > 1e-8, (
        "the two excision radii produced identical interiors; the pass never ran"
    )
    # in the continuum the exterior is causally disconnected from the excised
    # sphere; discretely the reconstruction stencil reaches across the horizon,
    # so a SMALL difference leaks outward and is damped as it goes. the honest
    # invariants: the leakage just outside the horizon is bounded well below the
    # dynamic range, and it DECAYS with distance from the horizon.
    diff = np.abs(a["rho"] - b["rho"])
    scale = np.abs(a["rho"]).max()
    near = (r > 2.0 + 2.0 * dx) & (r < 3.0)
    far = (r > 4.0)
    near_leak = diff[near].max() / scale
    far_leak = diff[far].max() / scale
    assert near_leak < 1e-3, f"near-horizon excision leakage too large: {near_leak:e}"
    assert far_leak < 0.2 * max(near_leak, 1e-300), (
        f"excision leakage does not decay outward: near {near_leak:e}, far {far_leak:e}"
    )
