# =============================================================================
# test_schwarzschild_swirl_2d.py
#
# the azimuthal-momentum (swirl) DOF on the 2D (r, theta) schwarzschild grid — the
# `_sph_swirl` kernel family: 3-component covariant momentum (S_r, S_theta, S_phi)
# recovered, fluxed, and updated on a 2-axis grid. two gates:
#
# zero-swirl reduction: the exact michel transonic profile with v_phi = 0 must be
# HELD at the same truncation level the 1D michel gate pins — every swirl kernel
# runs (the 3-momentum c2p, the DOF-3 sweeps, the lifted godunov law), and S_phi
# must stay identically zero (the axisymmetric metric never reads phi, so the
# covariant source on the suppressed slot vanishes by construction).
#
# uniform-angular-momentum advection: with axisymmetry the S_phi law has NO source,
# and combined with mass conservation the specific angular momentum l = S_phi / D
# obeys pure advection — a uniform-l state stays uniform along the accreting flow.
# initializing l = l0 everywhere on the michel background, the interior l must stay
# at l0 to truncation accuracy. this is the transport gate for the lifted momentum
# (a wrong gamma_{phi phi} in the c2p/flux, a lost lapse power on the S_phi law, or
# a spurious phi source all show up as l drift). requires the built cpu_ext backend.
# =============================================================================
import glob
import math
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

_END_TIME = 10.0
_NR = 128
_NPOLAR = 8
# interior window (radial): the boundary-adjacent cells feel the first-order ghosts.
_SKIP_INNER = 4
_SKIP_OUTER = 8
# the 1D michel gate holds the profile at interior L1 ~ 5e-5 (128 zones); the wedge run
# carries the same radial truncation per theta column. tolerance with ~3x margin.
_L1_TOL = 1.5e-4
# uniform specific angular momentum for the advection gate: small enough that the
# rotating background stays near the michel profile (l^2/r^2 << 1 centrifugal shift),
# large enough that a lost metric factor (gamma_{phi phi} ~ r^2 sin^2 theta ~ 10-1e4)
# is orders of magnitude, not noise. the drift is l0-INDEPENDENT in relative terms
# (measured 3.814e-3 at l0 = 0.05 vs 3.811e-3 at 0.01): it is the truncation of the
# S_phi transport through the steep transonic infall, NOT a stir effect — and it
# CONVERGES (128x8 -> 256x16 ratio 2.2), which the second assertion pins. tolerance
# = measured 3.8e-3 with ~3x margin; a metric-factor transport bug is orders of
# magnitude and resolution-INDEPENDENT, failing both assertions.
_L0 = 0.05
_L_DRIFT_TOL = 1.2e-2
_L_DRIFT_CONV = 1.5


def _swirl_problem(data_dir: str, l0: float, nr: int = _NR, npolar: int = _NPOLAR):
    from simbi_configs.examples.gr_bondi_2d import GrBondi2D
    from simbi_configs.examples.gr_michel import MichelSolution

    class GrMichelSwirl2D(GrBondi2D):
        """the michel profile on the (r, theta) wedge with a lifted azimuthal momentum:
        5-tuple gas rows (rho, v^r, v^theta, v^phi, p) select the `_sph_swirl` kernels.
        v^phi seeds the uniform specific angular momentum l0 (zero for the reduction gate)."""

        def initial_primitive_state(self):
            sol = MichelSolution(
                mass=self.schwarzschild_mass,
                gamma=self.adiabatic_index,
                rho_inf=self.rho_ambient,
                p_inf=self.p_ambient,
            )
            nr, npolar = self.resolution
            (rmin, rmax) = self.bounds[0]
            (tmin, tmax) = self.bounds[1]
            q = (rmax / rmin) ** (1.0 / nr)
            dth = (tmax - tmin) / npolar

            def gas_state():
                for jj in range(npolar):
                    theta = tmin + (jj + 0.5) * dth
                    st = math.sin(theta)
                    for ii in range(nr):
                        rl = rmin * q**ii
                        rh = rl * q
                        r = 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)
                        rho, v1, pre = sol.primitive(r)
                        # v^phi from the uniform specific angular momentum l = S_phi/D =
                        # h W gamma_{phi phi} v^phi: with W from the radial michel flow
                        # (the small swirl's W correction is O(l^2), inside tolerance).
                        f = 1.0 - 2.0 * self.schwarzschild_mass / r
                        ww = 1.0 / math.sqrt(1.0 - (v1 / math.sqrt(f)) ** 2)
                        hh = 1.0 + self.adiabatic_index / (self.adiabatic_index - 1.0) * pre / rho
                        vphi = l0 / (hh * ww * r * r * st * st)
                        yield (rho, v1, 0.0, vphi, pre)

            return gas_state

    p = GrMichelSwirl2D.from_cli(["--nr", str(nr), "--npolar", str(npolar)])
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME
    return p


def _read_interior_2d(chkpt_path: str, nr: int, npolar: int):
    """interior primitives on the (theta, r) storage grid, halo excluded."""
    with h5py.File(chkpt_path, "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        shp = prims["rho"].shape
        halo = [(s - n) // 2 for s, n in zip(shp, (npolar, nr))]
        sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (npolar, nr)))
        out = {k: prims[k][sl] for k in ("rho", "pre", "v1", "v2", "v3")}
    return out


def _run(l0: float, nr: int = _NR, npolar: int = _NPOLAR):
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = _swirl_problem(d, l0, nr, npolar)
        runner.run(p, compute_mode="cpu")
        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, f"swirl run (l0 = {l0}) crashed before completion"
        prims = _read_interior_2d(finals[0], nr, npolar)

        from simbi_configs.examples.gr_michel import MichelSolution

        sol = MichelSolution(
            mass=p.schwarzschild_mass,
            gamma=p.adiabatic_index,
            rho_inf=p.rho_ambient,
            p_inf=p.p_ambient,
        )
        (rmin, rmax) = p.bounds[0]
        q = (rmax / rmin) ** (1.0 / nr)
        rl = rmin * q ** np.arange(nr)
        rh = rl * q
        rc = 0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)
        ref_rho = np.array([sol.primitive(r)[0] for r in rc])
    return p, prims, rc, ref_rho


@needs_backend
def test_swirl_zero_reduces_to_held_michel() -> None:
    p, prims, rc, ref_rho = _run(0.0)

    assert prims["pre"].min() > 0.0, f"pressure went non-positive: {prims['pre'].min():.3e}"
    # the lifted azimuthal momentum must stay identically zero for zero-swirl data.
    assert np.abs(prims["v3"]).max() < 1e-12, (
        f"v^phi grew from exact zero: max {np.abs(prims['v3']).max():.3e}"
    )
    # every theta column holds the michel profile at the 1D gate's truncation level.
    cut = slice(_SKIP_INNER, _NR - _SKIP_OUTER)
    e = np.abs(prims["rho"][:, cut] / ref_rho[cut] - 1.0)
    assert e.mean() < _L1_TOL, f"interior L1 rho residual {e.mean():.3e}"


def _l_drift(nr: int, npolar: int) -> float:
    p, prims, rc, ref_rho = _run(_L0, nr, npolar)
    assert prims["pre"].min() > 0.0, f"pressure went non-positive: {prims['pre'].min():.3e}"
    # rebuild l = S_phi/D = h W gamma_{phi phi} v^phi from the evolved primitives.
    (tmin, tmax) = p.bounds[1]
    dth = (tmax - tmin) / npolar
    theta = tmin + (np.arange(npolar) + 0.5) * dth
    st = np.sin(theta)[:, None]
    f = 1.0 - 2.0 * p.schwarzschild_mass / rc[None, :]
    v_sq = prims["v1"] ** 2 / f + rc[None, :] ** 2 * prims["v2"] ** 2 \
        + rc[None, :] ** 2 * st**2 * prims["v3"] ** 2
    ww = 1.0 / np.sqrt(1.0 - v_sq)
    hh = 1.0 + p.adiabatic_index / (p.adiabatic_index - 1.0) * prims["pre"] / prims["rho"]
    ll = hh * ww * rc[None, :] ** 2 * st**2 * prims["v3"]
    cut = slice(_SKIP_INNER, nr - _SKIP_OUTER)
    return float(np.abs(ll[:, cut] / _L0 - 1.0).mean())


@needs_backend
def test_uniform_angular_momentum_is_advected() -> None:
    d_lo = _l_drift(_NR, _NPOLAR)
    d_hi = _l_drift(2 * _NR, 2 * _NPOLAR)
    assert d_lo < _L_DRIFT_TOL, (
        f"specific angular momentum drifted: mean |l/l0 - 1| = {d_lo:.3e}"
    )
    # truncation converges; a metric-factor transport error is resolution-independent.
    assert d_lo / d_hi > _L_DRIFT_CONV, (
        f"l drift does not converge: {d_lo:.3e} -> {d_hi:.3e}"
    )
