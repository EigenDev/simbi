# =============================================================================
# test_kerr_2d.py
#
# the spinning-kerr wiring gates on the 2D (r, theta) horizon-penetrating grid
# (the `_kerr` swirl kernel family: non-diagonal gamma_{r phi}, theta-dependent
# lapse, radial shift, Sigma sin(theta) covariant measure).
#
# zero-spin cross-check: kerr at a = 0 is the SAME physics as the a = 0
# kerr-schild chart through DIFFERENT kernel expressions (Sigma = r^2 + a^2 cos^2
# with a = 0 adds exact-zero terms), so a uniform-gas accretion transient run on
# both spacetimes must agree to near roundoff — the wiring-exactness gate for the
# whole `_kerr` family (metric, geometry moments, sources, shift, wave speeds).
#
# frame dragging: infalling gas seeded with ZERO angular momentum. axisymmetry
# keeps the S_phi law source-free (S_phi stays at truncation scale) — yet the
# recovered AZIMUTHAL velocity is
# v^phi = gamma^{phi r} S_r / (tau + D + p) != 0 through the non-diagonal inverse
# metric: the twist is spin-linear (antisymmetric under a -> -a, while the
# scalar flow is spin-even), strongest at the horizon. the defining kerr
# effect, gated oracle-free through its symmetry structure.
# requires the built cpu_ext backend; skipped otherwise.
# =============================================================================
import glob
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
_NR = 96
_NPOLAR = 16


def _kerr_problem(data_dir: str, spacetime: str, spin: float):
    from simbi_configs.examples.gr_bondi_2d import GrBondi2D
    from simbi.types import Spacetime

    class GrKerrBondi2D(GrBondi2D):
        """uniform gas at rest on the horizon-penetrating chart, swirl DOF lifted
        (5-tuple rows select the DOF = 3 kernels). the inner boundary sits BELOW
        the horizon r_plus = M + sqrt(M^2 - a^2), so the through-horizon inflow is
        supersonic and the inner ghost state is causally disconnected."""

        def initial_primitive_state(self):
            nr, npolar = self.resolution
            def gas_state():
                for _jj in range(npolar):
                    for _ii in range(nr):
                        yield (self.rho_ambient, 0.0, 0.0, 0.0, self.p_ambient)
            return gas_state

    p = GrKerrBondi2D.from_cli(["--nr", str(_NR), "--npolar", str(_NPOLAR)])
    p.spacetime = Spacetime(spacetime)
    p.kerr_spin = spin
    # horizon-penetrating: inner boundary below r_plus (= 2M at a = 0, 1.436M at a = 0.9).
    p.bounds[0] = (1.2, 100.0)
    p.end_time = _END_TIME
    p.data_directory = data_dir
    p.checkpoint_interval = _END_TIME
    return p


def _run(spacetime: str, spin: float) -> dict:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = _kerr_problem(d, spacetime, spin)
        runner.run(p, compute_mode="cpu")
        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, f"{spacetime} (a = {spin}) crashed before completion"
        with h5py.File(finals[0]) as h:
            g = h["level_0/partition_0/hydro/primitives"]
            shp = g["rho"].shape
            halo = [(s - n) // 2 for s, n in zip(shp, (_NPOLAR, _NR))]
            sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (_NPOLAR, _NR)))
            out = {k: g[k][sl] for k in ("rho", "pre", "v1", "v2", "v3")}
            c = h["level_0/conserved"]
            out["m3"] = c["m3"][sl]
    return out


@needs_backend
def test_kerr_at_zero_spin_matches_the_ks_chart() -> None:
    kerr = _run("kerr", 0.0)
    ks = _run("kerr_schild", 0.0)
    # same physics, different kernels AND a different CFL map (the kerr light-cone bound vs
    # the kerr-schild banyuls-font speeds), so the dt sequences differ and the trajectories
    # separate at the truncation floor — measured 1.6e-7 max after t = 10. a wiring error
    # (wrong metric / measure / source) sits orders above (the angular face-weight bug
    # measured 1e-2). differences normalize by the FIELD's domain scale (with the radial-flow
    # scale as the floor): v^theta is roundoff-noise-sized in this theta-symmetric flow, and
    # a per-cell relative error on noise is meaningless.
    v_scale = np.abs(ks["v1"]).max()
    for k in ("rho", "pre", "v1", "v2"):
        scale = max(np.abs(ks[k]).max(), 1e-3 * v_scale)
        e = np.abs(kerr[k] - ks[k]).max() / scale
        assert e < 1e-5, f"{k}: kerr(a=0) vs ks scaled max diff {e:.3e}"


@needs_backend
def test_frame_dragging_twists_zero_angular_momentum_infall() -> None:
    spin = 0.9
    out = _run("kerr", spin)
    out_m = _run("kerr", -spin)

    assert out["pre"].min() > 0.0, f"pressure went non-positive: {out['pre'].min():.3e}"
    # the S_phi law is source-free (axisymmetry), so angular momentum is generated only at
    # truncation level: the exact per-cell dragging cancellation gamma_{phi r} v^r +
    # gamma_{phi phi} v^phi = 0 breaks under independent reconstruction of the two velocities,
    # and HLL transports the residual. measured 5.2e-2 against near-horizon |S_r| ~ 10
    # (relative ~1e-3) at 96x16; an order-unity S_phi mishandling sits far above.
    assert np.abs(out["m3"]).max() < 0.15, (
        f"S_phi beyond truncation scale: {np.abs(out['m3']).max():.3e}"
    )
    # frame dragging: with S_phi = 0 the recovered azimuthal velocity is PURELY the
    # non-diagonal inverse-metric lift v^phi = gamma^{phi r} S_r / (tau + D + p),
    # gamma^{phi r} = a / Sigma — nonzero, spin-linear (antisymmetric under a -> -a),
    # strong near the horizon and decaying outward. sign-free structural assertions.
    eq = _NPOLAR // 2
    vphi_inner = out["v3"][eq, 2:8]
    vphi_outer = out["v3"][eq, -12:-6]
    assert np.abs(vphi_inner).min() > 1e-6, (
        f"no frame dragging near the horizon: v^phi inner = {vphi_inner}"
    )
    inner_sign = np.sign(vphi_inner)
    assert (inner_sign == inner_sign[0]).all(), (
        f"incoherent dragging direction: v^phi inner = {vphi_inner}"
    )
    assert np.abs(vphi_inner).mean() > 10.0 * np.abs(vphi_outer).mean(), (
        "frame dragging does not decay outward: "
        f"inner {np.abs(vphi_inner).mean():.3e} vs outer {np.abs(vphi_outer).mean():.3e}"
    )
    # antisymmetry: the twist is odd in the spin while the scalar flow is even.
    e_flip = np.abs(out["v3"] + out_m["v3"]) / (np.abs(out["v3"]) + 1e-300)
    assert e_flip.max() < 1e-9, (
        f"v^phi is not spin-antisymmetric: max rel {e_flip.max():.3e}"
    )
    e_even = np.abs(out["rho"] - out_m["rho"]) / np.abs(out["rho"])
    assert e_even.max() < 1e-9, (
        f"rho is not spin-even: max rel {e_even.max():.3e}"
    )
