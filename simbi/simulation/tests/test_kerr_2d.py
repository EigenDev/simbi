# =============================================================================
# test_kerr_2d.py
#
# the spinning-kerr wiring gates on the 2D (r, theta) horizon-penetrating grid
# (the `_kerr` swirl kernel family: non-diagonal gamma_{r phi}, theta-dependent
# lapse, radial shift, Sigma sin(theta) covariant measure).
#
# the gates split by what double precision can promise. IDENTITY gates demand
# machine precision: the ONE-STEP matched-dt kerr(a = 0) vs kerr-schild comparison
# (same state, same dt -> only ULP reassociation differs), the one-step S_phi from
# uniform data (nothing generates it in one step), the spin-PARITY of full
# trajectories (every kernel operation is IEEE parity-exact in a, and the +-a runs
# share the dt sequence, so v^phi is BITWISE odd and rho bitwise even), and the
# FULL-TRAJECTORY S_phi of zero-angular-momentum infall (both the flux and the
# ghost fill carry the angular-momentum variable w = v^phi + (gamma_{r phi}/
# gamma_{phi phi}) v^r, so a w = 0 state has no S_phi generator anywhere — interior
# faces or boundary ghosts). CONSISTENCY gates are truncation-bounded: the
# full-trajectory a = 0 cross-check (the two spacetimes use different CFL maps, so
# dt sequences differ and trajectories separate at the truncation floor).
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
    from simbi_configs.examples.grhd.gr_bondi_2d import GrBondi2D
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


def _one_step(spacetime: str, spin: float, nr: int = _NR, npolar: int = _NPOLAR) -> dict:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = _kerr_problem(d, spacetime, spin)
        p.nr = nr
        p.npolar = npolar
        p.resolution = (nr, npolar)
        # far below the CFL dt: both spacetimes take exactly this single step, so the
        # comparison isolates the KERNELS from any dt-sequence difference.
        p.end_time = 1e-6
        p.checkpoint_interval = 1.0
        runner.run(p, compute_mode="cpu")
        with h5py.File(glob.glob(os.path.join(d, "*final*.h5"))[0]) as h:
            c = h["level_0/conserved"]
            return {k: c[k][:] for k in ("den", "m1", "m2", "m3", "nrg")}


@needs_backend
def test_kerr_zero_spin_one_step_is_machine_exact() -> None:
    # the machine-precision wiring gate: same state, same dt, one step — the a = 0 kerr
    # kernels (Sigma = r^2 + a^2 cos^2 folding to exact zeros) against the kerr-schild
    # kernels. measured: den/nrg/m3 BITWISE equal, m1 at 1e-15 of its own scale, m2 at the
    # roundoff of the exact well-balanced cancellation (|m2| ~ 4e-23 while the real update
    # scale m1 ~ 2e-7). the absolute floor covers the cancellation-noise fields.
    kerr0 = _one_step("kerr", 0.0)
    ks = _one_step("kerr_schild", 0.0)
    for k in ("den", "m1", "m2", "m3", "nrg"):
        diff = np.abs(kerr0[k] - ks[k]).max()
        bound = 1e-12 * np.abs(ks[k]).max() + 1e-18
        assert diff <= bound, f"one-step {k}: |diff| {diff:.3e} exceeds {bound:.3e}"


@needs_backend
def test_one_step_generates_no_angular_momentum() -> None:
    # from uniform data (v = 0 everywhere) nothing sources or fluxes S_phi in a single
    # step: the axisymmetric covariant source is identically zero and the HLL flux of a
    # uniform state is the (zero) analytic flux. measured 1.7e-15.
    out = _one_step("kerr", 0.9)
    assert np.abs(out["m3"]).max() < 1e-12, (
        f"one-step S_phi from uniform data: {np.abs(out['m3']).max():.3e}"
    )


@needs_backend
def test_kerr_zero_spin_matched_dt_trajectory_is_machine_exact() -> None:
    # with the dt sequence PINNED (max_dt below both charts' CFL floors), the kerr(a = 0)
    # and kerr-schild trajectories differ only by ULP reassociation: measured 7e-15 on the
    # flowing fields over ~10^4 identical steps, m3 bitwise zero. m2 is the roundoff residue
    # of the exact well-balanced cancellation (noise-scale field) — normalized by the flow
    # scale like the transient check.
    def traj(spacetime):
        with tempfile.TemporaryDirectory() as d:
            d = d + "/"
            p = _kerr_problem(d, spacetime, 0.0)
            p.max_dt = 1e-3
            runner.run(p, compute_mode="cpu")
            with h5py.File(glob.glob(os.path.join(d, "*final*.h5"))[0]) as h:
                c = h["level_0/conserved"]
                return {k: c[k][:] for k in ("den", "m1", "m2", "m3", "nrg")}

    kerr0 = traj("kerr")
    ks = traj("kerr_schild")
    m_scale = np.abs(ks["m1"]).max()
    for k in ("den", "m1", "m2", "m3", "nrg"):
        scale = max(np.abs(ks[k]).max(), 1e-3 * m_scale)
        e = np.abs(kerr0[k] - ks[k]).max() / scale
        # 1e4 steps of ULP accumulation put the cancellation-noise m2 at ~1e-12 scaled.
        assert e < 1e-11, f"matched-dt {k}: scaled max diff {e:.3e}"


@needs_backend
def test_kerr_at_zero_spin_matches_the_ks_chart() -> None:
    kerr = _run("kerr", 0.0)
    ks = _run("kerr_schild", 0.0)
    # same physics, different kernels AND a different CFL map (the kerr light-cone bound vs
    # the kerr-schild banyuls-font speeds), so the dt sequences differ and the trajectories
    # separate at the truncation floor — measured 9.3e-8 max after t = 10. a wiring error
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
    # the S_phi law is source-free (axisymmetry) and BOTH S_phi generators are closed:
    # the flux kernel reconstructs the angular-momentum-carrying variable
    # w = v^phi + (gamma_{r phi}/gamma_{phi phi}) v^r (dragging states have w = 0 to
    # roundoff at every face), and the kerr ghost fill copies w too (a raw v^phi copy
    # would violate the dragging relation at the ghost's shifted radius — that boundary
    # generator alone put |S_phi| at 1.0e-3). S_phi is conserved to ROUNDOFF over the
    # full trajectory: measured 1.8e-15 at 96x16 over ~2000 steps.
    assert np.abs(out["m3"]).max() < 1e-12, (
        f"S_phi beyond roundoff: {np.abs(out['m3']).max():.3e}"
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
    # parity: every kernel operation is IEEE parity-exact in the spin and the +-a runs
    # share the dt sequence (the light-cone CFL map is even in a), so the trajectory-level
    # antisymmetry is BITWISE (measured exactly 0.0 over ~2000 steps).
    e_flip = np.abs(out["v3"] + out_m["v3"]).max() / np.abs(out["v3"]).max()
    assert e_flip < 1e-14, f"v^phi is not spin-antisymmetric: {e_flip:.3e}"
    e_even = np.abs(out["rho"] - out_m["rho"]).max() / np.abs(out["rho"]).max()
    assert e_even < 1e-14, f"rho is not spin-even: {e_even:.3e}"
    # roundoff-scale S_phi at the refined resolution too (measured 2.2e-15 at 192x32) —
    # a truncation-scale generator would grow visible orders above this floor.
    hi = _run_m3_max(2 * _NR, 2 * _NPOLAR)
    assert hi < 1e-12, f"S_phi beyond roundoff at 192x32: {hi:.3e}"


def _run_m3_max(nr: int, npolar: int) -> float:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = _kerr_problem(d, "kerr", 0.9)
        p.nr = nr
        p.npolar = npolar
        p.resolution = (nr, npolar)
        runner.run(p, compute_mode="cpu")
        with h5py.File(glob.glob(os.path.join(d, "*final*.h5"))[0]) as h:
            return float(np.abs(h["level_0/conserved/m3"][:]).max())


@needs_backend
def test_spinning_fm_disk_holds_through_half_an_orbit() -> None:
    # the paper-certified fishbone-moncrief disk at a = 0.9 on the horizon-penetrating
    # kerr grid (the science configuration): completes half an orbital period at its
    # pressure maximum with no floors, the disk core keeps its density and its
    # rotation sense, and the corona-fed through-horizon inflow stays positive.
    from simbi_configs.examples.grmhd.gr_fishbone_moncrief import GrFishboneMoncrief

    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        # a bound, warm spinning disk: r_in = 3.5 (well outside r_ms(0.9) ~ 2.32),
        # kappa = 1.03 -> r_max ~ 10, closed outer edge ~ 54, core p/rho ~ 6.5e-3
        # (the a = 0 defaults give an ultra-cold shallow disk at spin — the potential
        # depth is set by r_in relative to the SPIN-dependent marginally stable orbit).
        p = GrFishboneMoncrief.from_cli(
            ["--nr", "96", "--npolar", "32", "--kerr-spin", "0.9",
             "--r-in", "3.5", "--kappa", "1.03"]
        )
        p.end_time = 50.0
        p.data_directory = d
        p.checkpoint_interval = 50.0
        runner.run(p, compute_mode="cpu")
        finals = glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
        assert finals, "spinning FM disk crashed before completion"
        with h5py.File(finals[0]) as h:
            g = h["level_0/partition_0/hydro/primitives"]
            shp = g["rho"].shape
            halo = [(s - n) // 2 for s, n in zip(shp, (32, 96))]
            sl = tuple(slice(hh, hh + n) for hh, n in zip(halo, (32, 96)))
            rho, pre, v3 = g["rho"][sl], g["pre"][sl], g["v3"][sl]
    assert pre.min() > 0.0, f"pressure went non-positive: {pre.min():.3e}"
    # the disk core (well above the corona) survives and corotates with the hole.
    core = rho > 0.1
    assert core.sum() > 50, f"the disk core dispersed: {core.sum()} cells above 0.1"
    assert (v3[core] > 0).mean() > 0.95, "the disk core lost its rotation sense"
