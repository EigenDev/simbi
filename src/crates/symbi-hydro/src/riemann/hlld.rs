// =============================================================================
// riemann/hlld.rs
//
// the HLLD five-wave solver for RMHD: resolves fast magnetoacoustic, alfven,
// and contact waves via a secant pressure iteration (mignone, ugliano & bodo
// 2009). carrier-generic: every comparison goes through `S::cmp_*` / `S::select` /
// `S::branch`, the secant loop runs through `Scalar::iterate_vec` (fixed-step body,
// freeze-on-converged), and the HLLE fallback is computed eagerly with the final
// flux choice selected by a success mask. ONE Riemann source, two backends
// (host f64 + traced Gv).
// =============================================================================

use crate::energy::Zero;
use crate::eos::Eos;
use crate::isothermal_mhd::IsothermalMhd;
use crate::mhd_state::{IsoMhdCons, IsoMhdPrim, MhdCons, MhdPrim};
use crate::newtonian_mhd::NewtonianMhd;
use crate::regime::Regime;
use crate::riemann::hlle;
use crate::rmhd::Rmhd;
use crate::spatial_metric::SpatialMetric;
use crate::state::{Cons, ConsG, Prim};
use symbi_algebra::Tensor;
use symbi_ir::algebra::{Scalar, Selectable};

/// physical-consistency tolerance for the intermediate-state checks.
const CONSISTENCY_TOL: f64 = 1e-12;
/// secant convergence tolerance on the f-function.
const CONVERGENCE_TOL: f64 = 1e-12;
/// divergence guard: an f-value exceeding this is taken as an unphysical
/// state and triggers the HLLE fallback at the final select.
const DIVERGENCE_GUARD: f64 = 1e30;
/// initial secant perturbation off the pressure guess.
const SECANT_PERTURBATION: f64 = 1e-6;
/// low-B regime threshold: below `bn^2/p < this`, use the low-B quadratic
/// pressure estimate replaces the HLL recovered pressure.
const LOW_B_PRESSURE_RATIO: f64 = 0.01;
/// secant iteration count (`max_iter = 15`).
const SECANT_STEPS: usize = 15;
/// relative degeneracy tolerance for the transverse-star denominator at the
/// alfven resonance, shared by the newtonian and isothermal hlld solvers. the
/// transverse star fields divide by a denominator that vanishes when the fast
/// wave coincides with an alfven wave. switch to the no-rotation limit when
/// the denominator is small relative to the magnitude of its two
/// cancelling constituents.
const ALFVEN_DEGENERACY_TOL: f64 = 1e-3;

/// the vdiff helper result: intermediate states + the f-function value at
/// the trial pressure. unphysical inputs leave `f` saturated at
/// `DIVERGENCE_GUARD * 10` (so the secant + final-select treat them as
/// divergent without needing a host bool sentinel).
struct VdiffOut<S: Scalar, const D: usize> {
    f: S,
    vv: [Tensor<S, D>; 2],
    bv: [Tensor<S, D>; 2],
    alf: [S; 2],
    vc: Tensor<S, D>,
    bc: Tensor<S, D>,
}

/// branchless HLLD intermediate-state + f-function. carrier-generic: no
/// native `<`/`>`/`if`. unphysical states (x ~= 0 or any of the 8-way
/// consistency checks failing) saturate the returned `f` to a value the
/// outer divergence-guard catches.
fn hlld_vdiff<S: Scalar, const D: usize>(
    p: S,
    r: [&MhdCons<S, D>; 2],
    lam: [S; 2],
    bn: S,
    nhat: &Tensor<S, D>,
    metric: &SpatialMetric<S, D>,
) -> VdiffOut<S, D> {
    let one = S::ONE;
    let zero = S::ZERO;
    let rel = S::from_f64(32.0 * f64::EPSILON);
    let sgn_bn = S::select(bn.cmp_ge(zero), one, -one);

    let big = S::from_f64(DIVERGENCE_GUARD * 10.0);
    // start with `physical = true`; bitand failed checks at the end.
    let mut physical: S::Mask = bn.cmp_ge(bn); // tautology, all-true mask

    let mut eta = [zero; 2];
    let mut enthalpy = [zero; 2];
    let mut kv = [Tensor::<S, D>::zeros(); 2];
    let mut vv = [Tensor::<S, D>::zeros(); 2];
    let mut bv = [Tensor::<S, D>::zeros(); 2];

    for ii in 0..2 {
        let a_s = lam[ii];
        let rs = r[ii];
        // the SR MUB09 star-state reconstruction (Mignone, Ugliano & Bodo 2009). the GR path does
        // NOT drive this in the coordinate frame: it maps to the local ORTHONORMAL frame where the
        // metric is identity, runs THIS flat reconstruction, and maps the flux back (the metric
        // contractions below are euclidean at the flat metric the GR wrapper passes).
        let rmn = rs.mom.dot(nhat);
        let rmtrans = rs.mom - nhat.scale(rmn);
        let rbtrans = metric.project_transverse(&rs.mag, nhat); // transverse magnetic part
        let ret = rs.nrg + rs.den;

        // Eqs (26)-(30): coefficients of the linear system for the side state.
        let a = rmn - a_s * ret + p * (one - a_s * a_s);
        let g = metric.norm_sq_contra(&rbtrans); // |rbtrans|^2
        let ag = a + g;
        let c = rbtrans.dot(&rmtrans);
        let q = -ag + bn * bn * (one - a_s * a_s);
        let x = bn * (a * a_s * bn + c) - ag * (a_s * p + ret);

        let x_scale = (bn * (a * a_s * bn + c))
            .abs()
            .max((ag * (a_s * p + ret)).abs());
        let x_ok = x.abs().cmp_gt(rel * x_scale);
        physical = physical & x_ok;
        let safe_x = S::select(x_ok, x, one);
        let inv_x = one / safe_x;

        // Eqs (23)-(25): mignone, ugliano & bodo 2009.
        let term = c + bn * (a_s * rmn - ret);
        let vn = (bn * (a * bn + a_s * c) - ag * (p + rmn)) * inv_x;
        let vtrans = (rmtrans.scale(q) + rbtrans.scale(term)).scale(inv_x);

        // Eq (21).
        let den_v = a_s - vn;
        let den_v_ok = den_v.abs().cmp_gt(rel * a_s.abs().max(vn.abs()));
        physical = physical & den_v_ok;
        let var1 = one / S::select(den_v_ok, den_v, one);
        let btrans = (rbtrans - vtrans.scale(bn)).scale(var1);

        // Eq (31).
        let rdv = vn * rmn + vtrans.dot(&rmtrans);
        let wt = p + (ret - rdv) * var1;
        enthalpy[ii] = wt;

        // Eqs (35) & (43): note the eta sign flips per side, captured as a const.
        let sign_lit = if ii == 0 { -1.0 } else { 1.0 };
        let sign = S::from_f64(sign_lit);
        eta[ii] = sign * sgn_bn * wt.abs().sqrt();
        let eta_s = eta[ii];
        let den_k = a_s * p + ret + bn * eta_s;
        let den_k_scale = (a_s * p).abs().max(ret.abs()).max((bn * eta_s).abs());
        let den_k_ok = den_k.abs().cmp_gt(rel * den_k_scale);
        physical = physical & den_k_ok;
        let var2 = one / S::select(den_k_ok, den_k, one);
        let kn = (rmn + p + rs.mag.dot(nhat) * eta_s) * var2;
        let ktrans = (rmtrans + rbtrans.scale(eta_s)).scale(var2);

        bv[ii] = nhat.scale(bn) + btrans;
        vv[ii] = nhat.scale(vn) + vtrans;
        kv[ii] = nhat.scale(kn) + ktrans;
    }

    // K-vector + velocity are contravariant -> metric contractions.
    let alf_l = kv[0].dot(nhat);
    let alf_r = kv[1].dot(nhat);
    let alf = [alf_l, alf_r];
    let vn_l = vv[0].dot(nhat);
    let vn_r = vv[1].dot(nhat);

    // bn -> 0 degeneracy flag: the alfven waves collapse onto the contact, so the K-speed
    // spread `dkn = alf_r - alf_l` is DRIVEN to zero by the pressure iteration itself (f =
    // dkn is the residual on this path). a vanishing dkn is then the converged degenerate
    // geometry, not an unphysical state, so the dkn-proportional guards below act as pure
    // DIVISION guards here and must not fold into `physical`: folding them made the
    // success verdict the roundoff bit of whether the residual cleared the epsilon, and
    // that bit selects between the fan and the O(1)-different HLLE fallback.
    let b_scale = metric
        .norm_sq_contra(&bv[0])
        .max(metric.norm_sq_contra(&bv[1]))
        .sqrt();
    let bn_small = bn.abs().cmp_le(rel * b_scale);

    // Eq (45): contact B-field.
    let dkn = alf_r - alf_l;
    let dkn_scale = alf_r.abs().max(alf_l.abs());
    let dkn_ok = dkn.abs().cmp_gt(rel * dkn_scale);
    physical = physical & (dkn_ok | bn_small);
    let inv_dkn = one / S::select(dkn_ok, dkn, one);
    let bc = ((bv[1].scale(alf_r - vn_r) + vv[1].scale(bn))
        - (bv[0].scale(alf_l - vn_l) + vv[0].scale(bn)))
    .scale(inv_dkn);

    // Eq (47): contact velocity.
    let ksq_l = metric.norm_sq_contra(&kv[0]);
    let kdb_l = metric.contract_contra(&kv[0], &bc);
    let vc_den_l = eta[0] - kdb_l;
    let vc_den_l_ok = vc_den_l.abs().cmp_gt(rel * eta[0].abs().max(kdb_l.abs()));
    physical = physical & vc_den_l_ok;
    let vc_l = kv[0] - bc.scale((one - ksq_l) / S::select(vc_den_l_ok, vc_den_l, one));

    let ksq_r = metric.norm_sq_contra(&kv[1]);
    let kdb_r = metric.contract_contra(&kv[1], &bc);
    let vc_den_r = eta[1] - kdb_r;
    let vc_den_r_ok = vc_den_r.abs().cmp_gt(rel * eta[1].abs().max(kdb_r.abs()));
    physical = physical & vc_den_r_ok;
    let vc_r = kv[1] - bc.scale((one - ksq_r) / S::select(vc_den_r_ok, vc_den_r, one));

    // Eq (49) — denominator pre-clamped to avoid divide-by-zero in Gv.
    let denom_l = (eta[0] - kdb_l) * dkn;
    let denom_r = (eta[1] - kdb_r) * dkn;
    let denom_l_scale = (eta[0] * dkn).abs().max((kdb_l * dkn).abs());
    let denom_r_scale = (eta[1] * dkn).abs().max((kdb_r * dkn).abs());
    let denom_l_ok = denom_l.abs().cmp_gt(rel * denom_l_scale);
    let denom_r_ok = denom_r.abs().cmp_gt(rel * denom_r_scale);
    // dkn-proportional: division guards only when bn -> 0 (y_l/y_r feed f_default, which
    // the bn-small residual path never consumes).
    physical = physical & ((denom_l_ok & denom_r_ok) | bn_small);
    let y_l = (one - ksq_l) / S::select(denom_l_ok, denom_l, one);
    let y_r = (one - ksq_r) / S::select(denom_r_ok, denom_r, one);

    // Eq (48), three-way branchless select on the magnitude of dkn and bn.
    let dkn_small = !dkn_ok;
    let f_default = dkn * (one - bn * (y_r - y_l));
    let f_when_bn_small = dkn;
    let f_inner = S::select(bn_small, f_when_bn_small, f_default);
    let f_val_pre = S::select(dkn_small, zero, f_inner);

    // Eq (54): the 8-way physical consistency check, bitand of masks.
    let tol = S::from_f64(CONSISTENCY_TOL);
    let neg_tol = -tol;
    let c1 = (vn_l - alf_l).cmp_gt(neg_tol);
    let c2 = (alf_r - vn_r).cmp_gt(neg_tol);
    let c3 = (lam[0] - vn_l).cmp_lt(zero);
    let c4 = (lam[1] - vn_r).cmp_gt(zero);
    let c5 = (enthalpy[1] - p).cmp_gt(zero);
    let c6 = (enthalpy[0] - p).cmp_gt(zero);
    let c7 = (alf_l - lam[0]).cmp_gt(neg_tol);
    let c8 = (lam[1] - alf_r).cmp_gt(neg_tol);
    physical = physical & c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8;

    // unphysical -> sentinel saturating the outer divergence guard.
    let f = S::select(physical, f_val_pre, big);
    let vc = (vc_l + vc_r).scale(S::from_f64(0.5));

    VdiffOut {
        f,
        vv,
        bv,
        alf,
        vc,
        bc,
    }
}

/// the converged five-wave fan: the secant-final intermediate state + the success mask. SHARED by
/// the flux path (`hlld_rmhd`) and the UCT-HLLD edge-EMF star-state extraction (`hlld_rmhd_states`).
/// `success` is FALSE iff the secant diverged or the state is unphysical (the flux then falls back
/// to HLLE) — in that case every field below is GARBAGE and the consumer MUST NOT trust it.
struct HlldConverged<S: Scalar, const D: usize> {
    p_final: S,
    v: VdiffOut<S, D>,
    success: S,
}

/// run the secant pressure iteration to convergence and re-read the intermediate state at `p_final`.
/// `success` returned as a 0/1 scalar (1 = converged + physical). carrier-generic.
fn hlld_rmhd_converge<S: Scalar, const D: usize>(
    p_init: S,
    r_pair: [&MhdCons<S, D>; 2],
    lam: [S; 2],
    bn: S,
    nhat: &Tensor<S, D>,
    metric: &SpatialMetric<S, D>,
) -> HlldConverged<S, D> {
    let one = S::ONE;
    let zero = S::ZERO;
    let feps = S::from_f64(CONVERGENCE_TOL);
    let relative_eps = S::from_f64(32.0 * f64::EPSILON);
    let div_guard = S::from_f64(DIVERGENCE_GUARD);

    let p_perturbed = p_init * (one + S::from_f64(SECANT_PERTURBATION));
    let f_init = hlld_vdiff(p_init, r_pair, lam, bn, nhat, metric).f;

    let result_acc = S::iterate_vec::<4>(
        [p_init, f_init, p_perturbed, zero],
        SECANT_STEPS,
        |[p_prev, f_prev, p_cur, freeze]| {
            let frozen = freeze.cmp_gt(S::from_f64(0.5));
            let v = hlld_vdiff(p_cur, r_pair, lam, bn, nhat, metric);
            let f_cur = v.f;
            let f_too_big = f_cur.abs().cmp_gt(div_guard);
            let slope = f_cur - f_prev;
            let slope_scale = f_cur.abs().max(f_prev.abs());
            let slope_dead = slope.abs().cmp_le(relative_eps * slope_scale);
            let diverged = f_too_big | slope_dead;
            let slope_denom = S::select(slope_dead, one, slope);
            let dp = (p_cur - p_prev) / slope_denom * f_cur;
            let p_next_naive = p_cur - dp;
            let ptol = p_cur.abs() * feps;
            let converged_now = dp.abs().cmp_le(ptol) & f_cur.abs().cmp_lt(feps);
            let new_freeze = S::select(frozen | diverged | converged_now, one, zero);
            let p_prev_next = S::select(frozen, p_prev, p_cur);
            let f_prev_next = S::select(frozen, f_prev, f_cur);
            let p_cur_next = S::select(frozen, p_cur, p_next_naive);
            [p_prev_next, f_prev_next, p_cur_next, new_freeze]
        },
        |_prev, [_p_prev, _f_prev, _p_cur, freeze]| freeze.cmp_gt(S::from_f64(0.5)),
        2,
    );

    let v = hlld_vdiff(result_acc, r_pair, lam, bn, nhat, metric);
    let ok = v.f.abs().cmp_lt(div_guard) & v.f.abs().cmp_lt(feps * S::from_f64(1e6));
    let success = S::select(ok, one, zero);
    HlldConverged {
        p_final: result_acc,
        v,
        success,
    }
}

/// HLLD five-wave solver for RMHD. carrier-generic — single body for f64 host
/// and Gv-traced kernel. on unphysical state or secant divergence, falls back
/// to HLLE via a final mask-driven `select` over the flux. matches mignone,
/// ugliano & bodo (2009).
/// GR HLLD via the local ORTHONORMAL frame (TETRAD form). the tetrad E = `metric.orthonormal_basis(dir)`
/// maps the physical contravariant velocity/field into the frame where the spatial metric is the
/// identity: V_hat = E^{-1} v, B_hat = E^{-1} B (whence W_hat = W, |B_hat|^2 = |B|^2, V_hat.B_hat = v.B
/// — every Lorentz scalar preserved). the VALIDATED flat MUB09 solver runs there and the intercell
/// flux maps back EXACTLY, with the single normal factor E_dd = E[dir][dir]:
///   F_D   = E_dd F_hat_D,             F_tau = E_dd F_hat_tau,
///   F_B^i = E_dd (E F_hat_B)^i        (contravariant induction flux),
///   F_S_j = E_dd (E^{-T} F_hat_S)_j   (covariant momentum; the Valencia pressure term is the MIXED
///           p_tot delta^n_j, so it cancels exactly under the transform — no residual).
/// the ALE face speed rides along as vface_hat = vface / E_dd. exact for ANY symmetric-positive
/// spatial metric: diagonal Schwarzschild/KS (E diagonal, reduces to the sqrt(g_i) scaling) and the
/// non-diagonal spinning-Kerr gamma alike.
pub fn hlld_rmhd_gr_ortho<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    dir: usize,
    vface: S,
    metric: &SpatialMetric<S, D>,
) -> MhdCons<S, D> {
    let nhat = Tensor::<S, D>::unit(dir);
    let e = metric.orthonormal_basis(dir);
    let e_t = e.transpose();
    let e_dd = e[(dir, dir)];
    // the tetrad is orthonormal w.r.t. gamma, so the FORWARD map needs no matrix inverse:
    // V_hat^a = gamma(v, e_hat_a) = (E^T gamma v)_a. the covariant momentum flux maps back with
    // T^{-T} = gamma E (F_S = E_dd gamma (E F_hat_S)); the contravariant field maps with E.
    let to_hat = |p: &MhdPrim<S, D>| MhdPrim {
        hydro: Prim {
            rho: p.hydro.rho,
            vel: e_t.mul_vec(&metric.lower(&p.hydro.vel)),
            pre: p.hydro.pre,
        },
        mag: e_t.mul_vec(&metric.lower(&p.mag)),
    };
    let hat_l = to_hat(prim_l);
    let hat_r = to_hat(prim_r);
    let fhat = hlld_rmhd(
        &Rmhd,
        eos,
        &hat_l,
        &hat_r,
        &nhat,
        vface / e_dd,
        &SpatialMetric::flat(),
    );
    MhdCons {
        hydro: Cons {
            den: fhat.hydro.den * e_dd,
            mom: metric.lower(&e.mul_vec(&fhat.hydro.mom)).scale(e_dd),
            nrg: fhat.hydro.nrg * e_dd,
        },
        mag: e.mul_vec(&fhat.mag).scale(e_dd),
    }
}

/// the converged HLLD fan for the GR UCT-HLLD edge EMF, via the ORTHONORMAL frame (the states-only
/// analogue of `hlld_rmhd_gr_ortho`). the flat MUB09 star states are extracted in the tetrad frame
/// and mapped back to the coordinate frame with the SAME tetrad rules that make the wave-sum EMF
/// telescope to the coordinate B_t flux: a CONTRAVARIANT field/velocity maps as X = E X_hat; a wave
/// SPEED (dx^n/dt) maps as lambda = E_dd lambda_hat; the normal field maps as B^n = E_dd B_hat^n. in
/// the wave-sum Phi = sum |lambda_k| (B_t^{star} - B_t) these combine exactly to the contravariant
/// B_t flux, so Phi IS the coordinate EMF. exact for any symmetric-positive spatial metric (diagonal
/// Schwarzschild/KS and non-diagonal Kerr alike).
pub fn hlld_rmhd_states_gr_ortho<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    dir: usize,
    metric: &SpatialMetric<S, D>,
) -> HlldStates<S, D> {
    let nhat = Tensor::<S, D>::unit(dir);
    let e = metric.orthonormal_basis(dir);
    let e_t = e.transpose();
    let e_dd = e[(dir, dir)];
    // forward map (no inverse): V_hat^a = gamma(v, e_hat_a) = (E^T gamma v)_a.
    let to_hat = |p: &MhdPrim<S, D>| MhdPrim {
        hydro: Prim {
            rho: p.hydro.rho,
            vel: e_t.mul_vec(&metric.lower(&p.hydro.vel)),
            pre: p.hydro.pre,
        },
        mag: e_t.mul_vec(&metric.lower(&p.mag)),
    };
    let st = hlld_rmhd_states(
        &Rmhd,
        eos,
        &to_hat(prim_l),
        &to_hat(prim_r),
        &nhat,
        &SpatialMetric::flat(),
    );
    // contravariant field/velocity: X = E X_hat.
    let up = |v: &Tensor<S, D>| e.mul_vec(v);
    HlldStates {
        lam: [st.lam[0] * e_dd, st.lam[1] * e_dd],
        alf: [st.alf[0] * e_dd, st.alf[1] * e_dd],
        lstar: st.lstar * e_dd,
        bstar: [up(&st.bstar[0]), up(&st.bstar[1])],
        bc: up(&st.bc),
        vc: up(&st.vc),
        bn: st.bn * e_dd,
        success: st.success,
    }
}

pub fn hlld_rmhd<S: Scalar, const D: usize, R>(
    regime: &R,
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
    metric: &SpatialMetric<S, D>,
) -> MhdCons<S, D>
where
    R: Regime<S, D, Prim = MhdPrim<S, D>, Cons = MhdCons<S, D>>,
{
    let zero = S::ZERO;
    let one = S::ONE;
    let half = S::from_f64(0.5);

    // eagerly compute HLLE — fallback if HLLD reports divergence at the end.
    let hlle_flux = hlle(regime, eos, prim_l, prim_r, nhat, vface);

    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (a_l, a_r) = regime.extremal_speeds(eos, prim_l, prim_r, nhat);

    // wave-bracket: supersonic-L / supersonic-R arms are the plain side
    // fluxes minus the ALE vface drift. wrap as branches so traced kernels
    // short-circuit cleanly. all three arms get computed at S = Gv (S::branch
    // evaluates both); the select picks the right one.
    let supersonic_l = a_l.cmp_ge(vface);
    let supersonic_r = a_r.cmp_le(vface);

    let dwave = a_r - a_l;
    let dwave_ok = dwave
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * a_r.abs().max(a_l.abs()));
    let inv_dwave = one / S::select(dwave_ok, dwave, one);
    let hll_state = (u_r * a_r - u_l * a_l - f_r + f_l) * inv_dwave;
    let hll_flux = (f_l * a_r - f_r * a_l + (u_r - u_l) * (a_l * a_r)) * inv_dwave;
    let bn = hll_state.mag.dot(nhat); // B.n (contravariant)

    // r-vectors for prim_l and prim_r (Eq. 12).
    let r_l = u_l * a_l - f_l;
    let r_r = u_r * a_r - f_r;
    let r_pair = [&r_l, &r_r];
    let lam = [a_l, a_r];

    // initial pressure guess: L+R average of TOTAL pressure (gas + magnetic).
    // recovering the HLL state's primitive pressure via `regime.to_primitive`
    // would require `OrderedNumeric` and isn't
    // Gv-traceable (the c2p error code is host-only). the L+R average is
    // strictly positive, monotonic in inputs, and the secant converges fine
    // from it — the only cost is a few more iterations on extreme shocks.
    let p_total = |prim: &MhdPrim<S, D>| -> S {
        prim.hydro.pre + half * metric.norm_sq_contra(&prim.mag) // |B|^2 (contravariant)
    };
    let p_left = p_total(prim_l);
    let p_right = p_total(prim_r);
    let p_scale = p_left.abs().max(p_right.abs());
    let p_floor_val = S::from_f64(CONVERGENCE_TOL) * p_scale;
    let p_hll_raw = (p_left + p_right) * half;
    let p_hll = S::select(p_hll_raw.cmp_le(zero), p_floor_val, p_hll_raw);
    let _ = &regime; // silence unused-binding warning in this carrier-generic path

    // low-B branch: when bn^2 / p < ratio, use the quadratic pressure estimate.
    let lowb_thresh = S::from_f64(LOW_B_PRESSURE_RATIO);
    let lowb_mask = (bn * bn / p_hll).cmp_lt(lowb_thresh);

    let et_hll = hll_state.nrg + hll_state.den;
    let fet_hll = hll_flux.nrg + hll_flux.den;
    let mn_hll = hll_state.mom.dot(nhat);
    let fmn_hll = hll_flux.mom.dot(nhat);
    let bb_q = et_hll - fmn_hll;
    let cc_q = fet_hll * mn_hll - et_hll * fmn_hll;
    let disc = (bb_q * bb_q - S::from_f64(4.0) * cc_q).max(zero);
    let p_lowb_raw = half * (-bb_q + disc.sqrt());
    let p_lowb = S::select(p_lowb_raw.cmp_le(zero), p_floor_val, p_lowb_raw);

    let p_init = S::select(lowb_mask, p_lowb, p_hll);

    // secant pressure iteration -> converged intermediate state (shared with the UCT-HLLD edge-EMF
    // star-state extraction). `success` is a 0/1 scalar; the flux falls back to HLLE when it is 0.
    let conv = hlld_rmhd_converge(p_init, r_pair, lam, bn, nhat, metric);
    let p_final = conv.p_final;
    let v = conv.v;
    let vc = v.vc;
    let bc = v.bc;
    let alf = v.alf;
    let vv_iso = v.vv;
    let bv_iso = v.bv;
    let success = conv.success.cmp_gt(half) & dwave_ok;

    // pick fast-wave side via the contact-vs-vface test, branchless.
    let vnc = vc.dot(nhat); // v_c.n (contravariant)
    let on_left = vface.cmp_lt(vnc);

    // ---- fast-wave intermediate state (Section 3.1) ----
    let make_fast = |u_side: MhdCons<S, D>,
                     f_side: MhdCons<S, D>,
                     r_side: MhdCons<S, D>,
                     lc: S,
                     va: Tensor<S, D>,
                     ba: Tensor<S, D>|
     -> (MhdCons<S, D>, MhdCons<S, D>) {
        let vdba = metric.contract_contra(&va, &ba); // v.B (contravariant)
        let vna = va.dot(nhat); // v.n (contravariant)
        let denominator = lc - vna;
        let denominator_ok = denominator
            .abs()
            .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * lc.abs().max(vna.abs()));
        let inv_lc_vna = one / S::select(denominator_ok, denominator, one);
        let da = r_side.den * inv_lc_vna;
        let ea = (r_side.nrg + r_side.den + p_final * vna - vdba * bn) * inv_lc_vna;
        let ma = va.scale(ea + p_final) - ba.scale(vdba);
        let ua = MhdCons {
            hydro: Cons {
                den: da,
                mom: ma,
                nrg: ea - da,
            },
            mag: ba,
        };
        let fa = f_side + (ua - u_side) * lc;
        (ua, fa)
    };

    let (ua_l, fa_l) = make_fast(u_l, f_l, r_l, a_l, vv_iso[0], bv_iso[0]);
    let (ua_r, fa_r) = make_fast(u_r, f_r, r_r, a_r, vv_iso[1], bv_iso[1]);

    // pick per-side `ua` / `fa` / `lc` / `la` for the at-contact arm.
    let ua = MhdCons::select(on_left, ua_l, ua_r);
    let fa = MhdCons::select(on_left, fa_l, fa_r);
    let _lc = S::select(on_left, a_l, a_r); // fast-wave speed picked per-side; consumed by make_fast above
    let la = S::select(on_left, alf[0], alf[1]);

    // "between fast and Alfven" (upstream of the contact): flux is fa - ua * vface.
    // "between Alfven and contact": include the contact-wave correction.
    let at_contact_l = vface.cmp_ge(alf[0]);
    let at_contact_r = vface.cmp_le(alf[1]);
    let at_contact = S::Mask::select_mask(on_left, at_contact_l, at_contact_r);

    // ---- contact-wave state (Section 3.3) ----
    let vdbc = metric.contract_contra(&vc, &bc); // v_c.B_c (contravariant)
    let vna_used = S::select(on_left, vv_iso[0].dot(nhat), vv_iso[1].dot(nhat));
    // `la - vnc` is analytically ZERO when bn -> 0 (the alfven waves collapse onto the
    // contact), so this comparison decides on pure roundoff there; it must therefore gate
    // only the EMPTY contact wedge (see the flux_hlld select below), never an HLLE
    // fallback — either arm of a roundoff-decided select has to produce the same flux.
    let contact_denominator = la - vnc;
    let contact_ok = contact_denominator
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * la.abs().max(vnc.abs()));
    let inv_la_vnc = one / S::select(contact_ok, contact_denominator, one);
    let dc = ua.den * (la - vna_used) * inv_la_vnc;
    let man = ua.mom.dot(nhat);
    let ec = (ua.nrg + ua.den) * la;
    let ec = (ec - man + p_final * vnc - vdbc * bn) * inv_la_vnc;
    let mc = vc.scale(ec + p_final) - bc.scale(vdbc);
    let ut = MhdCons {
        hydro: Cons {
            den: dc,
            mom: mc,
            nrg: ec - dc,
        },
        mag: bc,
    };

    let flux_fast = fa - ua * vface;
    let flux_contact = fa + (ut - ua) * la - ut * vface;
    // degenerate contact (la -> vnc, i.e. bn -> 0): the alfven waves coincide with the
    // contact, the wedge between them is EMPTY, and the fan reduces to the per-side
    // single-star (fast) state — the HLLC-like limit, not a failure. routing this case
    // to flux_fast (instead of an HLLE fallback) keeps the flux continuous across the
    // roundoff-decided contact_ok boundary: with an empty wedge, flux_contact is never
    // the selected arm, so both sides of the select agree.
    let flux_hlld = MhdCons::select(at_contact & contact_ok, flux_contact, flux_fast);

    // wave-bracket select chain.
    let flux_supersonic_l = f_l - u_l * vface;
    let flux_supersonic_r = f_r - u_r * vface;

    let fast_den_l = a_l - vv_iso[0].dot(nhat);
    let fast_den_r = a_r - vv_iso[1].dot(nhat);
    let fast_ok_l = fast_den_l
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * a_l.abs().max(vv_iso[0].dot(nhat).abs()));
    let fast_ok_r = fast_den_r
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * a_r.abs().max(vv_iso[1].dot(nhat).abs()));
    let fast_ok = S::Mask::select_mask(on_left, fast_ok_l, fast_ok_r);
    let flux_inner = MhdCons::select(success & fast_ok, flux_hlld, hlle_flux);
    let flux_choose_r = MhdCons::select(supersonic_r, flux_supersonic_r, flux_inner);
    MhdCons::select(supersonic_l, flux_supersonic_l, flux_choose_r)
}

/// the converged HLLD five-wave fan for the UCT-HLLD edge EMF (Mignone & Del Zanna 2020, sec. 5.2). all
/// speeds + the per-side SINGLE-STAR (post-fast, Section 3.1) lab-frame B field, plus `success`.
/// CONSUMER CONTRACT: when `success == 0` the gas flux fell back to HLLE and EVERY field here is
/// garbage — the EMF must use the HLL coefficients there. ordering (when success): `lam[0] <= alf[0]
/// <= lstar <= alf[1] <= lam[1]`. `bstar[s]` is lab-frame B; transverse part = `bstar[s] - n*bn`.
pub struct HlldStates<S: Scalar, const D: usize> {
    pub lam: [S; 2],              // outermost FAST speeds [lambda^L, lambda^R]
    pub alf: [S; 2],              // rotational/Alfven speeds [lambda*^L, lambda*^R] (MUB09)
    pub lstar: S,                 // contact speed lambda* = vc . n
    pub bstar: [Tensor<S, D>; 2], // per-side single-star B (lab-frame): B*^{L}, B*^{R}
    pub bc: Tensor<S, D>, // CONTACT transverse field B_c (MUB09 Eq. 45) — B_t^{ss}, the EMF chi-jump target
    pub vc: Tensor<S, D>, // CONTACT velocity v_c (MUB09 Eq. 47); v_c.n == lstar
    pub bn: S,            // the (single, div-free) normal field
    pub success: S,       // 1.0 = HLLD converged + physical; 0.0 = HLLE fallback (garbage)
}

/// extract the converged HLLD fan for the edge-EMF coefficients WITHOUT building the flux. shares
/// `hlld_rmhd_converge` with `hlld_rmhd` (identical secant), so the states are bit-consistent with
/// the flux solve. carrier-generic.
pub fn hlld_rmhd_states<S: Scalar, const D: usize, R>(
    regime: &R,
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    metric: &SpatialMetric<S, D>,
) -> HlldStates<S, D>
where
    R: Regime<S, D, Prim = MhdPrim<S, D>, Cons = MhdCons<S, D>>,
{
    let zero = S::ZERO;
    let one = S::ONE;
    let half = S::from_f64(0.5);
    // the spatial metric at the face: `SpatialMetric::flat()` for the SR path (bit-identical to
    // euclidean), the metric-aware gamma for the GR path. the MUB09 star-state algebra
    // (`hlld_vdiff`/`hlld_rmhd_converge`) is fully metric-generic; only the L/R conserved/flux
    // (from `regime`, metric-aware for `RmhdGr`) and these contractions carry the frame.

    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (a_l, a_r) = regime.extremal_speeds(eos, prim_l, prim_r, nhat);
    let dwave = a_r - a_l;
    let dwave_ok = dwave
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * a_r.abs().max(a_l.abs()));
    let inv_dwave = one / S::select(dwave_ok, dwave, one);
    let hll_state = (u_r * a_r - u_l * a_l - f_r + f_l) * inv_dwave;
    let hll_flux = (f_l * a_r - f_r * a_l + (u_r - u_l) * (a_l * a_r)) * inv_dwave;
    let bn = hll_state.mag.dot(nhat); // B.n (contravariant)
    let r_l = u_l * a_l - f_l;
    let r_r = u_r * a_r - f_r;
    let r_pair = [&r_l, &r_r];
    let lam = [a_l, a_r];

    let p_total =
        |prim: &MhdPrim<S, D>| -> S { prim.hydro.pre + half * metric.norm_sq_contra(&prim.mag) };
    let p_left = p_total(prim_l);
    let p_right = p_total(prim_r);
    let p_floor_val = S::from_f64(CONVERGENCE_TOL) * p_left.abs().max(p_right.abs());
    let p_hll_raw = (p_left + p_right) * half;
    let p_hll = S::select(p_hll_raw.cmp_le(zero), p_floor_val, p_hll_raw);
    let lowb_thresh = S::from_f64(LOW_B_PRESSURE_RATIO);
    let lowb_mask = (bn * bn / p_hll).cmp_lt(lowb_thresh);
    let et_hll = hll_state.nrg + hll_state.den;
    let fet_hll = hll_flux.nrg + hll_flux.den;
    let mn_hll = hll_state.mom.dot(nhat);
    let fmn_hll = hll_flux.mom.dot(nhat);
    let bb_q = et_hll - fmn_hll;
    let cc_q = fet_hll * mn_hll - et_hll * fmn_hll;
    let disc = (bb_q * bb_q - S::from_f64(4.0) * cc_q).max(zero);
    let p_lowb_raw = half * (-bb_q + disc.sqrt());
    let p_lowb = S::select(p_lowb_raw.cmp_le(zero), p_floor_val, p_lowb_raw);
    let p_init = S::select(lowb_mask, p_lowb, p_hll);

    let conv = hlld_rmhd_converge(p_init, r_pair, lam, bn, nhat, metric);
    let v = conv.v;
    HlldStates {
        lam: [a_l, a_r],
        alf: v.alf,
        lstar: v.vc.dot(nhat), // v_c.n (contravariant)
        bstar: v.bv,
        bc: v.bc,
        vc: v.vc,
        bn,
        success: conv.success * S::select(dwave_ok, one, zero),
    }
}

// helper: mask-on-mask select. `Mask: BitAnd + BitOr + Not` from `algebra.rs`,
// but the carrier doesn't ship a `select_mask` on Mask itself — it is synthesized
// as `(on_left & a) | (!on_left & b)` so the `at_contact` per-side flag
// stays branchless.
trait MaskSelect: Sized + Copy {
    fn select_mask(c: Self, a: Self, b: Self) -> Self;
}
impl<M> MaskSelect for M
where
    M: Copy
        + std::ops::BitAnd<Output = M>
        + std::ops::BitOr<Output = M>
        + std::ops::Not<Output = M>,
{
    fn select_mask(c: Self, a: Self, b: Self) -> Self {
        (c & a) | (!c & b)
    }
}

/// the Newtonian (non-relativistic) ideal-MHD HLLD five-wave solver
/// (Miyoshi & Kusano 2005). closed-form intermediate states — NO pressure
/// iteration, unlike the relativistic `hlld_rmhd`. carrier-generic: native
/// comparisons are `cmp_*`/`select`, every denominator is clamped before the
/// divide (the Alfven-resonant / vanishing-Bn degeneracies select to the
/// single-state limit), and a physicality-gated HLLE fallback guarantees a
/// valid flux. ONE source, host f64 + traced Gv. wave structure (left->right):
/// S_L < S*_L (alfven) < S_M (contact) < S*_R (alfven) < S_R.
pub fn hlld_newtonian<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
) -> MhdCons<S, D> {
    let zero = S::ZERO;
    let one = S::ONE;
    let half = S::from_f64(0.5);
    let neg = S::from_f64(-1.0);
    let regime = NewtonianMhd;

    // the normal field is CONTINUOUS across the Riemann fan (div B = 0): Miyoshi-Kusano
    // assume a single constant B_x. PLM reconstruction of the CELL-centered field gives
    // bn_l != bn_r in 2D, and feeding those raw leaves a spurious normal remnant in the
    // transverse decomposition `B - n(B.n)` (-> corrupt star B -> negative pressure; the
    // Orszag-Tang bug, ABSENT in constant-Bx Brio-Wu). enforce a single normal field on
    // both states (the L/R average) — this also makes the normal-B flux F(Bn) = 0 exactly.
    let bn = (prim_l.mag.dot(nhat) + prim_r.mag.dot(nhat)) * half;
    let bn_sq = bn * bn;
    let with_bn = |p: &MhdPrim<S, D>| -> MhdPrim<S, D> {
        MhdPrim {
            hydro: p.hydro,
            mag: p.mag + nhat.scale(bn - p.mag.dot(nhat)),
        }
    };
    let pl = with_bn(prim_l);
    let pr = with_bn(prim_r);
    let prim_l = &pl;
    let prim_r = &pr;

    // HLLE fallback (selected when the HLLD star states are unphysical).
    let hlle_flux = hlle(&regime, eos, prim_l, prim_r, nhat, vface);

    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (sll, slr) = regime.wave_speeds(eos, prim_l, nhat);
    let (srl, srr) = regime.wave_speeds(eos, prim_r, nhat);
    let s_l = sll.min(srl);
    let s_r = slr.max(srr);

    let un_l = prim_l.vel.dot(nhat);
    let un_r = prim_r.vel.dot(nhat);
    let rho_l = prim_l.rho;
    let rho_r = prim_r.rho;
    let pt_l = prim_l.pre + half * prim_l.mag.dot(&prim_l.mag);
    let pt_r = prim_r.pre + half * prim_r.mag.dot(&prim_r.mag);

    // contact speed S_M + star total pressure (Miyoshi-Kusano eqs 38, 41).
    let cl = (s_l - un_l) * rho_l;
    let cr = (s_r - un_r) * rho_r;
    let dm = cr - cl;
    let dm_ok = dm
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * cr.abs().max(cl.abs()));
    let dm_s = S::select(dm_ok, dm, one);
    let s_m = (cr * un_r - cl * un_l - pt_r + pt_l) / dm_s;
    let pt_star = (cr * pt_l - cl * pt_r + cl * cr * (un_r - un_l)) / dm_s;

    // tangential part of a vector (subtract its normal projection).
    let tang = |v: &Tensor<S, D>, vn: S| -> Tensor<S, D> { *v - nhat.scale(vn) };

    // per-side single-star (*) state (eqs 43-48). the Alfven-resonant denominator
    // selects to the single-state limit (transverse fields unchanged).
    let star = |u_k: &MhdCons<S, D>,
                f_k: &MhdCons<S, D>,
                prim_k: &MhdPrim<S, D>,
                s_k: S,
                un_k: S,
                rho_k: S,
                pt_k: S|
     -> (MhdCons<S, D>, MhdCons<S, D>, Tensor<S, D>, Tensor<S, D>, S) {
        let smk = s_k - s_m;
        let smk_ok = smk
            .abs()
            .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_k.abs().max(s_m.abs()));
        let smk_s = S::select(smk_ok, smk, one);
        let rho_star = rho_k * (s_k - un_k) / smk_s;
        // the transverse star fields divide by den = rho_k(s_k-u_k)(s_k-s_m) - bn^2, which
        // vanishes at the alfven resonance (the rotational wave coincides with the
        // entropy wave — frequent in 2D when the field is nearly face-normal). there the
        // factors below diverge -> singular star state -> negative pressure. switch to the
        // no-rotation limit (transverse fields unchanged) on a relative threshold.
        let term = rho_k * (s_k - un_k) * smk;
        let den = term - bn_sq;
        let small = den
            .abs()
            .cmp_le(S::from_f64(ALFVEN_DEGENERACY_TOL) * (term.abs() + bn_sq));
        let den_s = S::select(small, one, den);
        let v_tang = tang(&prim_k.vel, un_k);
        let b_tang = tang(&prim_k.mag, bn);
        let fac_v = S::select(small, zero, bn * (s_m - un_k) / den_s);
        let fac_b = S::select(
            small,
            one,
            (rho_k * (s_k - un_k) * (s_k - un_k) - bn_sq) / den_s,
        );
        let v_star = nhat.scale(s_m) + (v_tang - b_tang.scale(fac_v));
        let b_star = nhat.scale(bn) + b_tang.scale(fac_b);
        let e_k = u_k.nrg; // newtonian total energy (no D split)
        let vdb_k = prim_k.vel.dot(&prim_k.mag);
        let vdb_s = v_star.dot(&b_star);
        let e_star =
            ((s_k - un_k) * e_k - pt_k * un_k + pt_star * s_m + bn * (vdb_k - vdb_s)) / smk_s;
        let u_star = MhdCons {
            hydro: Cons {
                den: rho_star,
                mom: v_star.scale(rho_star),
                nrg: e_star,
            },
            mag: b_star,
        };
        let f_star = *f_k + (u_star - *u_k) * s_k;
        (u_star, f_star, v_star, b_star, rho_star)
    };

    let (us_l, fs_l, vs_l, bs_l, rs_l) = star(&u_l, &f_l, prim_l, s_l, un_l, rho_l, pt_l);
    let (us_r, fs_r, vs_r, bs_r, rs_r) = star(&u_r, &f_r, prim_r, s_r, un_r, rho_r, pt_r);

    // Alfven speeds + double-star (**) states between them (eqs 51, 59-63).
    let sqrt_rl = rs_l.safe_sqrt();
    let sqrt_rr = rs_r.safe_sqrt();
    let bn_abs = bn.abs();
    let sqrt_l_ok = sqrt_rl.cmp_gt(zero);
    let sqrt_r_ok = sqrt_rr.cmp_gt(zero);
    let sa_l = s_m - bn_abs / S::select(sqrt_l_ok, sqrt_rl, one);
    let sa_r = s_m + bn_abs / S::select(sqrt_r_ok, sqrt_rr, one);

    let sgn = S::select(bn.cmp_ge(zero), one, neg);
    let sqrt_sum = sqrt_rl + sqrt_rr;
    let sqrt_sum_ok = sqrt_sum.cmp_gt(zero);
    let inv_sden = one / S::select(sqrt_sum_ok, sqrt_sum, one);
    let vsl_t = tang(&vs_l, s_m);
    let vsr_t = tang(&vs_r, s_m);
    let bsl_t = tang(&bs_l, bn);
    let bsr_t = tang(&bs_r, bn);
    let vss_t =
        (vsl_t.scale(sqrt_rl) + vsr_t.scale(sqrt_rr) + (bsr_t - bsl_t).scale(sgn)).scale(inv_sden);
    let bss_t = (bsr_t.scale(sqrt_rl)
        + bsl_t.scale(sqrt_rr)
        + (vsr_t - vsl_t).scale(sgn * sqrt_rl * sqrt_rr))
    .scale(inv_sden);
    let v_ss = nhat.scale(s_m) + vss_t;
    let b_ss = nhat.scale(bn) + bss_t;
    let vdb_ss = v_ss.dot(&b_ss);
    let e_ss_l = us_l.nrg - sqrt_rl * (vs_l.dot(&bs_l) - vdb_ss) * sgn;
    let e_ss_r = us_r.nrg + sqrt_rr * (vs_r.dot(&bs_r) - vdb_ss) * sgn;
    let uss_l = MhdCons {
        hydro: Cons {
            den: rs_l,
            mom: v_ss.scale(rs_l),
            nrg: e_ss_l,
        },
        mag: b_ss,
    };
    let uss_r = MhdCons {
        hydro: Cons {
            den: rs_r,
            mom: v_ss.scale(rs_r),
            nrg: e_ss_r,
        },
        mag: b_ss,
    };
    let fss_l = fs_l + (uss_l - us_l) * sa_l;
    let fss_r = fs_r + (uss_r - us_r) * sa_r;

    // ALE: each region's interface flux is F_region - vface * U_region.
    let reg = |f: MhdCons<S, D>, u: MhdCons<S, D>| -> MhdCons<S, D> { f - u * vface };
    let pick = MhdCons::select(
        vface.cmp_lt(s_l),
        reg(f_l, u_l),
        MhdCons::select(
            vface.cmp_lt(sa_l),
            reg(fs_l, us_l),
            MhdCons::select(
                vface.cmp_lt(s_m),
                reg(fss_l, uss_l),
                MhdCons::select(
                    vface.cmp_lt(sa_r),
                    reg(fss_r, uss_r),
                    MhdCons::select(vface.cmp_lt(s_r), reg(fs_r, us_r), reg(f_r, u_r)),
                ),
            ),
        ),
    );

    // physicality: any non-positive star density / pressure routes to HLLE.
    let smk_l_ok = (s_l - s_m)
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_l.abs().max(s_m.abs()));
    let smk_r_ok = (s_r - s_m)
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_r.abs().max(s_m.abs()));
    let ok = dm_ok
        & smk_l_ok
        & smk_r_ok
        & sqrt_l_ok
        & sqrt_r_ok
        & sqrt_sum_ok
        & rs_l.cmp_gt(zero)
        & rs_r.cmp_gt(zero)
        & pt_star.cmp_gt(zero);
    MhdCons::select(ok, pick, hlle_flux)
}

/// the UCT-HLLD edge-EMF coefficients `(a^L, d^L, d^R)` from L/R primitives (M&DZ 2020 Eq. 44-46).
/// it runs the SAME classical Miyoshi-Kusano fan as `hlld_newtonian` — single normal field, Davis
/// `wave_speeds`, contact `lambda*`, star densities `rho^{*s}`, rotational `lambda^{*s}` — and
/// extracts the master-form coefficients from that fan. so when the EMF feeds it the SAME
/// reconstructed face states the gas flux uses, the EMF's fan is identical to the flux's (the
/// CT-consistency M&DZ require). carrier-generic; NO clamp (verbatim Eq. 44).
pub fn hlld_newtonian_coeffs<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &MhdPrim<S, D>,
    prim_r: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
) -> (S, S, S) {
    let zero = S::ZERO;
    let one = S::ONE;
    let two = S::from_f64(2.0);
    let half = S::from_f64(0.5);
    let relative_eps = S::from_f64(32.0 * f64::EPSILON);
    let eps_deg = S::from_f64(1.0e-9);
    let regime = NewtonianMhd;

    // single continuous normal field (div B = 0), identical to `hlld_newtonian`.
    let bn = (prim_l.mag.dot(nhat) + prim_r.mag.dot(nhat)) * half;
    let with_bn = |p: &MhdPrim<S, D>| -> MhdPrim<S, D> {
        MhdPrim {
            hydro: p.hydro,
            mag: p.mag + nhat.scale(bn - p.mag.dot(nhat)),
        }
    };
    let pl = with_bn(prim_l);
    let pr = with_bn(prim_r);
    let prim_l = &pl;
    let prim_r = &pr;

    let (sll, slr) = regime.wave_speeds(eos, prim_l, nhat);
    let (srl, srr) = regime.wave_speeds(eos, prim_r, nhat);
    let s_l = sll.min(srl); // lambda^L
    let s_r = slr.max(srr); // lambda^R

    let un_l = prim_l.vel.dot(nhat);
    let un_r = prim_r.vel.dot(nhat);
    let pt_l = prim_l.pre + half * prim_l.mag.dot(&prim_l.mag);
    let pt_r = prim_r.pre + half * prim_r.mag.dot(&prim_r.mag);

    // contact lambda* (Miyoshi-Kusano eq 38) — same as the flux.
    let cl = (s_l - un_l) * prim_l.rho;
    let cr = (s_r - un_r) * prim_r.rho;
    let dm = cr - cl;
    let dm_ok = dm.abs().cmp_gt(relative_eps * cr.abs().max(cl.abs()));
    let dm_s = S::select(dm_ok, dm, one);
    let s_m = (cr * un_r - cl * un_l - pt_r + pt_l) / dm_s;

    // star densities rho^{*s} (eq 43) + rotational speeds lambda^{*s} (eq 51).
    let smk_l_raw = s_l - s_m;
    let smk_r_raw = s_r - s_m;
    let smk_l_ok = smk_l_raw
        .abs()
        .cmp_gt(relative_eps * s_l.abs().max(s_m.abs()));
    let smk_r_ok = smk_r_raw
        .abs()
        .cmp_gt(relative_eps * s_r.abs().max(s_m.abs()));
    let smk_l = S::select(smk_l_ok, smk_l_raw, one);
    let smk_r = S::select(smk_r_ok, smk_r_raw, one);
    let rs_l = prim_l.rho * (s_l - un_l) / smk_l;
    let rs_r = prim_r.rho * (s_r - un_r) / smk_r;
    let bn_abs = bn.abs();
    let sqrt_rl = rs_l.safe_sqrt();
    let sqrt_rr = rs_r.safe_sqrt();
    let sqrt_l_ok = sqrt_rl.cmp_gt(zero);
    let sqrt_r_ok = sqrt_rr.cmp_gt(zero);
    let sa_l = s_m - bn_abs / S::select(sqrt_l_ok, sqrt_rl, one); // lambda^{*L}
    let sa_r = s_m + bn_abs / S::select(sqrt_r_ok, sqrt_rr, one); // lambda^{*R}

    // chi~^s (Eq. By_chi, singular factor cancelled); nu^s, nu* with the degenerate guard (Eq. 46).
    let chi_den_l = sa_l + s_l - two * s_m;
    let chi_den_r = sa_r + s_r - two * s_m;
    let chi_l_ok = chi_den_l
        .abs()
        .cmp_gt(relative_eps * (sa_l.abs() + s_l.abs() + two * s_m.abs()));
    let chi_r_ok = chi_den_r
        .abs()
        .cmp_gt(relative_eps * (sa_r.abs() + s_r.abs() + two * s_m.abs()));
    let chitl = (un_l - s_m) * (s_l - s_m) / S::select(chi_l_ok, chi_den_l, one);
    let chitr = (un_r - s_m) * (s_r - s_m) / S::select(chi_r_ok, chi_den_r, one);
    let nu_den_l = sa_l.abs() + s_l.abs();
    let nu_den_r = sa_r.abs() + s_r.abs();
    let nustar_den = sa_r.abs() + sa_l.abs();
    let nu_l_ok = nu_den_l.cmp_gt(zero);
    let nu_r_ok = nu_den_r.cmp_gt(zero);
    let nustar_ok = nustar_den.cmp_gt(zero);
    let nul = (sa_l + s_l) / S::select(nu_l_ok, nu_den_l, one);
    let nur = (sa_r + s_r) / S::select(nu_r_ok, nu_den_r, one);
    let nustar_raw = (sa_r + sa_l) / S::select(nustar_ok, nustar_den, one);
    let nustar = S::select(
        (sa_r - sa_l).abs().cmp_gt(eps_deg * (s_r - s_l).abs()),
        nustar_raw,
        zero,
    );
    let dl = half * (nul - nustar) * chitl + half * (sa_l.abs() - nustar * sa_l);
    let dr = half * (nur - nustar) * chitr + half * (sa_r.abs() - nustar * sa_r);
    let valid = dm_ok
        & smk_l_ok
        & smk_r_ok
        & sqrt_l_ok
        & sqrt_r_ok
        & chi_l_ok
        & chi_r_ok
        & nu_l_ok
        & nu_r_ok
        & nustar_ok;
    let ap = s_r.max(zero);
    let am = (-s_l).max(zero);
    let hll_sum = ap + am;
    let hll_ok = hll_sum.cmp_gt(zero);
    let hll_den = S::select(hll_ok, hll_sum, one);
    let hll_al = S::select(hll_ok, ap / hll_den, half);
    let hll_d = S::select(hll_ok, ap * am / hll_den, zero);
    (
        S::select(valid, half * (one + nustar), hll_al),
        S::select(valid, dl, hll_d),
        S::select(valid, dr, hll_d),
    )
}

/// the UCT-HLLD edge-EMF coefficients `(a^L, d^L, d^R)` for ISOTHERMAL MHD (M&DZ 2020 Appendix A).
/// the isothermal fan has NO contact/entropy mode, so the appendix states the a/d/nu formulas (Eq.
/// 44-46) are UNCHANGED but chi~^s is built from the HLL central state:
///   rho* = rho^hll,  u* = F_rho^hll/rho^hll,  lambda^{*s} = u* -/+ |Bx|/sqrt(rho*),
///   chi~^s = (v_x^s - u*)(lambda^s - u*)/(lambda^{*s} + lambda^s - 2u*).
/// mirrors the `hlld_isothermal` flux's fan exactly (single rho*, u* = mass-flux/rho — the advective mass flux, where m_x/rho would give the wrong star velocity),
/// so the EMF fan == the flux fan at the same reconstructed face state. carrier-generic; NO clamp.
pub fn hlld_isothermal_coeffs<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &IsoMhdPrim<S, D>,
    prim_r: &IsoMhdPrim<S, D>,
    nhat: &Tensor<S, D>,
) -> (S, S, S) {
    let regime = IsothermalMhd;
    let zero = S::ZERO;
    let one = S::ONE;
    let two = S::from_f64(2.0);
    let half = S::from_f64(0.5);
    let relative_eps = S::from_f64(32.0 * f64::EPSILON);
    let eps_deg = S::from_f64(1.0e-9);

    // single continuous normal field (div B = 0), identical to the isothermal flux.
    let bn = (prim_l.mag.dot(nhat) + prim_r.mag.dot(nhat)) * half;
    let bn_abs = bn.abs();
    let with_bn = |p: &IsoMhdPrim<S, D>| -> IsoMhdPrim<S, D> {
        IsoMhdPrim {
            hydro: p.hydro,
            mag: p.mag + nhat.scale(bn - p.mag.dot(nhat)),
        }
    };
    let pl = with_bn(prim_l);
    let pr = with_bn(prim_r);
    let prim_l = &pl;
    let prim_r = &pr;

    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (sll, slr) = regime.wave_speeds(eos, prim_l, nhat);
    let (srl, srr) = regime.wave_speeds(eos, prim_r, nhat);
    let s_l = sll.min(srl); // lambda^L
    let s_r = slr.max(srr); // lambda^R
    let un_l = prim_l.vel.dot(nhat);
    let un_r = prim_r.vel.dot(nhat);

    // HLL central state: rho* = rho^hll, u* = F_rho^hll/rho^hll (Appendix A; same as the iso flux).
    let dwave = s_r - s_l;
    let dwave_ok = dwave
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_r.abs().max(s_l.abs()));
    let inv_dwave = one / S::select(dwave_ok, dwave, one);
    let u_hll = (u_r * s_r - u_l * s_l - f_r + f_l) * inv_dwave;
    let f_hll = (f_l * s_r - f_r * s_l + (u_r - u_l) * (s_l * s_r)) * inv_dwave;
    let rho_s = u_hll.den;
    let rho_ok = rho_s.cmp_gt(zero);
    let rho_s_safe = S::select(rho_ok, rho_s, one);
    let u_star = f_hll.den / rho_s_safe;
    let sqrt_rs = rho_s.safe_sqrt();
    let sqrt_ok = sqrt_rs.cmp_gt(zero);
    let cax = bn_abs / S::select(sqrt_ok, sqrt_rs, one);
    let sa_l = u_star - cax; // lambda^{*L}
    let sa_r = u_star + cax; // lambda^{*R}

    // chi~^s (Appendix A: u* + single rho* in place of the MK contact); nu/d/a unchanged (Eq. 44-46).
    let chi_den_l = sa_l + s_l - two * u_star;
    let chi_den_r = sa_r + s_r - two * u_star;
    let chi_l_ok = chi_den_l
        .abs()
        .cmp_gt(relative_eps * (sa_l.abs() + s_l.abs() + two * u_star.abs()));
    let chi_r_ok = chi_den_r
        .abs()
        .cmp_gt(relative_eps * (sa_r.abs() + s_r.abs() + two * u_star.abs()));
    let chitl = (un_l - u_star) * (s_l - u_star) / S::select(chi_l_ok, chi_den_l, one);
    let chitr = (un_r - u_star) * (s_r - u_star) / S::select(chi_r_ok, chi_den_r, one);
    let nu_den_l = sa_l.abs() + s_l.abs();
    let nu_den_r = sa_r.abs() + s_r.abs();
    let nustar_den = sa_r.abs() + sa_l.abs();
    let nu_l_ok = nu_den_l.cmp_gt(zero);
    let nu_r_ok = nu_den_r.cmp_gt(zero);
    let nustar_ok = nustar_den.cmp_gt(zero);
    let nul = (sa_l + s_l) / S::select(nu_l_ok, nu_den_l, one);
    let nur = (sa_r + s_r) / S::select(nu_r_ok, nu_den_r, one);
    let nustar_raw = (sa_r + sa_l) / S::select(nustar_ok, nustar_den, one);
    let nustar = S::select(
        (sa_r - sa_l).abs().cmp_gt(eps_deg * (s_r - s_l).abs()),
        nustar_raw,
        zero,
    );
    let dl = half * (nul - nustar) * chitl + half * (sa_l.abs() - nustar * sa_l);
    let dr = half * (nur - nustar) * chitr + half * (sa_r.abs() - nustar * sa_r);
    let valid = dwave_ok & rho_ok & sqrt_ok & chi_l_ok & chi_r_ok & nu_l_ok & nu_r_ok & nustar_ok;
    let ap = s_r.max(zero);
    let am = (-s_l).max(zero);
    let hll_sum = ap + am;
    let hll_ok = hll_sum.cmp_gt(zero);
    let hll_den = S::select(hll_ok, hll_sum, one);
    let hll_al = S::select(hll_ok, ap / hll_den, half);
    let hll_d = S::select(hll_ok, ap * am / hll_den, zero);
    (
        S::select(valid, half * (one + nustar), hll_al),
        S::select(valid, dl, hll_d),
        S::select(valid, dr, hll_d),
    )
}

// =============================================================================
// isothermal MHD HLLD — Mignone (2007). the THREE-state / four-wave solver: the
// isothermal closure (p = a^2 rho) removes the entropy/contact mode, so the fan
// is U_L* | U_c* | U_R* enclosed by two fast waves S_L/S_R and two Alfven waves
// S_L*/S_R* (NO middle contact wave, unlike adiabatic HLLD). closed form; the
// density is the HLL average so positivity is trivial (no energy to go negative).
// =============================================================================

/// isothermal MHD HLLD face flux (Mignone 2007, eqs 20-39). carrier-generic; the
/// normal field is made single-valued (constant-Bx assumption, as in the adiabatic
/// solver) so the substrate's staggered bface coupling feeds it consistently.
#[allow(clippy::too_many_lines)]
pub fn hlld_isothermal<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim_l: &IsoMhdPrim<S, D>,
    prim_r: &IsoMhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    vface: S,
) -> IsoMhdCons<S, D> {
    let regime = IsothermalMhd;
    let zero = S::ZERO;
    let one = S::ONE;
    let half = S::from_f64(0.5);
    let neg = S::from_f64(-1.0);

    // transverse projection (any nhat): v - n (v.n).
    let tang = |v: &Tensor<S, D>, vn: S| -> Tensor<S, D> { *v - nhat.scale(vn) };

    // single normal field (continuous across the fan; Mignone assumes Bx const).
    let bn = (prim_l.mag.dot(nhat) + prim_r.mag.dot(nhat)) * half;
    let bn_sq = bn * bn;
    let bn_abs = bn_sq.safe_sqrt();
    let with_bn = |p: &IsoMhdPrim<S, D>| -> IsoMhdPrim<S, D> {
        IsoMhdPrim {
            hydro: p.hydro,
            mag: p.mag + nhat.scale(bn - p.mag.dot(nhat)),
        }
    };
    let pl = with_bn(prim_l);
    let pr = with_bn(prim_r);
    let prim_l = &pl;
    let prim_r = &pr;

    // HLLE fallback (selected if the HLL density is non-positive).
    let hlle_flux = hlle(&regime, eos, prim_l, prim_r, nhat, vface);

    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);
    let (sll, slr) = regime.wave_speeds(eos, prim_l, nhat);
    let (srl, srr) = regime.wave_speeds(eos, prim_r, nhat);
    let s_l = sll.min(srl);
    let s_r = slr.max(srr);

    let un_l = prim_l.vel.dot(nhat);
    let un_r = prim_r.vel.dot(nhat);
    let rho_l = prim_l.rho;
    let rho_r = prim_r.rho;

    // HLL single-average state + flux (eqs 15, 17).
    let dwave = s_r - s_l;
    let dwave_ok = dwave
        .abs()
        .cmp_gt(S::from_f64(32.0 * f64::EPSILON) * s_r.abs().max(s_l.abs()));
    let inv_dwave = one / S::select(dwave_ok, dwave, one);
    let u_hll = (u_r * s_r - u_l * s_l - f_r + f_l) * inv_dwave;
    let f_hll = (f_l * s_r - f_r * s_l + (u_r - u_l) * (s_l * s_r)) * inv_dwave;

    // rho* = HLL density (eq 20); m_x* = HLL normal momentum (eq 21, constant in fan);
    // u* = F_rho^hll / rho^hll (eq 23 advective choice; m_x/rho would give the wrong star velocity).
    let rho_s = u_hll.den;
    let rho_ok = rho_s.cmp_gt(zero);
    let rho_s_safe = S::select(rho_ok, rho_s, one);
    let mx_hll = u_hll.mom.dot(nhat);
    let u_star = f_hll.den / rho_s_safe;

    // Alfven speeds (eq 29): S_L* = u* - |Bx|/sqrt(rho*), S_R* = u* + |Bx|/sqrt(rho*).
    let sqrt_rs = rho_s.safe_sqrt();
    let sqrt_ok = sqrt_rs.cmp_gt(zero);
    let sqrt_rs_safe = S::select(sqrt_ok, sqrt_rs, one);
    let cax = bn_abs / sqrt_rs_safe; // normal Alfven speed |Bn|/sqrt(rho*)
    let sa_l = u_star - cax;
    let sa_r = u_star + cax;

    // per-side star tangential state (eqs 30-33), vectorized over the transverse plane.
    // den = (s_k - s_l*)(s_k - s_r*) = (s_k - u*)^2 - cax^2, which vanishes at the alfven
    // resonance (the fast wave coincides with an alfven wave — a purely-normal field, where
    // c_fast -> cax). there fac_v / fac_b diverge -> singular star state. switch to the
    // no-rotation limit (transverse fields unchanged) on a relative threshold measured
    // against the magnitude of the two cancelling terms `(s_k-u*)^2` and `cax^2`.
    let star = |rho_k: S,
                un_k: S,
                s_k: S,
                vt_k: &Tensor<S, D>,
                bt_k: &Tensor<S, D>|
     -> (Tensor<S, D>, Tensor<S, D>) {
        let su = s_k - u_star;
        let den = (s_k - sa_l) * (s_k - sa_r);
        let small = den
            .abs()
            .cmp_le(S::from_f64(ALFVEN_DEGENERACY_TOL) * (su * su + cax * cax));
        let den_s = S::select(small, one, den);
        let fac_v = S::select(small, zero, bn * (u_star - un_k) / den_s);
        let fac_b = S::select(
            small,
            one,
            (rho_k * (s_k - un_k) * (s_k - un_k) - bn_sq) / (rho_s_safe * den_s),
        );
        // mv = rho* v_k* (eqs 30-31);  b* = B_k* (eqs 32-33).
        let mv = vt_k.scale(rho_s) - bt_k.scale(fac_v);
        let bs = bt_k.scale(fac_b);
        (mv, bs)
    };

    let (mv_l, bs_l) = star(
        rho_l,
        un_l,
        s_l,
        &tang(&prim_l.vel, un_l),
        &tang(&prim_l.mag, bn),
    );
    let (mv_r, bs_r) = star(
        rho_r,
        un_r,
        s_r,
        &tang(&prim_r.vel, un_r),
        &tang(&prim_r.mag, bn),
    );

    // conserved star state: den=rho*, normal mom = m_x^hll, transverse mom = mv, B = n*bn + b*.
    let mk_cons = |mv: Tensor<S, D>, bs: Tensor<S, D>| -> IsoMhdCons<S, D> {
        IsoMhdCons {
            hydro: ConsG {
                den: rho_s,
                mom: nhat.scale(mx_hll) + mv,
                nrg: Zero::default(),
            },
            mag: nhat.scale(bn) + bs,
        }
    };
    let us_l = mk_cons(mv_l, bs_l);
    let us_r = mk_cons(mv_r, bs_r);

    // central state (eqs 34-37): X = sqrt(rho*) sign(Bx).
    let sgn = S::select(bn.cmp_ge(zero), one, neg);
    let x = sqrt_rs * sgn;
    let x_safe = S::select(sqrt_ok, x, one);
    let mv_c = (mv_l + mv_r).scale(half) + (bs_r - bs_l).scale(x * half);
    let bs_c = (bs_l + bs_r).scale(half) + (mv_r - mv_l).scale(half / x_safe);
    let us_c = mk_cons(mv_c, bs_c);

    // telescoping fluxes: F_k* = F_k + S_k (U_k* - U_k); F_c* = F_L* + S_L*(U_c* - U_L*).
    let fs_l = f_l + (us_l - u_l) * s_l;
    let fs_r = f_r + (us_r - u_r) * s_r;
    let fs_c = fs_l + (us_c - us_l) * sa_l;

    // 5-region sample (eq 38), ALE: F_region - vface * U_region.
    let reg = |f: IsoMhdCons<S, D>, u: IsoMhdCons<S, D>| -> IsoMhdCons<S, D> { f - u * vface };
    let pick = IsoMhdCons::select(
        vface.cmp_lt(s_l),
        reg(f_l, u_l),
        IsoMhdCons::select(
            vface.cmp_lt(sa_l),
            reg(fs_l, us_l),
            IsoMhdCons::select(
                vface.cmp_lt(sa_r),
                reg(fs_c, us_c),
                IsoMhdCons::select(vface.cmp_lt(s_r), reg(fs_r, us_r), reg(f_r, u_r)),
            ),
        ),
    );

    // positivity fallback: rho* (the HLL density) must be positive.
    IsoMhdCons::select(dwave_ok & rho_ok & sqrt_ok, pick, hlle_flux)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::{IdealGas, Isothermal};
    use crate::rmhd::{Rmhd, RmhdGr};
    use crate::spatial_metric::{Gamma, GammaInv};
    use crate::state::{Prim, PrimG};

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn hlld_rmhd_uniform_state() {
        let eos = IdealGas { gamma: 2.0 };
        let regime = Rmhd;
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.3, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 1.0, 0.0]),
        };
        let nhat = Tensor::unit(0);
        let flux = hlld_rmhd(
            &regime,
            &eos,
            &prim,
            &prim,
            &nhat,
            0.0,
            &SpatialMetric::flat(),
        );
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(
            approx(flux.den, exact.den),
            "den: {} vs {}",
            flux.den,
            exact.den
        );
        for dd in 0..3 {
            assert!(
                approx(flux.mom[dd], exact.mom[dd]),
                "mom[{}]: {} vs {}",
                dd,
                flux.mom[dd],
                exact.mom[dd]
            );
        }
        assert!(
            approx(flux.nrg, exact.nrg),
            "nrg: {} vs {}",
            flux.nrg,
            exact.nrg
        );
    }

    #[test]
    fn hlld_rmhd_balsara_shock() {
        let eos = IdealGas { gamma: 2.0 };
        let regime = Rmhd;
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 1.0, 0.0]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.125,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 0.1,
            },
            mag: Tensor::new([0.5, -1.0, 0.0]),
        };
        let nhat = Tensor::unit(0);
        let flux = hlld_rmhd(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            0.0,
            &SpatialMetric::flat(),
        );
        assert!(
            flux.den > 0.0,
            "density flux should be positive: {}",
            flux.den
        );
    }

    #[test]
    fn hlld_rmhd_states_ordering_and_coplanarity() {
        // THE GATE for UCT-HLLD: the star-state extractor must give (1) a physically ORDERED fan
        // lam_L <= alf_L <= lstar <= alf_R <= lam_R, and (2) a SCALAR chi^s (B*^s_t parallel to
        // B^s_t — fast-wave coplanarity), tested with a genuine 3D transverse field (By AND Bz).
        // reduction state: asymmetric L/R (different fast speeds) + transverse-B jump + Bx != 0.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rmhd;
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.2, 0.1, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 0.8, 0.4]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.3,
                vel: Tensor::new([-0.1, -0.2, 0.05]),
                pre: 0.4,
            },
            mag: Tensor::new([0.5, -0.6, 0.3]),
        };
        let nhat = Tensor::<f64, 3>::unit(0);
        let s = hlld_rmhd_states(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            &SpatialMetric::flat(),
        );
        println!(
            "GATE: success={} lam={:?} alf={:?} lstar={} bn={}",
            s.success, s.lam, s.alf, s.lstar, s.bn
        );
        println!("GATE: B*_L={:?} B*_R={:?}", s.bstar[0], s.bstar[1]);
        assert!(
            s.success > 0.5,
            "HLLD must converge on this state (else the gate is meaningless)"
        );
        let t = 1e-9;
        assert!(
            s.lam[0] <= s.alf[0] + t,
            "lam_L <= alf_L: {} vs {}",
            s.lam[0],
            s.alf[0]
        );
        assert!(
            s.alf[0] <= s.lstar + t,
            "alf_L <= lstar: {} vs {}",
            s.alf[0],
            s.lstar
        );
        assert!(
            s.lstar <= s.alf[1] + t,
            "lstar <= alf_R: {} vs {}",
            s.lstar,
            s.alf[1]
        );
        assert!(
            s.alf[1] <= s.lam[1] + t,
            "alf_R <= lam_R: {} vs {}",
            s.alf[1],
            s.lam[1]
        );
        // coplanarity: B*^s_t parallel to B^s_t. chi^s is extracted by PROJECTION (robust to the
        // tiny non-parallel residual the HLLD star state carries — a componentwise ratio amplifies
        // it because chi itself is small). the GATE is that the NON-PARALLEL residual is negligible.
        for (side, prim) in [(0usize, &prim_l), (1usize, &prim_r)] {
            let bt = [prim.mag[1], prim.mag[2]]; // transverse (y,z) upstream
            let bst = [s.bstar[side][1], s.bstar[side][2]]; // transverse star
            let bt2 = bt[0] * bt[0] + bt[1] * bt[1];
            let chi = (bst[0] * bt[0] + bst[1] * bt[1]) / bt2 - 1.0; // projection
            // residual of B*_t off the (1+chi)B_t line, relative to |B*_t|.
            let res = [bst[0] - (1.0 + chi) * bt[0], bst[1] - (1.0 + chi) * bt[1]];
            let res_rel = (res[0] * res[0] + res[1] * res[1]).sqrt()
                / (bst[0] * bst[0] + bst[1] * bst[1]).sqrt().max(1e-30);
            println!("GATE: side {side} chi_proj={chi} non-parallel residual={res_rel:.2e}");
            // the MUB09 single-star B is APPROXIMATELY coplanar: the non-parallel residual scales
            // with the amplification |chi| (0.09% at chi=-0.05, 0.74% at chi=0.48 here). projection
            // is the principled scalar extraction; the residual is the HLLD approximation error and
            // is negligible vs the HLL/HLLD diffusion gap. MONITOR it for the high-sigma wind.
            assert!(
                res_rel < 2e-2,
                "B*^{side}_t coplanarity residual too large: {res_rel:.2e}"
            );
        }
    }

    #[test]
    fn hlld_rmhd_emf_reduces_to_by_flux() {
        // the grid-aligned reduction: in the double-star
        // region (lambda*^L < 0 < lambda*^R, the interface sits between the Alfven waves) the HLLD
        // induction flux is CONSTANT and equals the contact-state value:
        //   F_x[B_y] = v_x B_y - v_y B_x  ->  lambda* B_c^y - v_c^y B^x
        // (v_x = lambda*, B_y = B_c, v_y = v_c, B_x = B^x are all constant there). this validates the
        // EXACT contact quantities (B_c, v_c, lstar, bn) the D-formula is built from against the
        // proven `hlld_rmhd` flux. asymmetric L/R + B_y^L != B_y^R + B^x != 0 (the nontrivial case).
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rmhd;
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.2, 0.1, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 0.8, 0.4]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.3,
                vel: Tensor::new([-0.1, -0.2, 0.05]),
                pre: 0.4,
            },
            mag: Tensor::new([0.5, -0.6, 0.3]),
        };
        let nhat = Tensor::<f64, 3>::unit(0);
        let s = hlld_rmhd_states(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            &SpatialMetric::flat(),
        );
        assert!(
            s.success > 0.5,
            "HLLD must converge for the reduction gate to be meaningful"
        );
        assert!(
            s.alf[0] < 0.0 && s.alf[1] > 0.0,
            "need the double-star region to straddle the interface: alf={:?}",
            s.alf
        );
        let flux = hlld_rmhd(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            0.0,
            &SpatialMetric::flat(),
        );
        let f_by = flux.mag[1]; // F_x[B_y], the induction flux == -E_z
        let identity = s.lstar * s.bc[1] - s.vc[1] * s.bn; // lambda* B_c^y - v_c^y B^x
        println!(
            "CHECK3: F_x[B_y]={f_by:.10} vs lambda* B_c - v_c B^x={identity:.10}  (lstar={:.4} bc_y={:.4} vc_y={:.4} bn={:.4})",
            s.lstar, s.bc[1], s.vc[1], s.bn
        );
        assert!(
            (f_by - identity).abs() < 1e-8,
            "EMF reduction FAILED: hlld_rmhd B_y flux {f_by} != contact identity {identity} (diff {})",
            (f_by - identity).abs()
        );
    }

    #[test]
    fn hlld_rmhd_uct_telescopes_to_flux() {
        // the Mignone & Del Zanna (2020) master form Eq. (30) must
        // reconstruct the hlld_rmhd B_y flux EXACTLY from the per-side coefficients:
        //   F^[By] = a^L F^L + a^R F^R - (d^R B_y^R - d^L B_y^L),   F^s = v_x^s B_y^s - v_y^s B^x
        // with the BOUNDED chi-term substitution  d^s_chi . B_y^s = 1/2 (v^s-v*)(lambda*^s-lambda^s)(B_c^y - B_y^s).
        // if this == flux.mag[1], the relativistic coefficients are PROVEN consistent (no rebuild).
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rmhd;
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.2, 0.1, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 0.8, 0.4]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.3,
                vel: Tensor::new([-0.1, -0.2, 0.05]),
                pre: 0.4,
            },
            mag: Tensor::new([0.5, -0.6, 0.3]),
        };
        let nhat = Tensor::<f64, 3>::unit(0);
        let s = hlld_rmhd_states(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            &SpatialMetric::flat(),
        );
        assert!(s.success > 0.5, "HLLD must converge");
        let (lam, alf, bn, bc_y) = (s.lam, s.alf, s.bn, s.bc[1]);
        // speed-only weights (classical kernel forms): v^s, v*, a^L=(1+v*)/2.
        // per-side induction flux F^s = v_x^s B_y^s - v_y^s B^x.
        let by = [prim_l.mag[1], prim_r.mag[1]];
        let vx = [prim_l.vel[0], prim_r.vel[0]];
        let vy = [prim_l.vel[1], prim_r.vel[1]];
        let f = |i: usize| vx[i] * by[i] - vy[i] * bn;
        // single-star (between fast & Alfven) and double-star (== contact B_c) transverse fields.
        let by_ss_l = s.bstar[0][1]; // B_y^{sL}
        let by_ss_r = s.bstar[1][1]; // B_y^{sR}
        // the EXACT HLLD induction flux (M&DZ Eq. 39): central minus the per-wave |lambda| dissipation
        // over the ACTUAL star-field jumps. BOUNDED (all field differences); relativistically correct.
        let f_hat = 0.5
            * (f(0) + f(1)
                - lam[0].abs() * (by_ss_l - by[0])
                - alf[0].abs() * (bc_y - by_ss_l)
                - alf[1].abs() * (by_ss_r - bc_y)
                - lam[1].abs() * (by[1] - by_ss_r));
        let flux = hlld_rmhd(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            0.0,
            &SpatialMetric::flat(),
        );
        let f_ref = flux.mag[1];
        println!(
            "TELESCOPE: F_hat={f_hat:.10} vs flux.mag[1]={f_ref:.10}  diff={:.2e}",
            (f_hat - f_ref).abs()
        );
        assert!(
            (f_hat - f_ref).abs() < 1e-8,
            "UCT master form does NOT telescope to the HLLD flux: {f_hat} vs {f_ref} (diff {})",
            (f_hat - f_ref).abs()
        );
    }

    // a MILD diagonal SPD spatial metric (a physical Schwarzschild-scale perturbation of
    // identity: gamma_rr = 1.3, gamma_tt = gamma_pp = 1.15) that exercises every metric
    // contraction in the HLLD fan WITHOUT the r^2 blow-up that would map coordinate B to an
    // unphysical ultra-low-beta physical field.
    fn mild_curved_metric() -> SpatialMetric<f64, 3> {
        SpatialMetric::new(
            Gamma::new(symbi_algebra::Matrix::diag(Tensor::new([1.3, 1.15, 1.15]))),
            GammaInv::new(symbi_algebra::Matrix::diag(Tensor::new([
                1.0 / 1.3,
                1.0 / 1.15,
                1.0 / 1.15,
            ]))),
        )
    }

    // a MILD NON-DIAGONAL SPD spatial metric (the kerr-class case: off-diagonal gamma_r_phi-type
    // couplings), exercising the FULL tetrad path in the orthonormal-frame HLLD (including the
    // off-diagonal couplings beyond the diagonal sqrt(g) scaling). the inverse is computed exactly from the closed-form 3x3 inv.
    fn mild_nondiag_metric() -> SpatialMetric<f64, 3> {
        let gamma =
            symbi_algebra::Matrix::new([[1.3, 0.2, 0.05], [0.2, 1.15, 0.1], [0.05, 0.1, 1.15]]);
        SpatialMetric::new(Gamma::new(gamma), GammaInv::new(gamma.inv()))
    }

    #[test]
    fn hlld_rmhd_gr_telescopes_to_flux_on_a_curved_metric() {
        // the DECISIVE proof for the metric-generalized MUB09 solver: the M&DZ wave-sum
        // (Eq. 39) built from the metric-aware star states must telescope EXACTLY to the
        // metric-aware HLLD B_t flux — the star states and the flux are self-consistent on a
        // curved background (the CT-flux consistency the GR-UCT-HLLD EMF relies on). the
        // induction flux F(B^i) = B^i v^n - v^i B^n is metric-free in contravariant
        // components, so the telescoping FORMULA is unchanged; only the star fields carry gamma.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let m = mild_curved_metric();
        let gr = RmhdGr {
            metric: m,
            alpha: 0.9,
        };
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.15, 0.05, -0.03]),
                pre: 1.0,
            },
            mag: Tensor::new([0.4, 0.6, 0.3]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.3,
                vel: Tensor::new([-0.1, -0.08, 0.04]),
                pre: 0.4,
            },
            mag: Tensor::new([0.4, -0.5, 0.25]),
        };
        let dir = 0usize;
        let _ = &gr;
        assert!(m.norm_sq_contra(&prim_l.vel) < 1.0 && m.norm_sq_contra(&prim_r.vel) < 1.0);
        let s = hlld_rmhd_states_gr_ortho(&eos, &prim_l, &prim_r, dir, &m);
        assert!(
            s.success > 0.5,
            "GR HLLD must converge on the curved metric"
        );
        let (lam, alf, bn, bc_y) = (s.lam, s.alf, s.bn, s.bc[1]);
        let by = [prim_l.mag[1], prim_r.mag[1]];
        let vx = [prim_l.vel[0], prim_r.vel[0]];
        let vy = [prim_l.vel[1], prim_r.vel[1]];
        let f = |i: usize| vx[i] * by[i] - vy[i] * bn;
        let by_ss_l = s.bstar[0][1];
        let by_ss_r = s.bstar[1][1];
        let f_hat = 0.5
            * (f(0) + f(1)
                - lam[0].abs() * (by_ss_l - by[0])
                - alf[0].abs() * (bc_y - by_ss_l)
                - alf[1].abs() * (by_ss_r - bc_y)
                - lam[1].abs() * (by[1] - by_ss_r));
        let flux = hlld_rmhd_gr_ortho(&eos, &prim_l, &prim_r, dir, 0.0, &m);
        let f_ref = flux.mag[1];
        assert!(
            (f_hat - f_ref).abs() < 1e-8,
            "GR wave-sum does NOT telescope to the curved HLLD flux: {f_hat} vs {f_ref} (diff {})",
            (f_hat - f_ref).abs()
        );
    }

    #[test]
    fn hlld_rmhd_gr_telescopes_on_the_theta_direction() {
        // the theta-direction Riemann (nhat = unit(1)) — where the michel-monopole theta-momentum
        // instability lives. the wave-sum for the dissipated transverse component (here B_r,
        // component 0) must telescope to the metric HLLD flux for THAT component, or the
        // theta-face flux has a directional metric bug.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let m = mild_curved_metric();
        let gr = RmhdGr {
            metric: m,
            alpha: 0.9,
        };
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.05, 0.15, -0.03]),
                pre: 1.0,
            },
            mag: Tensor::new([0.6, 0.4, 0.3]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.3,
                vel: Tensor::new([-0.08, -0.1, 0.04]),
                pre: 0.4,
            },
            mag: Tensor::new([-0.5, 0.4, 0.25]),
        };
        let dir = 1usize;
        let _ = &gr;
        let s = hlld_rmhd_states_gr_ortho(&eos, &prim_l, &prim_r, dir, &m);
        assert!(s.success > 0.5, "GR HLLD (theta) must converge");
        // theta-Riemann dissipates B_r (component 0), normal B_theta (component 1).
        let (t, nrm) = (0usize, 1usize);
        let bt = [prim_l.mag[t], prim_r.mag[t]];
        // F(B_t) = B_t v^n - v_t B^n with n = theta: v^n = v^1, B^n = B^1.
        let vn = [prim_l.vel[nrm], prim_r.vel[nrm]];
        let vt = [prim_l.vel[t], prim_r.vel[t]];
        let bn = s.bn;
        let f = |i: usize| bt[i] * vn[i] - vt[i] * bn;
        let bt_ss_l = s.bstar[0][t];
        let bt_ss_r = s.bstar[1][t];
        let f_hat = 0.5
            * (f(0) + f(1)
                - s.lam[0].abs() * (bt_ss_l - bt[0])
                - s.alf[0].abs() * (s.bc[t] - bt_ss_l)
                - s.alf[1].abs() * (bt_ss_r - s.bc[t])
                - s.lam[1].abs() * (bt[1] - bt_ss_r));
        let flux = hlld_rmhd_gr_ortho(&eos, &prim_l, &prim_r, dir, 0.0, &m);
        let f_ref = flux.mag[t];
        assert!(
            (f_hat - f_ref).abs() < 1e-8,
            "GR theta-direction wave-sum does NOT telescope: {f_hat} vs {f_ref} (diff {})",
            (f_hat - f_ref).abs()
        );
    }

    #[test]
    fn hlld_rmhd_gr_ortho_reduces_to_flat_at_identity() {
        // the GR orthonormal wrapper at the identity metric (sqrt(g_i) = 1) must be BIT-IDENTICAL
        // to the flat MUB09 solver on a genuine L != R shock: no metric factor may perturb the SR
        // path. this is the guard that the SR-MHD path is untouched by the GR generalization.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let flat = SpatialMetric::<f64, 3>::flat();
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.2,
                vel: Tensor::new([0.1, -0.04, 0.02]),
                pre: 0.8,
            },
            mag: Tensor::new([0.3, 0.5, -0.2]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.5,
                vel: Tensor::new([-0.05, 0.06, -0.03]),
                pre: 0.5,
            },
            mag: Tensor::new([0.3, -0.4, 0.15]),
        };
        for d in 0..3usize {
            let nhat = Tensor::<f64, 3>::unit(d);
            let ortho = hlld_rmhd_gr_ortho(&eos, &prim_l, &prim_r, d, 0.0, &flat);
            let flat_flux = hlld_rmhd(&Rmhd, &eos, &prim_l, &prim_r, &nhat, 0.0, &flat);
            assert_eq!(
                ortho.den, flat_flux.den,
                "d{d} den not bit-identical at flat"
            );
            assert_eq!(
                ortho.nrg, flat_flux.nrg,
                "d{d} nrg not bit-identical at flat"
            );
            for k in 0..3 {
                assert_eq!(
                    ortho.mom[k], flat_flux.mom[k],
                    "d{d} mom{k} not bit-identical at flat"
                );
                assert_eq!(
                    ortho.mag[k], flat_flux.mag[k],
                    "d{d} mag{k} not bit-identical at flat"
                );
            }
        }
    }

    #[test]
    fn hlld_rmhd_gr_uniform_state_equals_flux() {
        // the SMOOTH-LIMIT gate: for identical L=R states on a CURVED metric, the HLLD intercell
        // flux MUST equal the metric-aware physical flux F(U) exactly (no wave dissipation for a
        // trivial Riemann). the orthonormal-frame wrapper reduces the GR solve to the validated
        // flat MUB09 solver and maps the flux back, so this holds to roundoff in every component
        // and every direction (the coordinate-frame reconstruction did not — it carried residual
        // covariant-variance sqrt(gamma) factors in the transverse star fields).
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let m = mild_curved_metric();
        let gr = RmhdGr {
            metric: m,
            alpha: 0.9,
        };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.2, 0.05, -0.03]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 0.6, 0.3]),
        };
        for d in 0..2 {
            let nhat = Tensor::<f64, 3>::unit(d);
            let flux = hlld_rmhd_gr_ortho(&eos, &prim, &prim, d, 0.0, &m);
            let exact = gr.to_flux(&prim, &nhat, &eos);
            assert!(
                (flux.den - exact.den).abs() < 1e-12,
                "d{d} den: {} vs {}",
                flux.den,
                exact.den
            );
            for k in 0..3 {
                assert!(
                    (flux.mom[k] - exact.mom[k]).abs() < 1e-12,
                    "d{d} mom{k}: {} vs {}",
                    flux.mom[k],
                    exact.mom[k]
                );
                assert!(
                    (flux.mag[k] - exact.mag[k]).abs() < 1e-12,
                    "d{d} mag{k}: {} vs {}",
                    flux.mag[k],
                    exact.mag[k]
                );
            }
            assert!(
                (flux.nrg - exact.nrg).abs() < 1e-12,
                "d{d} nrg: {} vs {}",
                flux.nrg,
                exact.nrg
            );
        }
    }

    #[test]
    fn hlld_rmhd_gr_ortho_nondiagonal_smooth_equals_flux() {
        // the TETRAD smooth-limit gate: on a NON-DIAGONAL (kerr-class) metric the orthonormal-frame
        // HLLD must STILL equal F(U) exactly for L = R. this exercises the full Gram-Schmidt tetrad
        // (off-diagonal gamma), beyond the diagonal sqrt(g) scaling. all three coordinate normals.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let m = mild_nondiag_metric();
        let gr = RmhdGr {
            metric: m,
            alpha: 0.9,
        };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.18, 0.06, -0.04]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 0.6, 0.3]),
        };
        for d in 0..3 {
            let nhat = Tensor::<f64, 3>::unit(d);
            let flux = hlld_rmhd_gr_ortho(&eos, &prim, &prim, d, 0.0, &m);
            let exact = gr.to_flux(&prim, &nhat, &eos);
            assert!(
                (flux.den - exact.den).abs() < 1e-11,
                "d{d} den: {} vs {}",
                flux.den,
                exact.den
            );
            for k in 0..3 {
                assert!(
                    (flux.mom[k] - exact.mom[k]).abs() < 1e-11,
                    "d{d} mom{k}: {} vs {}",
                    flux.mom[k],
                    exact.mom[k]
                );
                assert!(
                    (flux.mag[k] - exact.mag[k]).abs() < 1e-11,
                    "d{d} mag{k}: {} vs {}",
                    flux.mag[k],
                    exact.mag[k]
                );
            }
            assert!(
                (flux.nrg - exact.nrg).abs() < 1e-11,
                "d{d} nrg: {} vs {}",
                flux.nrg,
                exact.nrg
            );
        }
    }

    #[test]
    fn hlld_rmhd_gr_ortho_shift_smooth_equals_moving_flux() {
        // the SHIFTED smooth-limit gate: a moving interface vface (the kerr-schild/kerr shift
        // beta^n/alpha rides the fan this way) must give, for L = R, the intercell flux F(U) -
        // vface U EXACTLY — the kernel flux of the shifted valencia system (the godunov re-applies
        // alpha). validates the shifted HLLD; tested on the non-diagonal metric, all three normals.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let m = mild_nondiag_metric();
        let gr = RmhdGr {
            metric: m,
            alpha: 0.8,
        };
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.15, 0.05, -0.03]),
                pre: 1.0,
            },
            mag: Tensor::new([0.5, 0.6, 0.3]),
        };
        let vface = 0.2;
        for d in 0..3 {
            let nhat = Tensor::<f64, 3>::unit(d);
            let flux = hlld_rmhd_gr_ortho(&eos, &prim, &prim, d, vface, &m);
            let f = gr.to_flux(&prim, &nhat, &eos);
            let u = gr.to_conserved(&eos, &prim);
            assert!(
                (flux.den - (f.den - vface * u.den)).abs() < 1e-11,
                "d{d} den"
            );
            for k in 0..3 {
                assert!(
                    (flux.mom[k] - (f.mom[k] - vface * u.mom[k])).abs() < 1e-11,
                    "d{d} mom{k}"
                );
                assert!(
                    (flux.mag[k] - (f.mag[k] - vface * u.mag[k])).abs() < 1e-11,
                    "d{d} mag{k}"
                );
            }
            assert!(
                (flux.nrg - (f.nrg - vface * u.nrg)).abs() < 1e-11,
                "d{d} nrg"
            );
        }
    }

    #[test]
    fn hlld_rmhd_gr_ortho_nondiagonal_telescopes() {
        // the TETRAD wave-sum gate: on a NON-DIAGONAL metric the M&DZ wave-sum built from the tetrad
        // star states must telescope EXACTLY to the tetrad HLLD B_t flux (the CT-flux consistency the
        // GR-UCT-HLLD EMF relies on, now with the full frame-dragging-class metric). r-direction,
        // dissipated component B_y. the induction flux F(B^i) = B^i v^n - v^i B^n is metric-free in
        // contravariant components, so the telescoping FORMULA is unchanged; the tetrad enters only
        // through the star fields + speeds.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let m = mild_nondiag_metric();
        let dir = 0usize;
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.15, 0.05, -0.03]),
                pre: 1.0,
            },
            mag: Tensor::new([0.4, 0.6, 0.3]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.3,
                vel: Tensor::new([-0.1, -0.08, 0.04]),
                pre: 0.4,
            },
            mag: Tensor::new([0.4, -0.5, 0.25]),
        };
        assert!(m.norm_sq_contra(&prim_l.vel) < 1.0 && m.norm_sq_contra(&prim_r.vel) < 1.0);
        let s = hlld_rmhd_states_gr_ortho(&eos, &prim_l, &prim_r, dir, &m);
        assert!(
            s.success > 0.5,
            "tetrad HLLD must converge on the non-diagonal metric"
        );
        let (lam, alf, bn, bc_y) = (s.lam, s.alf, s.bn, s.bc[1]);
        let by = [prim_l.mag[1], prim_r.mag[1]];
        let vx = [prim_l.vel[0], prim_r.vel[0]];
        let vy = [prim_l.vel[1], prim_r.vel[1]];
        let f = |i: usize| vx[i] * by[i] - vy[i] * bn;
        let f_hat = 0.5
            * (f(0) + f(1)
                - lam[0].abs() * (s.bstar[0][1] - by[0])
                - alf[0].abs() * (bc_y - s.bstar[0][1])
                - alf[1].abs() * (s.bstar[1][1] - bc_y)
                - lam[1].abs() * (by[1] - s.bstar[1][1]));
        let flux = hlld_rmhd_gr_ortho(&eos, &prim_l, &prim_r, dir, 0.0, &m);
        assert!(
            (f_hat - flux.mag[1]).abs() < 1e-8,
            "tetrad wave-sum does NOT telescope: {f_hat} vs {} (diff {})",
            flux.mag[1],
            (f_hat - flux.mag[1]).abs()
        );
    }

    #[test]
    fn hlld_rmhd_gr_fan_is_ordered_on_a_curved_metric() {
        // the five-wave fan on the curved metric must stay ordered
        // lam^L <= alf^L <= lstar <= alf^R <= lam^R (MUB09), and the normal field single-valued.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let m = mild_curved_metric();
        let gr = RmhdGr {
            metric: m,
            alpha: 0.9,
        };
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.2,
                vel: Tensor::new([0.1, -0.04, 0.02]),
                pre: 0.8,
            },
            mag: Tensor::new([0.3, 0.5, -0.2]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.5,
                vel: Tensor::new([-0.05, 0.06, -0.03]),
                pre: 0.5,
            },
            mag: Tensor::new([0.3, -0.4, 0.15]),
        };
        let dir = 0usize;
        let _ = &gr;
        let s = hlld_rmhd_states_gr_ortho(&eos, &prim_l, &prim_r, dir, &m);
        assert!(s.success > 0.5, "GR HLLD must converge");
        let tol = 1e-9;
        assert!(s.lam[0] <= s.alf[0] + tol, "lamL <= alfL");
        assert!(s.alf[0] <= s.lstar + tol, "alfL <= lstar");
        assert!(s.lstar <= s.alf[1] + tol, "lstar <= alfR");
        assert!(s.alf[1] <= s.lam[1] + tol, "alfR <= lamR");
    }

    #[test]
    fn hlld_rmhd_states_bx_zero_is_finite() {
        // degenerate Bx=0 (the toroidal-wind regime at the equator): the rotational waves collapse
        // onto the contact; the extractor must stay finite (success may be 0 -> EMF uses HLL there).
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rmhd;
        let prim_l = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.1, 0.2, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, 0.8, 0.4]),
        };
        let prim_r = MhdPrim {
            hydro: Prim {
                rho: 0.5,
                vel: Tensor::new([-0.1, 0.1, 0.0]),
                pre: 0.5,
            },
            mag: Tensor::new([0.0, -0.5, 0.3]),
        };
        let nhat = Tensor::<f64, 3>::unit(0);
        let s = hlld_rmhd_states(
            &regime,
            &eos,
            &prim_l,
            &prim_r,
            &nhat,
            &SpatialMetric::flat(),
        );
        println!(
            "GATE Bx=0: success={} lam={:?} alf={:?} lstar={}",
            s.success, s.lam, s.alf, s.lstar
        );
        assert!(
            s.lstar.is_finite() && s.alf[0].is_finite() && s.alf[1].is_finite(),
            "states finite at Bx=0"
        );
        for k in 0..3 {
            assert!(
                s.bstar[0][k].is_finite() && s.bstar[1][k].is_finite(),
                "B* finite at Bx=0"
            );
        }
    }

    // ---- Newtonian HLLD (Miyoshi-Kusano) ----

    fn nm_prim(rho: f64, v: [f64; 3], p: f64, b: [f64; 3]) -> MhdPrim<f64, 3> {
        MhdPrim {
            hydro: Prim {
                rho,
                vel: Tensor::new(v),
                pre: p,
            },
            mag: Tensor::new(b),
        }
    }

    fn assert_flux_eq(got: &MhdCons<f64, 3>, want: &MhdCons<f64, 3>, ctx: &str) {
        assert!(
            approx(got.den, want.den),
            "{ctx} den: {} vs {}",
            got.den,
            want.den
        );
        for dd in 0..3 {
            assert!(
                approx(got.mom[dd], want.mom[dd]),
                "{ctx} mom[{dd}]: {} vs {}",
                got.mom[dd],
                want.mom[dd]
            );
        }
        assert!(
            approx(got.nrg, want.nrg),
            "{ctx} nrg: {} vs {}",
            got.nrg,
            want.nrg
        );
        for dd in 0..3 {
            assert!(
                approx(got.mag[dd], want.mag[dd]),
                "{ctx} mag[{dd}]: {} vs {}",
                got.mag[dd],
                want.mag[dd]
            );
        }
    }

    #[test]
    fn hlld_newtonian_uniform_is_physical_flux() {
        // consistency: F(U, U) == F(U) exactly (all star states collapse to U).
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let cases = [
            nm_prim(1.0, [0.0, 0.0, 0.0], 1.0, [0.5, 1.0, 0.0]), // static, oblique B
            nm_prim(0.7, [0.3, -0.2, 0.1], 0.9, [0.4, 0.2, -0.6]), // moving, full 3D B
            nm_prim(1.2, [-0.4, 0.0, 0.3], 1.5, [0.0, 0.8, -0.3]), // Bn = 0 (perpendicular)
        ];
        for (ii, prim) in cases.iter().enumerate() {
            let flux = hlld_newtonian(&eos, prim, prim, &nhat, 0.0);
            let exact = NewtonianMhd.to_flux(prim, &nhat, &eos);
            assert_flux_eq(&flux, &exact, &format!("uniform case {ii}"));
        }
    }

    #[test]
    fn hlld_newtonian_is_homogeneous_under_state_rescaling() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let left = nm_prim(1.1, [0.35, -0.12, 0.08], 0.8, [0.45, 0.3, -0.2]);
        let right = nm_prim(0.7, [-0.18, 0.09, -0.04], 0.5, [0.45, -0.15, 0.25]);
        let reference = hlld_newtonian(&eos, &left, &right, &nhat, 0.0);
        for factor in [1e-100_f64, 1.0, 1e100] {
            let field_factor = factor.sqrt();
            let scale = |prim: &MhdPrim<f64, 3>| {
                nm_prim(
                    prim.rho * factor,
                    prim.vel.data,
                    prim.pre * factor,
                    prim.mag.data.map(|value| value * field_factor),
                )
            };
            let flux = hlld_newtonian(&eos, &scale(&left), &scale(&right), &nhat, 0.0);
            assert!((flux.den / (reference.den * factor) - 1.0).abs() < 1e-12);
            assert!((flux.nrg / (reference.nrg * factor) - 1.0).abs() < 1e-12);
            for dd in 0..3 {
                assert!((flux.mom[dd] / (reference.mom[dd] * factor) - 1.0).abs() < 1e-12);
                if reference.mag[dd] != 0.0 {
                    assert!(
                        (flux.mag[dd] / (reference.mag[dd] * field_factor) - 1.0).abs() < 1e-12
                    );
                } else {
                    assert_eq!(flux.mag[dd], 0.0);
                }
            }
        }
    }

    #[test]
    fn hlld_newtonian_b_zero_matches_hydro_flux() {
        let eos = IdealGas { gamma: 1.4 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let prim = nm_prim(1.3, [0.4, -0.2, 0.1], 0.7, [0.0, 0.0, 0.0]);
        let flux = hlld_newtonian(&eos, &prim, &prim, &nhat, 0.0);
        let exact = NewtonianMhd.to_flux(&prim, &nhat, &eos);
        assert_flux_eq(&flux, &exact, "b=0 uniform");
    }

    #[test]
    fn hlld_newtonian_supersonic_right_upwinds_left() {
        // vn >> fast speed on BOTH sides -> S_L > 0 -> the whole fan is right-going,
        // so the interface flux at vface=0 is the LEFT physical flux F(U_L).
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let prim_l = nm_prim(1.0, [5.0, 0.2, 0.0], 1.0, [0.3, 0.5, 0.0]);
        let prim_r = nm_prim(0.5, [5.0, -0.1, 0.2], 0.4, [0.3, -0.4, 0.1]);
        let flux = hlld_newtonian(&eos, &prim_l, &prim_r, &nhat, 0.0);
        let f_l = NewtonianMhd.to_flux(&prim_l, &nhat, &eos);
        assert_flux_eq(&flux, &f_l, "supersonic-right");
    }

    #[test]
    fn hlld_newtonian_supersonic_left_upwinds_right() {
        // vn << -fast speed -> S_R < 0 -> the interface flux is the RIGHT flux F(U_R).
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let prim_l = nm_prim(1.0, [-5.0, 0.2, 0.0], 1.0, [0.3, 0.5, 0.0]);
        let prim_r = nm_prim(0.5, [-5.0, -0.1, 0.2], 0.4, [0.3, -0.4, 0.1]);
        let flux = hlld_newtonian(&eos, &prim_l, &prim_r, &nhat, 0.0);
        let f_r = NewtonianMhd.to_flux(&prim_r, &nhat, &eos);
        assert_flux_eq(&flux, &f_r, "supersonic-left");
    }

    #[test]
    fn hlld_newtonian_brio_wu_is_finite_and_physical() {
        // a Brio-Wu-like subsonic discontinuity: the HLLD flux must be finite and
        // its mass flux must lie within the L/R physical-flux bracket (consistency).
        let eos = IdealGas { gamma: 2.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let prim_l = nm_prim(1.0, [0.0, 0.0, 0.0], 1.0, [0.75, 1.0, 0.0]);
        let prim_r = nm_prim(0.125, [0.0, 0.0, 0.0], 0.1, [0.75, -1.0, 0.0]);
        let flux = hlld_newtonian(&eos, &prim_l, &prim_r, &nhat, 0.0);
        assert!(
            flux.den.is_finite() && flux.nrg.is_finite(),
            "flux must be finite"
        );
        for dd in 0..3 {
            assert!(
                flux.mom[dd].is_finite() && flux.mag[dd].is_finite(),
                "flux comp {dd} finite"
            );
        }
        // Bn is continuous across the fan -> the normal-B flux is 0 (induction F(Bn)=0).
        assert!(
            flux.mag[0].abs() < 1e-12,
            "normal-B flux must vanish: {}",
            flux.mag[0]
        );
    }

    // ---- isothermal HLLD (Mignone 2007) ----

    fn im_prim(rho: f64, v: [f64; 3], b: [f64; 3]) -> IsoMhdPrim<f64, 3> {
        IsoMhdPrim {
            hydro: PrimG {
                rho,
                vel: Tensor::new(v),
                pre: Zero::default(),
            },
            mag: Tensor::new(b),
        }
    }

    fn assert_iso_flux_eq(got: &IsoMhdCons<f64, 3>, want: &IsoMhdCons<f64, 3>, ctx: &str) {
        assert!(
            approx(got.den, want.den),
            "{ctx} den: {} vs {}",
            got.den,
            want.den
        );
        for dd in 0..3 {
            assert!(
                approx(got.mom[dd], want.mom[dd]),
                "{ctx} mom[{dd}]: {} vs {}",
                got.mom[dd],
                want.mom[dd]
            );
            assert!(
                approx(got.mag[dd], want.mag[dd]),
                "{ctx} mag[{dd}]: {} vs {}",
                got.mag[dd],
                want.mag[dd]
            );
        }
    }

    #[test]
    fn hlld_isothermal_uniform_is_physical_flux() {
        // consistency: F(U, U) == F(U). all star factors collapse (u* = u_n, fac_v = 0,
        // fac_b = 1), so the star state is U and the sampled flux is the exact physical flux.
        let eos = Isothermal { cs: 0.7 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let cases = [
            im_prim(1.0, [0.0, 0.0, 0.0], [0.3, 1.0, 0.0]), // static, weak normal B (no resonance)
            im_prim(0.8, [0.2, -0.1, 0.15], [0.2, 0.4, -0.5]), // moving, full 3D B
            im_prim(1.2, [-0.3, 0.0, 0.2], [0.0, 0.8, -0.3]), // Bn = 0 (perpendicular)
        ];
        for (ii, prim) in cases.iter().enumerate() {
            let flux = hlld_isothermal(&eos, prim, prim, &nhat, 0.0);
            let exact = IsothermalMhd.to_flux(prim, &nhat, &eos);
            assert_iso_flux_eq(&flux, &exact, &format!("iso uniform case {ii}"));
        }
    }

    #[test]
    fn hlld_isothermal_supersonic_upwinds() {
        // |v_n| >> fast speed -> the whole fan is one-sided, so the vface=0 flux is the
        // upwind physical flux. both directions.
        let eos = Isothermal { cs: 0.5 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let pl = im_prim(1.0, [5.0, 0.2, 0.0], [0.3, 0.5, 0.0]);
        let pr = im_prim(0.5, [5.0, -0.1, 0.2], [0.3, -0.4, 0.1]);
        let flux_r = hlld_isothermal(&eos, &pl, &pr, &nhat, 0.0);
        assert_iso_flux_eq(
            &flux_r,
            &IsothermalMhd.to_flux(&pl, &nhat, &eos),
            "iso supersonic-right",
        );

        let ql = im_prim(1.0, [-5.0, 0.2, 0.0], [0.3, 0.5, 0.0]);
        let qr = im_prim(0.5, [-5.0, -0.1, 0.2], [0.3, -0.4, 0.1]);
        let flux_l = hlld_isothermal(&eos, &ql, &qr, &nhat, 0.0);
        assert_iso_flux_eq(
            &flux_l,
            &IsothermalMhd.to_flux(&qr, &nhat, &eos),
            "iso supersonic-left",
        );
    }

    #[test]
    fn hlld_isothermal_normal_b_flux_vanishes() {
        // Bn made single-valued across the fan -> induction F(Bn) = 0 exactly.
        let eos = Isothermal { cs: 0.6 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let pl = im_prim(1.0, [0.1, 0.2, 0.0], [0.5, 0.6, 0.0]);
        let pr = im_prim(0.7, [-0.2, 0.1, 0.1], [0.5, -0.3, 0.2]);
        let flux = hlld_isothermal(&eos, &pl, &pr, &nhat, 0.0);
        assert!(
            flux.mag[0].abs() < 1e-12,
            "normal-B flux must vanish: {}",
            flux.mag[0]
        );
    }

    #[test]
    fn hlld_isothermal_alfven_resonance_stays_bounded() {
        // tier-1 #3b. a STRONG normal field (cax^2 = Bn^2/rho = 1.0 > cs^2 = 0.25) makes the
        // fast speed approach the Alfven speed near a vanishing tangential field, where the
        // transverse-star denominator den = (S_k - u*)^2 - cax^2 is smallest and the relative
        // guard engages. sweep By across that band on an asymmetric shock (u* != u_n) and
        // require the flux to stay finite and O(1). NOTE: unlike the newtonian solver (whose
        // energy-bearing star state goes singular -> negative pressure at the Orszag-Tang
        // resonance), the isothermal degeneracy is empirically BENIGN — the fast speed brackets
        // the Alfven speed so den >= 0 with the zero removable (0/0), and the HLL-average density
        // is always positive. so this is a robustness/regression pin reproducing no
        // catastrophic blow-up; the relative guard's value is regularizing den near-zero to the
        // physically-correct no-rotation limit (and matching the newtonian guard for the
        // identical den structure); no Inf is being rescued.
        let eos = Isothermal { cs: 0.5 };
        let nhat = Tensor::<f64, 3>::unit(0);
        for k in 0..12 {
            let by = 0.5 * 0.5_f64.powi(k); // 0.5, 0.25, ... ~1.2e-4 (crosses the guard band)
            let pl = im_prim(1.0, [0.3, 0.1, 0.0], [1.0, by, 0.0]);
            let pr = im_prim(0.6, [-0.2, -0.15, 0.0], [1.0, by, 0.0]);
            let flux = hlld_isothermal(&eos, &pl, &pr, &nhat, 0.0);
            assert!(
                flux.den.is_finite(),
                "den flux non-finite at By={by}: {}",
                flux.den
            );
            for dd in 0..3 {
                assert!(
                    flux.mom[dd].is_finite() && flux.mag[dd].is_finite(),
                    "comp {dd} non-finite at By={by}: mom {} mag {}",
                    flux.mom[dd],
                    flux.mag[dd],
                );
                assert!(
                    flux.mom[dd].abs() < 50.0,
                    "mom[{dd}] unbounded at By={by}: {}",
                    flux.mom[dd]
                );
                assert!(
                    flux.mag[dd].abs() < 50.0,
                    "mag[{dd}] unbounded at By={by}: {}",
                    flux.mag[dd]
                );
            }
        }
    }
}
