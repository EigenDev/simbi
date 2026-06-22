// =============================================================================
// rmhd/cons.rs
//
// the RMHD cons->prim recovery — Kastaun, Kalinani & Ciolfi (KKC) false-position.
// the rescale (Eqs. 22-25), the bracketing `find_mu_plus` (root of Eq. 49), the
// master residual `kkc_fmu44` (Eq. 44), and the 6-state false-position root `mu`
// (Illinois half-damp, sticky `done`), then the algebraic recovery. carrier-generic
// (every C++ branch is a traceable `select`); at S=Gv it lowers to one
// multi-accumulator IterateInline, at S=f64/f32 it is the false-position loop.
// =============================================================================

use symbi_algebra::{Tensor, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use crate::eos::Eos;
use crate::state::Prim;
use crate::mhd_state::{MhdPrim, MhdCons};
use crate::c2p_result::C2pResult;

/// host bound on the false-position; the substrate kernel bakes its own (build.rs).
const RMHD_MAX_ITER: usize = 100;
/// convergence tolerance for false-position iteration (also the B=0 divzero guard).
const CONVERGENCE_TOL: f64 = 1e-12;
/// pressure floor for specific internal energy bound (zero-T floor, RMHD KKC-specific:
/// keeps f_upper0 >= 0; matches the C++ `pfloor = 1e-3`). NOT a diagnostic threshold.
const PRESSURE_FLOOR_EPS: f64 = 1e-3;

/// KKC Eq. 49 bracketing function (enthalpy limit h0 = 1): `mu*sqrt(1 + rbar_sq) - 1`,
/// whose root brackets `mu_plus`. carrier-generic. TEST-ONLY: production `find_mu_plus`
/// collapsed to the constant `1` (its proof is `kkc_fmu49(1) >= 0` unconditionally — see the
/// doc on `find_mu_plus`), so this is no longer called on the hot path; it survives only to
/// pin the C++ `helpers::kkc_fmu49` parity + as the instrumented bracket-search probe.
#[cfg(test)]
fn kkc_fmu49<S: Scalar>(mu: S, bee_sq: S, rdb_sq: S, r: S) -> S {
    let x = S::ONE / (S::ONE + mu * bee_sq);
    let rbar_sq = r * r * x * x + mu * x * (S::ONE + x) * rdb_sq;
    mu * (S::ONE + rbar_sq).sqrt() - S::ONE
}

/// KKC Eq. 44 master function (h0 = 1): the residual `mu - muhat` whose root is the
/// c2p solution. carrier-generic — every C++ branch is a traceable `select`.
#[allow(clippy::too_many_arguments)]
fn kkc_fmu44<S: Scalar>(mu: S, r: S, rp_sq: S, bee_sq: S, rdb_sq: S, qq: S, dd: S, gamma: S) -> S {
    let half = S::from_f64(0.5);
    let x = S::ONE / (S::ONE + mu * bee_sq);
    let rbar_sq = r * r * x * x + mu * x * (S::ONE + x) * rdb_sq;
    let qbar = qq - half * (bee_sq + mu * mu * x * x * bee_sq * rp_sq);

    // velocity ceiling z_upper = r/h0 = r; v_limit = z/sqrt(1+z^2).
    let z_upper = r;
    let v_limit = z_upper / (S::ONE + z_upper * z_upper).sqrt();
    let vsq = (mu * mu * rbar_sq).min(v_limit * v_limit);
    let gbsq = vsq / (S::ONE - vsq);
    let g = (S::ONE + gbsq).sqrt();

    let rhohat = dd / g;
    let eps = g * (qbar - mu * rbar_sq) + gbsq / (S::ONE + g);
    let eps_min = S::from_f64(PRESSURE_FLOOR_EPS) / (rhohat * (gamma - S::ONE)); // zero-T floor
    let epshat = eps.max(eps_min);
    let phat = (gamma - S::ONE) * rhohat * epshat;
    let ahat = phat / (rhohat * (S::ONE + epshat));
    let nu_hat_a = (S::ONE + ahat) * (S::ONE + epshat) / g;
    let nu_hat_b = (S::ONE + ahat) * (S::ONE + qbar - mu * rbar_sq);
    let nu_hat = S::select(eps.cmp_lt(eps_min), nu_hat_a, nu_hat_a.max(nu_hat_b));
    let muhat = S::ONE / (nu_hat + rbar_sq * mu);
    mu - muhat
}

/// KKC `find_mu_plus` — the upper bracket for the false-position root of `kkc_fmu44`.
///
/// since `kkc_fmu49(0) = −1 < 0` and `kkc_fmu49(1) = sqrt(1+rbar²(1)) − 1 ≥ 0` for ANY
/// state (the rbar² in the sqrt is always non-negative), the root of `kkc_fmu49` is
/// always in `[0, 1]` regardless of `r`. so `mu = 1` is always a valid upper bracket.
///
/// the previous implementation ran 50 iters of bisection on `[0, 1]` to tighten the
/// bracket near the root for `r >= 1` cells (and threw the result away for `r < 1`,
/// where it's discarded by the final select). measurement on orszag_tang n=128 t=0.3
/// HLLE: the production 50-iter bisection cost 15% of wall time. false-position from
/// the wide initial bracket [0, 1] converges in 5-7 iters (same as from the tight
/// bracket on the same problem), so the bisection-tightening did not actually reduce
/// fp iters meaningfully. removing it: 3.89s → 3.32s (-15%).
///
/// see docs/c9fbdcb_perf_study/01_c2p.md + the c2p_iter_distribution_orszag_tang
/// test below for the empirical investigation.
fn find_mu_plus<S: Scalar>(_bee_sq: S, _rdb_sq: S, _r: S) -> S {
    S::ONE
}

/// the branch-free RMHD cons->prim recovery — THE single-source physics: the rescale
/// (KKC Eqs. 22-25), the KKC false-position root `mu` (the 6-state bracket `[mu_lo,
/// mu_hi, f_lo, f_hi, mu, done]` over `kkc_fmu44`, Illinois half-damp, sticky `done`),
/// and the algebraic recovery (Eqs. 26/38/39/32/41/42/43/68). NO guards — the host
/// `rmhd_to_primitive` wrapper adds the C2pResult diagnostics post-hoc. traced at
/// `S = Gv` (`symbi_discretize::rmhd_c2p_gv`) it lowers to ONE multi-accumulator
/// `IterateInline`; at `S = f64/f32` it is the false-position loop. `use_four_velocity
/// = false`: the returned velocity is the 3-velocity. RMHD vectors are always 3-comp.
pub fn rmhd_recover<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    cons: &MhdCons<S, D>,
    max_iter: usize,
) -> MhdPrim<S, D> {
    let dd = cons.den;
    let tau = cons.nrg;
    let bfield = cons.mag;
    let half = S::from_f64(0.5);
    let gamma = eos.gamma();
    let eps = S::from_f64(CONVERGENCE_TOL); // 1e-12: the bracket guard + the convergence tol

    // rescale the conserved (KKC Eqs. 22-25).
    let inv_d = S::ONE / dd;
    let isqrtd = inv_d.sqrt();
    let qq = tau * inv_d;
    let rvec = cons.mom.scale(inv_d);
    let r_sq = rvec.dot(&rvec);
    let r_mag = r_sq.sqrt();
    let hvec = bfield.scale(isqrtd);
    let bee_sq = hvec.dot(&hvec) + eps; // + epsilon: divzero guard when B = 0
    let rdb = rvec.dot(&hvec);
    let rdb_sq = rdb * rdb;
    let rparr = hvec.scale(rdb / bee_sq);
    let rperp = rvec - rparr;
    let rp_sq = rperp.dot(&rperp);

    // the master residual at a given mu (the 8 invariants are fixed).
    let kkc = |mu: S| kkc_fmu44(mu, r_mag, rp_sq, bee_sq, rdb_sq, qq, dd, gamma);

    // KKC false-position over `kkc_fmu44`, producing the root `mu`. the 6-state bracket
    // freezes on the OLD sticky `done` so the iteration that first converges still WRITES
    // its mu and the next freezes it (exact C++ do-while semantics; see iterate_vec).
    let muu0 = find_mu_plus(bee_sq, rdb_sq, r_mag);
    let f_lower0 = kkc(S::ZERO);
    let f_upper0 = kkc(muu0);
    let mu = S::iterate_vec(
        [S::ZERO, muu0, f_lower0, f_upper0, S::ZERO, S::ZERO], // [mul, muu, f_lo, f_hi, mu, done]
        max_iter,
        |s| {
            let (mul, muu, f_lo, f_hi, done) = (s[0], s[1], s[2], s[3], s[5]);
            let mu = (mul * f_hi - muu * f_lo) / (f_hi - f_lo);
            let ff = kkc(mu);
            // C++: ff*f_upper < 0 ? {mul=muu; f_lo=f_hi} : {f_lo *= 0.5}; muu=mu, f_hi=ff.
            let cond = (ff * f_hi).cmp_lt(S::ZERO);
            let mul_n = S::select(cond, muu, mul);
            let f_lo_n = S::select(cond, f_hi, half * f_lo);
            // C++ post-update test: |mul-muu| <= eps OR |ff| <= eps.
            let conv = (mul_n - mu).abs().min(ff.abs()).cmp_lt(eps);
            let done_n = done.max(S::select(conv, S::ONE, S::ZERO)); // sticky
            [mul_n, mu, f_lo_n, ff, mu, done_n]
        },
        |s, _next| half.cmp_lt(s[5]), // freeze once `done` was ALREADY set last step
        4,                            // result = mu
    );
    // bracketing guarantee (resolves the find_mu_plus=1 / kkc_fmu44-root concern):
    // false-position needs f_lower0 = kkc(0) and f_upper0 = kkc(muu0=1) to straddle zero.
    // f_lower0 = -muhat(0) < 0 always (muhat > 0). f_upper0 = 1 - muhat(1) >= 0 is forced
    // by TWO mechanisms baked into kkc_fmu44, not by the fmu49 proof: (1) the v_limit
    // ceiling (vsq <= v_limit^2 < 1, line ~50) keeps g/rhohat/eps finite for any mu in
    // [0,1]; (2) the eps_min zero-T floor drives nu_hat -> ~1+eps_min > 1 in the strong-
    // field / low-energy limit (where bee_sq is large), so muhat(1) < 1 and f_upper0 > 0.
    // verified: the 3M-state probe (docs) + rmhd_recover_brackets_evolved_high_w_high_b_states
    // (high-W, strong+weak B, low-density) find ZERO non-bracketing cells. a defensive
    // NaN-poison was prototyped and removed: it never fired on any reachable state and
    // added ops to the hottest kernel (c2p) — untestable, anti-perf. f_upper0 is consumed
    // below only as the iterate seed.

    // recovery (Eqs. 26/38/39/32/41/42/43/68); use_four_velocity = false.
    let x = S::ONE / (S::ONE + mu * bee_sq);
    let mu2 = mu * mu;
    let rbar_sq = r_sq * x * x + mu * x * (S::ONE + x) * rdb_sq;
    let qbar = qq - half * (bee_sq + mu2 * x * x * bee_sq * rp_sq);
    let vsq = mu2 * rbar_sq;
    let gbsq = vsq / (S::ONE - vsq);
    let ww = (S::ONE + gbsq).sqrt();
    let rho = dd / ww;
    let eps_e = ww * (qbar - mu * rbar_sq) + gbsq / (S::ONE + ww);
    let rho_gm1 = rho * (gamma - S::ONE);
    let epshat = eps_e.max(S::from_f64(PRESSURE_FLOOR_EPS) / rho_gm1); // zero-T floor
    let pre = rho_gm1 * epshat;
    let mu_x = mu * x;
    let rdb_mu = rdb * mu;
    let vel = (rvec + hvec.scale(rdb_mu)).scale(mu_x);

    MhdPrim { hydro: Prim { rho, vel, pre }, mag: bfield }
}

/// the host RMHD cons->prim: the branch-free `rmhd_recover` plus post-hoc C2pResult
/// diagnostics. no silent floor — the value is the raw recovered state, the ErrorCode
/// is the explicit signal (matches `Newtonian::to_primitive`; feedback_no_silent_floors).
///
/// **host-only** (Tier 1.7): `S: OrderedNumeric` because the diagnostic check uses
/// native `<` / `<=` / `==` on a host scalar. the kernel path is `rmhd_recover` above
/// (carrier-generic over `S: Scalar`), not this wrapper.
pub(crate) fn rmhd_to_primitive<S: Scalar + OrderedNumeric, const D: usize>(
    eos: &impl Eos<S>,
    cons: &MhdCons<S, D>,
) -> C2pResult<MhdPrim<S, D>> {
    let dd = cons.den;

    // input guard: clearly-invalid conserved density (host-only early-out, NOT in the
    // kernel path). B passes through. shared SRHD/RMHD guard (now NaN-checked too; tier-1 #5).
    if let Some(code) = crate::c2p_result::relativistic_density_guard(dd) {
        let floored = MhdPrim {
            hydro: Prim { rho: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR), vel: Tensor::zeros(),
                          pre: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR) },
            mag: cons.mag,
        };
        return C2pResult::err(floored, code);
    }

    let prim = rmhd_recover(eos, cons, RMHD_MAX_ITER);

    // post-hoc diagnostics on the raw recovered state (shared SRHD/RMHD contract; tier-1 #5).
    let v_sq = prim.vel.dot(&prim.vel);
    let code = crate::c2p_result::relativistic_c2p_code(prim.rho, prim.pre, v_sq);
    if code.is_ok() { C2pResult::ok(prim) } else { C2pResult::err(prim, code) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;

    // straight-Rust transcription of cpp_src/utility/helpers.hpp::kkc_fmu49.
    fn ref_fmu49(mu: f64, beesq: f64, beedrsq: f64, r: f64) -> f64 {
        let x = 1.0 / (1.0 + mu * beesq);
        let rbar_sq = r * r * x * x + mu * x * (1.0 + x) * beedrsq;
        mu * (1.0 + rbar_sq).sqrt() - 1.0
    }

    // straight-Rust transcription of cpp_src/utility/helpers.hpp::kkc_fmu44.
    #[allow(clippy::too_many_arguments)]
    fn ref_fmu44(mu: f64, r: f64, rperp: f64, beesq: f64, beedrsq: f64, qterm: f64, dterm: f64, gamma: f64) -> f64 {
        let x = 1.0 / (1.0 + mu * beesq);
        let rbar_sq = r * r * x * x + mu * x * (1.0 + x) * beedrsq;
        let qbar = qterm - 0.5 * (beesq + mu * mu * x * x * beesq * rperp);
        let z_upper = r;
        let v_limit = z_upper / (1.0 + z_upper * z_upper).sqrt();
        let vsq = (mu * mu * rbar_sq).min(v_limit * v_limit);
        let gbsq = vsq / (1.0 - vsq);
        let g = (1.0 + gbsq).sqrt();
        let rhohat = dterm / g;
        let eps = g * (qbar - mu * rbar_sq) + gbsq / (1.0 + g);
        let eps_min = 1.0e-3 / (rhohat * (gamma - 1.0));
        let epshat = eps.max(eps_min);
        let phat = (gamma - 1.0) * rhohat * epshat;
        let ahat = phat / (rhohat * (1.0 + epshat));
        let nu_hat_a = (1.0 + ahat) * (1.0 + epshat) / g;
        let nu_hat_b = (1.0 + ahat) * (1.0 + qbar - mu * rbar_sq);
        let nu_hat = if eps < eps_min { nu_hat_a } else { nu_hat_a.max(nu_hat_b) };
        let muhat = 1.0 / (nu_hat + rbar_sq * mu);
        mu - muhat
    }

    // transcription of the FULL helpers.hpp::find_mu_plus (doubling + bisection + break)
    // that the carrier-generic bisection elides; agreement to ~1e-9 proves the elision.
    // ORPHANED: kept as the C++ reference, but no test currently asserts against it
    // (the comparison was lost). allow(dead_code) preserves the reference until it is
    // re-wired into an elision-equivalence test or removed.
    #[allow(dead_code)]
    fn ref_find_mu_plus(beesq: f64, beedrsq: f64, r: f64) -> f64 {
        if r < 1.0 { return 1.0; }
        let mut mu_lower = 0.0;
        let mut mu_upper = 1.0;
        let mut f_upper = ref_fmu49(mu_upper, beesq, beedrsq, r);
        while f_upper < 0.0 { mu_upper *= 2.0; f_upper = ref_fmu49(mu_upper, beesq, beedrsq, r); }
        let eps = 1.0e-12;
        let mut mu_mid = 1.0;
        let mut f_lower = ref_fmu49(mu_lower, beesq, beedrsq, r);
        let mut iter = 0;
        while iter < 50 && (mu_upper - mu_lower) > eps {
            mu_mid = 0.5 * (mu_lower + mu_upper);
            let f_mid = ref_fmu49(mu_mid, beesq, beedrsq, r);
            if f_mid.abs() < eps { break; }
            if f_mid * f_lower < 0.0 { mu_upper = mu_mid; } else { mu_lower = mu_mid; f_lower = f_mid; }
            iter += 1;
        }
        mu_mid * 1.000001
    }

    #[test]
    fn kkc_fmu49_matches_cpp_reference() {
        let cases = [(0.2, 0.4, 0.01, 0.3), (0.5, 0.8, 0.05, 0.6), (0.8, 1.2, 0.1, 0.9), (1.5, 0.3, 0.02, 1.2)];
        for (mu, beesq, beedrsq, r) in cases {
            let got = kkc_fmu49::<f64>(mu, beesq, beedrsq, r);
            let want = ref_fmu49(mu, beesq, beedrsq, r);
            assert!((got - want).abs() < 1e-12, "fmu49(mu={mu}): {got} != {want}");
        }
    }

    #[test]
    fn kkc_fmu44_matches_cpp_reference() {
        let g = 5.0 / 3.0;
        let cases = [(0.2, 0.3, 0.05, 0.4, 0.01, 0.5, 1.0), (0.5, 0.6, 0.1, 0.8, 0.05, 1.0, 1.5), (0.8, 0.9, 0.2, 1.2, 0.1, 1.5, 2.0)];
        for (mu, r, rperp, beesq, beedrsq, q, d) in cases {
            let got = kkc_fmu44::<f64>(mu, r, rperp, beesq, beedrsq, q, d, g);
            let want = ref_fmu44(mu, r, rperp, beesq, beedrsq, q, d, g);
            assert!((got - want).abs() < 1e-12, "fmu44(mu={mu}): {got} != {want}");
        }
    }

    #[test]
    fn find_mu_plus_returns_unity() {
        // find_mu_plus was simplified to return `1` unconditionally. the proof
        // that 1 is always a valid upper bracket of the kkc_fmu49 root lives in
        // the doc comment on the production fn. validation that the WIDE initial
        // bracket [0, 1] still produces a converging false-position is the
        // c2p_iter_distribution_orszag_tang test below + the orszag_tang
        // checkpoint round-trip in `crates/symbi/examples`.
        let cases = [(0.3, 0.02, 1.2), (0.8, 0.1, 2.0), (1.5, 0.3, 5.0), (0.5, 0.05, 1.0), (0.4, 0.01, 0.5), (0.6, 0.05, 0.8)];
        for (beesq, beedrsq, r) in cases {
            let got = find_mu_plus::<f64>(beesq, beedrsq, r);
            assert_eq!(got, 1.0, "find_mu_plus(r={r}): expected unity, got {got}");
        }
    }

    // =========================================================================
    // c2p iter-distribution probe — 2026-06-07
    //
    // measures actual iter counts for `find_mu_plus` (bisection) and the main
    // false-position loop on orszag_tang IC states, with `break` instead of
    // fixed-iter freeze. answers: which of the two loops actually dominates?
    // can mu-cache (lever 5) help, or does find_mu_plus eat the budget anyway?
    //
    // run: `cargo test --release -p symbi-hydro c2p_iter_distribution -- --nocapture`
    // =========================================================================

    /// instrumented `find_mu_plus`: bisection over [0,1] with real `break` when
    /// the bracket falls below `tol`. returns (mu_plus, actual_iters).
    fn find_mu_plus_instrumented(bee_sq: f64, rdb_sq: f64, r: f64, tol: f64) -> (f64, usize) {
        let mut mu_l = 0.0_f64;
        let mut mu_u = 1.0_f64;
        // f_l = kkc_fmu49(0, ..) = -1 always (the bracket's left end); the full
        // expression simplifies to exactly this, so bind it directly.
        let mut f_l = -1.0_f64;
        let mut mu_mid = 0.5_f64;
        let max_iter = 60_usize;
        let mut iters = max_iter;
        for ii in 0..max_iter {
            let mid = 0.5 * (mu_l + mu_u);
            let f_mid = kkc_fmu49::<f64>(mid, bee_sq, rdb_sq, r);
            if (mu_u - mu_l).abs() < tol {
                iters = ii;
                mu_mid = mid;
                break;
            }
            let cond = f_mid * f_l < 0.0;
            if cond { mu_u = mid; } else { mu_l = mid; f_l = f_mid; }
            mu_mid = mid;
        }
        let result = if r < 1.0 { 1.0 } else { mu_mid * 1.000001 };
        (result, iters)
    }

    /// instrumented full c2p with real `break` on convergence. returns
    /// (mu_root, find_mu_plus_iters, false_position_iters).
    fn instrumented_c2p(
        cons: MhdCons<f64, 3>,
        gamma: f64,
        max_fp_iter: usize,
        tol: f64,
    ) -> (f64, usize, usize) {
        let dd = cons.den;
        let tau = cons.nrg;
        let bfield = cons.mag;
        let inv_d = 1.0 / dd;
        let isqrtd = inv_d.sqrt();
        let qq = tau * inv_d;
        let rvec = cons.mom.scale(inv_d);
        let r_sq = rvec.dot(&rvec);
        let r_mag = r_sq.sqrt();
        let hvec = bfield.scale(isqrtd);
        let bee_sq = hvec.dot(&hvec) + tol;
        let rdb = rvec.dot(&hvec);
        let rdb_sq = rdb * rdb;
        let rparr = hvec.scale(rdb / bee_sq);
        let rperp = rvec - rparr;
        let rp_sq = rperp.dot(&rperp);

        // 1. bracket via instrumented bisection.
        let (muu0, fmp_iters) = find_mu_plus_instrumented(bee_sq, rdb_sq, r_mag, tol);

        // 2. false-position with Illinois half-damp + real break.
        let mut mul = 0.0_f64;
        let mut muu = muu0;
        let mut f_lo = kkc_fmu44::<f64>(mul, r_mag, rp_sq, bee_sq, rdb_sq, qq, dd, gamma);
        let mut f_hi = kkc_fmu44::<f64>(muu, r_mag, rp_sq, bee_sq, rdb_sq, qq, dd, gamma);
        let mut mu = 0.0_f64;
        let mut fp_iters = max_fp_iter;
        for ii in 0..max_fp_iter {
            let mu_new = (mul * f_hi - muu * f_lo) / (f_hi - f_lo);
            let ff = kkc_fmu44::<f64>(mu_new, r_mag, rp_sq, bee_sq, rdb_sq, qq, dd, gamma);
            let cond = ff * f_hi < 0.0;
            let mul_n = if cond { muu } else { mul };
            let f_lo_n = if cond { f_hi } else { 0.5 * f_lo };
            // post-update convergence (matches production: |mul-mu| OR |f| below tol).
            let bracket_tight = (mul_n - mu_new).abs() < tol;
            let func_tight = ff.abs() < tol;
            if bracket_tight || func_tight {
                fp_iters = ii + 1;
                mu = mu_new;
                break;
            }
            mul = mul_n;
            f_lo = f_lo_n;
            muu = mu_new;
            f_hi = ff;
            mu = mu_new;
        }
        (mu, fmp_iters, fp_iters)
    }

    /// orszag_tang IC at coords (x, y), matching `crates/symbi/examples/rmhd_orszag_tang.rs`.
    fn orszag_tang_prim(x: f64, y: f64, gamma: f64, v0: f64, b0: f64) -> MhdPrim<f64, 3> {
        let pi = std::f64::consts::PI;
        let rho = gamma * gamma;
        let pre = gamma;
        let vx = -v0 * (2.0 * pi * y).sin();
        let vy =  v0 * (2.0 * pi * x).sin();
        let vz = 0.0;
        let bx = -b0 * (2.0 * pi * y).sin();
        let by =  b0 * (4.0 * pi * x).sin();
        let bz = 0.0;
        MhdPrim {
            hydro: Prim { rho, vel: Tensor::new([vx, vy, vz]), pre },
            mag: Tensor::new([bx, by, bz]),
        }
    }

    /// inline prim->cons mirroring `Rmhd::to_conserved`. avoids the trait import.
    fn rmhd_prim_to_cons(prim: &MhdPrim<f64, 3>, gamma: f64) -> MhdCons<f64, 3> {
        let v_sq = prim.vel.dot(&prim.vel);
        let w_sq = 1.0 / (1.0 - v_sq);
        let ww = w_sq.sqrt();
        let h = 1.0 + gamma * prim.pre / (prim.rho * (gamma - 1.0));
        let rho_h_w2 = prim.rho * h * w_sq;
        let bsq = prim.mag.dot(&prim.mag);
        let vdb = prim.vel.dot(&prim.mag);
        let b_mu_sq = bsq / w_sq + vdb * vdb;
        let p_tot = prim.pre + 0.5 * b_mu_sq;
        let den = prim.rho * ww;
        let mom = prim.vel.scale(rho_h_w2 + bsq) - prim.mag.scale(vdb);
        let nrg = rho_h_w2 + bsq - p_tot - den;
        MhdCons {
            hydro: crate::state::Cons { den, mom, nrg },
            mag: prim.mag,
        }
    }

    #[test]
    fn c2p_iter_distribution_orszag_tang() {
        // generate a 128×128 grid of orszag_tang IC cells, convert each to cons,
        // run instrumented c2p, accumulate histograms for find_mu_plus and fp.
        let gamma = 5.0 / 3.0;
        let v0 = 0.5;
        let b0 = 1.0;
        let n = 128_usize;
        let tol = 1e-12_f64;
        let max_fp = 100_usize;

        let mut fmp_hist = vec![0_usize; 65];
        let mut fp_hist  = vec![0_usize; max_fp + 1];
        let mut total = 0_usize;
        let mut fmp_sum = 0_usize;
        let mut fp_sum  = 0_usize;
        let mut fp_failed = 0_usize;
        let mut fmp_max = 0_usize;
        let mut fp_max  = 0_usize;

        for j in 0..n {
            for i in 0..n {
                let x = (i as f64 + 0.5) / n as f64;
                let y = (j as f64 + 0.5) / n as f64;
                let prim = orszag_tang_prim(x, y, gamma, v0, b0);
                let cons = rmhd_prim_to_cons(&prim, gamma);
                let (_, fmp_iters, fp_iters) = instrumented_c2p(cons, gamma, max_fp, tol);
                if fmp_iters < fmp_hist.len() { fmp_hist[fmp_iters] += 1; }
                if fp_iters < fp_hist.len() { fp_hist[fp_iters] += 1; }
                if fp_iters == max_fp { fp_failed += 1; }
                fmp_sum += fmp_iters;
                fp_sum  += fp_iters;
                fmp_max = fmp_max.max(fmp_iters);
                fp_max  = fp_max.max(fp_iters);
                total += 1;
            }
        }

        let fmp_avg = fmp_sum as f64 / total as f64;
        let fp_avg  = fp_sum  as f64 / total as f64;

        // percentiles from cumulative histogram.
        let pct = |hist: &[usize], pct: f64| -> usize {
            let target = (total as f64 * pct).ceil() as usize;
            let mut cum = 0_usize;
            for (k, &v) in hist.iter().enumerate() {
                cum += v;
                if cum >= target { return k; }
            }
            hist.len() - 1
        };
        let fmp_p50 = pct(&fmp_hist, 0.50);
        let fmp_p90 = pct(&fmp_hist, 0.90);
        let fmp_p99 = pct(&fmp_hist, 0.99);
        let fp_p50  = pct(&fp_hist,  0.50);
        let fp_p90  = pct(&fp_hist,  0.90);
        let fp_p99  = pct(&fp_hist,  0.99);

        eprintln!("\n=== c2p iter distribution on orszag_tang n={n} (tol={tol:e}) ===");
        eprintln!("total cells: {total}\n");
        eprintln!("find_mu_plus (bisection on [0,1]):");
        eprintln!("  avg = {:.2}, p50 = {}, p90 = {}, p99 = {}, max = {}", fmp_avg, fmp_p50, fmp_p90, fmp_p99, fmp_max);
        eprintln!("  production runs FIXED 50 iters always -> {} wasted iters per cell on avg",
            (50.0 - fmp_avg).max(0.0));
        eprintln!();
        eprintln!("false-position (Illinois, post-bracket):");
        eprintln!("  avg = {:.2}, p50 = {}, p90 = {}, p99 = {}, max = {}", fp_avg, fp_p50, fp_p90, fp_p99, fp_max);
        eprintln!("  production runs UP TO 100 iters w/ sticky-done freeze");
        eprintln!("  non-converged cells: {} / {} ({:.1}%)", fp_failed, total, 100.0 * fp_failed as f64 / total as f64);
        eprintln!();
        eprintln!("total avg iters/cell (both loops) = {:.1}", fmp_avg + fp_avg);
        eprintln!("production worst-case = 50 + 100 = 150 iters/cell\n");

        // dump compact histogram for fp.
        eprintln!("fp iter histogram (count per iter, 0..30):");
        for (k, &v) in fp_hist.iter().enumerate().take(31) {
            if v > 0 { eprintln!("  iter {:2}: {:5} cells ({:5.1}%)", k, v, 100.0 * v as f64 / total as f64); }
        }
        eprintln!();
        eprintln!("fmp iter histogram (count per iter, 0..60):");
        for (k, &v) in fmp_hist.iter().enumerate().take(61) {
            if v > 0 { eprintln!("  iter {:2}: {:5} cells ({:5.1}%)", k, v, 100.0 * v as f64 / total as f64); }
        }

        // sanity: any cell should converge well under the 100 cap.
        assert!(fp_failed == 0, "{} cells failed to converge in {} fp iters", fp_failed, max_fp);
    }

    // item 3: resolves the find_mu_plus=1 / kkc_fmu44-bracket concern. the proof on
    // `find_mu_plus` covers kkc_fmu49; the root actually solved is kkc_fmu44. these
    // EVOLVED-state cases (high Lorentz W, strong AND weak magnetization, low density —
    // the regime t=0 orszag_tang never reaches, where r can exceed 1) exercise the
    // production rmhd_recover end to end and confirm physical states ALWAYS bracket and
    // recover finite. zero non-bracketing cells => no kernel-path bracket guard needed
    // (the v_limit ceiling + eps_min zero-T floor force f_upper0 >= 0; see rmhd_recover).
    // pressures are kept above the eps_min zero-T floor (~PRESSURE_FLOOR_EPS) so the
    // round-trip is not floored.
    #[test]
    fn rmhd_recover_brackets_evolved_high_w_high_b_states() {
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        // (rho, |v|, pre, |B|): push W and magnetization well past orszag_tang.
        let cases = [
            (1.0_f64, 0.99,   0.1,  5.0),  // W~7, strong field
            (1.0,     0.999,  0.1,  10.0), // W~22, very strong field
            (0.1,     0.9999, 1.0,  1.0),  // W~71, low density
            (10.0,    0.95,   0.1,  50.0), // magnetization-dominated
            (1.0,     0.5,    0.1,  1e-3), // near-vacuum field
        ];
        for (rho, v, pre, b) in cases {
            // physical prim: velocity along x, B along y (non-aligned => rdb != 0).
            let prim = MhdPrim::<f64, 3> {
                hydro: Prim { rho, vel: Tensor::new([v, 0.0, 0.0]), pre },
                mag: Tensor::new([0.0, b, 0.0]),
            };
            let cons = rmhd_prim_to_cons(&prim, eos.gamma);

            // the bracket itself must straddle zero (f_lower0 < 0 < f_upper0) — this is
            // the invariant the false-position relies on, pinned directly.
            let (f_lo0, f_hi0) = bracket_endpoints(&cons, eos.gamma);
            assert!(
                f_lo0 < 0.0 && f_hi0 >= 0.0,
                "kkc_fmu44 bracket does not straddle for evolved state \
                 (rho={rho}, v={v}, pre={pre}, b={b}): f_lo0={f_lo0}, f_hi0={f_hi0}"
            );

            let got = rmhd_recover(&eos, &cons, RMHD_MAX_ITER);
            assert!(
                got.hydro.rho.is_finite() && got.hydro.pre.is_finite()
                    && got.hydro.vel.dot(&got.hydro.vel).is_finite(),
                "physical evolved state (rho={rho}, v={v}, pre={pre}, b={b}) failed to \
                 recover finite: rho={}, pre={}",
                got.hydro.rho, got.hydro.pre
            );
            assert!(
                (got.hydro.pre - pre).abs() < 1e-6 * pre.abs().max(1.0),
                "evolved-state c2p pressure mismatch (rho={rho}, v={v}, pre={pre}, b={b}): {} vs {pre}",
                got.hydro.pre
            );
        }
    }

    /// the false-position bracket endpoints f_lower0 = kkc_fmu44(0), f_upper0 =
    /// kkc_fmu44(muu0=1) for a cons state — the invariant rmhd_recover relies on.
    fn bracket_endpoints(cons: &MhdCons<f64, 3>, gamma: f64) -> (f64, f64) {
        let dd = cons.den;
        let inv_d = 1.0 / dd;
        let isqrtd = inv_d.sqrt();
        let qq = cons.nrg * inv_d;
        let rvec = cons.mom.scale(inv_d);
        let r_mag = rvec.dot(&rvec).sqrt();
        let hvec = cons.mag.scale(isqrtd);
        let bee_sq = hvec.dot(&hvec) + CONVERGENCE_TOL;
        let rdb = rvec.dot(&hvec);
        let rdb_sq = rdb * rdb;
        let rparr = hvec.scale(rdb / bee_sq);
        let rperp = rvec - rparr;
        let rp_sq = rperp.dot(&rperp);
        let f = |mu: f64| kkc_fmu44::<f64>(mu, r_mag, rp_sq, bee_sq, rdb_sq, qq, dd, gamma);
        (f(0.0), f(1.0))
    }
}
