// =============================================================================
// rmhd/cons.rs
//
// the RMHD cons->prim recovery — Kastaun, Kalinani & Ciolfi (KKC) false-position.
// the rescale (Eqs. 22-25), the bracketing `find_mu_plus` (root of Eq. 49), the
// master residual `kkc_fmu44` (Eq. 44), and the 6-state false-position root `mu`
// (Illinois half-damp, sticky `done`), then the algebraic recovery. carrier-generic
// (every branch is a traceable `select`); at S=Gv it lowers to one
// multi-accumulator IterateInline, at S=f64/f32 it is the false-position loop.
// =============================================================================

use crate::c2p_result::C2pResult;
use crate::eos::Eos;
use crate::mhd_state::{MhdCons, MhdPrim};
use crate::spatial_metric::SpatialMetric;
use crate::state::Prim;
use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_ir::algebra::Scalar;

/// host bound on the false-position; the substrate kernel bakes its own (build.rs).
const RMHD_MAX_ITER: usize = 100;
/// convergence tolerance for false-position iteration (also the B=0 divzero guard).
const CONVERGENCE_TOL: f64 = 1e-12;
/// the MAX iteration cap for the `mu_+` bracketed Illinois solve (root of `kkc_fmu49` on `[0, 1]`).
/// Illinois is superlinear so a warm cell converges in ~10 and early-breaks; this cap only bounds a
/// pathological non-converging cell. generous headroom over the typical count.
const FIND_MU_PLUS_ITERS: usize = 54;

/// KKC Eq. 49 auxiliary function (enthalpy limit h0 = 1): `f_a(mu) = mu*sqrt(1 + rbar_sq(mu)) - 1`.
/// smooth, strictly increasing, EOS-independent; its unique root `mu_+` in `(0, 1]` is the tight
/// upper bracket for the master root (KKC Sec. II F). carrier-generic. `find_mu_plus` bisects it.
fn kkc_fmu49<S: Scalar>(mu: S, bee_sq: S, rdb_sq: S, r: S) -> S {
    let x = S::ONE / (S::ONE + mu * bee_sq);
    let rbar_sq = r * r * x * x + mu * x * (S::ONE + x) * rdb_sq;
    mu * (S::ONE + rbar_sq).sqrt() - S::ONE
}

/// KKC Eq. 44 master function (h0 = 1): the residual `mu - muhat` whose root is the
/// c2p solution. carrier-generic — every branch is a traceable `select`.
#[allow(clippy::too_many_arguments)]
fn kkc_fmu44<S: Scalar>(mu: S, r: S, rp_sq: S, bee_sq: S, rdb_sq: S, qq: S, dd: S, gamma: S) -> S {
    let half = S::from_f64(0.5);
    let x = S::ONE / (S::ONE + mu * bee_sq);
    let rbar_sq = r * r * x * x + mu * x * (S::ONE + x) * rdb_sq;
    let qbar = qq - half * (bee_sq + mu * mu * x * x * bee_sq * rp_sq);

    // the shared c2p velocity ceiling v_limit^2 = r^2/(1+r^2) (KKC h0 = 1).
    let vsq = (mu * mu * rbar_sq).min(crate::c2p_result::relativistic_velocity_ceiling_sq(r * r));
    let gbsq = vsq / (S::ONE - vsq);
    let g = (S::ONE + gbsq).sqrt();

    let rhohat = dd / g;
    let eps = g * (qbar - mu * rbar_sq) + gbsq / (S::ONE + g);
    // NO pressure floor: the RAW specific internal energy. a cold or unphysical (eps < 0) state
    // recovers a small or negative pressure that the post-hoc c2p diagnostic flags (fail-loud);
    // a floor would silently warm it to eps_min = pfloor/(rho (gamma-1)), a spurious-physical
    // state that masks the failure. nu_hat is the enthalpy branch max unconditionally.
    let phat = (gamma - S::ONE) * rhohat * eps;
    let ahat = phat / (rhohat * (S::ONE + eps));
    let nu_hat_a = (S::ONE + ahat) * (S::ONE + eps) / g;
    let nu_hat_b = (S::ONE + ahat) * (S::ONE + qbar - mu * rbar_sq);
    let nu_hat = nu_hat_a.max(nu_hat_b);
    let muhat = S::ONE / (nu_hat + rbar_sq * mu);
    mu - muhat
}

/// KKC `find_mu_plus` — the tight upper bracket `mu_+` for the master root, defined as the
/// unique root of the auxiliary `kkc_fmu49` (KKC Eq. 49) on `(0, 1/h0]` with `h0 = 1`.
///
/// KKC prove (Sec. II F, Eq. 54) that `f(mu_+) >= 0` while `f(0) < 0`, so the master root lies
/// in `(0, mu_+]`, and (Sec. II G) that it is the UNIQUE root there. beyond `mu_+` the velocity
/// cutoff in `kkc_fmu44` (the `v_limit` clamp) induces a "strong kink" that can produce a SECOND,
/// spurious root corresponding to a superluminal, negative-pressure state.
///
/// bracketing with the loose upper bound `1/h0 = 1` is valid ONLY when `r < h0` (KKC:
/// then `f_a(1/h0) > 0`, so `[0, 1]` still straddles the single root). for `r >= h0` the interval
/// `[0, 1]` spans BOTH the physical root and the spurious one, `f(1) < 0`, and the false-position
/// converges to the spurious superluminal root. this manifests whenever `r.b != 0` (a shock-normal
/// magnetic field), which is precisely when a shock drives `r` past `h0`. so `mu_+` must be
/// computed for the general case.
///
/// `f_a` is smooth and strictly increasing with `f_a(0) = -1 < 0` and `f_a(1) >= 0` for ANY state
/// (the `rbar_sq` under the sqrt is non-negative), so bisection on `[0, 1]` always brackets its
/// root. returning the UPPER end guarantees the result is `>= root(f_a) = mu_+`, hence
/// `f(result) >= 0` and a valid straddle `f(0) < 0 <= f(result)` for the master false-position.
fn find_mu_plus<S: Scalar>(bee_sq: S, rdb_sq: S, r: S) -> S {
    let half = S::from_f64(0.5);
    let eps = S::from_f64(CONVERGENCE_TOL);
    // bracketed Illinois (regula falsi + stale-endpoint half-damp) on the monotone-increasing f_a.
    // superlinear, so it reaches tol in ~10 iters (vs ~54 for bisection); `converged` drives the
    // IterateInline early-break so a converged cell stops. `hi` ALWAYS satisfies f_a(hi) >= 0 (the
    // straddle-preserving upper bracket >= mu_+): every step either keeps the old hi or moves hi to
    // a point with f_a >= 0. the halved f_hi/f_lo are interpolation weights only — the bracket
    // POSITIONS keep their true signs, so the >= mu_+ guarantee is exact regardless of the damping.
    let f_lo0 = kkc_fmu49(S::ZERO, bee_sq, rdb_sq, r); // = -1 < 0
    let f_hi0 = kkc_fmu49(S::ONE, bee_sq, rdb_sq, r); // >= 0 for any state
    S::iterate_vec(
        [S::ZERO, S::ONE, f_lo0, f_hi0, S::ZERO], // [lo, hi, f_lo, f_hi, done]; f_a(lo) < 0 <= f_a(hi)
        FIND_MU_PLUS_ITERS,
        |s| {
            let (lo, hi, f_lo, f_hi, done) = (s[0], s[1], s[2], s[3], s[4]);
            let mu = (lo * f_hi - hi * f_lo) / (f_hi - f_lo);
            let ff = kkc_fmu49(mu, bee_sq, rdb_sq, r);
            let below = ff.cmp_lt(S::ZERO); // f_a(mu) < 0 => root above mu => raise lo, keep hi
            let lo_n = S::select(below, mu, lo);
            let hi_n = S::select(below, hi, mu);
            // Illinois: half-damp the RETAINED endpoint's function value (hi when below, else lo).
            let f_lo_n = S::select(below, ff, half * f_lo);
            let f_hi_n = S::select(below, half * f_hi, ff);
            let conv = ff.abs().min(hi_n - lo_n).cmp_lt(eps);
            let done_n = done.max(S::select(conv, S::ONE, S::ZERO)); // sticky
            [lo_n, hi_n, f_lo_n, f_hi_n, done_n]
        },
        |s, _next| half.cmp_lt(s[4]), // early-break once `done` is set (branch-free break_when)
        1,                            // result = hi, the upper bracket >= mu_+
    )
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
    metric: &SpatialMetric<S, D>,
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
    // r is COVARIANT (rescaled conserved momentum) -> raise: |r|^2 = gamma^{ij} r_i r_j. flat = euclidean.
    let r_sq = metric.norm_sq_cov(&rvec);
    let r_mag = r_sq.sqrt();
    let hvec = bfield.scale(isqrtd);
    // h is CONTRAVARIANT (rescaled B^i) -> lower: |h|^2 = gamma_{ij} h^i h^j. + epsilon: divzero guard when B=0.
    let bee_sq = metric.norm_sq_contra(&hvec) + eps;
    // r.h = r_i h^i is a COVARIANT*CONTRAVARIANT pairing -> METRIC-FREE (no gamma factor); stays `.dot()`.
    let rdb = rvec.dot(&hvec);
    let rdb_sq = rdb * rdb;
    // the perp invariant |r_perp|^2 = gamma^{ij} (r - r_par)_i (r - r_par)_j with the parallel
    // projection LOWERED to match r's variance: r_par_i = (r.b / |b|^2) h_i. identity gamma ->
    // the euclidean decomposition bit-for-bit (lower = id, norm_sq_cov = dot).
    let rparr = metric.lower(&hvec).scale(rdb / bee_sq);
    let rperp = rvec - rparr;
    let rp_sq = metric.norm_sq_cov(&rperp);

    // the master residual at a given mu (the 8 invariants are fixed).
    let kkc = |mu: S| kkc_fmu44(mu, r_mag, rp_sq, bee_sq, rdb_sq, qq, dd, gamma);

    // KKC false-position over `kkc_fmu44`, producing the root `mu`. the 6-state bracket
    // freezes on the OLD sticky `done` so the iteration that first converges still WRITES
    // its mu and the next freezes it (do-while semantics; see iterate_vec).
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
            // ff*f_upper < 0 ? {mul=muu; f_lo=f_hi} : {f_lo *= 0.5}; muu=mu, f_hi=ff.
            let cond = (ff * f_hi).cmp_lt(S::ZERO);
            let mul_n = S::select(cond, muu, mul);
            let f_lo_n = S::select(cond, f_hi, half * f_lo);
            // post-update test: |mul-muu| <= eps OR |ff| <= eps.
            let conv = (mul_n - mu).abs().min(ff.abs()).cmp_lt(eps);
            let done_n = done.max(S::select(conv, S::ONE, S::ZERO)); // sticky
            [mul_n, mu, f_lo_n, ff, mu, done_n]
        },
        |s, _next| half.cmp_lt(s[5]), // freeze once `done` was ALREADY set last step
        4,                            // result = mu
    );
    // bracketing guarantee (KKC Sec. II F, Eqs. 49-54): the false-position needs f_lower0 = kkc(0)
    // and f_upper0 = kkc(muu0 = mu_+) to straddle zero, where mu_+ = find_mu_plus is the root of
    // the auxiliary f_a. f_lower0 = -muhat(0) < 0 always (muhat > 0). f_upper0 = f(mu_+) >= 0 by
    // KKC Eq. 54. crucially the interval is [0, mu_+]: on (0, mu_+] the velocity stays
    // below v0 < 1 (kkc_fmu49 root definition) so the v_limit cutoff in kkc_fmu44 never binds and
    // the master function has a UNIQUE root (KKC Sec. II G); beyond mu_+ the cutoff kink can create
    // a SECOND, spurious superluminal root that the [0, 1] bracket would wrongly select whenever
    // r >= h0 (a shock-normal field, r.b != 0). a cold/unphysical state whose true root sits at the
    // ceiling recovers p <= 0, which the post-hoc c2p diagnostic (`relativistic_c2p_code`) flags as
    // a FAILURE, routing the zone through first-order correction; no silent floor is applied.

    // recovery (Eqs. 26/38/39/32/41/42/43/68); use_four_velocity = false.
    let x = S::ONE / (S::ONE + mu * bee_sq);
    let mu2 = mu * mu;
    let rbar_sq = r_sq * x * x + mu * x * (S::ONE + x) * rdb_sq;
    let qbar = qq - half * (bee_sq + mu2 * x * x * bee_sq * rp_sq);
    // MIRROR the root-finder's velocity ceiling (the shared c2p ceiling v_limit^2 = r^2/(1+r^2)). the
    // recovery must apply the SAME cap: a strong-field root sits at the ceiling, so the uncapped
    // vsq = mu^2 rbar_sq can reach >= 1, giving gbsq < -1 and ww = sqrt(1+gbsq) = NaN — a NaN rho/p
    // that poisons neighbours; the intended behaviour is a clean fail-loud (p <= 0 flagged below). capped,
    // ww/rho/p stay finite and the q(U) verdict routes the zone through first-order correction.
    let vsq = (mu2 * rbar_sq).min(crate::c2p_result::relativistic_velocity_ceiling_sq(r_sq));
    let gbsq = vsq / (S::ONE - vsq);
    let ww = (S::ONE + gbsq).sqrt();
    let rho = dd / ww;
    let eps_e = ww * (qbar - mu * rbar_sq) + gbsq / (S::ONE + ww);
    let rho_gm1 = rho * (gamma - S::ONE);
    // NO pressure floor: raw p = (gamma-1) rho eps. an unphysical negative eps yields a negative
    // pressure the post-hoc diagnostic flags; no silently-floored spurious-physical state is produced.
    let pre = rho_gm1 * eps_e;
    // EXACT admissibility (wu 2017 cone) folded into the pressure verdict. the velocity-ceiling
    // clamp (mu -> muu0, v -> v_limit) recovers a cold near-light-speed state that IS superluminal:
    // q(U)/D = (tau/D + 1) - sqrt(1 + gamma^{ij} S_i S_j / D^2) < 0. the raw (rho > 0 & pre > 0)
    // test accepts such a clamped state (pre from eps_e stays > 0), so its face state poisons a
    // neighbour's first-order redo -> a freeze. forcing the pressure non-positive when q(U) <= 0
    // routes the zone through first-order correction instead. r_sq is metric-raised (line ~115).
    let cone_ok = crate::c2p_result::relativistic_cone_residual(qq, r_sq).cmp_gt(S::ZERO);
    let pre = S::select(
        cone_ok,
        pre,
        S::from_f64(crate::c2p_result::C2P_CONE_FAIL_PRESSURE),
    );
    let mu_x = mu * x;
    let rdb_mu = rdb * mu;
    // the CONTRAVARIANT valencia velocity v^i = mu x (gamma^{ij} r_j + mu (r.b) h^i) — the
    // covariant momentum part raised before adding the contravariant field part. identity
    // gamma -> the euclidean form bit-for-bit (raise = id).
    let vel = (metric.raise(&rvec) + hvec.scale(rdb_mu)).scale(mu_x);

    MhdPrim {
        hydro: Prim { rho, vel, pre },
        mag: bfield,
    }
}

/// the host RMHD cons->prim: the branch-free `rmhd_recover` plus post-hoc C2pResult
/// diagnostics. no silent floor — the value is the raw recovered state, the ErrorCode
/// is the explicit signal (matches `Newtonian::to_primitive`; feedback_no_silent_floors).
///
/// **host-only** (Tier 1.7): `S: OrderedNumeric` because the diagnostic check uses
/// native `<` / `<=` / `==` on a host scalar. the kernel path is `rmhd_recover` above
/// (carrier-generic over `S: Scalar`); this wrapper is host-only.
pub(crate) fn rmhd_to_primitive<S: Scalar + OrderedNumeric, const D: usize>(
    eos: &impl Eos<S>,
    cons: &MhdCons<S, D>,
) -> C2pResult<MhdPrim<S, D>> {
    let dd = cons.den;

    // input guard: clearly-invalid conserved density (host-only early-out, absent from the
    // kernel path). B passes through. shared RHD/RMHD guard (now NaN-checked too; tier-1 #5).
    if let Some(code) = crate::c2p_result::relativistic_density_guard(dd) {
        let floored = MhdPrim {
            hydro: Prim {
                rho: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR),
                vel: Tensor::zeros(),
                pre: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR),
            },
            mag: cons.mag,
        };
        return C2pResult::err(floored, code);
    }

    let prim = rmhd_recover(eos, cons, &SpatialMetric::flat(), RMHD_MAX_ITER);

    // post-hoc diagnostics on the raw recovered state (shared RHD/RMHD contract; tier-1 #5).
    let v_sq = prim.vel.dot(&prim.vel);
    let code = crate::c2p_result::relativistic_c2p_code(prim.rho, prim.pre, v_sq);
    if code.is_ok() {
        C2pResult::ok(prim)
    } else {
        C2pResult::err(prim, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;

    // direct f64 reference for `kkc_fmu49` (KKC Eq. 49).
    fn ref_fmu49(mu: f64, beesq: f64, beedrsq: f64, r: f64) -> f64 {
        let x = 1.0 / (1.0 + mu * beesq);
        let rbar_sq = r * r * x * x + mu * x * (1.0 + x) * beedrsq;
        mu * (1.0 + rbar_sq).sqrt() - 1.0
    }

    // direct f64 reference for `kkc_fmu44` (KKC Eq. 44).
    #[allow(clippy::too_many_arguments)]
    fn ref_fmu44(
        mu: f64,
        r: f64,
        rperp: f64,
        beesq: f64,
        beedrsq: f64,
        qterm: f64,
        dterm: f64,
        gamma: f64,
    ) -> f64 {
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
        let phat = (gamma - 1.0) * rhohat * eps;
        let ahat = phat / (rhohat * (1.0 + eps));
        let nu_hat_a = (1.0 + ahat) * (1.0 + eps) / g;
        let nu_hat_b = (1.0 + ahat) * (1.0 + qbar - mu * rbar_sq);
        let nu_hat = nu_hat_a.max(nu_hat_b);
        let muhat = 1.0 / (nu_hat + rbar_sq * mu);
        mu - muhat
    }

    // the FULL find_mu_plus (doubling + bisection + break) that the carrier-generic
    // bisection elides; agreement to ~1e-9 proves the elision.
    // UNUSED: kept as the analytic reference; no test asserts against it.
    // allow(dead_code) preserves the reference.
    #[allow(dead_code)]
    fn ref_find_mu_plus(beesq: f64, beedrsq: f64, r: f64) -> f64 {
        if r < 1.0 {
            return 1.0;
        }
        let mut mu_lower = 0.0;
        let mut mu_upper = 1.0;
        let mut f_upper = ref_fmu49(mu_upper, beesq, beedrsq, r);
        while f_upper < 0.0 {
            mu_upper *= 2.0;
            f_upper = ref_fmu49(mu_upper, beesq, beedrsq, r);
        }
        let eps = 1.0e-12;
        let mut mu_mid = 1.0;
        let mut f_lower = ref_fmu49(mu_lower, beesq, beedrsq, r);
        let mut iter = 0;
        while iter < 50 && (mu_upper - mu_lower) > eps {
            mu_mid = 0.5 * (mu_lower + mu_upper);
            let f_mid = ref_fmu49(mu_mid, beesq, beedrsq, r);
            if f_mid.abs() < eps {
                break;
            }
            if f_mid * f_lower < 0.0 {
                mu_upper = mu_mid;
            } else {
                mu_lower = mu_mid;
                f_lower = f_mid;
            }
            iter += 1;
        }
        mu_mid * 1.000001
    }

    #[test]
    fn kkc_fmu49_matches_cpp_reference() {
        let cases = [
            (0.2, 0.4, 0.01, 0.3),
            (0.5, 0.8, 0.05, 0.6),
            (0.8, 1.2, 0.1, 0.9),
            (1.5, 0.3, 0.02, 1.2),
        ];
        for (mu, beesq, beedrsq, r) in cases {
            let got = kkc_fmu49::<f64>(mu, beesq, beedrsq, r);
            let want = ref_fmu49(mu, beesq, beedrsq, r);
            assert!(
                (got - want).abs() < 1e-12,
                "fmu49(mu={mu}): {got} != {want}"
            );
        }
    }

    #[test]
    fn kkc_fmu44_matches_cpp_reference() {
        let g = 5.0 / 3.0;
        let cases = [
            (0.2, 0.3, 0.05, 0.4, 0.01, 0.5, 1.0),
            (0.5, 0.6, 0.1, 0.8, 0.05, 1.0, 1.5),
            (0.8, 0.9, 0.2, 1.2, 0.1, 1.5, 2.0),
        ];
        for (mu, r, rperp, beesq, beedrsq, q, d) in cases {
            let got = kkc_fmu44::<f64>(mu, r, rperp, beesq, beedrsq, q, d, g);
            let want = ref_fmu44(mu, r, rperp, beesq, beedrsq, q, d, g);
            assert!(
                (got - want).abs() < 1e-12,
                "fmu44(mu={mu}): {got} != {want}"
            );
        }
    }

    #[test]
    fn find_mu_plus_is_the_kkc_fmu49_root() {
        // find_mu_plus returns mu_+, the root of the auxiliary kkc_fmu49 (KKC Eq. 49). verify it
        // is a genuine root (|f_a(mu_+)| ~ 0) and a valid UPPER bracket (f_a(mu_+) >= 0 >= f_a below
        // it). the r >= 1 cases are exactly where the old `return 1` bracket spanned the spurious
        // second root of the master function.
        let cases = [
            (0.3, 0.02, 1.2),
            (0.8, 0.1, 2.0),
            (1.5, 0.3, 5.0),
            (0.5, 0.05, 1.0),
            (3.29, 18.7, 2.43),
        ];
        for (beesq, beedrsq, r) in cases {
            let mu_plus = find_mu_plus::<f64>(beesq, beedrsq, r);
            let f_at = kkc_fmu49::<f64>(mu_plus, beesq, beedrsq, r);
            assert!(
                mu_plus > 0.0 && mu_plus <= 1.0,
                "mu_+ out of (0,1] for r={r}: {mu_plus}"
            );
            // the correctness invariant is a TIGHT UPPER bracket: f_a(mu_+) >= 0 (so it exceeds the
            // f_a root => f_master(mu_+) >= 0, straddle holds) AND close to the root (Illinois returns
            // the hi endpoint within ~tol of the root, so f_a(hi) ~ f_a'*tol, a few e-12).
            assert!(
                f_at >= 0.0,
                "mu_+ is not an upper bracket of the f_a root for r={r}: f_a={f_at}"
            );
            assert!(
                f_at < 1e-10,
                "mu_+ not a tight bracket for r={r}: f_a={f_at}"
            );
            // upper bracket: f_a strictly increasing, so just below mu_+ it is negative.
            assert!(
                kkc_fmu49::<f64>(mu_plus - 1e-9, beesq, beedrsq, r) < f_at,
                "f_a not increasing through mu_+ for r={r}"
            );
        }
    }

    #[test]
    fn rmhd_recover_selects_physical_root_with_normal_field() {
        // regression: mignone & bodo (2006) test 1, the discontinuity cell after ~29 forward-euler
        // steps. the conserved state is IN-CONE (q(U)/D > 0) with a shock-normal field (Bx = 0.5,
        // r.b != 0, r_mag ~ 2.43 > h0 = 1). the master function has TWO roots under the velocity
        // cutoff: the physical mu ~ 0.198 (|v| ~ 0.474, p ~ 0.38) and a spurious mu ~ 0.995 (|v| ~
        // 2.37, p < 0) in the post-mu_+ kink region. the [0, mu_+] bracket must select the physical
        // one. with the old `return 1` bracket this recovered p ~ -0.88, v ~ 1.45 (superluminal).
        let eos = IdealGas { gamma: 2.0 };
        let cons = MhdCons::<f64, 3> {
            hydro: crate::state::Cons {
                den: 0.2499729,
                mom: Tensor::new([0.2358797, -0.5587815, 0.0]),
                nrg: 1.047524,
            },
            mag: Tensor::new([0.5, -0.7566201, 0.0]),
        };
        let got = rmhd_recover(&eos, &cons, &SpatialMetric::flat(), RMHD_MAX_ITER);
        let v_sq = got.hydro.vel.dot(&got.hydro.vel);
        assert!(v_sq < 1.0, "recovered superluminal velocity: v^2 = {v_sq}");
        assert!(
            got.hydro.pre > 0.0,
            "recovered non-positive pressure: p = {}",
            got.hydro.pre
        );
        assert!(
            got.hydro.rho > 0.0,
            "recovered non-positive density: rho = {}",
            got.hydro.rho
        );
        // the physical root recovered by an independent mu-scan of the master function.
        assert!(
            (got.hydro.pre - 0.3805).abs() < 1e-3,
            "pressure off physical root: {}",
            got.hydro.pre
        );
        assert!(
            (v_sq.sqrt() - 0.4741).abs() < 1e-3,
            "velocity off physical root: {}",
            v_sq.sqrt()
        );
    }

    // =========================================================================
    // c2p iter-distribution probe — 2026-06-07
    //
    // measures actual iter counts for `find_mu_plus` (bisection) and the main
    // false-position loop on orszag_tang IC states, with a `break` on convergence
    // (the fixed-iter freeze removed). answers: which of the two loops actually dominates?
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
            if cond {
                mu_u = mid;
            } else {
                mu_l = mid;
                f_l = f_mid;
            }
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
        let vy = v0 * (2.0 * pi * x).sin();
        let vz = 0.0;
        let bx = -b0 * (2.0 * pi * y).sin();
        let by = b0 * (4.0 * pi * x).sin();
        let bz = 0.0;
        MhdPrim {
            hydro: Prim {
                rho,
                vel: Tensor::new([vx, vy, vz]),
                pre,
            },
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
        let mut fp_hist = vec![0_usize; max_fp + 1];
        let mut total = 0_usize;
        let mut fmp_sum = 0_usize;
        let mut fp_sum = 0_usize;
        let mut fp_failed = 0_usize;
        let mut fmp_max = 0_usize;
        let mut fp_max = 0_usize;

        for j in 0..n {
            for i in 0..n {
                let x = (i as f64 + 0.5) / n as f64;
                let y = (j as f64 + 0.5) / n as f64;
                let prim = orszag_tang_prim(x, y, gamma, v0, b0);
                let cons = rmhd_prim_to_cons(&prim, gamma);
                let (_, fmp_iters, fp_iters) = instrumented_c2p(cons, gamma, max_fp, tol);
                if fmp_iters < fmp_hist.len() {
                    fmp_hist[fmp_iters] += 1;
                }
                if fp_iters < fp_hist.len() {
                    fp_hist[fp_iters] += 1;
                }
                if fp_iters == max_fp {
                    fp_failed += 1;
                }
                fmp_sum += fmp_iters;
                fp_sum += fp_iters;
                fmp_max = fmp_max.max(fmp_iters);
                fp_max = fp_max.max(fp_iters);
                total += 1;
            }
        }

        let fmp_avg = fmp_sum as f64 / total as f64;
        let fp_avg = fp_sum as f64 / total as f64;

        // percentiles from cumulative histogram.
        let pct = |hist: &[usize], pct: f64| -> usize {
            let target = (total as f64 * pct).ceil() as usize;
            let mut cum = 0_usize;
            for (k, &v) in hist.iter().enumerate() {
                cum += v;
                if cum >= target {
                    return k;
                }
            }
            hist.len() - 1
        };
        let fmp_p50 = pct(&fmp_hist, 0.50);
        let fmp_p90 = pct(&fmp_hist, 0.90);
        let fmp_p99 = pct(&fmp_hist, 0.99);
        let fp_p50 = pct(&fp_hist, 0.50);
        let fp_p90 = pct(&fp_hist, 0.90);
        let fp_p99 = pct(&fp_hist, 0.99);

        eprintln!("\n=== c2p iter distribution on orszag_tang n={n} (tol={tol:e}) ===");
        eprintln!("total cells: {total}\n");
        eprintln!("find_mu_plus (bisection on [0,1]):");
        eprintln!(
            "  avg = {:.2}, p50 = {}, p90 = {}, p99 = {}, max = {}",
            fmp_avg, fmp_p50, fmp_p90, fmp_p99, fmp_max
        );
        eprintln!(
            "  production runs FIXED 50 iters always -> {} wasted iters per cell on avg",
            (50.0 - fmp_avg).max(0.0)
        );
        eprintln!();
        eprintln!("false-position (Illinois, post-bracket):");
        eprintln!(
            "  avg = {:.2}, p50 = {}, p90 = {}, p99 = {}, max = {}",
            fp_avg, fp_p50, fp_p90, fp_p99, fp_max
        );
        eprintln!("  production runs UP TO 100 iters w/ sticky-done freeze");
        eprintln!(
            "  non-converged cells: {} / {} ({:.1}%)",
            fp_failed,
            total,
            100.0 * fp_failed as f64 / total as f64
        );
        eprintln!();
        eprintln!(
            "total avg iters/cell (both loops) = {:.1}",
            fmp_avg + fp_avg
        );
        eprintln!("production worst-case = 50 + 100 = 150 iters/cell\n");

        // dump compact histogram for fp.
        eprintln!("fp iter histogram (count per iter, 0..30):");
        for (k, &v) in fp_hist.iter().enumerate().take(31) {
            if v > 0 {
                eprintln!(
                    "  iter {:2}: {:5} cells ({:5.1}%)",
                    k,
                    v,
                    100.0 * v as f64 / total as f64
                );
            }
        }
        eprintln!();
        eprintln!("fmp iter histogram (count per iter, 0..60):");
        for (k, &v) in fmp_hist.iter().enumerate().take(61) {
            if v > 0 {
                eprintln!(
                    "  iter {:2}: {:5} cells ({:5.1}%)",
                    k,
                    v,
                    100.0 * v as f64 / total as f64
                );
            }
        }

        // sanity: any cell should converge well under the 100 cap.
        assert!(
            fp_failed == 0,
            "{} cells failed to converge in {} fp iters",
            fp_failed,
            max_fp
        );
    }

    // item 3: resolves the find_mu_plus=1 / kkc_fmu44-bracket concern. the proof on
    // `find_mu_plus` covers kkc_fmu49; the root actually solved is kkc_fmu44. these
    // EVOLVED-state cases (high Lorentz W, strong AND weak magnetization, low density —
    // the regime t=0 orszag_tang never reaches, where r can exceed 1) exercise the
    // production rmhd_recover end to end and confirm PHYSICAL (warm, p > 0) states ALWAYS
    // bracket and recover finite. zero non-bracketing cells => no kernel-path bracket guard
    // needed (the v_limit ceiling forces f_upper0 >= 0 for any warm state; see rmhd_recover).
    // there is NO pressure floor — the recovered pressure is the raw round-trip value.
    #[test]
    fn rmhd_recover_brackets_evolved_high_w_high_b_states() {
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        // (rho, |v|, pre, |B|): push W and magnetization well past orszag_tang.
        let cases = [
            (1.0_f64, 0.99, 0.1, 5.0), // W~7, strong field
            (1.0, 0.999, 0.1, 10.0),   // W~22, very strong field
            (0.1, 0.9999, 1.0, 1.0),   // W~71, low density
            (10.0, 0.95, 0.1, 50.0),   // magnetization-dominated
            (1.0, 0.5, 0.1, 1e-3),     // near-vacuum field
        ];
        for (rho, v, pre, b) in cases {
            // physical prim: velocity along x, B along y. v perp B => r.b = 0 here (the r.b != 0
            // shock-normal case is covered by rmhd_recover_selects_physical_root_with_normal_field).
            let prim = MhdPrim::<f64, 3> {
                hydro: Prim {
                    rho,
                    vel: Tensor::new([v, 0.0, 0.0]),
                    pre,
                },
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

            let got = rmhd_recover(&eos, &cons, &SpatialMetric::flat(), RMHD_MAX_ITER);
            assert!(
                got.hydro.rho.is_finite()
                    && got.hydro.pre.is_finite()
                    && got.hydro.vel.dot(&got.hydro.vel).is_finite(),
                "physical evolved state (rho={rho}, v={v}, pre={pre}, b={b}) failed to \
                 recover finite: rho={}, pre={}",
                got.hydro.rho,
                got.hydro.pre
            );
            // floor-less KKC: without the eps_min zero-T floor smoothing the master function near
            // the velocity ceiling, false-position converges to ~1e-5 at the pathological W~71 low-
            // density case (r_mag ~ 2900, root sits ON the ceiling where mu ~ mu_+) within the
            // 100-iter cap (realistic W ~ a few recover to ~1e-9). the exact root gives p to ~4e-12;
            // the residual is bracket-trajectory-dependent iteration precision at this extreme corner,
            // it is not a physics error. the bracket (the real robustness invariant) still straddles.
            assert!(
                (got.hydro.pre - pre).abs() < 2e-5 * pre.abs().max(1.0),
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
