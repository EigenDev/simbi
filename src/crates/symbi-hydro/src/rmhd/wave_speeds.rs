// =============================================================================
// rmhd/wave_speeds.rs
//
// the RMHD characteristic wave speeds: the full Mignone & Del Zanna magnetosonic
// quartic (Eq. 56) with fast paths for vsq~0 (Eq. 57) and bn~0 (Eq. 58), plus the
// polynomial solvers it needs (resolvent cubic + quartic min/max root). this is
// the SINGLE source both the RMHD flux's HLLE and the CFL map consume.
// GPU-traceable: all paths computed unconditionally, selected via S::select.
// =============================================================================

use crate::eos::Eos;
use crate::mhd_state::MhdPrim;
use crate::rhd;
use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// RMHD wave speeds via the full quartic dispersion relation (Mignone & Del Zanna
/// Eq. 56), with fast paths for vsq~0 (Eq. 57) and bn~0 (Eq. 58). returns
/// (sl, sr) — left and right signal speeds along `nhat`.
pub(crate) fn rmhd_wave_speeds<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
) -> (S, S) {
    let rho = prim.rho;
    let hh = rhd::enthalpy(eos, rho, prim.pre);
    let vsq = prim.vel.dot(&prim.vel);
    let vn = prim.vel.dot(nhat);
    let ww = rhd::lorentz_factor(vsq);
    let w2 = ww * ww;
    let cssq = eos.sound_speed(rho, prim.pre) * eos.sound_speed(rho, prim.pre) / hh;

    let bsq = prim.mag.dot(&prim.mag);
    let vdb = prim.vel.dot(&prim.mag);
    let bn = prim.mag.dot(nhat);

    let bmu0 = ww * vdb;
    let bmu_sq = bsq / w2 + vdb * vdb;
    let bmun = bn / ww + ww * vn * vdb;

    let eps = S::from_f64(1e-14);

    // the (sl, sr) path selection as a nested `cond_vec<2>` (the lazy-branch
    // dual of iterate, vector form): exactly ONE path's intermediates are
    // computed at render time. CRUCIALLY the full quartic (Eq. 56 — the
    // resolvent cubic + ~10 transcendentals via `solve_quartic_minmax`) lives
    // in the innermost else arm, so it is SKIPPED entirely when `vsq ~ 0`
    // (Eq. 57) or `bn ~ 0` (Eq. 58) — the `wave_speeds` early-`return` chain
    // skips compute-all-paths via `S::select`. the
    // cheap shared prefix above (rho, hh, w2, cssq, bmu*) stays unconditional.
    let cond_vsq = vsq.cmp_lt(eps);
    let magnetic_scale = bsq + rho * hh;
    let cond_bn = (bn * bn).cmp_lt(eps * magnetic_scale);

    let [sl, sr] = S::cond_vec(
        cond_vsq,
        // path 1: vsq ~ 0 (Eq. 57) — quadratic in lambda^2.
        || {
            let fac = S::ONE / (rho * hh + bmu_sq);
            let bb_1 = S::ZERO - (bmu_sq + rho * hh * cssq + bn * bn * cssq) * fac;
            let cc_1 = cssq * bn * bn * fac;
            let disq_1 = (bb_1 * bb_1 - S::from_f64(4.0) * cc_1).safe_sqrt();
            let lambda_r_1 = (S::from_f64(0.5) * (S::ZERO - bb_1 + disq_1)).safe_sqrt();
            [S::ZERO - lambda_r_1, lambda_r_1]
        },
        || {
            S::cond_vec(
                cond_bn,
                // path 2: bn ~ 0 (Eq. 58) — quadratic in lambda.
                || {
                    let vdbperp = vdb - vn * bn;
                    let qq = bmu_sq - cssq * vdbperp * vdbperp;
                    let a2_p2 = rho * hh * (cssq + w2 * (S::ONE - cssq)) + qq;
                    let a1_p2 = S::ZERO - S::from_f64(2.0) * rho * hh * w2 * vn * (S::ONE - cssq);
                    let a0_p2 = rho * hh * (S::ZERO - cssq + w2 * vn * vn * (S::ONE - cssq)) - qq;
                    let disq_2 = (a1_p2 * a1_p2 - S::from_f64(4.0) * a2_p2 * a0_p2).max(S::ZERO);
                    let sr_2 = S::from_f64(0.5) * (S::ZERO - a1_p2 + disq_2.sqrt()) / a2_p2;
                    let sl_2 = S::from_f64(0.5) * (S::ZERO - a1_p2 - disq_2.sqrt()) / a2_p2;
                    [sl_2, sr_2]
                },
                // path 3: the FULL quartic (Eq. 56) — computed ONLY when both fast
                // paths fail. this is the ~750-op + ~10-transcendental body.
                || {
                    let vn2 = vn * vn;
                    let a4 = S::ZERO - bmu0 * bmu0 * cssq + bmu_sq * w2 - cssq * w2 * w2 * hh * rho
                        + cssq * w2 * hh * rho
                        + w2 * w2 * hh * rho;
                    let inv_a4 = S::ONE / a4;

                    let a3 = inv_a4
                        * (S::from_f64(2.0) * bmu0 * bmun * cssq
                            - S::from_f64(2.0) * bmu_sq * w2 * vn
                            + S::from_f64(4.0) * cssq * w2 * w2 * hh * rho * vn
                            - S::from_f64(2.0) * cssq * w2 * hh * rho * vn
                            - S::from_f64(4.0) * w2 * w2 * hh * rho * vn);

                    let a2_q = inv_a4
                        * (bmu0 * bmu0 * cssq + bmu_sq * w2 * vn2
                            - bmu_sq * w2
                            - bmun * bmun * cssq
                            - S::from_f64(6.0) * cssq * w2 * w2 * hh * rho * vn2
                            + cssq * w2 * hh * rho * vn2
                            - cssq * w2 * hh * rho
                            + S::from_f64(6.0) * w2 * w2 * hh * rho * vn2);

                    let a1_q = inv_a4
                        * (S::ZERO - S::from_f64(2.0) * bmu0 * bmun * cssq
                            + S::from_f64(2.0) * bmu_sq * w2 * vn
                            + S::from_f64(4.0) * cssq * w2 * w2 * hh * rho * vn * vn2
                            + S::from_f64(2.0) * cssq * w2 * hh * rho * vn
                            - S::from_f64(4.0) * w2 * w2 * hh * rho * vn * vn2);

                    let a0_q = inv_a4
                        * (S::ZERO - bmu_sq * w2 * vn2 + bmun * bmun * cssq
                            - cssq * w2 * w2 * hh * rho * vn2 * vn2
                            - cssq * w2 * hh * rho * vn2
                            + w2 * w2 * hh * rho * vn2 * vn2);

                    let (sl_3, sr_3) = solve_quartic_minmax(a3, a2_q, a1_q, a0_q);
                    [sl_3, sr_3]
                },
            )
        },
    );

    (sl, sr)
}

/// the magnetosonic UPPER BOUND on the RMHD fast wave speed — for the CFL TIMESTEP ONLY.
///
/// `c_f^2 = c_s^2 + c_A^2 - c_s^2 c_A^2` (textbook no-rotation magnetosonic bound), a STRICT
/// upper bound on the Mignone & Del Zanna quartic fast speed (`rmhd_wave_speeds`). the CFL
/// needs only a stable UPPER bound — this is ~25x cheaper
/// (~30 ops + 1 sqrt vs ~750 ops + ~10 transcendentals, all of which trace into the kernel
/// because `S::select` evaluates every arm). it stays CFL-safe because it never UNDER-
/// estimates the true signal speed. do NOT route the Riemann/flux path (`extremal_speeds`)
/// here — HLLE diffusion needs the tight quartic.
///
/// returns `(sl, sr)` along `nhat` via the SR relativistic velocity-addition. NaN-PRESERVING:
/// `cf_sq` uses the product form `1 - (1-cs^2)(1-cA^2)` (manifestly < 1 for physical inputs,
/// so `denom = 1 - vsq*cf_sq > 0` with NO clamp) and the outputs are UNCLAMPED — a NaN prim
/// from an unphysical c2p propagates straight to the CFL max-reduction + dt guard, where it
/// halts the run; a light-cone `.min`/`.max` would mask it by dropping the NaN.
/// over-estimating a speed only shrinks dt, which is safe.
pub fn rmhd_magnetosonic_cfl_speeds<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
) -> (S, S) {
    let rho = prim.rho;
    let hh = rhd::enthalpy(eos, rho, prim.pre);
    let cs = eos.sound_speed(rho, prim.pre);
    let cssq = cs * cs / hh; // relativistic sound speed squared (matches rmhd_wave_speeds)

    let vsq = prim.vel.dot(&prim.vel);
    let vn = prim.vel.dot(nhat);
    let w2 = rhd::lorentz_factor_sq(vsq);

    let bsq = prim.mag.dot(&prim.mag);
    let vdb = prim.vel.dot(&prim.mag);
    let b_mu_sq = bsq / w2 + vdb * vdb; // co-moving |B|^2 (matches rmhd_wave_speeds)
    let rho_h = rho * hh;
    let va_sq = b_mu_sq / (rho_h + b_mu_sq); // alfven speed squared, in [0, 1)

    // magnetosonic c_f^2 via the PRODUCT form: 1 - (1-cs^2)(1-cA^2). strictly < 1 for
    // physical cs^2, cA^2 in [0,1) (so denom below stays positive without a clamp), and it
    // preserves NaN (1 - (1-NaN)(.) = NaN) so an unphysical prim surfaces unmasked.
    let cf_sq = S::ONE - (S::ONE - cssq) * (S::ONE - va_sq);

    // SR relativistic addition of the normal flow velocity and the magnetosonic speed.
    let one_m_vsq = S::ONE - vsq;
    let denom = S::ONE / (S::ONE - vsq * cf_sq);
    // clamp the discriminant before the sqrt (Gv traces both arms); cf_sq's NaN still
    // reaches the outputs via the (1 - cf_sq) and denom terms, so this does not mask it.
    let disc = (one_m_vsq * cf_sq).safe_sqrt();
    let sl = (vn * (S::ONE - cf_sq) - disc) * denom;
    let sr = (vn * (S::ONE - cf_sq) + disc) * denom;
    (sl, sr)
}

/// the curved-background magnetosonic UPPER BOUND — for the CFL TIMESTEP ONLY.
///
/// the SAME product-form bound as `rmhd_magnetosonic_cfl_speeds`, with the euclidean
/// contractions replaced by the spatial metric and the coordinate-frame factors of the
/// Banyuls-Font transform applied: `gamma^{nn}` scales the discriminant and `alpha` the
/// whole fan (the shift is the caller's coordinate correction). the discriminant
/// `gamma^{nn} (1 - v^2) c_f^2` bounds the exact Banyuls-Font radical
/// `(1 - v^2)(gamma^{nn}(1 - v^2 c_f^2) - vn^2(1 - c_f^2))` because
/// `gamma^{nn} >= gamma^{nn}(1 - v^2 c_f^2) - vn^2(1 - c_f^2)` for physical inputs, so the
/// fan never under-estimates a signal speed and stays CFL-safe. at `alpha = 1`,
/// `gamma = delta` every extra factor is an exact 1.0 multiply, so the bound equals the
/// flat `rmhd_magnetosonic_cfl_speeds` — minkowski-limit runs take the flat timestep.
/// do NOT route the Riemann/flux path here — HLLE diffusion needs the tight quartic.
/// nan-preserving for the same reasons as the flat form: no clamp, no light-cone cap.
pub fn rmhd_magnetosonic_cfl_speeds_gr<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim: &MhdPrim<S, D>,
    nhat: &Tensor<S, D>,
    metric: &crate::spatial_metric::SpatialMetric<S, D>,
    alpha: S,
) -> (S, S) {
    let rho = prim.rho;
    let hh = rhd::enthalpy(eos, rho, prim.pre);
    let cs = eos.sound_speed(rho, prim.pre);
    let cssq = cs * cs / hh;

    let vsq = metric.norm_sq_contra(&prim.vel);
    let vn = prim.vel.dot(nhat);
    let w2 = rhd::lorentz_factor_sq(vsq);

    let bsq = metric.norm_sq_contra(&prim.mag);
    let vdb = metric.contract_contra(&prim.vel, &prim.mag);
    let b_mu_sq = bsq / w2 + vdb * vdb;
    let rho_h = rho * hh;
    let va_sq = b_mu_sq / (rho_h + b_mu_sq);

    let cf_sq = S::ONE - (S::ONE - cssq) * (S::ONE - va_sq);

    let gamma_nn = metric.norm_sq_cov(nhat);
    let one_m_vsq = S::ONE - vsq;
    let denom = S::ONE / (S::ONE - vsq * cf_sq);
    let disc = (gamma_nn * one_m_vsq * cf_sq).safe_sqrt();
    let sl = alpha * (vn * (S::ONE - cf_sq) - disc) * denom;
    let sr = alpha * (vn * (S::ONE - cf_sq) + disc) * denom;
    (sl, sr)
}

/// solve quartic x^4 + bx^3 + cx^2 + dx + e = 0 and return (min_root, max_root).
/// uses resolvent cubic method. for RMHD wave speed computation.
/// GPU-traceable: all root pairs computed unconditionally, invalid roots
/// masked via sentinel values so they don't affect min/max.
fn solve_quartic_minmax<S: Scalar>(b: S, c: S, d: S, e: S) -> (S, S) {
    let p = c - S::from_f64(0.375) * b * b;
    let q = S::from_f64(0.125) * b * b * b - S::from_f64(0.5) * b * c + d;
    let m = solve_cubic_resolvent(
        p,
        S::from_f64(0.25) * p * p + S::from_f64(0.01171875) * b * b * b * b - e
            + S::from_f64(0.25) * b * d
            - S::from_f64(0.0625) * b * b * c,
        S::from_f64(-0.125) * q * q,
    );

    let m_valid = m.cmp_ge(S::ZERO);
    let safe_m = m.max(S::ZERO);
    let sqrt_2m = (S::from_f64(2.0) * safe_m).sqrt();
    let eps = S::from_f64(1e-14);
    let qb4 = S::from_f64(-0.25) * b;
    let half = S::from_f64(0.5);
    let sent_hi = S::from_f64(1e30);
    let sent_lo = S::from_f64(-1e30);

    // safe divisor for q / sqrt_2m (only used when q != 0)
    let safe_sqrt = S::select(sqrt_2m.cmp_gt(S::ZERO), sqrt_2m, S::ONE);
    let q_over_s = q / safe_sqrt;

    // ---- q ~ 0 path: roots from degenerate biquadratic ----
    let disc_q0 = S::ZERO - safe_m - p;
    let delta_q0 = (S::from_f64(2.0) * disc_q0.max(S::ZERO)).sqrt();
    let r0_z = qb4 + half * (sqrt_2m - delta_q0);
    let r1_z = qb4 - half * (sqrt_2m - delta_q0);
    let r2_z = qb4 + half * (sqrt_2m + delta_q0);
    let r3_z = qb4 - half * (sqrt_2m + delta_q0);
    // valid if discriminant >= -eps (covers both 4-root and degenerate 2-root cases)
    let q0_valid = disc_q0.cmp_ge(S::ZERO - eps);
    let smin_q0 = S::select(q0_valid, r0_z.min(r1_z).min(r2_z).min(r3_z), S::ZERO);
    let smax_q0 = S::select(q0_valid, r0_z.max(r1_z).max(r2_z).max(r3_z), S::ZERO);

    // ---- q != 0 path: two independent root pairs ----
    let disc1 = S::ZERO - safe_m - p + q_over_s;
    let disc2 = S::ZERO - safe_m - p - q_over_s;
    let d1_valid = disc1.cmp_ge(S::ZERO);
    let d2_valid = disc2.cmp_ge(S::ZERO);

    // root pair 1 (from disc1)
    let delta1 = (S::from_f64(2.0) * disc1.max(S::ZERO)).sqrt();
    let r0_nz = half * (S::ZERO - sqrt_2m + delta1) + qb4;
    let r1_nz = half * (S::ZERO - sqrt_2m - delta1) + qb4;

    // root pair 2 (from disc2)
    let delta2 = (S::from_f64(2.0) * disc2.max(S::ZERO)).sqrt();
    let r2_nz = half * (sqrt_2m + delta2) + qb4;
    let r3_nz = half * (sqrt_2m - delta2) + qb4;

    // mask invalid roots with sentinels so they don't affect min/max
    let r0_lo = S::select(d1_valid, r0_nz, sent_hi);
    let r1_lo = S::select(d1_valid, r1_nz, sent_hi);
    let r2_lo = S::select(d2_valid, r2_nz, sent_hi);
    let r3_lo = S::select(d2_valid, r3_nz, sent_hi);
    let r0_hi = S::select(d1_valid, r0_nz, sent_lo);
    let r1_hi = S::select(d1_valid, r1_nz, sent_lo);
    let r2_hi = S::select(d2_valid, r2_nz, sent_lo);
    let r3_hi = S::select(d2_valid, r3_nz, sent_lo);

    // OR of validity (d1_valid || d2_valid), expressed via select so it stays traceable:
    // `cmp_*` returns a Bool node under the tracing carrier (Gv), so arithmetic on it
    // (`1-(1-a)(1-b)`) is a type error — select keeps it a clean 0/1 value.
    let any_valid = S::select(d1_valid, S::ONE, S::select(d2_valid, S::ONE, S::ZERO));
    let has_roots = any_valid.cmp_gt(S::from_f64(0.5));
    let smin_nz = S::select(has_roots, r0_lo.min(r1_lo).min(r2_lo).min(r3_lo), S::ZERO);
    let smax_nz = S::select(has_roots, r0_hi.max(r1_hi).max(r2_hi).max(r3_hi), S::ZERO);

    // outer select: q ~ 0 or q != 0
    let q_near_zero = q.abs().cmp_lt(eps);
    let smin = S::select(q_near_zero, smin_q0, smin_nz);
    let smax = S::select(q_near_zero, smax_q0, smax_nz);

    // guard against m < 0 (no real roots)
    (
        S::select(m_valid, smin, S::ZERO),
        S::select(m_valid, smax, S::ZERO),
    )
}

/// solve resolvent cubic x^3 + bx^2 + cx + d = 0 for one real root.
/// GPU-traceable via `S::cond` (the DUAL of iterate): ONLY the taken case's
/// transcendental pair is evaluated — the carrier-portable form of the
/// `solve_cubic` early-`return`. cheap, symmetric
/// sub-choices (the cube-root sign) stay branch-free `S::select`. clamps are
/// kept for bit-equivalence with the host f64 path (they are identity when the
/// owning arm is taken).
fn solve_cubic_resolvent<S: Scalar>(b: S, c: S, d: S) -> S {
    let p = c - b * b / S::from_f64(3.0);
    let q = S::from_f64(2.0) * b * b * b / S::from_f64(27.0) - b * c / S::from_f64(3.0) + d;
    let eps = S::from_f64(1e-14);
    let third = S::ONE / S::from_f64(3.0);
    let b3 = b * third;

    // outer case split: p ~ 0 -> q ~ 0 -> general (each tier's expensive work
    // is inside its `cond` closure, so only the chosen tier computes it).
    S::cond(
        p.abs().cmp_lt(eps),
        // case 1: p ~ 0 -> cube-root formula (the `powf` lives ONLY here).
        || {
            let aq = q.abs();
            let cq = aq.powf(third);
            let scq = S::select(q.cmp_gt(S::ZERO), cq, S::ZERO - cq); // cheap sign flip
            S::ZERO - scq - b3
        },
        || {
            S::cond(
                q.abs().cmp_lt(eps),
                // case 2: q ~ 0 -> trivial root.
                || S::ZERO - b3,
                // general case: the shared sqrt/divide setup, then the four
                // mutually-exclusive transcendental cases — each evaluated only
                // when its branch is taken.
                || {
                    let safe_p = S::select(p.abs().cmp_gt(eps), p, S::ONE);
                    let t = (p.abs() / S::from_f64(3.0)).sqrt();
                    let g = S::from_f64(1.5) * q / (safe_p * t);
                    let disc = S::from_f64(4.0) * p * p * p + S::from_f64(27.0) * q * q;
                    S::cond(
                        p.cmp_gt(S::ZERO),
                        // case 3: p > 0 -> sinh formula.
                        || S::ZERO - S::from_f64(2.0) * t * (g.asinh() * third).sinh() - b3,
                        || {
                            S::cond(
                                disc.cmp_lt(S::ZERO),
                                // case 4: three real roots -> cos formula (g in [-1,1]).
                                || {
                                    let g_clamp = g.max(-S::ONE).min(S::ONE);
                                    S::from_f64(2.0) * t * (g_clamp.acos() * third).cos() - b3
                                },
                                || {
                                    S::cond(
                                        q.cmp_gt(S::ZERO),
                                        // case 5: q > 0 -> cosh formula (negative sign).
                                        || {
                                            let ng_clamp = (S::ZERO - g).max(S::ONE);
                                            S::ZERO
                                                - S::from_f64(2.0)
                                                    * t
                                                    * (ng_clamp.acosh() * third).cosh()
                                                - b3
                                        },
                                        // case 6: q <= 0 -> cosh formula (positive sign).
                                        || {
                                            let g_clamp_hi = g.max(S::ONE);
                                            S::from_f64(2.0)
                                                * t
                                                * (g_clamp_hi.acosh() * third).cosh()
                                                - b3
                                        },
                                    )
                                },
                            )
                        },
                    )
                },
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use crate::regime::Regime;
    use crate::rmhd::Rmhd;
    use crate::state::Prim;
    use symbi_ir::gv::{Gv, begin_trace, end_trace};
    use symbi_ir::passes::{cse, pressure, scalarize};

    // trace rmhd_wave_speeds at S=Gv (axis 0), scalarize through the same
    // pipeline the real kernel emit takes, run CSE, and return the peak
    // register-pressure report for the resulting kernel.
    fn trace_and_measure_pressure() -> pressure::PressureReport {
        begin_trace();
        let rho = Gv::field("prim_rho", "prim.rho");
        let vel: [Gv; 3] =
            std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), &format!("prim.vel[{k}]")));
        let pre = Gv::field("prim_pre", "prim.pre");
        let mag: [Gv; 3] =
            std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), &format!("prim.mag[{k}]")));
        let gamma = Gv::scalar("gamma");
        let eos = IdealGas { gamma };
        let prim = MhdPrim::<Gv, 3> {
            hydro: Prim {
                rho,
                vel: Tensor::new(vel),
                pre,
            },
            mag: Tensor::new(mag),
        };
        let nhat = Tensor::<Gv, 3>::unit(0);
        let (sl, sr) = rmhd_wave_speeds(&eos, &prim, &nhat);
        let outputs = [sl.node(), sr.node()];
        let graph = end_trace().graph;
        let mut k = scalarize::scalarize_kernel(&graph, &outputs);
        cse::cse_kernel(&mut k);
        pressure::peak_pressure_kernel(&k)
    }

    /// a loose blow-up guard (250) on the static `peak_pressure` estimate for the traced
    /// rmhd_wave_speeds kernel.
    ///
    /// the estimate counts every ancestor-scope `Let` as live and ignores true liveness, so it is a
    /// conservative UPPER bound and NOT the SASS register count a device compiler picks. the full
    /// Mignone & Del Zanna quartic (~200 ops) sits inside the lazily-selected branch taken only when
    /// the flow is neither near-static (`vsq ~ 0`) nor field-aligned-degenerate (`bn ~ 0`); the
    /// estimate sums that whole branch body as simultaneously live at the deepest scope, with no
    /// cross-path CSE to compact it, which is why it reads ~239 while a liveness-aware count reads
    /// near ~90. branch-local temps have short live ranges and cost the same whether they sit at
    /// function scope or inside an arm, and the lazy branch only DELETES work on the degenerate
    /// paths, so a value in that range carries no register regression.
    ///
    /// the guard therefore catches a gross (>2x) structural blow-up only. the load-bearing evidence
    /// for the branch structure is the emitted kernel shape and wallclock, not this number.
    #[test]
    fn rmhd_wave_speeds_under_pressure_bound() {
        let report = trace_and_measure_pressure();
        assert!(
            report.peak <= 250,
            "rmhd_wave_speeds peak pressure {} exceeds the loose blow-up bound 250 \
             at scope {:?} (the conservative metric over-counts lazy branches; see \
             the test doc — investigate a real regression, do not just bump this)",
            report.peak,
            report.at_scope_path,
        );
    }

    // CFL safety: the magnetosonic upper bound must NEVER under-estimate the exact
    // Mignone & Del Zanna quartic signal speed — otherwise the CFL dt would be unsafe. sweep a
    // battery of physical RMHD states (varying density, velocity direction/magnitude, pressure,
    // field strength/orientation) and assert max|s_ub| >= max|s_quartic| per axis.
    #[test]
    fn magnetosonic_bound_never_underestimates_quartic() {
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let rhos = [0.1_f64, 1.0, 10.0];
        let pres = [0.01_f64, 1.0, 100.0];
        let vels = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.6, 0.3],
            [0.7, -0.4, 0.2],
            [0.9, 0.0, 0.0],
            [-0.3, 0.3, 0.85],
        ];
        let mags = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.5, -1.0, 3.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ];
        let mut worst_margin = f64::INFINITY;
        for &rho in &rhos {
            for &pre in &pres {
                for v in &vels {
                    let vsq: f64 = v.iter().map(|x| x * x).sum();
                    if vsq >= 1.0 {
                        continue;
                    } // physical only
                    for b in &mags {
                        let prim = MhdPrim::<f64, 3> {
                            hydro: Prim {
                                rho,
                                vel: Tensor::new(*v),
                                pre,
                            },
                            mag: Tensor::new(*b),
                        };
                        for axis in 0..3 {
                            let nhat = Tensor::<f64, 3>::unit(axis);
                            let (slq, srq) = rmhd_wave_speeds(&eos, &prim, &nhat);
                            let (slu, sru) = rmhd_magnetosonic_cfl_speeds(&eos, &prim, &nhat);
                            let exact = slq.abs().max(srq.abs());
                            let bound = slu.abs().max(sru.abs());
                            let margin = bound - exact;
                            worst_margin = worst_margin.min(margin);
                            assert!(
                                margin >= -1e-9,
                                "magnetosonic bound UNDER-estimates the quartic (UNSAFE CFL): \
                                 rho={rho}, pre={pre}, v={v:?}, b={b:?}, axis={axis}: \
                                 bound={bound} < exact={exact}"
                            );
                        }
                    }
                }
            }
        }
        // sanity: the bound is genuinely an over-estimate (a strict margin everywhere).
        assert!(worst_margin >= -1e-9, "worst margin {worst_margin}");
    }

    #[test]
    fn quartic_wave_speeds_are_invariant_to_state_units() {
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let nhat = Tensor::<f64, 3>::unit(0);
        let reference = MhdPrim {
            hydro: Prim {
                rho: 1.3,
                vel: Tensor::new([0.31, -0.17, 0.08]),
                pre: 0.47,
            },
            mag: Tensor::new([2e-8, 0.61, -0.23]),
        };
        let expected = rmhd_wave_speeds(&eos, &reference, &nhat);

        for factor in [1e-100_f64, 1.0, 1e100] {
            let field_factor = factor.sqrt();
            let scaled = MhdPrim {
                hydro: Prim {
                    rho: reference.rho * factor,
                    vel: reference.vel,
                    pre: reference.pre * factor,
                },
                mag: reference.mag.scale(field_factor),
            };
            let got = rmhd_wave_speeds(&eos, &scaled, &nhat);
            assert!((got.0 - expected.0).abs() < 1e-12);
            assert!((got.1 - expected.1).abs() < 1e-12);
        }
    }

    // the CFL upper bound must PRESERVE NaN — an unphysical prim (NaN from a failed c2p) must
    // yield NaN speeds so it reaches the NaN-propagating CFL reduction + dt guard, which halts
    // the run; a clamped finite speed would silently mask it.
    #[test]
    fn magnetosonic_bound_propagates_nan() {
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        let nan = f64::NAN;
        let cases = [
            // NaN in density, pressure, a velocity component, a field component.
            MhdPrim::<f64, 3> {
                hydro: Prim {
                    rho: nan,
                    vel: Tensor::new([0.1, 0.0, 0.0]),
                    pre: 1.0,
                },
                mag: Tensor::new([1.0, 0.0, 0.0]),
            },
            MhdPrim::<f64, 3> {
                hydro: Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.1, 0.0, 0.0]),
                    pre: nan,
                },
                mag: Tensor::new([1.0, 0.0, 0.0]),
            },
            MhdPrim::<f64, 3> {
                hydro: Prim {
                    rho: 1.0,
                    vel: Tensor::new([nan, 0.0, 0.0]),
                    pre: 1.0,
                },
                mag: Tensor::new([1.0, 0.0, 0.0]),
            },
            MhdPrim::<f64, 3> {
                hydro: Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.1, 0.0, 0.0]),
                    pre: 1.0,
                },
                mag: Tensor::new([nan, 0.0, 0.0]),
            },
        ];
        for (i, prim) in cases.iter().enumerate() {
            let nhat = Tensor::<f64, 3>::unit(0);
            let (sl, sr) = rmhd_magnetosonic_cfl_speeds(&eos, prim, &nhat);
            assert!(
                !(sl.abs().max(sr.abs())).is_finite(),
                "case {i}: NaN prim produced a finite CFL speed (sl={sl}, sr={sr}); the bound \
                 must propagate NaN so the dt guard catches it"
            );
        }
    }

    // monic quartic coefficients from 4 real roots: x^4 + b x^3 + c x^2 + d x + e.
    fn quartic_from_roots(r: [f64; 4]) -> (f64, f64, f64, f64) {
        let b = -(r[0] + r[1] + r[2] + r[3]);
        let c = r[0] * r[1] + r[0] * r[2] + r[0] * r[3] + r[1] * r[2] + r[1] * r[3] + r[2] * r[3];
        let d =
            -(r[0] * r[1] * r[2] + r[0] * r[1] * r[3] + r[0] * r[2] * r[3] + r[1] * r[2] * r[3]);
        let e = r[0] * r[1] * r[2] * r[3];
        (b, c, d, e)
    }

    #[test]
    fn solve_quartic_minmax_recovers_known_real_roots() {
        // build a quartic from KNOWN real roots and require the solver to return their
        // (min, max). semantics-independent of the cubic resolvent's branch choice.
        let root_sets = [
            [-2.0, -1.0, 1.0, 2.0],
            [0.1, 0.2, 0.3, 0.5],
            [-0.5, 0.0, 0.5, 1.0],
            [-3.0, -2.0, 2.0, 3.0],
            [-0.9, -0.4, 0.4, 0.9],
        ];
        for r in &root_sets {
            let (b, c, d, e) = quartic_from_roots(*r);
            let (lo, hi) = solve_quartic_minmax::<f64>(b, c, d, e);
            assert!(
                (lo - r[0]).abs() <= 1e-7 * (1.0 + r[0].abs()),
                "lo: {lo} != {}",
                r[0]
            );
            assert!(
                (hi - r[3]).abs() <= 1e-7 * (1.0 + r[3].abs()),
                "hi: {hi} != {}",
                r[3]
            );
        }
    }

    #[test]
    fn solve_cubic_resolvent_is_a_root() {
        // the resolvent returns ONE real root of x^3 + b x^2 + c x + d; verify it nulls
        // the polynomial across the trig / hyperbolic branches.
        let cases = [
            (0.0, -1.0, 0.0),
            (-1.0, 0.5, 0.2),
            (2.0, -2.0, 1.0),
            (-3.0, 1.0, -0.5),
            (1.5, 0.3, -0.7),
        ];
        for (b, c, d) in cases {
            let x = solve_cubic_resolvent::<f64>(b, c, d);
            let val = x * x * x + b * x * x + c * x + d;
            assert!(
                val.abs() < 1e-7,
                "cubic({b},{c},{d}) root {x} leaves residual {val}"
            );
        }
    }

    // direct analytic rmhd wave_speeds (3-velocity / lab B), the ground-truth
    // reference the substrate Expr form is validated against.
    fn ref_wave_speeds(
        rho: f64,
        vel: [f64; 3],
        p: f64,
        mag: [f64; 3],
        gamma: f64,
        dir: usize,
    ) -> (f64, f64) {
        let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let eps = 1e-12;
        let vsq = dot(&vel, &vel);
        let w = 1.0 / (1.0 - vsq).sqrt();
        let h = 1.0 + gamma / (gamma - 1.0) * p / rho;
        let cssq = gamma * p / (rho * h);
        let vdb = dot(&vel, &mag);
        let bmu0 = w * vdb;
        let bmu_s = [
            mag[0] / w + w * vel[0] * vdb,
            mag[1] / w + w * vel[1] * vdb,
            mag[2] / w + w * vel[2] * vdb,
        ];
        let bmusq = -bmu0 * bmu0 + bmu_s[0] * bmu_s[0] + bmu_s[1] * bmu_s[1] + bmu_s[2] * bmu_s[2];
        let bn = mag[dir];
        let bnsq = bn * bn;
        let vn = vel[dir];
        if vsq < eps {
            let fac = 1.0 / (rho * h + bmusq);
            let b = -(bmusq + rho * h * cssq + bnsq * cssq) * fac;
            let cc = cssq * bnsq * fac;
            let disq = (b * b - 4.0 * cc).sqrt();
            let lr = (0.5 * (-b + disq)).sqrt();
            (-lr, lr)
        } else if bnsq < eps {
            let g2 = w * w;
            let vdbperp = vdb - vn * bn;
            let q = bmusq - cssq * vdbperp * vdbperp;
            let a2 = rho * h * (cssq + g2 * (1.0 - cssq)) + q;
            let a1 = -2.0 * rho * h * g2 * vn * (1.0 - cssq);
            let a0 = rho * h * (-cssq + g2 * vn * vn * (1.0 - cssq)) - q;
            let disq = a1 * a1 - 4.0 * a2 * a0;
            (
                0.5 * (-a1 - disq.sqrt()) / a2,
                0.5 * (-a1 + disq.sqrt()) / a2,
            )
        } else {
            let bmun = bmu_s[dir];
            let w2 = w * w;
            let vn2 = vn * vn;
            let a4 = -bmu0 * bmu0 * cssq + bmusq * w2 - cssq * w2 * w2 * h * rho
                + cssq * w2 * h * rho
                + w2 * w2 * h * rho;
            let fac = 1.0 / a4;
            let a3 = fac
                * (2.0 * bmu0 * bmun * cssq - 2.0 * bmusq * w2 * vn
                    + 4.0 * cssq * w2 * w2 * h * rho * vn
                    - 2.0 * cssq * w2 * h * rho * vn
                    - 4.0 * w2 * w2 * h * rho * vn);
            let a2 = fac
                * (bmu0 * bmu0 * cssq + bmusq * w2 * vn2
                    - bmusq * w2
                    - bmun * bmun * cssq
                    - 6.0 * cssq * w2 * w2 * h * rho * vn2
                    + cssq * w2 * h * rho * vn2
                    - cssq * w2 * h * rho
                    + 6.0 * w2 * w2 * h * rho * vn2);
            let a1 = fac
                * (-2.0 * bmu0 * bmun * cssq
                    + 2.0 * bmusq * w2 * vn
                    + 4.0 * cssq * w2 * w2 * h * rho * vn * vn2
                    + 2.0 * cssq * w2 * h * rho * vn
                    - 4.0 * w2 * w2 * h * rho * vn * vn2);
            let a0 = fac
                * (-bmusq * w2 * vn2 + bmun * bmun * cssq
                    - cssq * w2 * w2 * h * rho * vn2 * vn2
                    - cssq * w2 * h * rho * vn2
                    + w2 * w2 * h * rho * vn2 * vn2);
            let (ll, lr) = {
                let (b, c, d, e) = (a3, a2, a1, a0);
                let p = c - 0.375 * b * b;
                let q = 0.125 * b * b * b - 0.5 * b * c + d;
                let m = solve_cubic_ref(
                    p,
                    0.25 * p * p + 0.01171875 * b * b * b * b - e + 0.25 * b * d
                        - 0.0625 * b * b * c,
                    -0.125 * q * q,
                );
                quartic_minmax_ref(b, p, q, m)
            };
            if ll.is_nan() { (0.0, 0.0) } else { (ll, lr) }
        }
    }

    fn solve_cubic_ref(b: f64, c: f64, d: f64) -> f64 {
        let p = c - b * b / 3.0;
        let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
        if p.abs() < 1e-12 {
            return q.powf(1.0 / 3.0);
        }
        if q.abs() < 1e-12 {
            return 0.0;
        }
        let t = (p.abs() / 3.0).sqrt();
        let g = 1.5 * q / (p * t);
        if p > 0.0 {
            -2.0 * t * (g.asinh() / 3.0).sinh() - b / 3.0
        } else if 4.0 * p * p * p + 27.0 * q * q < 0.0 {
            2.0 * t * (g.acos() / 3.0).cos() - b / 3.0
        } else if q > 0.0 {
            -2.0 * t * ((-g).acosh() / 3.0).cosh() - b / 3.0
        } else {
            2.0 * t * (g.acosh() / 3.0).cosh() - b / 3.0
        }
    }

    fn quartic_minmax_ref(b: f64, p: f64, q: f64, m: f64) -> (f64, f64) {
        let nan = f64::NAN;
        let (mut smin, mut smax) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut track = |r: f64| {
            if r < smin {
                smin = r;
            }
            if r > smax {
                smax = r;
            }
        };
        if q.abs() < 1e-12 {
            if m < 0.0 {
                return (nan, nan);
            }
            let s = (2.0 * m).sqrt();
            if -m - p > 0.0 {
                let dl = (2.0 * (-m - p)).sqrt();
                track(-0.25 * b + 0.5 * (s - dl));
                track(-0.25 * b - 0.5 * (s - dl));
                track(-0.25 * b + 0.5 * (s + dl));
                track(-0.25 * b - 0.5 * (s + dl));
            }
            if (-m - p).abs() < 1e-12 {
                track(-0.25 * b - 0.5 * s);
                track(-0.25 * b + 0.5 * s);
            }
        } else {
            if m < 0.0 {
                return (nan, nan);
            }
            let s = (2.0 * m).sqrt();
            if -m - p + q / s >= 0.0 {
                let dl = (2.0 * (-m - p + q / s)).sqrt();
                track(0.5 * (-s + dl) - 0.25 * b);
                track(0.5 * (-s - dl) - 0.25 * b);
            }
            if -m - p - q / s >= 0.0 {
                let dl = (2.0 * (-m - p - q / s)).sqrt();
                track(0.5 * (s + dl) - 0.25 * b);
                track(0.5 * (s - dl) - 0.25 * b);
            }
        }
        if smin > smax {
            return (nan, nan);
        }
        (smin, smax)
    }

    #[test]
    fn rmhd_wave_speeds_match_cpp_reference() {
        // Rmhd::wave_speeds vs the analytic ground truth across all three dispersion regimes
        // (Eq.57 vsq~0, Eq.58 bn~0, Eq.56 full quartic). this is the function the flux HLLE
        // AND the spliced CFL map both call — one validated source.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let states = [
            ([0.0, 0.0, 0.0], [0.5, 0.3, 0.2], 0usize), // Eq.57: vsq ~ 0
            ([0.2, 0.1, 0.0], [0.0, 0.4, 0.3], 0),      // Eq.58: bn ~ 0
            ([0.3, 0.1, 0.2], [0.5, 0.3, 0.2], 0),      // Eq.56: quartic
            ([0.1, 0.4, 0.1], [0.6, 0.1, 0.4], 0),
            ([0.3, 0.1, 0.2], [0.5, 0.3, 0.2], 1), // direction indexing
            ([0.3, 0.1, 0.2], [0.5, 0.3, 0.2], 2),
        ];
        for (i, (vel, mag, dir)) in states.iter().enumerate() {
            let (rho, p) = (1.0, 1.0);
            let prim = MhdPrim {
                hydro: Prim {
                    rho,
                    vel: Tensor::new(*vel),
                    pre: p,
                },
                mag: Tensor::new(*mag),
            };
            let mut nh = [0.0; 3];
            nh[*dir] = 1.0;
            let (sl, sr) = Rmhd.wave_speeds(&eos, &prim, &Tensor::new(nh));
            let (wl, wr) = ref_wave_speeds(rho, *vel, p, *mag, 5.0 / 3.0, *dir);
            assert!(
                (sl - wl).abs() <= 1e-9 * (1.0 + wl.abs()),
                "state {i} dir {dir} sl: {sl} != {wl}"
            );
            assert!(
                (sr - wr).abs() <= 1e-9 * (1.0 + wr.abs()),
                "state {i} dir {dir} sr: {sr} != {wr}"
            );
            assert!(
                sl <= sr + 1e-12 && sr.abs() <= 1.0 + 1e-9,
                "state {i}: unphysical ({sl}, {sr})"
            );
        }
    }

    // wall-clock A/B: the production carrier cubic resolvent at S=f64 vs the
    // native single-branch reference. `solve_cubic_resolvent` now uses nested
    // `S::cond` (the lazy-branch dual of iterate), so at f64 it takes ONE arm —
    // matching the native branch. BEFORE the cond rewrite this was nested
    // `S::select` (computes ALL 4 transcendental cases, then blends) and ran
    // 2.16x slower; AFTER it is ~0.95x (parity). isolates that the
    // compute-all-paths tax is gone. run: cargo test -p symbi-hydro --release
    //   cubic_resolvent_select_tax_wallclock -- --ignored --nocapture
    #[test]
    #[ignore = "wall-clock A/B; timing is not a deterministic assertion. run with --release \
                --ignored --nocapture"]
    fn cubic_resolvent_select_tax_wallclock() {
        use std::hint::black_box;
        use std::time::Instant;

        // deterministic LCG giving representative (b,c,d) coefficient triples spanning
        // all four resolvent cases (p>0, three-real, q-sign). no rand dep.
        let n = 5_000_000usize;
        let mut inputs = Vec::with_capacity(n);
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut nxt = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 // [0,1)
        };
        for _ in 0..n {
            inputs.push((nxt() * 4.0 - 2.0, nxt() * 4.0 - 2.0, nxt() * 4.0 - 2.0));
        }

        // warm + time native single-branch.
        let t0 = Instant::now();
        let mut acc_n = 0.0;
        for &(b, c, d) in &inputs {
            acc_n += solve_cubic_ref(black_box(b), black_box(c), black_box(d));
        }
        let dt_native = t0.elapsed();
        black_box(acc_n);

        // time the carrier `cond` cubic at S=f64 — the production CPU path.
        let t1 = Instant::now();
        let mut acc_c = 0.0;
        for &(b, c, d) in &inputs {
            acc_c += super::solve_cubic_resolvent::<f64>(black_box(b), black_box(c), black_box(d));
        }
        let dt_carrier = t1.elapsed();
        black_box(acc_c);

        let ns_native = dt_native.as_nanos() as f64 / n as f64;
        let ns_carrier = dt_carrier.as_nanos() as f64 / n as f64;
        eprintln!("\n=== cubic resolvent: compute-all-paths tax (n={n}) ===");
        eprintln!("native single-branch : {ns_native:6.2} ns/call  ({dt_native:?})");
        eprintln!("carrier cond   (f64) : {ns_carrier:6.2} ns/call  ({dt_carrier:?})");
        eprintln!(
            "ratio                : {:.2}x  (was 2.16x under nested select)",
            ns_carrier / ns_native
        );
    }
}
