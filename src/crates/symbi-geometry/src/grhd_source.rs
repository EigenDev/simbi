// =============================================================================
// grhd_source.rs
//
// the GENERIC GRHD geodesic (gravity) source, carrier-generic over the scalar S (f64 host /
// f32 / Gv trace). computes the valencia momentum + energy sources by contracting the perfect-fluid
// stress-energy T^{mu nu} with the metric geometry, INSTEAD of a hand-coded closed form per
// spacetime:
//   S_{S_r}^gravity = (1/2) [ T^{tt} d_r g_tt + 2 T^{tr} d_r g_tr + T^{rr} d_r g_rr ]   (the t-r block;
//                     the angular 2p/r geometric term rides the flat curvilinear momentum source)
//   S_tau          = alpha ( T^{mu 0} d_mu ln alpha - T^{mu nu} Gamma^0_{mu nu} )
//
// this is the numerically-validated `test_kerr_schild_sources.py` oracle ported to rust. the caller
// supplies the ADM radial block (alpha, beta^r, gamma_rr) + its radial derivatives (the metric's
// analytic d_r), plus the fluid state (E = rho eta W^2 = D + tau + p, the orthonormal radial
// velocity V, and p). SPHERICAL background (the GR metrics are spherical); the suppressed angular
// directions enter the energy source via g_{theta theta} = r^2, g_{phi phi} = r^2 sin^2(theta), whose
// theta-dependence cancels, so the equatorial (sin theta = 1) evaluation is exact for the radial 1D
// source.
//
// usage:
//   let (s_mom_grav, s_tau) = grhd_radial_geodesic_source(r, alpha, beta_r, gamma_rr,
//       d_alpha, d_beta_r, d_gamma_rr, e, big_v, p);
// =============================================================================

use symbi_algebra::{Matrix, Tensor};
use symbi_ir::algebra::Scalar;
use symbi_ir::dual::Dual;

use crate::metric::Metric;

/// the radial-block ADM derivatives a curved metric supplies for the geodesic source: the analytic
/// `d_r` of the lapse, the radial shift, and the radial spatial-metric coefficient.
#[derive(Clone, Copy, Debug)]
pub struct AdmRadialDerivs<S> {
    pub d_lapse: S,     // d_r alpha
    pub d_shift_r: S,   // d_r beta^r
    pub d_gamma_rr: S,  // d_r gamma_{rr}
}

impl<S: Scalar> AdmRadialDerivs<S> {
    /// the flat / static-diagonal-with-no-radial-stretch value: every derivative zero (Minkowski,
    /// and the flat curvilinear metrics whose radial coefficient is constant).
    pub fn zero() -> Self {
        Self { d_lapse: S::ZERO, d_shift_r: S::ZERO, d_gamma_rr: S::ZERO }
    }
}

/// the generic GRHD geodesic gravity source on a SPHERICAL curved background, radial component.
/// returns `(S_{S_r}^gravity, S_tau)` — the momentum GRAVITY source (the t-r block; excludes the flat
/// 2p/r), and the full energy source. carrier-generic: at `S = Gv` this traces the kernel expression,
/// at `S = f64` it evaluates directly (the `test_grhd_source` validation + the python oracle).
#[allow(clippy::too_many_arguments)]
pub fn grhd_radial_geodesic_source<S: Scalar>(
    r: S,
    alpha: S,
    beta_r: S,      // beta^r (contravariant radial shift)
    gamma_rr: S,    // gamma_{rr}
    d_alpha: S,     // d_r alpha
    d_beta_r: S,    // d_r beta^r
    d_gamma_rr: S,  // d_r gamma_{rr}
    e: S,           // E = rho eta W^2 = D + tau + p
    big_v: S,       // orthonormal radial velocity V
    p: S,
) -> (S, S) {
    let two = S::from_f64(2.0);
    let half = S::from_f64(0.5);

    // ---- the (t, r) block of the 4-metric from the ADM decomposition ----
    // g_tt = -alpha^2 + gamma_rr (beta^r)^2,  g_tr = gamma_rr beta^r,  g_rr = gamma_rr.
    let g_tt = S::ZERO - alpha * alpha + gamma_rr * beta_r * beta_r;
    let g_tr = gamma_rr * beta_r;
    let g_rr = gamma_rr;
    // radial derivatives of the block.
    let dg_tt = S::ZERO - two * alpha * d_alpha
        + d_gamma_rr * beta_r * beta_r
        + gamma_rr * two * beta_r * d_beta_r;
    let dg_tr = d_gamma_rr * beta_r + gamma_rr * d_beta_r;
    let dg_rr = d_gamma_rr;

    // ---- the inverse (t, r) block (2x2) ----
    let det2 = g_tt * g_rr - g_tr * g_tr;
    let inv_tt = g_rr / det2;
    let inv_tr = (S::ZERO - g_tr) / det2;
    let inv_rr = g_tt / det2;

    // ---- the fluid stress-energy T^{mu nu} = rho eta u^mu u^nu + p g^{mu nu} ----
    // u^mu = W * uhat^mu, uhat = (1/alpha, V/sqrt(gamma_rr) - beta^r/alpha); rho eta u^mu u^nu =
    // E * uhat^mu uhat^nu (E = rho eta W^2). angular components zero (radial flow).
    let uhat_t = S::ONE / alpha;
    let uhat_r = big_v / gamma_rr.sqrt() - beta_r / alpha;
    let t_tt = e * uhat_t * uhat_t + p * inv_tt;
    let t_tr = e * uhat_t * uhat_r + p * inv_tr;
    let t_rr = e * uhat_r * uhat_r + p * inv_rr;
    // angular: g^{theta theta} = 1/r^2, g^{phi phi} = 1/(r^2 sin^2), so T^{theta theta} = p/r^2 etc.
    let t_ang = p / (r * r); // == T^{theta theta} = T^{phi phi} (sin theta = 1 equatorial)

    // ---- momentum gravity source: (1/2)(T^tt dg_tt + 2 T^tr dg_tr + T^rr dg_rr) ----
    let s_mom = half * (t_tt * dg_tt + two * t_tr * dg_tr + t_rr * dg_rr);

    // ---- energy source S_tau = alpha (T^{r0} d_r ln alpha - T^{mu nu} Gamma^0_{mu nu}) ----
    // Gamma^t from the (t,r) block + the angular pieces (only g^{t.} rows, only d_r nonzero).
    let gt_tt = S::ZERO - half * inv_tr * dg_tt;
    let gt_tr = half * inv_tt * dg_tt;
    let gt_rr = inv_tt * dg_tr + half * inv_tr * dg_rr;
    // Gamma^t_{theta theta} = Gamma^t_{phi phi} = -r g^{tr} (from -(1/2)g^{tr} d_r g_ang, d_r g_ang=2r).
    let gt_ang = S::ZERO - r * inv_tr;
    let t_gamma = t_tt * gt_tt
        + two * t_tr * gt_tr
        + t_rr * gt_rr
        + t_ang * gt_ang   // theta theta
        + t_ang * gt_ang;  // phi phi
    let dln_alpha = d_alpha / alpha;
    let s_tau = alpha * (t_tr * dln_alpha - t_gamma);

    (s_mom, s_tau)
}

/// the FULL covariant valencia geodesic source on a general STATIC background — the momentum
/// source for every coordinate slot plus the energy source, from one forward-autodiff pass per
/// coordinate axis:
///   S_j   = (1/2) T^{mu nu} d_j g_{mu nu}                       (all blocks, per coordinate j)
///   S_tau = alpha ( T^{t j} d_j ln(alpha) - T^{mu nu} Gamma^t_{mu nu} )
/// with T^{mu nu} = E uhat^mu uhat^nu + p g^{mu nu} (E = rho h W^2), uhat^t = 1/alpha,
/// uhat^i = v^i - beta^i/alpha (v^i the CONTRAVARIANT valencia velocity).
///
/// unlike [`grhd_radial_geodesic_source`] — the 1D (t, r)-block specialization whose angular
/// pressure blocks ride the flat curvilinear source — this contraction carries EVERY metric
/// block: the angular/centrifugal/pressure terms included. it therefore serves any coordinate
/// system (spherical, cartesian kerr-schild) and any momentum count: evaluate at the metric's
/// full coordinate dimension `D` regardless of the grid dimension. a symmetry axis the metric
/// never reads (axisymmetric phi) yields zero tangents, so the suppressed-axis momentum source
/// vanishes by construction (angular-momentum conservation). the metric supplies only its ADM
/// line element; autodiff differentiates it — no hand-derived christoffels.
///
/// carrier-generic: `S = f64` evaluates on the host (the validation tests); `S = Gv` traces
/// the kernel expression (the host loop over axes unrolls at trace time).
pub fn grhd_covariant_source<S: Scalar, M, const D: usize>(
    g: &M,
    x: Tensor<S, D>,
    e: S,
    v: Tensor<S, D>,
    p: S,
) -> (Tensor<S, D>, S)
where
    M: Metric<Dual<S>, D>,
{
    debug_assert!(D <= 3, "the padded 4-metric blocks hold at most 3 spatial axes");
    let half = S::from_f64(0.5);

    // one dual pass per axis: seed x_kk, harvest the ADM block and its d_kk. the values are
    // identical across passes; take them from the first.
    let mut alpha = S::ZERO;
    let mut beta = Tensor::<S, D>::zeros();
    let mut gam = Matrix::<S, D>::zeros();
    let mut gam_inv = Matrix::<S, D>::zeros();
    let mut d_alpha = [S::ZERO; D];
    let mut d_beta = [[S::ZERO; D]; D]; // d_beta[kk][ii] = d_kk beta^ii
    let mut d_gam: Vec<Matrix<S, D>> = Vec::with_capacity(D);
    for kk in 0..D {
        let xd = Tensor::<Dual<S>, D>::from_fn(|ii| {
            if ii == kk { Dual::variable(x[ii]) } else { Dual::constant(x[ii]) }
        });
        let a = g.lapse(xd);
        let b = g.shift(xd);
        let gm = g.spatial_metric(xd);
        if kk == 0 {
            alpha = a.value;
            beta = Tensor::from_fn(|ii| b[ii].value);
            gam = Matrix::from_fn(|ii, jj| gm[(ii, jj)].value);
            gam_inv = g.spatial_metric_inv(xd).map(|d| d.value);
        }
        d_alpha[kk] = a.tangent;
        for ii in 0..D {
            d_beta[kk][ii] = b[ii].tangent;
        }
        d_gam.push(gm.map(|d| d.tangent));
    }

    // ---- the 4-metric blocks (index 0 = t, 1..=D spatial), static: d_t g = 0 ----
    // g_tt = -alpha^2 + beta_i beta^i, g_ti = beta_i = gamma_ij beta^j, g_ij = gamma_ij.
    let beta_low = Tensor::<S, D>::from_fn(|ii| gam.row(ii).dot(&beta));
    let mut g4 = [[S::ZERO; 4]; 4];
    g4[0][0] = S::ZERO - alpha * alpha + beta_low.dot(&beta);
    for ii in 0..D {
        g4[0][ii + 1] = beta_low[ii];
        g4[ii + 1][0] = beta_low[ii];
        for jj in 0..D {
            g4[ii + 1][jj + 1] = gam[(ii, jj)];
        }
    }
    // radial-style derivatives per axis kk.
    let mut dg4 = [[[S::ZERO; 4]; 4]; 3];
    for kk in 0..D {
        let mut d_beta_low = [S::ZERO; D];
        for ii in 0..D {
            let mut s = S::ZERO;
            for jj in 0..D {
                s = s + d_gam[kk][(ii, jj)] * beta[jj] + gam[(ii, jj)] * d_beta[kk][jj];
            }
            d_beta_low[ii] = s;
        }
        let mut dtt = S::ZERO - S::from_f64(2.0) * alpha * d_alpha[kk];
        for ii in 0..D {
            dtt = dtt + d_beta_low[ii] * beta[ii] + beta_low[ii] * d_beta[kk][ii];
        }
        dg4[kk][0][0] = dtt;
        for ii in 0..D {
            dg4[kk][0][ii + 1] = d_beta_low[ii];
            dg4[kk][ii + 1][0] = d_beta_low[ii];
            for jj in 0..D {
                dg4[kk][ii + 1][jj + 1] = d_gam[kk][(ii, jj)];
            }
        }
    }

    // ---- the inverse 4-metric (ADM closed form) ----
    // g^tt = -1/alpha^2, g^ti = beta^i/alpha^2, g^ij = gamma^ij - beta^i beta^j/alpha^2.
    let ia2 = S::ONE / (alpha * alpha);
    let mut gi4 = [[S::ZERO; 4]; 4];
    gi4[0][0] = S::ZERO - ia2;
    for ii in 0..D {
        gi4[0][ii + 1] = beta[ii] * ia2;
        gi4[ii + 1][0] = beta[ii] * ia2;
        for jj in 0..D {
            gi4[ii + 1][jj + 1] = gam_inv[(ii, jj)] - beta[ii] * beta[jj] * ia2;
        }
    }

    // ---- the fluid stress-energy T^{mu nu} = E uhat^mu uhat^nu + p g^{mu nu} ----
    let mut uhat = [S::ZERO; 4];
    uhat[0] = S::ONE / alpha;
    for ii in 0..D {
        uhat[ii + 1] = v[ii] - beta[ii] / alpha;
    }
    let mut t4 = [[S::ZERO; 4]; 4];
    for mm in 0..=D {
        for nn in 0..=D {
            t4[mm][nn] = e * uhat[mm] * uhat[nn] + p * gi4[mm][nn];
        }
    }

    // ---- momentum sources: S_j = (1/2) T^{mu nu} d_j g_{mu nu} (full symmetric double sum) ----
    let s_mom = Tensor::<S, D>::from_fn(|kk| {
        let mut s = S::ZERO;
        for mm in 0..=D {
            for nn in 0..=D {
                s = s + t4[mm][nn] * dg4[kk][mm][nn];
            }
        }
        half * s
    });

    // ---- energy source: S_tau = alpha (T^{t j} d_j ln(alpha) - T^{mu nu} Gamma^t_{mu nu}) ----
    // Gamma^t_{mu nu} = (1/2) g^{t sigma} (d_mu g_{sigma nu} + d_nu g_{sigma mu} - d_sigma g_{mu nu});
    // static background: the t-derivative vanishes, so d_mu indexes the spatial passes only.
    let dg = |mu: usize, aa: usize, bb: usize| -> S {
        if mu == 0 { S::ZERO } else { dg4[mu - 1][aa][bb] }
    };
    let mut t_gamma = S::ZERO;
    for mm in 0..=D {
        for nn in 0..=D {
            let mut gt = S::ZERO;
            for ss in 0..=D {
                gt = gt + gi4[0][ss] * (dg(mm, ss, nn) + dg(nn, ss, mm) - dg(ss, mm, nn));
            }
            t_gamma = t_gamma + t4[mm][nn] * half * gt;
        }
    }
    let mut work = S::ZERO;
    for kk in 0..D {
        work = work + t4[0][kk + 1] * d_alpha[kk] / alpha;
    }
    let s_tau = alpha * (work - t_gamma);

    (s_mom, s_tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-11 * (1.0 + a.abs().max(b.abs()))
    }

    // the schwarzschild-coordinate ADM block + its radial derivatives (f = 1 - 2M/r).
    fn schwarzschild_adm(r: f64, m: f64) -> (f64, f64, f64, AdmRadialDerivs<f64>) {
        let f = 1.0 - 2.0 * m / r;
        let df = 2.0 * m / (r * r); // d_r f
        let alpha = f.sqrt();
        let d = AdmRadialDerivs {
            d_lapse: df / (2.0 * f.sqrt()),
            d_shift_r: 0.0,
            d_gamma_rr: -df / (f * f), // gamma_rr = 1/f
        };
        (alpha, 0.0, 1.0 / f, d)
    }

    // the ingoing kerr-schild ADM block (h = 1 + 2M/r).
    fn kerr_schild_adm(r: f64, m: f64) -> (f64, f64, f64, AdmRadialDerivs<f64>) {
        let b = 2.0 * m / r;
        let h = 1.0 + b;
        let db = -2.0 * m / (r * r); // d_r (2M/r)
        let alpha = 1.0 / h.sqrt();
        let d = AdmRadialDerivs {
            d_lapse: -db / (2.0 * h.powf(1.5)), // d_r (h^{-1/2})
            d_shift_r: -2.0 * m / (r + 2.0 * m).powi(2),
            d_gamma_rr: db, // gamma_rr = h
        };
        (alpha, b / h, h, d) // beta^r = (2M/r)/h = 2M/(r+2M)
    }

    #[test]
    fn generic_source_matches_schwarzschild_closed_form() {
        let m = 1.0;
        let (e, big_v, p) = (2.3, -0.4, 0.05);
        for &r in &[8.0, 3.0, 2.5] {
            let (a, br, grr, d) = schwarzschild_adm(r, m);
            let (s_mom, s_tau) =
                grhd_radial_geodesic_source(r, a, br, grr, d.d_lapse, d.d_shift_r, d.d_gamma_rr, e, big_v, p);
            let f = 1.0 - 2.0 * m / r;
            // closed forms (gv/godunov.rs Schwarzschild arms): gravity part only for momentum.
            let s_mom_cf = -m * e * (1.0 + big_v * big_v) / (r * r * f);
            let s_tau_cf = -a * e * big_v * m / (r * r * f);
            assert!(approx(s_mom, s_mom_cf), "S_Sr r={r}: {s_mom} != {s_mom_cf}");
            assert!(approx(s_tau, s_tau_cf), "S_tau r={r}: {s_tau} != {s_tau_cf}");
        }
    }

    // the covariant contraction at the metric's full D = 3, radial flow: the radial momentum
    // source must equal the (t, r)-block oracle PLUS the angular pressure blocks
    // (1/2)(T^{theta theta} d_r g_{theta theta} + T^{phi phi} d_r g_{phi phi}) = 2p/r; the
    // polar source is the pressure term p cot(theta); the azimuthal source vanishes
    // (axisymmetry); the energy source equals the oracle (which carries the angular
    // Gamma^t blocks already).
    #[test]
    fn covariant_source_radial_flow_matches_oracle_plus_angular_blocks() {
        use crate::metric::{Schwarzschild, SchwarzschildKS};
        let m = 1.0;
        let (e, big_v, p) = (2.3_f64, -0.4_f64, 0.05_f64);
        let theta = 1.1_f64;
        for &r in &[8.0_f64, 3.0, 2.5] {
            for ks in [false, true] {
                if !ks && r <= 2.0 * m {
                    continue;
                }
                let (a, br, grr, d) = if ks { kerr_schild_adm(r, m) } else { schwarzschild_adm(r, m) };
                let (mom_cf, tau_cf) = grhd_radial_geodesic_source(
                    r, a, br, grr, d.d_lapse, d.d_shift_r, d.d_gamma_rr, e, big_v, p,
                );
                let vr = big_v / grr.sqrt(); // contravariant v^r from the orthonormal V
                let x = Tensor::new([r, theta, 0.0]);
                let v = Tensor::new([vr, 0.0, 0.0]);
                let (s, s_tau) = if ks {
                    let g = SchwarzschildKS { mass: Dual::constant(m) };
                    grhd_covariant_source(&g, x, e, v, p)
                } else {
                    let g = Schwarzschild { mass: Dual::constant(m) };
                    grhd_covariant_source(&g, x, e, v, p)
                };
                assert!(approx(s[0], mom_cf + 2.0 * p / r), "S_r r={r} ks={ks}: {} != {}", s[0], mom_cf + 2.0 * p / r);
                assert!(approx(s[1], p * theta.cos() / theta.sin()), "S_theta r={r} ks={ks}: {}", s[1]);
                assert!(s[2].abs() < 1e-14, "S_phi must vanish (axisymmetry): {}", s[2]);
                assert!(approx(s_tau, tau_cf), "S_tau r={r} ks={ks}: {s_tau} != {tau_cf}");
            }
        }
    }

    // rotating (swirl) flow on schwarzschild, off the equator: the closed forms follow from
    // the diagonal metric — S_theta = (E (v^phi)^2 + p g^{phi phi}) r^2 sin(theta) cos(theta),
    // S_r gains the azimuthal centrifugal block E (v^phi)^2 r sin^2(theta), and the energy
    // source is unchanged from the radial oracle (zero shift: only T^{tr} couples to Gamma^t).
    #[test]
    fn covariant_source_swirl_closed_form_schwarzschild() {
        use crate::metric::Schwarzschild;
        let m = 1.0;
        let (e, p) = (2.3_f64, 0.05_f64);
        let (r, theta) = (6.0_f64, 1.1_f64);
        let (vr, vphi) = (-0.1_f64, 0.02_f64);
        let (st, ct) = (theta.sin(), theta.cos());

        let g = Schwarzschild { mass: Dual::constant(m) };
        let (s, s_tau) = grhd_covariant_source(
            &g, Tensor::new([r, theta, 0.0]), e, Tensor::new([vr, 0.0, vphi]), p,
        );

        let (a, br, grr, d) = schwarzschild_adm(r, m);
        let big_v = vr * grr.sqrt();
        let (mom_cf, tau_cf) = grhd_radial_geodesic_source(
            r, a, br, grr, d.d_lapse, d.d_shift_r, d.d_gamma_rr, e, big_v, p,
        );
        let s_r_cf = mom_cf + 2.0 * p / r + e * vphi * vphi * r * st * st;
        let s_th_cf = (e * vphi * vphi + p / (r * r * st * st)) * r * r * st * ct;
        assert!(approx(s[0], s_r_cf), "S_r: {} != {s_r_cf}", s[0]);
        assert!(approx(s[1], s_th_cf), "S_theta: {} != {s_th_cf}", s[1]);
        assert!(s[2].abs() < 1e-14, "S_phi must vanish: {}", s[2]);
        assert!(approx(s_tau, tau_cf), "S_tau: {s_tau} != {tau_cf}");
    }

    // the M = 0 limit is flat spherical: the contraction must reproduce the covariant
    // curvilinear inertial + pressure source with NO gravity and ZERO energy source.
    #[test]
    fn covariant_source_flat_limit_is_curvilinear_inertial() {
        use crate::metric::Schwarzschild;
        let (e, p) = (2.3_f64, 0.05_f64);
        let (r, theta) = (6.0_f64, 1.1_f64);
        let (vr, vth, vphi) = (-0.1_f64, 0.03_f64, 0.02_f64);
        let (st, ct) = (theta.sin(), theta.cos());

        let g = Schwarzschild { mass: Dual::constant(0.0) };
        let (s, s_tau) = grhd_covariant_source(
            &g, Tensor::new([r, theta, 0.0]), e, Tensor::new([vr, vth, vphi]), p,
        );

        let s_r_cf = e * (vth * vth * r + vphi * vphi * r * st * st) + 2.0 * p / r;
        let s_th_cf = (e * vphi * vphi + p / (r * r * st * st)) * r * r * st * ct;
        assert!(approx(s[0], s_r_cf), "flat S_r: {} != {s_r_cf}", s[0]);
        assert!(approx(s[1], s_th_cf), "flat S_theta: {} != {s_th_cf}", s[1]);
        assert!(s[2].abs() < 1e-14 && s_tau.abs() < 1e-14, "flat gravity must vanish");
    }

    // finite-difference mirror of the covariant contraction: the SAME T^{mu nu} algebra, but
    // every metric derivative taken numerically — validates the Dual (autodiff) plumbing through
    // an arbitrary theta-dependent, non-diagonal metric.
    fn covariant_source_fd<M: crate::metric::Metric<f64, 3>>(
        g: &M,
        x: Tensor<f64, 3>,
        e: f64,
        v: Tensor<f64, 3>,
        p: f64,
    ) -> (Tensor<f64, 3>, f64) {
        let eps = 1e-6;
        let block = |x: Tensor<f64, 3>| -> [[f64; 4]; 4] {
            let alpha = g.lapse(x);
            let beta = g.shift(x);
            let gm = g.spatial_metric(x);
            let beta_low: [f64; 3] =
                std::array::from_fn(|ii| (0..3).map(|jj| gm[(ii, jj)] * beta[jj]).sum());
            let mut g4 = [[0.0; 4]; 4];
            g4[0][0] = -alpha * alpha + (0..3).map(|ii| beta_low[ii] * beta[ii]).sum::<f64>();
            for ii in 0..3 {
                g4[0][ii + 1] = beta_low[ii];
                g4[ii + 1][0] = beta_low[ii];
                for jj in 0..3 {
                    g4[ii + 1][jj + 1] = gm[(ii, jj)];
                }
            }
            g4
        };
        let mut dg = [[[0.0; 4]; 4]; 3];
        for kk in 0..2 {
            // r and theta only; the phi derivative is zero by axisymmetry.
            let mut xp = x;
            let mut xm = x;
            xp[kk] += eps;
            xm[kk] -= eps;
            let (bp, bm) = (block(xp), block(xm));
            for aa in 0..4 {
                for bb in 0..4 {
                    dg[kk][aa][bb] = (bp[aa][bb] - bm[aa][bb]) / (2.0 * eps);
                }
            }
        }
        let alpha = g.lapse(x);
        let beta = g.shift(x);
        let gi = g.spatial_metric_inv(x);
        let ia2 = 1.0 / (alpha * alpha);
        let mut gi4 = [[0.0; 4]; 4];
        gi4[0][0] = -ia2;
        for ii in 0..3 {
            gi4[0][ii + 1] = beta[ii] * ia2;
            gi4[ii + 1][0] = beta[ii] * ia2;
            for jj in 0..3 {
                gi4[ii + 1][jj + 1] = gi[(ii, jj)] - beta[ii] * beta[jj] * ia2;
            }
        }
        let mut uhat = [0.0; 4];
        uhat[0] = 1.0 / alpha;
        for ii in 0..3 {
            uhat[ii + 1] = v[ii] - beta[ii] / alpha;
        }
        let mut t4 = [[0.0; 4]; 4];
        for mm in 0..4 {
            for nn in 0..4 {
                t4[mm][nn] = e * uhat[mm] * uhat[nn] + p * gi4[mm][nn];
            }
        }
        let s_mom = Tensor::<f64, 3>::from_fn(|kk| {
            let mut acc = 0.0;
            for mm in 0..4 {
                for nn in 0..4 {
                    acc += t4[mm][nn] * dg[kk][mm][nn];
                }
            }
            0.5 * acc
        });
        let d = |mu: usize, aa: usize, bb: usize| if mu == 0 { 0.0 } else { dg[mu - 1][aa][bb] };
        let mut t_gamma = 0.0;
        for mm in 0..4 {
            for nn in 0..4 {
                let mut gt = 0.0;
                for ss in 0..4 {
                    gt += gi4[0][ss] * (d(mm, ss, nn) + d(nn, ss, mm) - d(ss, mm, nn));
                }
                t_gamma += t4[mm][nn] * 0.5 * gt;
            }
        }
        let d_alpha = {
            let mut out = [0.0; 3];
            for kk in 0..2 {
                let mut xp = x;
                let mut xm = x;
                xp[kk] += eps;
                xm[kk] -= eps;
                out[kk] = (g.lapse(xp) - g.lapse(xm)) / (2.0 * eps);
            }
            out
        };
        let work: f64 = (0..3).map(|kk| t4[0][kk + 1] * d_alpha[kk] / alpha).sum();
        (s_mom, alpha * (work - t_gamma))
    }

    #[test]
    fn covariant_source_on_kerr_matches_finite_differences() {
        use crate::metric::KerrKS;
        // a rotating state with all three velocity components, off the equator, sampled
        // inside and outside the horizon — the autodiff contraction must agree with the
        // finite-difference mirror through the non-diagonal theta-dependent metric.
        let fd_close = |x: f64, y: f64, s: f64| (x - y).abs() < 1e-7 * (1.0 + s.abs());
        for &a in &[0.9_f64, -0.5] {
            let g = KerrKS { mass: 1.0_f64, spin: a };
            let gd = KerrKS { mass: Dual::constant(1.0_f64), spin: Dual::constant(a) };
            let (e, p) = (2.3_f64, 0.05_f64);
            for &(r, th) in &[(1.4_f64, 1.2_f64), (2.5, 0.8), (8.0, 1.9)] {
                let x = Tensor::new([r, th, 0.0]);
                let v = Tensor::new([-0.2_f64, 0.03, 0.02]);
                let (s_ad, tau_ad) = grhd_covariant_source(&gd, x, e, v, p);
                let (s_fd, tau_fd) = covariant_source_fd(&g, x, e, v, p);
                for kk in 0..3 {
                    assert!(fd_close(s_ad[kk], s_fd[kk], e),
                        "S_mom[{kk}] a={a} r={r}: {} vs {}", s_ad[kk], s_fd[kk]);
                }
                assert!(fd_close(tau_ad, tau_fd, e), "S_tau a={a} r={r}: {tau_ad} vs {tau_fd}");
            }
        }
    }

    #[test]
    fn covariant_source_on_kerr_at_zero_spin_matches_schwarzschild_ks() {
        use crate::metric::{KerrKS, SchwarzschildKS};
        // a = 0 kerr must reproduce the schwarzschild-KS source values for arbitrary
        // rotating states (different expression paths, same physics).
        let close = |x: f64, y: f64| (x - y).abs() < 1e-11 * (1.0 + x.abs().max(y.abs()));
        let kerr = KerrKS { mass: Dual::constant(1.0_f64), spin: Dual::constant(0.0) };
        let ks = SchwarzschildKS { mass: Dual::constant(1.0_f64) };
        let (e, p) = (2.3_f64, 0.05_f64);
        for &(r, th) in &[(1.5_f64, 1.0_f64), (3.0, 1.9), (12.0, 1.3)] {
            let x = Tensor::new([r, th, 0.0]);
            let v = Tensor::new([-0.3_f64, 0.02, 0.015]);
            let (sk, tk) = grhd_covariant_source(&kerr, x, e, v, p);
            let (ss, ts) = grhd_covariant_source(&ks, x, e, v, p);
            for kk in 0..3 {
                assert!(close(sk[kk], ss[kk]), "S_mom[{kk}] r={r}: {} vs {}", sk[kk], ss[kk]);
            }
            assert!(close(tk, ts), "S_tau r={r}: {tk} vs {ts}");
        }
    }

    #[test]
    fn generic_source_matches_kerr_schild_closed_form() {
        let m = 1.0;
        let (e, big_v, p) = (2.3, -0.4, 0.05);
        for &r in &[8.0, 3.0, 2.0, 1.5, 1.2] {
            let (a, br, grr, d) = kerr_schild_adm(r, m);
            let (s_mom, s_tau) =
                grhd_radial_geodesic_source(r, a, br, grr, d.d_lapse, d.d_shift_r, d.d_gamma_rr, e, big_v, p);
            let b = 2.0 * m / r;
            let h = 1.0 + b;
            let s_mom_cf = -m * e * (1.0 + big_v).powi(2) / (r * r * h);
            let s_tau_cf = -m / (r * r * h.powf(1.5)) * (e * big_v * (1.0 + (2.0 + b) * big_v) - p * (2.0 + 3.0 * b));
            assert!(approx(s_mom, s_mom_cf), "S_Sr r={r}: {s_mom} != {s_mom_cf}");
            assert!(approx(s_tau, s_tau_cf), "S_tau r={r}: {s_tau} != {s_tau_cf}");
        }
    }
}
