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

use symbi_ir::algebra::Scalar;

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
