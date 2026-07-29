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
// validated against the closed-form schwarzschild and kerr-schild geodesic sources. the caller
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
    pub d_lapse: S,    // d_r alpha
    pub d_shift_r: S,  // d_r beta^r
    pub d_gamma_rr: S, // d_r gamma_{rr}
}

impl<S: Scalar> AdmRadialDerivs<S> {
    /// the flat / static-diagonal-with-no-radial-stretch value: every derivative zero (Minkowski,
    /// and the flat curvilinear metrics whose radial coefficient is constant).
    pub fn zero() -> Self {
        Self {
            d_lapse: S::ZERO,
            d_shift_r: S::ZERO,
            d_gamma_rr: S::ZERO,
        }
    }
}

/// the generic GRHD geodesic gravity source on a SPHERICAL curved background, radial component.
/// returns `(S_{S_r}^gravity, S_tau)` — the momentum GRAVITY source (the t-r block; excludes the flat
/// 2p/r), and the full energy source. carrier-generic: at `S = Gv` this traces the kernel expression,
/// at `S = f64` it evaluates directly.
#[allow(clippy::too_many_arguments)]
pub fn grhd_radial_geodesic_source<S: Scalar>(
    r: S,
    alpha: S,
    beta_r: S,     // beta^r (contravariant radial shift)
    gamma_rr: S,   // gamma_{rr}
    d_alpha: S,    // d_r alpha
    d_beta_r: S,   // d_r beta^r
    d_gamma_rr: S, // d_r gamma_{rr}
    e: S,          // E = rho eta W^2 = D + tau + p
    big_v: S,      // orthonormal radial velocity V
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
        + t_ang * gt_ang; // phi phi
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
    let adm = adm_contraction_blocks(g, x);
    // ---- the fluid stress-energy T^{mu nu} = E uhat^mu uhat^nu + p g^{mu nu} ----
    let mut uhat = [S::ZERO; 4];
    uhat[0] = S::ONE / adm.alpha;
    for ii in 0..D {
        uhat[ii + 1] = v[ii] - adm.beta[ii] / adm.alpha;
    }
    let mut t4 = [[S::ZERO; 4]; 4];
    for mm in 0..=D {
        for nn in 0..=D {
            t4[mm][nn] = e * uhat[mm] * uhat[nn] + p * adm.gi4[mm][nn];
        }
    }
    contract_stress(&adm, &t4)
}

/// the rest-mass-subtracted covariant (killing) energy from the valencia energy `tau`:
///   E_hat = sqrt(gamma) ( alpha tau + (alpha - 1) D - beta^i S_i )   (docs/covariant_energy.md)
/// evolving `E_hat` in the energy slot conserves the relativistic bernoulli invariant `h u_t` to
/// roundoff on ANY stationary background (the killing energy density minus the zero-source baryon
/// density), where the eulerian `tau` carries a geodesic source that does not vanish. reduces to
/// `tau` at alpha = 1, beta = 0, sqrt(gamma) = 1. `shift` is the CONTRAVARIANT beta^i and `mom` the
/// COVARIANT valencia momentum S_i, so `shift.dot(mom) = beta^i S_i`. carrier-generic (S = f64 host
/// / S = Gv traces the kernel expression at the godunov energy assembly).
pub fn tau_to_ehat<S: Scalar, const D: usize>(
    tau: S,
    den: S,
    mom: Tensor<S, D>,
    alpha: S,
    shift: Tensor<S, D>,
    sqrt_gamma: S,
) -> S {
    sqrt_gamma * (alpha * tau + (alpha - S::ONE) * den - shift.dot(&mom))
}

/// invert [`tau_to_ehat`]: recover `tau` from the evolved `E_hat` and the (known, fixed) background
/// metric, so the existing c2p newton runs on an unchanged `tau` input. the `1/alpha` is bounded
/// away from zero by the metric guard (alpha clamped for r < M/2).
pub fn ehat_to_tau<S: Scalar, const D: usize>(
    ehat: S,
    den: S,
    mom: Tensor<S, D>,
    alpha: S,
    shift: Tensor<S, D>,
    sqrt_gamma: S,
) -> S {
    (ehat / sqrt_gamma + (S::ONE - alpha) * den + shift.dot(&mom)) / alpha
}

/// the coordinate free-index-down covariant energy conserved density and flux (HARM: Gammie et al.
/// 2003; AthenaK: Stone et al. 2024, eq. 20), in the code's POSITIVE-energy convention:
///   E_hat  = -sqrt(-g) ( T^t_t + rho u^t ),   F^j = -sqrt(-g) ( T^j_t + rho u^j )
/// with `T^mu_nu = w u^mu u_nu + p delta^mu_nu` (hydro; `w = rho h` the total enthalpy density),
/// `sqrt(-g) = alpha sqrt(det gamma)`, and the coordinate 4-velocity `u^t = W/alpha`,
/// `u^i = W(v^i - beta^i/alpha)`. the overall minus flips AthenaK's negative-energy `T^t_t` (which
/// is `-rho eps` at rest) to `+tau` at rest, matching the Valencia energy sign. the energy source is
/// IDENTICALLY ZERO on a stationary metric (the t-row of `(1/2)(d_nu g_ab)T^ab`), so this evolves
/// the killing energy current exactly. equals the ADM `E_hat = sqrt(gamma)(alpha tau + (alpha-1) D -
/// beta^i S_i)` (docs/covariant_energy.md), but `sqrt(-g) T^j_t` is the fully-densitized covariant
/// flux — no alpha/shift/sqrt(gamma) reassembly.
/// `rho` = rest density, `w` = rho h, `v` = valencia contravariant v^i, `p` = pressure.
pub fn coord_energy_cons_flux<S: Scalar, M, const D: usize>(
    g: &M,
    x: Tensor<S, D>,
    rho: S,
    w: S,
    v: Tensor<S, D>,
    p: S,
) -> (S, Tensor<S, D>)
where
    M: Metric<Dual<S>, D>,
{
    let adm = adm_contraction_blocks(g, x);
    let sqrt_neg_g = adm.alpha * adm.sqrt_gamma;
    // W = 1 / sqrt(1 - gamma_ij v^i v^j)
    let v_low = Tensor::<S, D>::from_fn(|ii| adm.gam.row(ii).dot(&v));
    let ww = S::ONE / (S::ONE - v.dot(&v_low)).sqrt();
    // coordinate 4-velocity u^t = W/alpha, u^i = W(v^i - beta^i/alpha)
    let ut = ww / adm.alpha;
    let u_sp = Tensor::<S, D>::from_fn(|ii| ww * (v[ii] - adm.beta[ii] / adm.alpha));
    // u_t = g_tt u^t + beta_j u^j, with g_tt = -alpha^2 + beta^k beta_k and g_tj = beta_j
    let beta_low = Tensor::<S, D>::from_fn(|ii| adm.gam.row(ii).dot(&adm.beta));
    let g_tt = S::ZERO - adm.alpha * adm.alpha + adm.beta.dot(&beta_low);
    let u_t = g_tt * ut + beta_low.dot(&u_sp);
    // T^t_t = w u^t u_t + p; T^j_t = w u^j u_t; positive-convention E_hat = -sqrt(-g)(T^*_t + rho u^*)
    let e_cons = S::ZERO - sqrt_neg_g * (w * ut * u_t + p + rho * ut);
    let flux =
        Tensor::<S, D>::from_fn(|jj| S::ZERO - sqrt_neg_g * (w * u_sp[jj] * u_t + rho * u_sp[jj]));
    (e_cons, flux)
}

#[cfg(test)]
mod ehat_tests {
    use super::*;
    use crate::metric::SchwarzschildKS;

    #[test]
    fn tau_ehat_round_trip_and_flat_limit() {
        let mom = Tensor::<f64, 3>::new([1.7, -0.4, 0.9]);
        let shift = Tensor::<f64, 3>::new([0.44, 0.1, 0.0]);
        let (tau, den, alpha, sg) = (0.55_f64, 2.3, 0.83, 1.9);
        // E_hat -> tau inverts to roundoff (the D and beta.S terms cancel algebraically; only the
        // sqrt(gamma) multiply/divide slips), so the recovered tau feeds the newton unchanged.
        let ehat = tau_to_ehat(tau, den, mom, alpha, shift, sg);
        let rec = ehat_to_tau(ehat, den, mom, alpha, shift, sg);
        assert!(
            (rec - tau).abs() < 1e-14 * tau.abs().max(1.0),
            "round-trip {rec} vs {tau}"
        );
        // flat background: alpha = 1, beta = 0, sqrt(gamma) = 1 -> E_hat == tau exactly.
        assert_eq!(tau_to_ehat(tau, den, mom, 1.0, Tensor::zeros(), 1.0), tau);
    }

    #[test]
    fn ehat_flux_is_conserved_on_the_michel_solution() {
        // on the EXACT michel transonic solution (gamma = 4/3, M = 1) in the ingoing kerr-schild
        // chart, the re-split covariant-energy flux F_Ehat = alpha F_E - beta F_S - F_D is
        // r-invariant to roundoff (= jm*(h_inf-1)), while the eulerian energy flux F_E is not. this
        // is the numerical statement of "E_hat conserves the bernoulli invariant, tau needs a
        // source". points (r, rho, u=|u^r|, p) generated from simbi_configs/.../gr_michel.py.
        const M: f64 = 1.0;
        const G: f64 = 4.0 / 3.0;
        const JM: f64 = 527.9439529572;
        const H_INF: f64 = 1.0400000000;
        let pts: [(f64, f64, f64, f64); 4] = [
            (
                3.0,
                8.779950259276684e1,
                6.681181269277645e-1,
                3.902318750504499e0,
            ),
            (
                6.0,
                3.457351834737535e1,
                4.241717506740386e-1,
                1.126310423270443e0,
            ),
            (
                10.0,
                1.799067745636194e1,
                2.934541816103347e-1,
                4.714077194002101e-1,
            ),
            (
                30.0,
                5.162852963799085e0,
                1.136202011344974e-1,
                8.923180448359856e-2,
            ),
        ];
        let mut f_ehat = Vec::new();
        let mut f_e = Vec::new();
        for (r, rho, u, p) in pts {
            // ingoing kerr-schild ADM for schwarzschild
            let a2 = 2.0 * M / r;
            let alpha = 1.0 / (1.0 + a2).sqrt();
            let beta = 2.0 * M / (r + 2.0 * M);
            let grr = 1.0 + a2;
            let sqrtg = r * r * grr.sqrt();
            // u^r infalling; solve g_tt X^2 + 2 g_tr X u^r + (g_rr u^r^2 + 1) = 0 (future-pointing)
            let ur = -u;
            let (gtt, gtr) = (-(1.0 - 2.0 * M / r), 2.0 * M / r);
            let (aa, bb, cc) = (gtt, 2.0 * gtr * ur, grr * ur * ur + 1.0);
            let disc = bb * bb - 4.0 * aa * cc;
            let mut ut = (-bb - disc.sqrt()) / (2.0 * aa);
            if ut <= 0.0 {
                ut = (-bb + disc.sqrt()) / (2.0 * aa);
            }
            let ww = alpha * ut;
            let vr = (ur / ut + beta) / alpha; // valencia v^r
            let h = 1.0 + G / (G - 1.0) * (p / rho);
            let den = rho * ww;
            let s_cov = rho * h * ww * ww * (grr * vr); // covariant S_r
            let s_up = rho * h * ww * ww * vr;
            let e = rho * h * ww * ww - p; // eulerian energy
            let tau = e - den;
            // densitized fluxes (Andersson & Comer 2021, eqs 11.41 / 11.45 / 11.27)
            let fe = sqrtg * (alpha * s_up - e * beta);
            let srr = p + grr * rho * h * ww * ww * vr * vr;
            let fs = sqrtg * (alpha * srr - s_cov * beta);
            let fd = sqrtg * den * (alpha * vr - beta);
            f_ehat.push(alpha * fe - beta * fs - fd);
            f_e.push(fe);
            // the in-crate primitive reproduces the covariant energy density and inverts.
            let mom = Tensor::<f64, 1>::new([s_cov]);
            let shift = Tensor::<f64, 1>::new([beta]);
            let ehat = tau_to_ehat(tau, den, mom, alpha, shift, sqrtg);
            let rec = ehat_to_tau(ehat, den, mom, alpha, shift, sqrtg);
            assert!(
                (rec - tau).abs() < 1e-12 * tau.abs().max(1.0),
                "round-trip at r={r}"
            );
            // AthenaK/HARM coordinate form (positive convention): -sqrt(-g)(T^t_t + rho u^t) and its
            // flux, computed straight from the primitives via the SchwarzschildKS metric, must agree
            // with the ADM E_hat / F_Ehat above — the two are the same conserved current. evaluated
            // in the full 3d spherical chart at the equator (theta = pi/2, radial flow v^theta =
            // v^phi = 0) so that gam.det() = r^2 sqrt(h) picks up the r^2 the reduced 1d chart drops,
            // matching the manual sqrtg = r^2 sqrt(h).
            let (e_coord, f_coord) = coord_energy_cons_flux(
                &SchwarzschildKS {
                    mass: Dual::constant(M),
                },
                Tensor::<f64, 3>::new([r, std::f64::consts::FRAC_PI_2, 0.0]),
                rho,
                rho * h,
                Tensor::<f64, 3>::new([vr, 0.0, 0.0]),
                p,
            );
            assert!(
                (e_coord - ehat).abs() < 1e-9 * ehat.abs().max(1.0),
                "coord energy vs ADM E_hat at r={r}: {e_coord} vs {ehat}"
            );
            assert!(
                (f_coord[0] - (alpha * fe - beta * fs - fd)).abs() < 1e-9 * fe.abs().max(1.0),
                "coord flux vs ADM F_Ehat at r={r}: {} vs {}",
                f_coord[0],
                alpha * fe - beta * fs - fd
            );
        }
        let mean: f64 = f_ehat.iter().sum::<f64>() / f_ehat.len() as f64;
        let spread = (f_ehat.iter().cloned().fold(f64::MIN, f64::max)
            - f_ehat.iter().cloned().fold(f64::MAX, f64::min))
            / mean.abs();
        assert!(
            spread < 1e-10,
            "F_Ehat not conserved: rel spread {spread:.2e}"
        );
        // sanity: the per-steradian constant equals the analytic killing-minus-baryon flux
        // jm*(h_inf-1) (negative for infall).
        let analytic = -JM * (H_INF - 1.0);
        assert!(
            (mean - analytic).abs() < 1e-6 * analytic.abs(),
            "F_Ehat {mean} vs {analytic}"
        );
        // and the eulerian flux is NOT conserved (varies O(1)).
        let e_mean: f64 = f_e.iter().sum::<f64>() / f_e.len() as f64;
        let e_spread = (f_e.iter().cloned().fold(f64::MIN, f64::max)
            - f_e.iter().cloned().fold(f64::MAX, f64::min))
            / e_mean.abs();
        assert!(
            e_spread > 1e-2,
            "eulerian flux unexpectedly flat: {e_spread:.2e}"
        );
    }
}

/// the full GRMHD covariant valencia source — [`grhd_covariant_source`] with the ideal-MHD
/// stress `T^{mu nu} = (rho h + b^2) u^mu u^nu + (p + b^2/2) g^{mu nu} - b^mu b^nu` in the SAME
/// per-axis contraction — the electromagnetic stress enters only through `T`. the caller
/// supplies the METRIC-FREE rest enthalpy density `rho_h = rho + Gamma/(Gamma-1) p`, the
/// contravariant valencia `v^i`, the isotropic-block pressure `p`, and the contravariant
/// eulerian field `B^i`; the lorentz factor and the magnetic four-vector assemble in here from
/// the harvested metric:
///   alpha b^t = W (v.B),   b^i = B^i/W + alpha b^t uhat^i,   b^2 = B^2/W^2 + (v.B)^2,
///   (rho h + b^2) u^mu u^nu = (rho_h + b^2) W^2 uhat^mu uhat^nu.
/// B = 0 reduces exactly to the hydro contraction at e = rho_h W^2. axisymmetry still zeroes
/// the suppressed-slot momentum source (the metric never reads phi), B or not —
/// angular-momentum conservation survives magnetization.
pub fn grmhd_covariant_source<S: Scalar, M, const D: usize>(
    g: &M,
    x: Tensor<S, D>,
    rho_h: S,
    v: Tensor<S, D>,
    p: S,
    bfield: Tensor<S, D>,
) -> (Tensor<S, D>, S)
where
    M: Metric<Dual<S>, D>,
{
    let half = S::from_f64(0.5);
    let adm = adm_contraction_blocks(g, x);
    let mut uhat = [S::ZERO; 4];
    uhat[0] = S::ONE / adm.alpha;
    for ii in 0..D {
        uhat[ii + 1] = v[ii] - adm.beta[ii] / adm.alpha;
    }
    // frame-local magnetic invariants with the harvested gamma.
    let v_low = Tensor::<S, D>::from_fn(|ii| adm.gam.row(ii).dot(&v));
    let b_low = Tensor::<S, D>::from_fn(|ii| adm.gam.row(ii).dot(&bfield));
    let v_sq = v_low.dot(&v);
    let w_sq = S::ONE / (S::ONE - v_sq);
    let ww = w_sq.sqrt();
    let vdb = v_low.dot(&bfield);
    let bsq = b_low.dot(&bfield);
    let b_mu_sq = bsq / w_sq + vdb * vdb;
    // the magnetic four-vector in uhat components: b^t = W (v.B)/alpha = wvb uhat^t,
    // b^i = B^i/W + wvb uhat^i.
    let wvb = ww * vdb;
    let mut b4 = [S::ZERO; 4];
    b4[0] = wvb * uhat[0];
    for ii in 0..D {
        b4[ii + 1] = bfield[ii] / ww + wvb * uhat[ii + 1];
    }
    let inertia = (rho_h + b_mu_sq) * w_sq;
    let p_tot = p + half * b_mu_sq;
    let mut t4 = [[S::ZERO; 4]; 4];
    for mm in 0..=D {
        for nn in 0..=D {
            t4[mm][nn] = inertia * uhat[mm] * uhat[nn] + p_tot * adm.gi4[mm][nn] - b4[mm] * b4[nn];
        }
    }
    contract_stress(&adm, &t4)
}

/// the harvested ADM blocks + derivatives a covariant-source contraction consumes: the 4-metric
/// coordinate derivatives `dg4[kk] = d_kk g_{mu nu}` (one Dual pass per axis on a STATIC
/// background), the inverse 4-metric `gi4`, and the point values of the 3+1 split.
struct AdmContractionBlocks<S: Scalar, const D: usize> {
    alpha: S,
    beta: Tensor<S, D>,
    gam: Matrix<S, D>,
    sqrt_gamma: S,
    d_alpha: [S; D],
    dg4: [[[S; 4]; 4]; 3],
    gi4: [[S; 4]; 4],
}

fn adm_contraction_blocks<S: Scalar, M, const D: usize>(
    g: &M,
    x: Tensor<S, D>,
) -> AdmContractionBlocks<S, D>
where
    M: Metric<Dual<S>, D>,
{
    debug_assert!(
        D <= 3,
        "the padded 4-metric blocks hold at most 3 spatial axes"
    );

    // one dual pass per axis: seed x_kk, harvest the ADM block and its d_kk. the values are
    // identical across passes; take them from the first.
    let mut alpha = S::ZERO;
    let mut beta = Tensor::<S, D>::zeros();
    let mut gam = Matrix::<S, D>::zeros();
    let mut sqrt_gamma = S::ZERO;
    let mut gam_inv = Matrix::<S, D>::zeros();
    let mut d_alpha = [S::ZERO; D];
    let mut d_beta = [[S::ZERO; D]; D]; // d_beta[kk][ii] = d_kk beta^ii
    let mut d_gam: Vec<Matrix<S, D>> = Vec::with_capacity(D);
    for kk in 0..D {
        let xd = Tensor::<Dual<S>, D>::from_fn(|ii| {
            if ii == kk {
                Dual::variable(x[ii])
            } else {
                Dual::constant(x[ii])
            }
        });
        let a = g.lapse(xd);
        let b = g.shift(xd);
        let gm = g.spatial_metric(xd);
        if kk == 0 {
            alpha = a.value;
            beta = Tensor::from_fn(|ii| b[ii].value);
            gam = Matrix::from_fn(|ii, jj| gm[(ii, jj)].value);
            sqrt_gamma = g.sqrt_det_gamma(xd).value;
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

    AdmContractionBlocks {
        alpha,
        beta,
        gam,
        sqrt_gamma,
        d_alpha,
        dg4,
        gi4,
    }
}

/// contract a stress tensor against the harvested metric derivatives:
///   S_j   = (1/2) T^{mu nu} d_j g_{mu nu}                       (all blocks, per coordinate j)
///   S_tau = alpha ( T^{t j} d_j ln(alpha) - T^{mu nu} Gamma^t_{mu nu} )
fn contract_stress<S: Scalar, const D: usize>(
    adm: &AdmContractionBlocks<S, D>,
    t4: &[[S; 4]; 4],
) -> (Tensor<S, D>, S) {
    let half = S::from_f64(0.5);
    let (alpha, d_alpha, dg4, gi4) = (adm.alpha, &adm.d_alpha, &adm.dg4, &adm.gi4);

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
        if mu == 0 {
            S::ZERO
        } else {
            dg4[mu - 1][aa][bb]
        }
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

    #[test]
    fn cartesian_ks_covariant_source_is_x_y_symmetric() {
        // the cartesian kerr-schild metric is exactly symmetric under the x <-> y coordinate +
        // index swap, so the geodesic source must satisfy S_x(x, y) = S_y(y, x) (and S_tau
        // symmetric). a resolution-independent x <-> y asymmetry in a cartesian GR run would show
        // up here if the D-generic contraction dropped an axis.
        use crate::metric::SchwarzschildKSCartesian;
        let m = SchwarzschildKSCartesian {
            mass: Dual::constant(1.0_f64),
        };
        let (px, py, e, p) = (3.0_f64, 5.0, 1.5, 0.1);
        let (s1, tau1) = grhd_covariant_source(
            &m,
            Tensor::new([px, py, 0.0]),
            e,
            Tensor::new([0.1, 0.2, 0.0]),
            p,
        );
        let (s2, tau2) = grhd_covariant_source(
            &m,
            Tensor::new([py, px, 0.0]),
            e,
            Tensor::new([0.2, 0.1, 0.0]),
            p,
        );
        assert!(
            (s1[0] - s2[1]).abs() < 1e-12,
            "S_x(x,y) = {} != S_y(y,x) = {}",
            s1[0],
            s2[1]
        );
        assert!(
            (s1[1] - s2[0]).abs() < 1e-12,
            "S_y(x,y) = {} != S_x(y,x) = {}",
            s1[1],
            s2[0]
        );
        assert!(
            s1[2].abs() < 1e-12 && s2[2].abs() < 1e-12,
            "z-momentum source nonzero on the slice"
        );
        assert!(
            (tau1 - tau2).abs() < 1e-12,
            "S_tau asymmetric: {tau1} vs {tau2}"
        );
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
    fn schwarzschild_ks_adm(r: f64, m: f64) -> (f64, f64, f64, AdmRadialDerivs<f64>) {
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
            let (s_mom, s_tau) = grhd_radial_geodesic_source(
                r,
                a,
                br,
                grr,
                d.d_lapse,
                d.d_shift_r,
                d.d_gamma_rr,
                e,
                big_v,
                p,
            );
            let f = 1.0 - 2.0 * m / r;
            // schwarzschild closed forms: gravity part only for momentum.
            let s_mom_cf = -m * e * (1.0 + big_v * big_v) / (r * r * f);
            let s_tau_cf = -a * e * big_v * m / (r * r * f);
            assert!(approx(s_mom, s_mom_cf), "S_Sr r={r}: {s_mom} != {s_mom_cf}");
            assert!(
                approx(s_tau, s_tau_cf),
                "S_tau r={r}: {s_tau} != {s_tau_cf}"
            );
        }
    }

    // the covariant contraction at the metric's full D = 3, radial flow: the radial momentum
    // source must equal the (t, r)-block source PLUS the angular pressure blocks
    // (1/2)(T^{theta theta} d_r g_{theta theta} + T^{phi phi} d_r g_{phi phi}) = 2p/r; the
    // polar source is the pressure term p cot(theta); the azimuthal source vanishes
    // (axisymmetry); the energy source equals the (t, r)-block source (which carries the angular
    // Gamma^t blocks already).
    #[test]
    fn covariant_source_radial_flow_matches_oracle_plus_angular_blocks() {
        use crate::metric::SchwarzschildKS;
        let m = 1.0;
        let (e, big_v, p) = (2.3_f64, -0.4_f64, 0.05_f64);
        let theta = 1.1_f64;
        for &r in &[8.0_f64, 3.0, 2.5] {
            {
                let (a, br, grr, d) = schwarzschild_ks_adm(r, m);
                let (mom_cf, tau_cf) = grhd_radial_geodesic_source(
                    r,
                    a,
                    br,
                    grr,
                    d.d_lapse,
                    d.d_shift_r,
                    d.d_gamma_rr,
                    e,
                    big_v,
                    p,
                );
                let vr = big_v / grr.sqrt(); // contravariant v^r from the orthonormal V
                let x = Tensor::new([r, theta, 0.0]);
                let v = Tensor::new([vr, 0.0, 0.0]);
                let g = SchwarzschildKS {
                    mass: Dual::constant(m),
                };
                let (s, s_tau) = grhd_covariant_source(&g, x, e, v, p);
                assert!(
                    approx(s[0], mom_cf + 2.0 * p / r),
                    "S_r r={r}: {} != {}",
                    s[0],
                    mom_cf + 2.0 * p / r
                );
                assert!(
                    approx(s[1], p * theta.cos() / theta.sin()),
                    "S_theta r={r}: {}",
                    s[1]
                );
                assert!(
                    s[2].abs() < 1e-14,
                    "S_phi must vanish (axisymmetry): {}",
                    s[2]
                );
                assert!(
                    approx(s_tau, tau_cf),
                    "S_tau r={r}: {s_tau} != {tau_cf}"
                );
            }
        }
    }

    // rotating (swirl) flow off the equator. the MOMENTUM closed forms follow from the diagonal
    // spatial metric — S_theta = (E (v^phi)^2 + p g^{phi phi}) r^2 sin(theta) cos(theta), S_r gains
    // the azimuthal centrifugal block E (v^phi)^2 r sin^2(theta), and S_phi vanishes by axisymmetry.
    //
    // the ENERGY source needs one term the radial-block oracle cannot supply. that oracle carries
    // the angular blocks at ISOTROPIC PRESSURE only, so the fluid's angular kinetic blocks are
    // missing from `T^{mu nu} Gamma^t_{mu nu}`. on this chart
    //
    //     Gamma^t_{kk} = -(1/2) g^{tr} d_r g_{kk},        g^{tr} = beta^r / alpha^2
    //
    // (every other term of Gamma^t_{kk} carries a time or azimuthal derivative of a static,
    // axisymmetric metric), so each angular slot contributes `-alpha E (v^k)^2 Gamma^t_{kk}`:
    //
    //     dS_tau = (beta^r / alpha) E [ (v^theta)^2 r + (v^phi)^2 r sin^2(theta) ]
    //
    // which vanishes identically when beta^r does. that is why a static chart's energy source reads
    // off the radial block alone, and why a horizon-penetrating one does not — the shift is exactly
    // the coupling.
    #[test]
    fn covariant_source_swirl_closed_form_including_the_shifted_energy_coupling() {
        use crate::metric::SchwarzschildKS;
        let m = 1.0;
        let (e, p) = (2.3_f64, 0.05_f64);
        let (r, theta) = (6.0_f64, 1.1_f64);
        let (vr, vphi) = (-0.1_f64, 0.02_f64);
        let (st, ct) = (theta.sin(), theta.cos());

        let g = SchwarzschildKS {
            mass: Dual::constant(m),
        };
        let (s, s_tau) = grhd_covariant_source(
            &g,
            Tensor::new([r, theta, 0.0]),
            e,
            Tensor::new([vr, 0.0, vphi]),
            p,
        );

        let (a, br, grr, d) = schwarzschild_ks_adm(r, m);
        let big_v = vr * grr.sqrt();
        let (mom_cf, tau_cf) = grhd_radial_geodesic_source(
            r,
            a,
            br,
            grr,
            d.d_lapse,
            d.d_shift_r,
            d.d_gamma_rr,
            e,
            big_v,
            p,
        );
        let s_r_cf = mom_cf + 2.0 * p / r + e * vphi * vphi * r * st * st;
        let s_th_cf = (e * vphi * vphi + p / (r * r * st * st)) * r * r * st * ct;
        assert!(approx(s[0], s_r_cf), "S_r: {} != {s_r_cf}", s[0]);
        assert!(approx(s[1], s_th_cf), "S_theta: {} != {s_th_cf}", s[1]);
        assert!(s[2].abs() < 1e-14, "S_phi must vanish: {}", s[2]);
        // the shift-coupled angular kinetic block, derived above. v^theta = 0 here, so only the
        // azimuthal slot contributes.
        let beta_r = br;
        let tau_swirl = tau_cf + (beta_r / a) * e * vphi * vphi * r * st * st;
        assert!(
            approx(s_tau, tau_swirl),
            "S_tau: {s_tau} != {tau_swirl} (radial block {tau_cf} + azimuthal coupling {})",
            tau_swirl - tau_cf
        );
    }

    // the same coupling with BOTH angular slots carrying fluid motion, so neither term of
    // `dS_tau = (beta^r/alpha) E [ (v^theta)^2 r + (v^phi)^2 r sin^2(theta) ]` can hide the other.
    // asserts the ENERGY source only: a nonzero v^theta also moves S_r and S_theta, whose closed
    // forms the swirl test above already pins at v^theta = 0.
    #[test]
    fn shifted_chart_energy_source_carries_both_angular_kinetic_blocks() {
        use crate::metric::SchwarzschildKS;
        let m = 1.0;
        let (e, p) = (2.3_f64, 0.05_f64);
        let (r, theta) = (6.0_f64, 1.1_f64);
        let (vr, vth, vphi) = (-0.1_f64, 0.04_f64, 0.02_f64);
        let st = theta.sin();

        let g = SchwarzschildKS {
            mass: Dual::constant(m),
        };
        let (_, s_tau) = grhd_covariant_source(
            &g,
            Tensor::new([r, theta, 0.0]),
            e,
            Tensor::new([vr, vth, vphi]),
            p,
        );

        let (a, br, grr, d) = schwarzschild_ks_adm(r, m);
        let (_, tau_cf) = grhd_radial_geodesic_source(
            r,
            a,
            br,
            grr,
            d.d_lapse,
            d.d_shift_r,
            d.d_gamma_rr,
            e,
            vr * grr.sqrt(),
            p,
        );
        let want = tau_cf + (br / a) * e * (vth * vth * r + vphi * vphi * r * st * st);
        assert!(
            approx(s_tau, want),
            "S_tau with both angular slots moving: {s_tau} != {want}"
        );
        // the premise: the coupling must be RESOLVED here, not lost in the radial block's roundoff.
        assert!(
            (want - tau_cf).abs() > 1e-6 * tau_cf.abs().max(1.0),
            "the angular kinetic coupling is negligible at these parameters ({}); the gate would \
             pass with the term omitted",
            want - tau_cf
        );
    }

    // the M = 0 limit is flat spherical: the contraction must reproduce the covariant
    // curvilinear inertial + pressure source with NO gravity and ZERO energy source.
    #[test]
    fn covariant_source_flat_limit_is_curvilinear_inertial() {
        use crate::metric::SchwarzschildKS;
        let (e, p) = (2.3_f64, 0.05_f64);
        let (r, theta) = (6.0_f64, 1.1_f64);
        let (vr, vth, vphi) = (-0.1_f64, 0.03_f64, 0.02_f64);
        let (st, ct) = (theta.sin(), theta.cos());

        let g = SchwarzschildKS {
            mass: Dual::constant(0.0),
        };
        let (s, s_tau) = grhd_covariant_source(
            &g,
            Tensor::new([r, theta, 0.0]),
            e,
            Tensor::new([vr, vth, vphi]),
            p,
        );

        let s_r_cf = e * (vth * vth * r + vphi * vphi * r * st * st) + 2.0 * p / r;
        let s_th_cf = (e * vphi * vphi + p / (r * r * st * st)) * r * r * st * ct;
        assert!(approx(s[0], s_r_cf), "flat S_r: {} != {s_r_cf}", s[0]);
        assert!(approx(s[1], s_th_cf), "flat S_theta: {} != {s_th_cf}", s[1]);
        assert!(
            s[2].abs() < 1e-14 && s_tau.abs() < 1e-14,
            "flat gravity must vanish"
        );
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
            let g = KerrKS {
                mass: 1.0_f64,
                spin: a,
            };
            let gd = KerrKS {
                mass: Dual::constant(1.0_f64),
                spin: Dual::constant(a),
            };
            let (e, p) = (2.3_f64, 0.05_f64);
            for &(r, th) in &[(1.4_f64, 1.2_f64), (2.5, 0.8), (8.0, 1.9)] {
                let x = Tensor::new([r, th, 0.0]);
                let v = Tensor::new([-0.2_f64, 0.03, 0.02]);
                let (s_ad, tau_ad) = grhd_covariant_source(&gd, x, e, v, p);
                let (s_fd, tau_fd) = covariant_source_fd(&g, x, e, v, p);
                for kk in 0..3 {
                    assert!(
                        fd_close(s_ad[kk], s_fd[kk], e),
                        "S_mom[{kk}] a={a} r={r}: {} vs {}",
                        s_ad[kk],
                        s_fd[kk]
                    );
                }
                assert!(
                    fd_close(tau_ad, tau_fd, e),
                    "S_tau a={a} r={r}: {tau_ad} vs {tau_fd}"
                );
            }
        }
    }

    #[test]
    fn covariant_source_on_kerr_at_zero_spin_matches_schwarzschild_ks() {
        use crate::metric::{KerrKS, SchwarzschildKS};
        // a = 0 kerr must reproduce the schwarzschild-KS source values for arbitrary
        // rotating states (different expression paths, same physics).
        let close = |x: f64, y: f64| (x - y).abs() < 1e-11 * (1.0 + x.abs().max(y.abs()));
        let kerr = KerrKS {
            mass: Dual::constant(1.0_f64),
            spin: Dual::constant(0.0),
        };
        let ks = SchwarzschildKS {
            mass: Dual::constant(1.0_f64),
        };
        let (e, p) = (2.3_f64, 0.05_f64);
        for &(r, th) in &[(1.5_f64, 1.0_f64), (3.0, 1.9), (12.0, 1.3)] {
            let x = Tensor::new([r, th, 0.0]);
            let v = Tensor::new([-0.3_f64, 0.02, 0.015]);
            let (sk, tk) = grhd_covariant_source(&kerr, x, e, v, p);
            let (ss, ts) = grhd_covariant_source(&ks, x, e, v, p);
            for kk in 0..3 {
                assert!(
                    close(sk[kk], ss[kk]),
                    "S_mom[{kk}] r={r}: {} vs {}",
                    sk[kk],
                    ss[kk]
                );
            }
            assert!(close(tk, ts), "S_tau r={r}: {tk} vs {ts}");
        }
    }

    #[test]
    fn generic_source_matches_kerr_schild_closed_form() {
        let m = 1.0;
        let (e, big_v, p) = (2.3, -0.4, 0.05);
        for &r in &[8.0, 3.0, 2.0, 1.5, 1.2] {
            let (a, br, grr, d) = schwarzschild_ks_adm(r, m);
            let (s_mom, s_tau) = grhd_radial_geodesic_source(
                r,
                a,
                br,
                grr,
                d.d_lapse,
                d.d_shift_r,
                d.d_gamma_rr,
                e,
                big_v,
                p,
            );
            let b = 2.0 * m / r;
            let h = 1.0 + b;
            let s_mom_cf = -m * e * (1.0 + big_v).powi(2) / (r * r * h);
            let s_tau_cf = -m / (r * r * h.powf(1.5))
                * (e * big_v * (1.0 + (2.0 + b) * big_v) - p * (2.0 + 3.0 * b));
            assert!(approx(s_mom, s_mom_cf), "S_Sr r={r}: {s_mom} != {s_mom_cf}");
            assert!(
                approx(s_tau, s_tau_cf),
                "S_tau r={r}: {s_tau} != {s_tau_cf}"
            );
        }
    }

    // finite-difference mirror with the FULL ideal-MHD stress: the same T^{mu nu} =
    // (rho h + b^2) u u + (p + b^2/2) g^{-1} - b b algebra with every metric derivative
    // numerical — validates the b^mu assembly + contraction through the non-diagonal
    // theta-dependent kerr metric.
    fn covariant_source_fd_mhd<M: crate::metric::Metric<f64, 3>>(
        g: &M,
        x: Tensor<f64, 3>,
        rho_h: f64,
        v: Tensor<f64, 3>,
        p: f64,
        bfield: Tensor<f64, 3>,
    ) -> (Tensor<f64, 3>, f64) {
        let eps = 1e-6;
        let alpha = g.lapse(x);
        let beta = g.shift(x);
        let gm = g.spatial_metric(x);
        let gi = g.spatial_metric_inv(x);
        let mut uhat = [0.0; 4];
        uhat[0] = 1.0 / alpha;
        for ii in 0..3 {
            uhat[ii + 1] = v[ii] - beta[ii] / alpha;
        }
        let dot_g = |a: &Tensor<f64, 3>, b: &Tensor<f64, 3>| -> f64 {
            (0..3)
                .map(|ii| (0..3).map(|jj| gm[(ii, jj)] * a[ii] * b[jj]).sum::<f64>())
                .sum()
        };
        let v_sq = dot_g(&v, &v);
        let w_sq = 1.0 / (1.0 - v_sq);
        let ww = w_sq.sqrt();
        let vdb = dot_g(&v, &bfield);
        let bsq = dot_g(&bfield, &bfield);
        let b_mu_sq = bsq / w_sq + vdb * vdb;
        let wvb = ww * vdb;
        let mut b4 = [0.0; 4];
        b4[0] = wvb / alpha;
        for ii in 0..3 {
            b4[ii + 1] = bfield[ii] / ww + wvb * uhat[ii + 1];
        }
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
        let inertia = (rho_h + b_mu_sq) * w_sq;
        let p_tot = p + 0.5 * b_mu_sq;
        let mut t4 = [[0.0; 4]; 4];
        for mm in 0..4 {
            for nn in 0..4 {
                t4[mm][nn] = inertia * uhat[mm] * uhat[nn] + p_tot * gi4[mm][nn] - b4[mm] * b4[nn];
            }
        }
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
        let s_mom = Tensor::<f64, 3>::from_fn(|kk| {
            let mut s = 0.0;
            for mm in 0..4 {
                for nn in 0..4 {
                    s += t4[mm][nn] * dg[kk][mm][nn];
                }
            }
            0.5 * s
        });
        let d_alpha: [f64; 3] = std::array::from_fn(|kk| {
            if kk == 2 {
                return 0.0;
            }
            let mut xp = x;
            let mut xm = x;
            xp[kk] += eps;
            xm[kk] -= eps;
            (g.lapse(xp) - g.lapse(xm)) / (2.0 * eps)
        });
        let dgf = |mu: usize, aa: usize, bb: usize| -> f64 {
            if mu == 0 { 0.0 } else { dg[mu - 1][aa][bb] }
        };
        let mut t_gamma = 0.0;
        for mm in 0..4 {
            for nn in 0..4 {
                let mut gt = 0.0;
                for ss in 0..4 {
                    gt += gi4[0][ss] * (dgf(mm, ss, nn) + dgf(nn, ss, mm) - dgf(ss, mm, nn));
                }
                t_gamma += t4[mm][nn] * 0.5 * gt;
            }
        }
        let mut work = 0.0;
        for kk in 0..3 {
            work += t4[0][kk + 1] * d_alpha[kk] / alpha;
        }
        (s_mom, alpha * (work - t_gamma))
    }

    #[test]
    fn grmhd_source_reduces_to_hydro_at_zero_field() {
        use crate::metric::{KerrKS, SchwarzschildKS};
        // B = 0 zeroes every magnetic invariant, so the MHD contraction must equal the
        // hydro one to roundoff (the T4 assembly differs only by exact-zero terms).
        let close = |x: f64, y: f64, s: f64| (x - y).abs() < 1e-12 * (1.0 + s.abs());
        let (rho_h, p) = (2.3_f64, 0.05_f64);
        let v = Tensor::new([-0.2_f64, 0.03, 0.02]);
        let b0 = Tensor::new([0.0_f64, 0.0, 0.0]);
        let ks_f = SchwarzschildKS { mass: 1.0_f64 };
        let kerr_f = KerrKS {
            mass: 1.0_f64,
            spin: 0.7_f64,
        };
        let ks = SchwarzschildKS {
            mass: Dual::constant(1.0_f64),
        };
        let kerr = KerrKS {
            mass: Dual::constant(1.0_f64),
            spin: Dual::constant(0.7_f64),
        };
        // the hydro call takes e = rho_h W^2 (the same inertia the mhd call builds internally).
        let w_sq = |gm: symbi_algebra::Matrix<f64, 3>| -> f64 {
            let v_sq: f64 = (0..3)
                .map(|ii| (0..3).map(|jj| gm[(ii, jj)] * v[ii] * v[jj]).sum::<f64>())
                .sum();
            1.0 / (1.0 - v_sq)
        };
        for &(r, th) in &[(2.5_f64, 0.8_f64), (8.0, 1.9)] {
            let x = Tensor::new([r, th, 0.0]);
            let e_ks = rho_h * w_sq(crate::metric::Metric::<f64, 3>::spatial_metric(&ks_f, x));
            let (sh, th_h) = grhd_covariant_source(&ks, x, e_ks, v, p);
            let (sm, th_m) = grmhd_covariant_source(&ks, x, rho_h, v, p, b0);
            for kk in 0..3 {
                assert!(
                    close(sh[kk], sm[kk], e_ks),
                    "ks S[{kk}] r={r}: {} vs {}",
                    sh[kk],
                    sm[kk]
                );
            }
            assert!(close(th_h, th_m, e_ks), "ks S_tau r={r}");
            let e_kerr = rho_h * w_sq(crate::metric::Metric::<f64, 3>::spatial_metric(&kerr_f, x));
            let (sh, th_h) = grhd_covariant_source(&kerr, x, e_kerr, v, p);
            let (sm, th_m) = grmhd_covariant_source(&kerr, x, rho_h, v, p, b0);
            for kk in 0..3 {
                assert!(
                    close(sh[kk], sm[kk], e_kerr),
                    "kerr S[{kk}] r={r}: {} vs {}",
                    sh[kk],
                    sm[kk]
                );
            }
            assert!(close(th_h, th_m, e_kerr), "kerr S_tau r={r}");
        }
    }

    #[test]
    fn grmhd_source_on_kerr_matches_finite_differences() {
        use crate::metric::KerrKS;
        // a rotating magnetized state with all velocity AND field components, off the
        // equator, inside and outside the horizon — the autodiff contraction with the
        // b^mu assembly must agree with the finite-difference mirror.
        let fd_close = |x: f64, y: f64, s: f64| (x - y).abs() < 1e-7 * (1.0 + s.abs());
        for &a in &[0.9_f64, -0.5] {
            let g = KerrKS {
                mass: 1.0_f64,
                spin: a,
            };
            let gd = KerrKS {
                mass: Dual::constant(1.0_f64),
                spin: Dual::constant(a),
            };
            let (rho_h, p) = (2.3_f64, 0.05_f64);
            let v = Tensor::new([-0.2_f64, 0.03, 0.02]);
            let b = Tensor::new([0.4_f64, -0.15, 0.08]);
            for &(r, th) in &[(1.4_f64, 1.2_f64), (2.5, 0.8), (8.0, 1.9)] {
                let x = Tensor::new([r, th, 0.0]);
                let (s_ad, tau_ad) = grmhd_covariant_source(&gd, x, rho_h, v, p, b);
                let (s_fd, tau_fd) = covariant_source_fd_mhd(&g, x, rho_h, v, p, b);
                for kk in 0..3 {
                    assert!(
                        fd_close(s_ad[kk], s_fd[kk], rho_h),
                        "S_mom[{kk}] a={a} r={r}: {} vs {}",
                        s_ad[kk],
                        s_fd[kk]
                    );
                }
                assert!(
                    fd_close(tau_ad, tau_fd, rho_h),
                    "S_tau a={a} r={r}: {tau_ad} vs {tau_fd}"
                );
            }
        }
    }

    #[test]
    fn grmhd_axisymmetric_azimuthal_source_vanishes_with_field() {
        use crate::metric::KerrKS;
        // the metric never reads phi, so the suppressed-slot momentum source is zero by
        // construction — magnetization does not break angular-momentum conservation.
        let gd = KerrKS {
            mass: Dual::constant(1.0_f64),
            spin: Dual::constant(0.9_f64),
        };
        let v = Tensor::new([-0.2_f64, 0.03, 0.02]);
        let b = Tensor::new([0.4_f64, -0.15, 0.08]);
        for &(r, th) in &[(2.5_f64, 0.8_f64), (8.0, 1.9)] {
            let x = Tensor::new([r, th, 0.0]);
            let (s, _) = grmhd_covariant_source(&gd, x, 2.3, v, 0.05, b);
            assert!(s[2].abs() < 1e-14, "S_phi source with B: {}", s[2]);
        }
    }
}
