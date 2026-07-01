// =============================================================================
// rhd/wave_speeds.rs
//
// the RHD relativistic acoustic wave speeds (Mignone & Bodo 2005). the core
// `rhd_speeds_from_vn` is a pure function of (cs^2, normal velocity) — the SINGLE
// source both the nhat Riemann projection and the CFL axis projection call.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;
use crate::eos::Eos;
use crate::state::Prim;
use crate::rhd::sound_speed_sq;

/// the Mignone & Bodo (2005) relativistic acoustic wave speeds (eqs. 21-23) as a function of
/// the sound speed squared and the NORMAL velocity `vn` — the single core both projections
/// call: the nhat Riemann form (`vn = vel . nhat`) and the CFL axis form (`vn = vel[axis]`).
/// accounts for the relativistic dispersion relation; tighter than davis.
#[inline]
pub(crate) fn rhd_speeds_from_vn<S: Scalar>(cs_sq: S, vn: S) -> (S, S) {
    let vn_sq = vn * vn;
    let w_sq = S::ONE / (S::ONE - vn_sq); // W^2
    let ss = cs_sq / (w_sq * (S::ONE - cs_sq));
    let qf = S::ONE / (S::ONE + ss);
    let fac = (ss * (S::ONE - vn_sq + ss)).sqrt();
    ((vn - fac) * qf, (vn + fac) * qf)
}

/// the Banyuls-Font (1997) COORDINATE-frame acoustic speeds on a STATIC DIAGONAL metric (shift
/// beta = 0): the RHD dispersion relation with the inverse-metric normal component `gamma_nn`
/// threaded INTO the discriminant and the lapse `alpha` scaling the result:
///   `disc = (1 - vn^2)( gamma_nn(1 - vn^2 cs^2) - vn^2(1 - cs^2) )`
///   `lambda_pm = alpha [ vn(1 - cs^2) +/- cs sqrt(disc) ] / (1 - vn^2 cs^2)`.
/// at `gamma_nn = 1, alpha = 1` it reduces EXACTLY (in value) to `rhd_speeds_from_vn` — the flat
/// limit — so Minkowski is unchanged. the curved `gamma_nn`/`alpha` come from the spacetime metric
/// (Schwarzschild: `gamma_nn = f = 1-2M/r`, `alpha = sqrt(f)`). drives the GR CFL wave-speed map;
/// `gamma_nn` inside the radical (NOT a post-multiply by alpha) is what pins the sonic point.
///
/// the CFL wave-speed map uses the algebraically-EQUIVALENT factored form (the Schwarzschild
/// `gamma^{rr}=alpha^2` identity collapses this to `alpha^2 * lambda_SR` radial / `alpha` angular,
/// so the map reuses the SR speed + a per-axis correction). this canonical BF form is kept as the
/// verified reference (its unit test pins the reduction + the Schwarzschild values) and is the
/// general-metric path for the B.6 Riemann coordinate speeds.
#[allow(dead_code)]
#[inline]
pub(crate) fn rhd_speeds_from_vn_gr<S: Scalar>(cs_sq: S, vn: S, gamma_nn: S, alpha: S) -> (S, S) {
    let vn_sq = vn * vn;
    let one_m_v2cs2 = S::ONE - vn_sq * cs_sq;
    let disc = (S::ONE - vn_sq) * (gamma_nn * one_m_v2cs2 - vn_sq * (S::ONE - cs_sq));
    let term = vn * (S::ONE - cs_sq);
    let rad = cs_sq.sqrt() * disc.sqrt(); // cs * sqrt(disc)
    let inv = alpha / one_m_v2cs2;
    ((term - rad) * inv, (term + rad) * inv)
}

/// davis wave speed estimates for RHD (simpler, less tight bounds).
/// kept as a reference but not used — the Regime impl uses the Mignone-Bodo speeds.
#[inline]
#[allow(dead_code)]
fn davis_wave_speeds_reference<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    prim: &Prim<S, D>,
    nhat: &Tensor<S, D>,
) -> (S, S) {
    let vn = prim.vel.dot(nhat);
    let cs = sound_speed_sq(eos, prim.rho, prim.pre).sqrt();
    let sl = (vn - cs) / (S::ONE - cs * vn);
    let sr = (vn + cs) / (S::ONE + cs * vn);
    (sl, sr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhd::Rhd;
    use crate::regime::Regime;
    use crate::eos::IdealGas;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn wave_speeds_stationary() {
        // v=0: symmetric wave speeds
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
        let (sl, sr) = Rhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
        assert!(approx(sl, -sr));
        assert!(sr > 0.0);
        assert!(sr < 1.0); // subluminal
    }

    #[test]
    fn wave_speeds_subluminal() {
        // wave speeds must be in (-1, 1) for any physical state
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        for &v in &[0.0, 0.3, 0.5, 0.9, 0.99] {
            for &pre in &[0.01, 1.0, 100.0] {
                let prim = Prim { rho: 1.0, vel: Tensor::new([v]), pre };
                let (sl, sr) = Rhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
                assert!(sl > -1.0, "sl={} at v={}, p={}", sl, v, pre);
                assert!(sr < 1.0, "sr={} at v={}, p={}", sr, v, pre);
                assert!(sl < sr, "sl={} >= sr={} at v={}, p={}", sl, sr, v, pre);
            }
        }
    }

    #[test]
    fn kerr_schild_coordinate_speed_is_ingoing_at_and_inside_horizon() {
        // the horizon-penetrating CFL guarantee. the kerr-schild coordinate wave speed is the
        // factored form the CFL map uses, lambda_coord = alpha^2 * lambda^SR - beta^r, with
        // alpha^2 = 1/(1 + 2M/r) and beta^r = 2M/(r + 2M) (M = 1). BOTH characteristic roots are
        // strictly < 0 at and inside the horizon r <= 2M for EVERY physical fluid state, so the
        // numerical domain of dependence is entirely interior-directed and the inner outflow
        // boundary is causal. |lambda^SR| < 1 subluminal => alpha^2 |lambda^SR| < alpha^2 = beta^r
        // at the horizon, so lambda_coord < 0.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        for &r in &[2.0_f64, 1.7, 1.2, 1.0] {
            let alpha_sq = 1.0 / (1.0 + 2.0 / r);
            let beta_r = 2.0 / (r + 2.0);
            for &v in &[-0.99_f64, -0.5, 0.0, 0.5, 0.99] {
                for &pre in &[0.01, 1.0, 100.0] {
                    let prim = Prim { rho: 1.0, vel: Tensor::new([v]), pre };
                    let (sl, sr) = Rhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
                    let (ll, lr) = (alpha_sq * sl - beta_r, alpha_sq * sr - beta_r);
                    assert!(ll < 0.0 && lr < 0.0,
                        "coord speed not ingoing at r={r}, v={v}, p={pre}: ll={ll}, lr={lr}");
                }
            }
        }
    }

    #[test]
    fn davis_matches_mb_at_zero_v() {
        // at v=0, davis and mignone-bodo should give same result
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
        let (sl_mb, sr_mb) = Rhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
        // davis formula gives same result at v=0
        let cs_sq: f64 = sound_speed_sq(&eos, 1.0, 1.0);
        let cs = cs_sq.sqrt();
        let sl_d = -cs;
        let sr_d = cs;
        // both should give +/- cs
        assert!(approx(sl_mb, sl_d));
        assert!(approx(sr_mb, sr_d));
    }

    #[test]
    fn extremal_speeds_bracket_zero() {
        // Rhd::extremal_speeds always has s_l <= 0 and s_r >= 0
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let left = Prim { rho: 1.0, vel: Tensor::new([0.9]), pre: 1.0 };
        let right = Prim { rho: 1.0, vel: Tensor::new([0.9]), pre: 1.0 };
        let (sl, sr) = regime.extremal_speeds(&eos, &left, &right, &Tensor::unit(0));
        assert!(sl <= 0.0, "sl={} should be <= 0", sl);
        assert!(sr >= 0.0, "sr={} should be >= 0", sr);
    }

    #[test]
    fn max_wave_speed_3d() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5, -0.3, 0.1]),
            pre: 1.0,
        };
        let smax = regime.max_wave_speed(&eos, &prim);
        assert!(smax > 0.0);
        assert!(smax < 1.0); // subluminal
    }

    #[test]
    fn gr_speeds_reduce_to_flat_exactly() {
        // the Banyuls-Font path at (gamma_nn = 1, alpha = 1) reduces EXACTLY (to 1e-12) to the SR
        // form across the state space -> Minkowski wave speeds are unchanged.
        for &cs_sq in &[0.05_f64, 0.2, 0.33] {
            for &vn in &[-0.8_f64, -0.3, 0.0, 0.3, 0.8] {
                let (sl, sr) = rhd_speeds_from_vn(cs_sq, vn);
                let (gl, gr) = rhd_speeds_from_vn_gr(cs_sq, vn, 1.0, 1.0);
                assert!(approx(sl, gl) && approx(sr, gr), "flat reduction ({sl},{sr}) vs ({gl},{gr})");
            }
        }
    }

    #[test]
    fn gr_speeds_damp_under_schwarzschild() {
        // Schwarzschild at r=10, M=1 -> f = 0.8: gamma_nn = f, alpha = sqrt(f). the lapse + the
        // in-radical gamma_nn DAMP the COORDINATE speeds below the flat ones (gravity slows
        // coordinate-time propagation). hand-computed lambda+ ~ 0.54684, lambda- ~ -0.10965.
        let (cs_sq, vn, f) = (0.2_f64, 0.3_f64, 0.8_f64);
        let (sl_flat, sr_flat) = rhd_speeds_from_vn(cs_sq, vn);
        let (sl_gr, sr_gr) = rhd_speeds_from_vn_gr(cs_sq, vn, f, f.sqrt());
        assert!(sr_gr.abs() < sr_flat.abs(), "right speed must damp: {sr_gr} vs {sr_flat}");
        assert!(sl_gr.abs() < sl_flat.abs(), "left speed must damp: {sl_gr} vs {sl_flat}");
        assert!((sr_gr - 0.54684).abs() < 1e-4, "lambda+ = {sr_gr}");
        assert!((sl_gr + 0.10965).abs() < 1e-4, "lambda- = {sl_gr}");
        // still subluminal-physical (no superluminal coordinate speed for this state).
        assert!(sr_gr < 1.0 && sl_gr > -1.0);
    }
}
