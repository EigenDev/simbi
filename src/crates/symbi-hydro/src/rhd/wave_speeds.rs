// =============================================================================
// rhd/wave_speeds.rs
//
// the RHD relativistic acoustic wave speeds (Mignone & Bodo 2005). the core
// `rhd_speeds_from_vn` is a pure function of (cs^2, normal velocity) — the single
// source both the nhat Riemann projection and the CFL axis projection call.
// =============================================================================

use crate::eos::Eos;
use crate::rhd::sound_speed_sq;
use crate::state::Prim;
use symbi_algebra::Tensor;
use symbi_carrier::Scalar;

/// the Mignone & Bodo (2005) relativistic acoustic wave speeds (eqs. 21-23) as a function of
/// the sound speed squared and the normal velocity `vn` — the single core both projections
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

/// the Banyuls-Font coordinate-frame acoustic speeds on a static diagonal metric (shift beta = 0),
/// Font (2008) eq (37). critical: it takes two distinct velocities that a single-argument form would
/// wrongly conflate:
///   - `vn` = the contravariant normal velocity v^n = v^i n_i (Font's `v^x`), in the transport term
///     and inside the radical's second term.
///   - `v_sq` = gamma_ij v^i v^j, the physical speed squared (|V|^2), only in the `(1 - v^2)` factors.
///   `disc = (1 - v_sq)( gamma_nn (1 - v_sq cs^2) - vn^2 (1 - cs^2) )`
///   `lambda_pm = alpha [ vn(1 - cs^2) +/- cs sqrt(disc) ] / (1 - v_sq cs^2)`, `gamma_nn = gamma^{nn}`.
/// feeding the physical velocity to both slots drives `disc < 0` once |V| approaches
/// alpha (near the horizon), collapsing the Riemann fan to NaN.
/// with the correct split, cauchy-schwarz `vn^2 <= gamma^{nn} v_sq` guarantees `disc >= gamma^{nn}
/// (1 - v_sq)^2 >= 0`; for Schwarzschild (`gamma^{rr} = alpha^2`, v^r = alpha V) `disc = alpha^2
/// (1-V^2)^2` and `lambda_pm = alpha^2 (V +/- cs)/(1 +/- V cs)` (sonic point at |V| = cs).
/// at `gamma_nn = 1, alpha = 1, v_sq = vn^2` it reduces exactly to the flat `rhd_speeds_from_vn`.
#[inline]
pub(crate) fn rhd_speeds_from_vn_gr<S: Scalar>(
    cs_sq: S,
    vn: S,
    v_sq: S,
    gamma_nn: S,
    alpha: S,
) -> (S, S) {
    let one_m_v2cs2 = S::ONE - v_sq * cs_sq;
    let disc = (S::ONE - v_sq) * (gamma_nn * one_m_v2cs2 - vn * vn * (S::ONE - cs_sq));
    let term = vn * (S::ONE - cs_sq);
    let rad = cs_sq.sqrt() * disc.sqrt(); // cs * sqrt(disc)
    let inv = alpha / one_m_v2cs2;
    ((term - rad) * inv, (term + rad) * inv)
}

/// the davis wave-speed estimate for RHD: `s_l = (vn - cs)/(1 - cs vn)`, `s_r = (vn + cs)/(1 + cs vn)`,
/// the relativistic addition of +/- cs to the fluid's normal velocity. a valid but looser HLL bound
/// than the Mignone-Bodo characteristic speeds the Regime impl evolves with; retained as the
/// comparison baseline for solver diffusivity, so it has no caller.
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
    use symbi_algebra::{FaceNormal, Normalized};
    use crate::eos::IdealGas;
    use crate::regime::Regime;
    use crate::rhd::Rhd;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn wave_speeds_stationary() {
        // v=0: symmetric wave speeds
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        };
        let (sl, sr) = Rhd.wave_speeds(&eos, &prim, &Normalized::axis(0));
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
                let prim = Prim {
                    rho: 1.0,
                    vel: Tensor::new([v]),
                    pre,
                };
                let (sl, sr) = Rhd.wave_speeds(&eos, &prim, &Normalized::axis(0));
                assert!(sl > -1.0, "sl={} at v={}, p={}", sl, v, pre);
                assert!(sr < 1.0, "sr={} at v={}, p={}", sr, v, pre);
                assert!(sl < sr, "sl={} >= sr={} at v={}, p={}", sl, sr, v, pre);
            }
        }
    }

    #[test]
    fn schwarzschild_ks_coordinate_speed_is_ingoing_at_and_inside_horizon() {
        // the horizon-penetrating CFL guarantee. the kerr-schild coordinate wave speed is the
        // factored form the CFL map uses, lambda_coord = alpha^2 * lambda^SR - beta^r, with
        // alpha^2 = 1/(1 + 2M/r) and beta^r = 2M/(r + 2M) (M = 1). both characteristic roots are
        // strictly < 0 at and inside the horizon r <= 2M for every physical fluid state, so the
        // numerical domain of dependence is entirely interior-directed and the inner outflow
        // boundary is causal. |lambda^SR| < 1 subluminal => alpha^2 |lambda^SR| < alpha^2 = beta^r
        // at the horizon, so lambda_coord < 0.
        let eos = IdealGas { gamma: 4.0 / 3.0 };
        for &r in &[2.0_f64, 1.7, 1.2, 1.0] {
            let alpha_sq = 1.0 / (1.0 + 2.0 / r);
            let beta_r = 2.0 / (r + 2.0);
            for &v in &[-0.99_f64, -0.5, 0.0, 0.5, 0.99] {
                for &pre in &[0.01, 1.0, 100.0] {
                    let prim = Prim {
                        rho: 1.0,
                        vel: Tensor::new([v]),
                        pre,
                    };
                    let (sl, sr) = Rhd.wave_speeds(&eos, &prim, &Normalized::axis(0));
                    let (ll, lr) = (alpha_sq * sl - beta_r, alpha_sq * sr - beta_r);
                    assert!(
                        ll < 0.0 && lr < 0.0,
                        "coord speed not ingoing at r={r}, v={v}, p={pre}: ll={ll}, lr={lr}"
                    );
                }
            }
        }
    }

    #[test]
    fn davis_matches_mb_at_zero_v() {
        // at v=0, davis and mignone-bodo should give same result
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        };
        let (sl_mb, sr_mb) = Rhd.wave_speeds(&eos, &prim, &Normalized::axis(0));
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
        let left = Prim {
            rho: 1.0,
            vel: Tensor::new([0.9]),
            pre: 1.0,
        };
        let right = Prim {
            rho: 1.0,
            vel: Tensor::new([0.9]),
            pre: 1.0,
        };
        let (sl, sr) = regime.extremal_speeds(&eos, &left, &right, &Normalized::axis(0));
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
        // the Banyuls-Font path at (gamma_nn = 1, alpha = 1, v_sq = vn^2) reduces exactly (to 1e-12)
        // to the SR form across the state space -> Minkowski wave speeds are unchanged.
        for &cs_sq in &[0.05_f64, 0.2, 0.33] {
            for &vn in &[-0.8_f64, -0.3, 0.0, 0.3, 0.8] {
                let (sl, sr) = rhd_speeds_from_vn(cs_sq, vn);
                let (gl, gr) = rhd_speeds_from_vn_gr(cs_sq, vn, vn * vn, 1.0, 1.0);
                assert!(
                    approx(sl, gl) && approx(sr, gr),
                    "flat reduction ({sl},{sr}) vs ({gl},{gr})"
                );
            }
        }
    }

    #[test]
    fn gr_speeds_match_schwarzschild_closed_form() {
        // Schwarzschild (gamma^{rr} = f = alpha^2, v^r = alpha V): the discriminant is a perfect
        // square, disc = alpha^2 (1 - V^2)^2 >= 0, so lambda_pm = alpha^2 (V +/- cs)/(1 +/- V cs)
        // with the sonic point exactly at |V| = cs. this is the value the fan must return; the old
        // bug (physical velocity in both slots) drove disc < 0 -> NaN for V >~ alpha.
        for &f in &[0.8_f64, 0.5, 0.34] {
            // f = 1 - 2M/r; alpha = sqrt(f), gamma^{rr} = f. sweep the physical velocity V incl. V > alpha.
            let (alpha, grr_inv) = (f.sqrt(), f);
            for &cs_sq in &[0.05_f64, 0.2, 0.33] {
                for &big_v in &[-0.9_f64, -0.5, -0.1, 0.0, 0.3, 0.7, 0.9] {
                    let vr = alpha * big_v; // contravariant v^r = alpha V
                    let v_sq = big_v * big_v; // physical |v|^2 = V^2 (radial)
                    let (sl, sr) = rhd_speeds_from_vn_gr(cs_sq, vr, v_sq, grr_inv, alpha);
                    let cs = cs_sq.sqrt();
                    let a2 = alpha * alpha;
                    let (want_l, want_r) = (
                        a2 * (big_v - cs) / (1.0 - big_v * cs),
                        a2 * (big_v + cs) / (1.0 + big_v * cs),
                    );
                    assert!(
                        sl.is_finite() && sr.is_finite(),
                        "NaN fan at V={big_v}, f={f}, cs^2={cs_sq}"
                    );
                    assert!(
                        approx(sl, want_l) && approx(sr, want_r),
                        "V={big_v} f={f} cs^2={cs_sq}: ({sl},{sr}) vs ({want_l},{want_r})"
                    );
                }
            }
        }
    }

    #[test]
    fn gr_fan_is_finite_at_transonic_inner_state() {
        // at the near-horizon steady inner state of a spherical accretion flow (r ~ 3.05, M=1, so
        // f ~ 0.344, alpha ~ 0.587) with a transonic physical velocity |V| ~ 0.64 > alpha, the fan
        // must be real: a negative discriminant here would collapse both wave speeds to NaN.
        let (cs_sq, alpha) = (0.0132_f64, 0.587_f64); // cs ~ 0.115, alpha^2 ~ f
        let grr_inv = alpha * alpha; // gamma^{rr} = alpha^2
        for &big_v in &[-0.64_f64, -0.7, -0.85, -0.95] {
            let vr = alpha * big_v;
            let (sl, sr) = rhd_speeds_from_vn_gr(cs_sq, vr, big_v * big_v, grr_inv, alpha);
            assert!(
                sl.is_finite() && sr.is_finite() && sl < sr,
                "fan collapsed at transonic inner state V={big_v}: ({sl},{sr})"
            );
        }
    }

    #[test]
    fn conflating_the_physical_velocity_into_the_contravariant_slot_collapses_the_fan() {
        // asserting the fan is real at a transonic near-horizon state says nothing on its own — a
        // formula that never produced a NaN would satisfy it too. this pins the other side: the
        // specific slot conflation does collapse the fan at that same state, so the finiteness
        // assertion exerts real pressure rather than passing vacuously.
        //
        // the defect is a slot conflation. `vn` is the contravariant normal velocity v^n and
        // `v_sq` the physical speed squared gamma_ij v^i v^j; on a static diagonal chart they are
        // related by v^n = alpha V, so passing the physical V into the contravariant slot inflates
        // vn by 1/alpha. the discriminant carries `- vn^2 (1 - cs^2)`, so the inflated value drives
        // it negative once |V| approaches alpha and the square root returns NaN.
        //
        // this is chart-independent, which is why it belongs here rather than in an accretion run:
        // it needs only gamma_nn != 1 and a fast enough flow, not a particular problem on a
        // particular chart whose observer happens to see the gas approach c.
        let (cs_sq, alpha) = (0.0132_f64, 0.587_f64);
        let grr_inv = alpha * alpha;
        let mut collapsed = 0;
        for &big_v in &[-0.64_f64, -0.7, -0.85, -0.95] {
            // the conflation: the physical velocity handed to the contravariant slot.
            let (sl, sr) = rhd_speeds_from_vn_gr(cs_sq, big_v, big_v * big_v, grr_inv, alpha);
            if !(sl.is_finite() && sr.is_finite()) {
                collapsed += 1;
            }
        }
        assert!(
            collapsed > 0,
            "the conflated velocity slot no longer collapses the fan at any transonic state, so \
             `gr_fan_is_finite_at_transonic_inner_state` defends nothing: either the discriminant \
             changed or these states stopped being transonic"
        );
    }

    #[test]
    fn the_discriminant_is_non_negative_for_every_admissible_state() {
        // the theorem behind the slot split, exercised directly instead of through a problem that
        // happens to reach the dangerous regime. with the slots split correctly, cauchy-schwarz gives
        // vn^2 <= gamma^{nn} v_sq, hence
        //   disc = (1 - v^2)( gamma^{nn}(1 - v^2 cs^2) - vn^2 (1 - cs^2) ) >= gamma^{nn}(1 - v^2)^2
        // which is non-negative for any subluminal state on any positive-definite metric. so the
        // fan is real everywhere, not merely at the states some accretion run visits: no chart, no
        // lapse and no flow speed can produce the NaN, which is a strictly stronger statement than
        // any single-problem regression can make.
        for &gamma_nn in &[0.2_f64, 0.5, 1.0, 2.5, 8.0] {
            for &alpha in &[0.15_f64, 0.6, 1.0] {
                for &cs_sq in &[1e-6_f64, 0.05, 0.33] {
                    for &v_sq in &[0.0_f64, 0.25, 0.81, 0.9801] {
                        // the cauchy-schwarz extreme: vn^2 at its maximum gamma^{nn} v_sq, where
                        // the bound is tight and the discriminant is smallest.
                        for sign in [-1.0_f64, 1.0] {
                            let vn = sign * (gamma_nn * v_sq).sqrt();
                            let (sl, sr) = rhd_speeds_from_vn_gr(cs_sq, vn, v_sq, gamma_nn, alpha);
                            assert!(
                                sl.is_finite() && sr.is_finite() && sl <= sr,
                                "fan not real at gamma_nn={gamma_nn} alpha={alpha} \
                                 cs^2={cs_sq} v^2={v_sq} vn={vn}: ({sl},{sr})"
                            );
                        }
                    }
                }
            }
        }
    }
}
