// =============================================================================
// srhd/wave_speeds.rs
//
// the SRHD relativistic acoustic wave speeds (Mignone & Bodo 2005). the core
// `srhd_speeds_from_vn` is a pure function of (cs^2, normal velocity) — the SINGLE
// source both the nhat Riemann projection and the CFL axis projection call.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;
use crate::eos::Eos;
use crate::state::Prim;
use crate::srhd::sound_speed_sq;

/// the Mignone & Bodo (2005) relativistic acoustic wave speeds (eqs. 21-23) as a function of
/// the sound speed squared and the NORMAL velocity `vn` — the single core both projections
/// call: the nhat Riemann form (`vn = vel . nhat`) and the CFL axis form (`vn = vel[axis]`).
/// accounts for the relativistic dispersion relation; tighter than davis.
#[inline]
pub(crate) fn srhd_speeds_from_vn<S: Scalar>(cs_sq: S, vn: S) -> (S, S) {
    let vn_sq = vn * vn;
    let w_sq = S::ONE / (S::ONE - vn_sq); // W^2
    let ss = cs_sq / (w_sq * (S::ONE - cs_sq));
    let qf = S::ONE / (S::ONE + ss);
    let fac = (ss * (S::ONE - vn_sq + ss)).sqrt();
    ((vn - fac) * qf, (vn + fac) * qf)
}

/// davis wave speed estimates for SRHD (simpler, less tight bounds).
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
    use crate::srhd::Srhd;
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
        let (sl, sr) = Srhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
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
                let (sl, sr) = Srhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
                assert!(sl > -1.0, "sl={} at v={}, p={}", sl, v, pre);
                assert!(sr < 1.0, "sr={} at v={}, p={}", sr, v, pre);
                assert!(sl < sr, "sl={} >= sr={} at v={}, p={}", sl, sr, v, pre);
            }
        }
    }

    #[test]
    fn davis_matches_mb_at_zero_v() {
        // at v=0, davis and mignone-bodo should give same result
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
        let (sl_mb, sr_mb) = Srhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
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
        // Srhd::extremal_speeds always has s_l <= 0 and s_r >= 0
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let left = Prim { rho: 1.0, vel: Tensor::new([0.9]), pre: 1.0 };
        let right = Prim { rho: 1.0, vel: Tensor::new([0.9]), pre: 1.0 };
        let (sl, sr) = regime.extremal_speeds(&eos, &left, &right, &Tensor::unit(0));
        assert!(sl <= 0.0, "sl={} should be <= 0", sl);
        assert!(sr >= 0.0, "sr={} should be >= 0", sr);
    }

    #[test]
    fn max_wave_speed_3d() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5, -0.3, 0.1]),
            pre: 1.0,
        };
        let smax = regime.max_wave_speed(&eos, &prim);
        assert!(smax > 0.0);
        assert!(smax < 1.0); // subluminal
    }
}
