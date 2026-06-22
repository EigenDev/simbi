// =============================================================================
// riemann/hlle.rs
//
// the HLLE approximate riemann solver: a two-wave (regime-generic) solver that
// works for ANY `Regime`. the regime supplies `to_conserved` / `to_flux` /
// `extremal_speeds`; HLLE combines them. pure math, GPU-callable (S::branch).
//
// usage:
//   let nhat = Tensor::unit(0);
//   let flux = hlle(&regime, &eos, &prim_l, &prim_r, &nhat, 0.0);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;
use crate::eos::Eos;
use crate::regime::Regime;

/// HLLE approximate riemann solver. two-wave solver, any regime.
/// pure math — no allocation, GPU-callable. `vface` is the grid velocity at the
/// face (nhat direction) for ALE moving meshes; pass 0 for a static mesh.
///
/// computes the fan speeds from the L/R states (`extremal_speeds`) then combines. when the
/// caller ALREADY has the fan speeds (e.g. read from a per-cell wave-speed field — see
/// `rmhd_wave_speeds_cell_gv`), call `hlle_with_speeds` directly to skip the (expensive)
/// wave-speed recomputation.
pub fn hlle<S: Scalar, const D: usize, R: Regime<S, D>>(
    regime: &R,
    eos: &impl Eos<S>,
    prim_l: &R::Prim,
    prim_r: &R::Prim,
    nhat: &Tensor<S, D>,
    vface: S,
) -> R::Cons {
    let (s_l, s_r) = regime.extremal_speeds(eos, prim_l, prim_r, nhat);
    hlle_with_speeds(regime, eos, prim_l, prim_r, nhat, vface, s_l, s_r)
}

/// the HLLE combine with the fan speeds `(s_l, s_r)` supplied by the caller — the body of
/// `hlle` after `extremal_speeds`. lets a face flux reuse wave speeds materialized once per
/// cell instead of re-solving the (quartic, for RMHD) speed at every face. `s_l <= 0 <= s_r`
/// is the caller's responsibility (the regime's extremal-speed zero-clamp).
pub fn hlle_with_speeds<S: Scalar, const D: usize, R: Regime<S, D>>(
    regime: &R,
    eos: &impl Eos<S>,
    prim_l: &R::Prim,
    prim_r: &R::Prim,
    nhat: &Tensor<S, D>,
    vface: S,
    s_l: S,
    s_r: S,
) -> R::Cons {
    let u_l = regime.to_conserved(eos, prim_l);
    let u_r = regime.to_conserved(eos, prim_r);
    let f_l = regime.to_flux(prim_l, nhat, eos);
    let f_r = regime.to_flux(prim_r, nhat, eos);

    S::branch(s_l.cmp_ge(vface),
        || f_l - u_l * vface,
        || S::branch(s_r.cmp_le(vface),
            || f_r - u_r * vface,
            || {
                let inv = S::ONE / (s_r - s_l);
                let f_hll = (f_l * s_r - f_r * s_l + (u_r - u_l) * (s_l * s_r)) * inv;
                let u_hll = (u_r * s_r - u_l * s_l - f_r + f_l) * inv;
                f_hll - u_hll * vface
            }
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::newtonian::Newtonian;
    use crate::state::Prim;
    use crate::eos::IdealGas;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn hlle_uniform_state_1d() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.5]), pre: 1.0 };
        let nhat = Tensor::unit(0);
        let flux = hlle(&regime, &eos, &prim, &prim, &nhat, 0.0);
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
        assert!(approx(flux.mom[0], exact.mom[0]));
        assert!(approx(flux.nrg, exact.nrg));
    }

    // the split is a pure refactor: `hlle` MUST equal `hlle_with_speeds` fed the same
    // `extremal_speeds`, for any L/R states. this pins that the per-cell-wave-speed path
    // (which calls hlle_with_speeds directly) is bit-identical to the inline path when the
    // supplied speeds match. covers the subsonic fan + both supersonic branches.
    #[test]
    fn hlle_equals_hlle_with_speeds() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let nhat = Tensor::unit(0);
        let cases = [
            // (L, R): subsonic contact, left-supersonic, right-supersonic
            ((1.0, 0.2, 1.0), (0.5, -0.3, 0.6)),
            ((1.0, 3.0, 1.0), (0.5, 3.0, 0.6)),   // both moving right fast (s_l >= 0)
            ((1.0, -3.0, 1.0), (0.5, -3.0, 0.6)), // both moving left fast (s_r <= 0)
        ];
        for ((rl, vl, pl), (rr, vr, pr)) in cases {
            let l = Prim { rho: rl, vel: Tensor::new([vl]), pre: pl };
            let r = Prim { rho: rr, vel: Tensor::new([vr]), pre: pr };
            let inline = hlle(&regime, &eos, &l, &r, &nhat, 0.0);
            let (s_l, s_r) = regime.extremal_speeds(&eos, &l, &r, &nhat);
            let split = hlle_with_speeds(&regime, &eos, &l, &r, &nhat, 0.0, s_l, s_r);
            assert_eq!(inline.den, split.den, "den mismatch");
            assert_eq!(inline.mom[0], split.mom[0], "mom mismatch");
            assert_eq!(inline.nrg, split.nrg, "nrg mismatch");
        }
    }
}
