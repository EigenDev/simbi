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

use crate::eos::Eos;
use crate::regime::Regime;
use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// HLLE approximate riemann solver. two-wave solver, any regime.
/// pure math — no allocation, GPU-callable. `vface` is the grid velocity at the
/// face (nhat direction) for ALE moving meshes; pass 0 for a static mesh.
///
/// computes the fan speeds from the L/R states (`extremal_speeds`) then combines. when the
/// caller ALREADY has the fan speeds (e.g., read from a per-cell wave-speed field — see
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
/// cell instead of re-solving the (quartic, for RMHD) speed at every face.
///
/// the fan is the CLAMPED closed form: with `bm = min(s_l - vface, 0)` and
/// `bp = max(s_r - vface, 0)`, the single expression
/// `(bp*(f_l - vface*u_l) - bm*(f_r - vface*u_r) + bp*bm*(u_r - u_l)) / (bp - bm)`
/// reduces algebraically to the upwind flux `f_l - vface*u_l` when `s_l >= vface`
/// (bm = 0), to `f_r - vface*u_r` when `s_r <= vface` (bp = 0), and to the
/// galilean moving-face HLL average `f_hll - vface*u_hll` in the subsonic fan —
/// one branch-free expression instead of a three-way wave select. the guard
/// covers the degenerate fan `bp == bm == 0` (both waves riding the face),
/// where the closed form is 0/0; the upwind states coincide there.
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

    let bm = (s_l - vface).min(S::ZERO);
    let bp = (s_r - vface).max(S::ZERO);
    let dn = bp - bm;
    let ok = dn.cmp_gt(S::ZERO);
    // the denominator select keeps the unselected arm division-safe (algebra.rs
    // select invariant: both arms of a traced select are evaluated).
    let inv = S::ONE / S::select(ok, dn, S::ONE);
    let f_l_face = f_l - u_l * vface;
    let f_r_face = f_r - u_r * vface;
    S::branch(
        ok,
        || (f_l_face * bp - f_r_face * bm + (u_r - u_l) * (bp * bm)) * inv,
        || f_l_face,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use crate::newtonian::Newtonian;
    use crate::state::Prim;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn hlle_uniform_state_1d() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5]),
            pre: 1.0,
        };
        let nhat = Tensor::unit(0);
        let flux = hlle(&regime, &eos, &prim, &prim, &nhat, 0.0);
        let exact = regime.to_flux(&prim, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
        assert!(approx(flux.mom[0], exact.mom[0]));
        assert!(approx(flux.nrg, exact.nrg));
    }

    // the clamped closed form must reduce to the pure upwind flux outside the fan:
    // s_l >= vface collapses bm to zero and the expression to f_l - vface*u_l (up to
    // the bp*(x)*(1/bp) rounding of the closed form); s_r <= vface mirrors to the
    // right state. tolerance-level, not bitwise — the closed form multiplies and
    // divides by the surviving wave speed.
    #[test]
    fn hlle_fan_reduces_to_upwind() {
        let regime = Newtonian;
        let eos = IdealGas { gamma: 1.4 };
        let nhat = Tensor::unit(0);
        let l = Prim {
            rho: 1.0,
            vel: Tensor::new([3.0]),
            pre: 1.0,
        };
        let r = Prim {
            rho: 0.5,
            vel: Tensor::new([3.0]),
            pre: 0.6,
        };
        // both states supersonic to the right: the flux is the left upwind flux.
        let flux = hlle(&regime, &eos, &l, &r, &nhat, 0.0);
        let exact = regime.to_flux(&l, &nhat, &eos);
        assert!(approx(flux.den, exact.den));
        assert!(approx(flux.mom[0], exact.mom[0]));
        assert!(approx(flux.nrg, exact.nrg));

        // mirrored: both states supersonic to the left picks the right state.
        let l2 = Prim {
            rho: 1.0,
            vel: Tensor::new([-3.0]),
            pre: 1.0,
        };
        let r2 = Prim {
            rho: 0.5,
            vel: Tensor::new([-3.0]),
            pre: 0.6,
        };
        let flux2 = hlle(&regime, &eos, &l2, &r2, &nhat, 0.0);
        let exact2 = regime.to_flux(&r2, &nhat, &eos);
        assert!(approx(flux2.den, exact2.den));
        assert!(approx(flux2.mom[0], exact2.mom[0]));
        assert!(approx(flux2.nrg, exact2.nrg));

        // a face moving faster than every wave upwinds to the right state in
        // the face frame: f_r - vface*u_r.
        let vf = 10.0;
        let flux3 = hlle(&regime, &eos, &l, &r, &nhat, vf);
        let u_r = regime.to_conserved(&eos, &r);
        let f_r = regime.to_flux(&r, &nhat, &eos);
        assert!(approx(flux3.den, f_r.den - vf * u_r.den));
        assert!(approx(flux3.mom[0], f_r.mom[0] - vf * u_r.mom[0]));
        assert!(approx(flux3.nrg, f_r.nrg - vf * u_r.nrg));
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
            ((1.0, 3.0, 1.0), (0.5, 3.0, 0.6)), // both moving right fast (s_l >= 0)
            ((1.0, -3.0, 1.0), (0.5, -3.0, 0.6)), // both moving left fast (s_r <= 0)
        ];
        for ((rl, vl, pl), (rr, vr, pr)) in cases {
            let l = Prim {
                rho: rl,
                vel: Tensor::new([vl]),
                pre: pl,
            };
            let r = Prim {
                rho: rr,
                vel: Tensor::new([vr]),
                pre: pr,
            };
            let inline = hlle(&regime, &eos, &l, &r, &nhat, 0.0);
            let (s_l, s_r) = regime.extremal_speeds(&eos, &l, &r, &nhat);
            let split = hlle_with_speeds(&regime, &eos, &l, &r, &nhat, 0.0, s_l, s_r);
            assert_eq!(inline.den, split.den, "den mismatch");
            assert_eq!(inline.mom[0], split.mom[0], "mom mismatch");
            assert_eq!(inline.nrg, split.nrg, "nrg mismatch");
        }
    }
}
