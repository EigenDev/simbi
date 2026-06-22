// =============================================================================
// srhd/cons.rs
//
// the SRHD cons->prim recovery — a carrier-generic Newton-Raphson on the pressure
// root (`Scalar::iterate`), then the algebraic velocity/Lorentz/density recovery.
// branch-free core (`srhd_recover`, what the substrate c2p kernel computes) + the
// host wrapper (`srhd_to_primitive`) that adds the C2pResult diagnostics post-hoc.
// =============================================================================

use symbi_algebra::{Tensor, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use crate::eos::Eos;
use crate::state::{Prim, Cons};
use crate::c2p_result::C2pResult;
use crate::srhd::lorentz_factor;

/// maximum newton-raphson iterations for SRHD cons2prim on the HOST (early-break).
/// the substrate kernel bakes its own fixed count (build.rs passes 20 to the gv
/// builder) — both share the `srhd_recover` body; the count is a knob, not physics.
const MAX_ITER: usize = 100;

/// the branch-free SRHD cons->prim recovery — THE single-source physics: the pressure
/// is the root of a 1D equation found by a carrier-generic Newton (`Scalar::iterate`),
/// then velocity/Lorentz/density follow algebraically. NO floors, NO guards — exactly
/// what the substrate c2p kernel computes (the `srhd_to_primitive` wrapper adds the
/// host C2pResult diagnostics post-hoc, matching `Newtonian::to_primitive`).
///
/// EOS-generic: uses `eos.pressure()` / `eos.sound_speed_sq()`, not gamma. traced at
/// `S = Gv` (`symbi_discretize::srhd_c2p_gv`) it lowers to the IterateInline c2p kernel;
/// at `S = f64/f32` it runs as a plain early-breaking loop. `max_iter` caps the Newton.
pub fn srhd_recover<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    cons: &Cons<S, D>,
    max_iter: usize,
) -> Prim<S, D> {
    let dd = cons.den;
    let tau = cons.nrg;
    let s_mag = cons.mom.dot(&cons.mom).sqrt();

    // initial pressure guess: |S - D - tau|
    let p_init = (s_mag - dd - tau).abs();

    // ONE newton step on the pressure: given a guess, derive (v, W, rho, eps), form
    // f(p) = eos.pressure - p and g = df/dp = cs_rel^2*v^2 - 1, return the updated guess.
    let newton_step = |p_eq: S| -> S {
        let et = tau + dd + p_eq;
        let v_sq = s_mag * s_mag / (et * et);
        let ww = S::ONE / (S::ONE - v_sq).sqrt();
        let rho = dd / ww;
        let eps = (tau + (S::ONE - ww) * dd + (S::ONE - ww * ww) * p_eq) / (dd * ww);
        let p_eos = eos.pressure(rho, eps);
        let ff = p_eos - p_eq;
        let h = S::ONE + eps + p_eos / rho;
        let cs_rel_sq = eos.sound_speed_sq(rho, p_eos) / h;
        let gg = cs_rel_sq * v_sq - S::ONE;
        p_eq - ff / gg
    };

    // convergence predicate |dp| = |cur - prev| < tol: drives the f64/f32 host early-break
    // AND the Gv carrier's sticky-done freeze (so the baked fixed-count kernel returns the
    // same recovered pressure the host does — see Scalar::iterate / Gv::iterate).
    let p_eq = p_init.iterate(max_iter, &newton_step, |prev, cur| {
        let tol = dd * S::from_f64(1e-12);
        (cur - prev).abs().cmp_lt(tol)
    });

    // recover primitive variables: 3-velocity v = S/(tau+D+p), W = 1/sqrt(1-v.v), rho = D/W.
    let et = tau + dd + p_eq;
    let vel = cons.mom.map(|s| s / et);
    let ww = lorentz_factor(vel.dot(&vel));
    let rho = dd / ww;
    Prim { rho, vel, pre: p_eq }
}

/// the host SRHD cons->prim: the branch-free `srhd_recover` plus post-hoc C2pResult
/// diagnostics. no silent floor — the value is the raw recovered state, the ErrorCode
/// is the explicit signal (matches `Newtonian::to_primitive`; feedback_no_silent_floors).
///
/// **host-only** (Tier 1.7): `S: OrderedNumeric` because the diagnostic check uses
/// native `<` / `<=` / `==` on a host scalar. the kernel path is `srhd_recover` above
/// (carrier-generic over `S: Scalar`), not this wrapper.
pub(crate) fn srhd_to_primitive<S: Scalar + OrderedNumeric, const D: usize>(
    eos: &impl Eos<S>,
    cons: &Cons<S, D>,
) -> C2pResult<Prim<S, D>> {
    let dd = cons.den;

    // input guard: clearly-invalid conserved density. a host-only early-out (NOT in the
    // kernel path) that keeps the recovery's `dd/ww`, `tau+dd+p` finite for callers.
    if let Some(code) = crate::c2p_result::relativistic_density_guard(dd) {
        let floored = Prim {
            rho: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR), vel: Tensor::zeros(),
            pre: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR),
        };
        return C2pResult::err(floored, code);
    }

    let prim = srhd_recover(eos, cons, MAX_ITER);

    // post-hoc diagnostics on the raw recovered state (shared SRHD/RMHD contract; tier-1 #5).
    let v_sq = prim.vel.dot(&prim.vel);
    let code = crate::c2p_result::relativistic_c2p_code(prim.rho, prim.pre, v_sq);
    if code.is_ok() { C2pResult::ok(prim) } else { C2pResult::err(prim, code) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::srhd::Srhd;
    use crate::regime::Regime;
    use crate::eos::IdealGas;

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn roundtrip_stationary() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx_rel(prim.rho, prim2.rho, 1e-10));
        assert!(approx_rel(prim.vel[0], prim2.vel[0], 1e-10));
        assert!(approx_rel(prim.pre, prim2.pre, 1e-10));
    }

    #[test]
    fn roundtrip_mildly_relativistic() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.5]), pre: 1.0 };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx_rel(prim.rho, prim2.rho, 1e-10));
        assert!(approx_rel(prim.vel[0], prim2.vel[0], 1e-10));
        assert!(approx_rel(prim.pre, prim2.pre, 1e-10));
    }

    #[test]
    fn roundtrip_ultra_relativistic() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.99]), pre: 10.0 };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx_rel(prim.rho, prim2.rho, 1e-8));
        assert!(approx_rel(prim.vel[0], prim2.vel[0], 1e-8));
        assert!(approx_rel(prim.pre, prim2.pre, 1e-8));
    }

    #[test]
    fn roundtrip_3d() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let prim = Prim {
            rho: 2.0,
            vel: Tensor::new([0.3, -0.2, 0.1]),
            pre: 0.5,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx_rel(prim.rho, prim2.rho, 1e-10));
        for dd in 0..3 {
            assert!(approx_rel(prim.vel[dd], prim2.vel[dd], 1e-10));
        }
        assert!(approx_rel(prim.pre, prim2.pre, 1e-10));
    }

    #[test]
    fn roundtrip_high_density_ratio() {
        // density contrast like a relativistic sod problem
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        for &(rho, pre) in &[(10.0, 13.33), (1.0, 1e-6), (1e-2, 1e-4)] {
            let prim = Prim { rho, vel: Tensor::new([0.0]), pre };
            let cons = regime.to_conserved(&eos, &prim);
            let prim2 = regime.to_primitive(&eos, &cons).unwrap();
            assert!(
                approx_rel(prim.rho, prim2.rho, 1e-8),
                "rho: {} vs {} (input rho={}, pre={})",
                prim.rho, prim2.rho, rho, pre
            );
            assert!(
                approx_rel(prim.pre, prim2.pre, 1e-8),
                "pre: {} vs {} (input rho={}, pre={})",
                prim.pre, prim2.pre, rho, pre
            );
        }
    }

    #[test]
    fn negative_density_detected() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let cons = Cons { den: -1.0, mom: Tensor::new([0.0]), nrg: 1.0 };
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.error.contains(crate::c2p_result::ErrorCode::NEGATIVE_DENSITY));
        assert!(result.value.rho > 0.0);
    }

    #[test]
    fn unphysical_cons_returns_error() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        // huge momentum, tiny density+energy -> superluminal or negative pressure
        let cons = Cons { den: 1e-14, mom: Tensor::new([100.0]), nrg: 1e-14 };
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.error.is_err());
    }

    // DELIBERATE no-clamp pin: the branch-free kernel body (`srhd_recover`, what the
    // substrate c2p kernel computes) must return a NON-FINITE prim for unphysical cons
    // (s_mag > d + tau => v_sq >= 1 => 1/sqrt(1-v_sq) blows up). this matches the C++
    // reference (helpers::newton_fg is unguarded; the outer loop detects !isfinite) and
    // is REQUIRED by feedback_no_silent_floors: the kernel path has no ErrorCode channel,
    // so the NaN must propagate to the CFL max-reduction (now NaN-propagating, item 1)
    // and trip check_dt_or_panic. clamping v_sq here would silently recover a garbage
    // finite state and defeat that guard. if this test fails because someone added a
    // clamp, the clamp is the bug.
    #[test]
    fn kernel_path_unphysical_cons_is_non_finite() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        // s_mag = 10 >> d + tau = 1.1, so the very first Newton iterate has v_sq > 1.
        let cons: Cons<f64, 1> = Cons { den: 1.0, mom: Tensor::new([10.0]), nrg: 0.1 };
        let prim = srhd_recover(&eos, &cons, MAX_ITER);
        let finite = prim.rho.is_finite() && prim.pre.is_finite() && prim.vel[0].is_finite();
        assert!(
            !finite,
            "unphysical cons silently recovered to a finite prim (rho={}, pre={}, v={}); \
             the kernel path must surface NaN, not mask it",
            prim.rho, prim.pre, prim.vel[0]
        );
    }

    #[test]
    fn valid_srhd_state_is_ok() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Srhd;
        let prim = Prim { rho: 1.0, vel: Tensor::new([0.3]), pre: 1.0 };
        let cons = regime.to_conserved(&eos, &prim);
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.is_ok());
    }
}
