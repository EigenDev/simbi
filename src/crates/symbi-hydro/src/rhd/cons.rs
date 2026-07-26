// =============================================================================
// rhd/cons.rs
//
// the RHD cons->prim recovery — a carrier-generic Newton-Raphson on the pressure
// root (`Scalar::iterate`), then the algebraic velocity/Lorentz/density recovery.
// branch-free core (`rhd_recover`, what the substrate c2p kernel computes) + the
// host wrapper (`rhd_to_primitive`) that adds the C2pResult diagnostics post-hoc.
// =============================================================================

use crate::c2p_result::C2pResult;
use crate::eos::Eos;
use crate::rhd::lorentz_factor;
use crate::spatial_metric::SpatialMetric;
use crate::state::{Cons, Prim};
use symbi_algebra::{OrderedNumeric, Tensor};
use symbi_ir::algebra::Scalar;

/// maximum newton-raphson iterations for RHD cons2prim on the HOST (early-break).
/// the substrate kernel bakes its own fixed count (build.rs passes 20 to the gv
/// builder) — both share the `rhd_recover` body; the count is a tunable knob.
const MAX_ITER: usize = 100;

/// the branch-free RHD cons->prim recovery — THE single-source physics: the pressure
/// is the root of a 1D equation found by a carrier-generic Newton (`Scalar::iterate`),
/// then velocity/Lorentz/density follow algebraically. NO floors, NO guards — exactly
/// what the substrate c2p kernel computes (the `rhd_to_primitive` wrapper adds the
/// host C2pResult diagnostics post-hoc, matching `Newtonian::to_primitive`).
///
/// EOS-generic: works through `eos.pressure()` / `eos.sound_speed_sq()`, so no gamma is hardcoded. traced at
/// `S = Gv` (`symbi_discretize::rhd_c2p_gv`) it lowers to the IterateInline c2p kernel;
/// at `S = f64/f32` it runs as a plain early-breaking loop. `max_iter` caps the Newton.
pub fn rhd_recover<S: Scalar, const D: usize>(
    eos: &impl Eos<S>,
    cons: &Cons<S, D>,
    metric: &SpatialMetric<S, D>,
    max_iter: usize,
) -> Prim<S, D> {
    let dd = cons.den;
    let tau = cons.nrg;
    // the conserved-momentum norm |S|^2 = gamma^{ij} S_i S_j (S_i is COVARIANT -> contract with
    // the inverse spatial metric). flat/orthonormal -> identity -> bit-identical to euclidean S.S.
    // THIS is the SR->GR distinction: the metric is a carrier-generic value the physics contracts
    // with, transported by the homomorphism like `eos`.
    let s_mag = metric.norm_sq_cov(&cons.mom).sqrt();
    // the rescaled conserved-momentum norm r^2 = |S|^2 / D^2 and the shared c2p velocity ceiling
    // v_limit^2 = r^2/(1+r^2) (KKC h0 = 1). clamping every recovered v^2 to this keeps the Lorentz
    // factor finite for an OUT-of-cone input (no NaN to poison a neighbour) while leaving an
    // in-cone recovery bit-identical (its true v is strictly below the ceiling). the cone test at
    // the end drives the pressure non-positive to signal the out-of-cone case. same contract as
    // rmhd_recover — see c2p_result::relativistic_velocity_ceiling_sq / relativistic_cone_residual.
    let r_sq = s_mag * s_mag / (dd * dd);
    let v_ceiling_sq = crate::c2p_result::relativistic_velocity_ceiling_sq(r_sq);

    // initial pressure guess: |S - D - tau|
    let p_init = (s_mag - dd - tau).abs();

    // ONE newton step on the pressure: given a guess, derive (v, W, rho, eps), form
    // f(p) = eos.pressure - p and g = df/dp = cs_rel^2*v^2 - 1, return the updated guess.
    let newton_step = |p_eq: S| -> S {
        let et = tau + dd + p_eq;
        // an undershooting iterate (et < |S|) would give v^2 > 1 and a NaN Lorentz factor; the
        // ceiling keeps the iterate finite. inactive for an in-cone root, so the recovered p is
        // unchanged for a valid state.
        let v_sq = (s_mag * s_mag / (et * et)).min(v_ceiling_sq);
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

    // recover primitives: the CONTRAVARIANT 3-velocity v^i = gamma^{ij} S_j / (tau+D+p) (RAISE the
    // covariant conserved momentum), W = 1/sqrt(1 - v.v), rho = D/W. flat/orthonormal -> gamma^{ij} =
    // identity -> v^i = S_i/et bit-identically; a real (GR) gamma raises the index (the Valencia
    // recovery) so `norm_sq_contra(v)` = gamma_ij v^i v^j = |S|^2/et^2 is consistent with the Newton's
    // norm_sq_cov (without the raise, S_i/et is the COVARIANT velocity — wrong for non-identity gamma).
    let et = tau + dd + p_eq;
    let vel = metric.raise(&cons.mom).map(|s| s / et);
    // density from the CEILING-clamped Lorentz factor: an out-of-cone state recovers a finite (if
    // flagged) rho; the clamp keeps it off NaN. the velocity VECTOR is left exact (finite, possibly
    // superluminal — the post-hoc diagnostic flags it); the clamp only sanitizes rho.
    let ww = lorentz_factor(metric.norm_sq_contra(&vel).min(v_ceiling_sq));
    let rho = dd / ww;
    // Wu-2017 cone test: q(U)/D <= 0 means no physical subluminal p > 0 solution exists. drive the
    // pressure to the shared non-positive sentinel so the FOFC probe and the c2p diagnostic flag
    // the zone (fail-loud); no spurious recovered pressure is accepted. identical contract
    // and math to rmhd_recover.
    let qq = tau / dd;
    let cone_ok = crate::c2p_result::relativistic_cone_residual(qq, r_sq).cmp_gt(S::ZERO);
    let pre = S::select(
        cone_ok,
        p_eq,
        S::from_f64(crate::c2p_result::C2P_CONE_FAIL_PRESSURE),
    );
    Prim { rho, vel, pre }
}

/// the host RHD cons->prim: the branch-free `rhd_recover` plus post-hoc C2pResult
/// diagnostics. no silent floor — the value is the raw recovered state, the ErrorCode
/// is the explicit signal (matches `Newtonian::to_primitive`; feedback_no_silent_floors).
///
/// **host-only** (Tier 1.7): `S: OrderedNumeric` because the diagnostic check uses
/// native `<` / `<=` / `==` on a host scalar. the kernel path is `rhd_recover` above
/// (carrier-generic over `S: Scalar`); this wrapper is host-only.
pub(crate) fn rhd_to_primitive<S: Scalar + OrderedNumeric, const D: usize>(
    eos: &impl Eos<S>,
    cons: &Cons<S, D>,
) -> C2pResult<Prim<S, D>> {
    let dd = cons.den;

    // input guard: clearly-invalid conserved density. a host-only early-out (absent from the
    // kernel path) that keeps the recovery's `dd/ww`, `tau+dd+p` finite for callers.
    if let Some(code) = crate::c2p_result::relativistic_density_guard(dd) {
        let floored = Prim {
            rho: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR),
            vel: Tensor::zeros(),
            pre: S::from_f64(crate::c2p_result::C2P_FAILURE_FLOOR),
        };
        return C2pResult::err(floored, code);
    }

    // flat/orthonormal frame -> the spatial metric is identity (bit-identical to euclidean norms);
    // a genuine GR metric is threaded here once the conserved state is densitized.
    let prim = rhd_recover(eos, cons, &SpatialMetric::flat(), MAX_ITER);

    // post-hoc diagnostics on the raw recovered state (shared RHD/RMHD contract; tier-1 #5).
    let v_sq = prim.vel.dot(&prim.vel);
    let code = crate::c2p_result::relativistic_c2p_code(prim.rho, prim.pre, v_sq);
    if code.is_ok() {
        C2pResult::ok(prim)
    } else {
        C2pResult::err(prim, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;
    use crate::regime::Regime;
    use crate::rhd::Rhd;

    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn roundtrip_stationary() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx_rel(prim.rho, prim2.rho, 1e-10));
        assert!(approx_rel(prim.vel[0], prim2.vel[0], 1e-10));
        assert!(approx_rel(prim.pre, prim2.pre, 1e-10));
    }

    #[test]
    fn roundtrip_mildly_relativistic() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.5]),
            pre: 1.0,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx_rel(prim.rho, prim2.rho, 1e-10));
        assert!(approx_rel(prim.vel[0], prim2.vel[0], 1e-10));
        assert!(approx_rel(prim.pre, prim2.pre, 1e-10));
    }

    #[test]
    fn roundtrip_ultra_relativistic() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.99]),
            pre: 10.0,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let prim2 = regime.to_primitive(&eos, &cons).unwrap();
        assert!(approx_rel(prim.rho, prim2.rho, 1e-8));
        assert!(approx_rel(prim.vel[0], prim2.vel[0], 1e-8));
        assert!(approx_rel(prim.pre, prim2.pre, 1e-8));
    }

    #[test]
    fn roundtrip_3d() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
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
        let regime = Rhd;
        for &(rho, pre) in &[(10.0, 13.33), (1.0, 1e-6), (1e-2, 1e-4)] {
            let prim = Prim {
                rho,
                vel: Tensor::new([0.0]),
                pre,
            };
            let cons = regime.to_conserved(&eos, &prim);
            let prim2 = regime.to_primitive(&eos, &cons).unwrap();
            assert!(
                approx_rel(prim.rho, prim2.rho, 1e-8),
                "rho: {} vs {} (input rho={}, pre={})",
                prim.rho,
                prim2.rho,
                rho,
                pre
            );
            assert!(
                approx_rel(prim.pre, prim2.pre, 1e-8),
                "pre: {} vs {} (input rho={}, pre={})",
                prim.pre,
                prim2.pre,
                rho,
                pre
            );
        }
    }

    #[test]
    fn negative_density_detected() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let cons = Cons {
            den: -1.0,
            mom: Tensor::new([0.0]),
            nrg: 1.0,
        };
        let result = regime.to_primitive(&eos, &cons);
        assert!(
            result
                .error
                .contains(crate::c2p_result::ErrorCode::NEGATIVE_DENSITY)
        );
        assert!(result.value.rho > 0.0);
    }

    #[test]
    fn unphysical_cons_returns_error() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        // huge momentum, tiny density+energy -> superluminal or negative pressure
        let cons = Cons {
            den: 1e-14,
            mom: Tensor::new([100.0]),
            nrg: 1e-14,
        };
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.error.is_err());
    }

    // the unified relativistic-c2p contract (shared with rmhd_recover): the branch-free kernel body
    // recovers a FINITE state for an out-of-cone conserved input — density from the ceiling-clamped
    // Lorentz factor, pressure driven to the shared non-positive `C2P_CONE_FAIL_PRESSURE` sentinel
    // by the Wu-2017 cone test — a finite sentinel. rationale for the change from the old NaN convention:
    // the FOFC probe's `finite_pos(pre)` rejects the non-positive sentinel IDENTICALLY to a NaN (so
    // the fail-loud is preserved), while a finite sentinel cannot poison a neighbour's
    // reconstruction the way an absorbing NaN does (the demonstrated FM-torus / RMHD failure mode).
    // where FOFC is inactive, the sentinel still fails loud: a sound speed `sqrt(gamma p / rho h)`
    // on p < 0 goes non-finite and trips the CFL check. feedback_no_silent_floors holds — the state
    // is FLAGGED (non-positive pressure), never floored to a spurious-physical value.
    #[test]
    fn kernel_path_unphysical_cons_recovers_finite_and_flagged() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        // s_mag = 10 >> d + tau = 1.1 => out of cone (tau + d < sqrt(d^2 + s_mag^2)).
        let cons: Cons<f64, 1> = Cons {
            den: 1.0,
            mom: Tensor::new([10.0]),
            nrg: 0.1,
        };
        let prim = rhd_recover(&eos, &cons, &SpatialMetric::flat(), MAX_ITER);
        assert!(
            prim.rho.is_finite() && prim.pre.is_finite() && prim.vel[0].is_finite(),
            "unified c2p must recover a FINITE state (rho={}, pre={}, v={}), not a NaN",
            prim.rho,
            prim.pre,
            prim.vel[0]
        );
        assert!(
            prim.rho > 0.0,
            "density must stay positive/finite, got {}",
            prim.rho
        );
        assert!(
            prim.pre <= 0.0,
            "out-of-cone state must flag a non-positive pressure (the FOFC-visible signal), got {}",
            prim.pre
        );
    }

    #[test]
    fn valid_rhd_state_is_ok() {
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let regime = Rhd;
        let prim = Prim {
            rho: 1.0,
            vel: Tensor::new([0.3]),
            pre: 1.0,
        };
        let cons = regime.to_conserved(&eos, &prim);
        let result = regime.to_primitive(&eos, &cons);
        assert!(result.is_ok());
    }
}
