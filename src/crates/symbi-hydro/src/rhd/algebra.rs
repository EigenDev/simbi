// =============================================================================
// rhd/algebra.rs
//
// the relativistic-hydrodynamic elementals — pure pointwise functions: lorentz
// factor, relativistic enthalpy, relativistic sound speed, the inertial density
// rho*h*W^2. shared with RMHD's hydrodynamic core (rho/vel/pre quantities); the
// magnetic four-vector helpers live in rmhd/algebra.rs. carrier-generic
// (S: Scalar), GPU-callable. conventions: c = 1, flat Minkowski (+---).
// preconditions: callers ensure 0 <= v^2 < 1 (the c2p admissibility guards);
// these helpers do not clamp — that would hide infeasibility from the recover step.
// =============================================================================

use symbi_ir::algebra::Scalar;
use crate::eos::Eos;

/// lorentz factor from velocity squared: W = 1/sqrt(1 - v^2).
#[inline]
pub fn lorentz_factor<S: Scalar>(v_sq: S) -> S {
    S::ONE / (S::ONE - v_sq).sqrt()
}

/// lorentz factor squared: W^2 = 1/(1 - v^2).
#[inline]
pub fn lorentz_factor_sq<S: Scalar>(v_sq: S) -> S {
    S::ONE / (S::ONE - v_sq)
}

/// relativistic specific enthalpy: h = 1 + e_int + p/rho.
/// for ideal gas: h = 1 + gamma*p / (rho*(gamma-1)).
/// EOS-generic — works for any EOS providing internal_energy().
#[inline]
pub fn enthalpy<S: Scalar>(eos: &impl Eos<S>, rho: S, pre: S) -> S {
    S::ONE + eos.internal_energy(rho, pre) + pre / rho
}

/// relativistic sound speed squared: cs_rel^2 = cs_newt^2 / h.
/// sqrt-free: `Eos::sound_speed_sq` returns cs_newt^2 directly (IdealGas: gamma*p/rho),
/// so the per-cell sqrt-then-square the newton step paid for is gone.
#[inline]
pub fn sound_speed_sq<S: Scalar>(eos: &impl Eos<S>, rho: S, pre: S) -> S {
    eos.sound_speed_sq(rho, pre) / enthalpy(eos, rho, pre)
}

/// the relativistic INERTIAL DENSITY: rho * h * W^2 (the `wgam2` of the literature).
/// appears in mom = rho h W^2 v, nrg = rho h W^2 - p - D, the c2p Newton residual,
/// and the curvilinear geometric source. one named quantity instead of three inline copies.
#[inline]
pub fn enthalpy_density<S: Scalar>(eos: &impl Eos<S>, rho: S, pre: S, v_sq: S) -> S {
    rho * enthalpy(eos, rho, pre) * lorentz_factor_sq(v_sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eos::IdealGas;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * a.abs().max(b.abs()).max(1.0)
    }
    fn approx_rel(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn lorentz_factor_zero_velocity() {
        assert!(approx(lorentz_factor(0.0), 1.0));
    }

    #[test]
    fn lorentz_factor_half_c() {
        // v = 0.5c, v^2 = 0.25, W = 1/sqrt(0.75)
        let w = lorentz_factor(0.25);
        assert!(approx(w, 1.0 / 0.75_f64.sqrt()));
    }

    #[test]
    fn lorentz_factor_090c() {
        // v = 0.9c, W = 1/sqrt(1-0.81) = 1/sqrt(0.19) ~ 2.294
        let w = lorentz_factor(0.81);
        assert!(approx(w, 1.0 / 0.19_f64.sqrt()));
    }

    #[test]
    fn lorentz_factor_099c() {
        // v = 0.99c, W ~ 7.089
        let w = lorentz_factor(0.99 * 0.99);
        assert!(approx(w, 1.0 / (1.0 - 0.99 * 0.99_f64).sqrt()));
    }

    #[test]
    fn enthalpy_newtonian_limit() {
        // for low pressure (p << rho), h -> 1 (newtonian limit)
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let h = enthalpy(&eos, 1.0, 1e-10);
        assert!(approx_rel(h, 1.0, 1e-8));
    }

    #[test]
    fn enthalpy_ideal_gas() {
        // h = 1 + gamma*p / (rho*(gamma-1))
        // gamma=5/3, rho=1, p=1: h = 1 + (5/3)*1 / (1*(2/3)) = 1 + 5/2 = 3.5
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let h = enthalpy(&eos, 1.0, 1.0);
        assert!(approx(h, 3.5));
    }

    #[test]
    fn enthalpy_formula_check() {
        // verify h = 1 + e_int + p/rho for arbitrary state
        let eos = IdealGas { gamma: 1.4 };
        let rho = 2.0;
        let pre = 3.0;
        let e_int = eos.internal_energy(rho, pre);
        let h = enthalpy(&eos, rho, pre);
        assert!(approx(h, 1.0 + e_int + pre / rho));
    }

    #[test]
    fn sound_speed_sq_newtonian_limit() {
        // for h ~ 1 (low p), cs_rel^2 ~ cs_newt^2
        let eos = IdealGas { gamma: 1.4 };
        let rho = 1.0;
        let pre = 1e-10;
        let cs2 = sound_speed_sq(&eos, rho, pre);
        let cs2_newt = eos.sound_speed(rho, pre).powi(2);
        assert!(approx_rel(cs2, cs2_newt, 1e-8));
    }

    #[test]
    fn sound_speed_sq_relativistic() {
        // cs_rel^2 = gamma*p / (rho*h) = cs_newt^2 / h
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let rho = 1.0;
        let pre = 1.0;
        let h = enthalpy(&eos, rho, pre);
        let cs2 = sound_speed_sq(&eos, rho, pre);
        let expected = (5.0 / 3.0) * 1.0 / (1.0 * h);
        assert!(approx(cs2, expected));
    }

    #[test]
    fn enthalpy_density_zero_velocity() {
        // at rest W = 1, so rho*h*W^2 == rho*h.
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        let (rho, pre) = (2.0, 1.5);
        let wgam2 = enthalpy_density(&eos, rho, pre, 0.0);
        assert!(approx(wgam2, rho * enthalpy(&eos, rho, pre)));
    }

    #[test]
    fn enthalpy_density_matches_inline_form() {
        // matches the rho * enthalpy * lorentz_factor_sq expression.
        let eos = IdealGas { gamma: 1.4 };
        let (rho, pre, v_sq) = (1.3, 0.7, 0.36); // |v| = 0.6c
        let want = rho * enthalpy(&eos, rho, pre) * lorentz_factor_sq(v_sq);
        assert!(approx(enthalpy_density(&eos, rho, pre, v_sq), want));
    }

    #[test]
    fn sound_speed_subluminal() {
        // cs must be < 1 for all physical states
        let eos = IdealGas { gamma: 5.0 / 3.0 };
        for &pre in &[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
            let cs2 = sound_speed_sq(&eos, 1.0, pre);
            assert!(cs2 < 1.0, "cs^2={} >= 1 at p={}", cs2, pre);
        }
    }
}
