// =============================================================================
// bondi.rs
//
// the analytic transonic bondi solution (docs/ideas/accretor.md §8): the
// initial-condition generator and the validation target for the emergent-rate
// drain. code units G*M = 1, c_inf = 1, rho_inf = 1, so the bondi radius
// r_B = 1 and every length below is in bondi radii.
//
// adiabatic (gamma > 1), at each r the algebraic system
//   bernoulli:  u^2/2 + c^2/(gamma-1) - 1/r = 1/(gamma-1)
//   mass flux:  r^2 rho u = lambda_c(gamma)          (Mdot_B / 4 pi)
//   eos:        c^2 = rho^(gamma-1)
// eliminating rho = lambda/(r^2 u) leaves g(u; r) with
// dg/du = (u^2 - c^2)/u — the stationary point IS the local sonic condition,
// at u_min = (lambda/r^2)^((gamma-1)/(gamma+1)). the subsonic root lies below
// u_min, the supersonic root above; bisection on each side is unconditionally
// bracketed. the transonic solution takes the subsonic branch for r > r_s and
// the supersonic branch for r < r_s, r_s = (5 - 3 gamma)/4.
//
// isothermal (gamma = 1): bernoulli u^2/2 + ln(rho) - 1/r = 0, c = 1,
// r_s = 1/2, u_min = 1 — the same construction with the log energy term.
//
// usage:
//   let s = bondi_profile(r, gamma);       // BondiState { rho, u, pre }
//   let mdot = mdot_bondi(gamma);          // 4 pi lambda_c(gamma)
//   let rs = sonic_radius(gamma);
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_ir::algebra::Scalar;

/// bondi-hoyle accretion rate coefficient lambda(gamma).
/// isothermal (gamma=1): exp(1.5)/4, adiabatic (gamma=5/3): 0.25.
/// general: 0.25 * (2/(5-3*gamma))^((5-3*gamma)/(2*gamma-2)).
pub fn accretion_coefficient<S: Scalar + OrderedNumeric>(gamma: S) -> S {
    let one = S::ONE;
    let diff_iso = (gamma - one).abs();

    if diff_iso < S::from_f64(1e-5) {
        // isothermal
        return S::from_f64(std::f64::consts::E.powf(1.5) / 4.0);
    }

    let five_thirds = S::from_f64(5.0 / 3.0);
    let diff_adi = (gamma - five_thirds).abs();

    if diff_adi < S::from_f64(1e-5) {
        return S::from_f64(0.25);
    }

    // general case
    let five = S::from_f64(5.0);
    let three = S::from_f64(3.0);
    let two = S::from_f64(2.0);
    let quarter = S::from_f64(0.25);

    let num = five - three * gamma;
    let den = two * gamma - two;
    let base = two / num;
    let exponent = num / den;
    quarter * base.powf(exponent)
}

/// the local transonic bondi state at radius `r` (bondi radii): density,
/// INFLOW speed magnitude (the radial velocity is `-u * rhat`), and pressure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondiState {
    pub rho: f64,
    pub u: f64,
    pub pre: f64,
}

/// the sonic radius of the spherical transonic solution, in bondi radii:
/// r_s = (5 - 3 gamma)/4 (isothermal 1/2; degenerate 0 at gamma = 5/3, where
/// the sonic surface attaches to the accretor — the validation edge).
pub fn sonic_radius(gamma: f64) -> f64 {
    (5.0 - 3.0 * gamma) / 4.0
}

/// the analytic bondi accretion rate in code units: 4 pi lambda_c(gamma).
pub fn mdot_bondi(gamma: f64) -> f64 {
    4.0 * std::f64::consts::PI * accretion_coefficient(gamma)
}

/// the transonic bondi profile at radius `r > 0` (bondi radii): subsonic
/// branch outside the sonic radius, supersonic inside, matched at r_s.
/// `1 <= gamma < 5/3` is the well-posed domain; gamma = 5/3 is accepted
/// (r_s = 0: the profile is supersonic-free — the subsonic branch everywhere).
pub fn bondi_profile(r: f64, gamma: f64) -> BondiState {
    assert!(r > 0.0, "bondi_profile: r must be positive (got {r})");
    assert!(
        (1.0..=5.0 / 3.0 + 1e-12).contains(&gamma),
        "bondi_profile: gamma must lie in [1, 5/3] (got {gamma})",
    );
    let lambda = accretion_coefficient(gamma);
    let iso = (gamma - 1.0).abs() < 1e-5;
    let a = lambda / (r * r);
    let supersonic = r < sonic_radius(gamma);

    // g(u): the bernoulli residual at fixed r after eliminating rho. the
    // stationary point u_min separates the two roots.
    let g = |u: f64| -> f64 {
        let rho = a / u;
        if iso {
            0.5 * u * u + rho.ln() - 1.0 / r
        } else {
            0.5 * u * u + rho.powf(gamma - 1.0) / (gamma - 1.0) - 1.0 / r - 1.0 / (gamma - 1.0)
        }
    };
    let u_min = if iso { 1.0 } else { a.powf((gamma - 1.0) / (gamma + 1.0)) };

    // bracket the requested branch. g -> +inf at both u -> 0 and u -> inf, so
    // stepping outward from u_min until g > 0 always closes the bracket.
    let (mut lo, mut hi) = if supersonic {
        let mut hi = u_min.max(1e-12) * 2.0;
        while g(hi) < 0.0 {
            hi *= 2.0;
        }
        (u_min, hi)
    } else {
        let mut lo = u_min.min(1.0) * 0.5;
        while g(lo) < 0.0 {
            lo *= 0.5;
        }
        (lo, u_min)
    };

    // exactly at r_s the two roots coincide at u_min and g(u_min) ~ 0; the
    // bisection below then converges to u_min from either side.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let (gm, want_neg_side_low) = (g(mid), supersonic);
        // on the supersonic branch g goes negative BELOW the root (toward
        // u_min) and positive above; subsonic is the mirror image.
        if (gm < 0.0) == want_neg_side_low {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1e-15 * hi.max(1.0) {
            break;
        }
    }
    let u = 0.5 * (lo + hi);
    let rho = a / u;
    let pre = if iso { rho } else { rho.powf(gamma) / gamma };
    BondiState { rho, u, pre }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_values_from_the_spec() {
        // lambda_c: isothermal e^{3/2}/4, gamma=1.4 -> 0.625, 5/3 -> 0.25.
        let pi4 = 4.0 * std::f64::consts::PI;
        assert!((mdot_bondi(1.0) / pi4 - std::f64::consts::E.powf(1.5) / 4.0).abs() < 1e-12);
        assert!((mdot_bondi(1.4) / pi4 - 0.625).abs() < 1e-12);
        assert!((mdot_bondi(5.0 / 3.0) / pi4 - 0.25).abs() < 1e-12);
        assert_eq!(sonic_radius(1.0), 0.5);
        assert!((sonic_radius(1.4) - 0.2).abs() < 1e-15);
        assert!(sonic_radius(5.0 / 3.0).abs() < 1e-15);
    }

    // the bernoulli invariant, reconstructed from the returned state, must
    // vanish at every radius on both branches.
    fn bernoulli_residual(r: f64, gamma: f64, s: &BondiState) -> f64 {
        if (gamma - 1.0).abs() < 1e-5 {
            0.5 * s.u * s.u + s.rho.ln() - 1.0 / r
        } else {
            let c2 = s.rho.powf(gamma - 1.0);
            0.5 * s.u * s.u + c2 / (gamma - 1.0) - 1.0 / r - 1.0 / (gamma - 1.0)
        }
    }

    #[test]
    fn bernoulli_and_mass_flux_hold_on_both_branches() {
        for gamma in [1.0, 1.2, 1.4, 1.5] {
            let rs = sonic_radius(gamma);
            for r in [0.05, 0.3 * rs.max(0.1), 0.9 * rs.max(0.1), 1.5 * rs.max(0.1), 1.0, 3.0, 10.0]
            {
                let s = bondi_profile(r, gamma);
                assert!(
                    bernoulli_residual(r, gamma, &s).abs() < 1e-10,
                    "gamma {gamma} r {r}: bernoulli residual {}",
                    bernoulli_residual(r, gamma, &s),
                );
                let flux = r * r * s.rho * s.u;
                let lambda = accretion_coefficient(gamma);
                assert!(
                    (flux - lambda).abs() < 1e-10 * lambda,
                    "gamma {gamma} r {r}: mass flux {flux} != lambda {lambda}",
                );
            }
        }
    }

    #[test]
    fn branches_are_transonic_and_match_at_the_sonic_radius() {
        for gamma in [1.0, 1.2, 1.4] {
            let rs = sonic_radius(gamma);
            let cs = |s: &BondiState| -> f64 {
                if (gamma - 1.0).abs() < 1e-5 { 1.0 } else { s.rho.powf(gamma - 1.0).sqrt() }
            };
            let inside = bondi_profile(0.5 * rs, gamma);
            let outside = bondi_profile(2.0 * rs, gamma);
            assert!(inside.u > cs(&inside), "gamma {gamma}: not supersonic inside r_s");
            assert!(outside.u < cs(&outside), "gamma {gamma}: not subsonic outside r_s");
            // the two branches meet at r_s: u -> u_s = sqrt(1/(2 r_s)) from both sides.
            let u_s = (1.0 / (2.0 * rs)).sqrt();
            let just_in = bondi_profile(rs * (1.0 - 1e-6), gamma);
            let just_out = bondi_profile(rs * (1.0 + 1e-6), gamma);
            assert!((just_in.u - u_s).abs() < 1e-3 * u_s, "gamma {gamma}: inner limit off u_s");
            assert!((just_out.u - u_s).abs() < 1e-3 * u_s, "gamma {gamma}: outer limit off u_s");
        }
    }

    #[test]
    fn far_field_approaches_ambient() {
        for gamma in [1.0, 1.2, 1.4, 1.5] {
            let s = bondi_profile(100.0, gamma);
            assert!((s.rho - 1.0).abs() < 2e-2, "gamma {gamma}: rho(100) = {}", s.rho);
            assert!(s.u < 1e-3, "gamma {gamma}: u(100) = {}", s.u);
            let p_inf = if (gamma - 1.0).abs() < 1e-5 { 1.0 } else { 1.0 / gamma };
            assert!((s.pre - p_inf).abs() < 3e-2, "gamma {gamma}: pre(100) = {}", s.pre);
        }
    }

    #[test]
    fn gamma_five_thirds_is_the_degenerate_edge() {
        // r_s = 0: the subsonic branch everywhere, still bernoulli-consistent.
        let s = bondi_profile(0.05, 5.0 / 3.0);
        assert!(bernoulli_residual(0.05, 5.0 / 3.0, &s).abs() < 1e-10);
    }
}
