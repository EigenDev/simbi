// =============================================================================
// sink.rs
//
// the analytic Bondi accretion coefficient lambda(gamma): the transonic
// mass-accretion rate `Mdot = 4 pi lambda(gamma) (GM)^2 rho_inf / c_inf^3`. the
// validation target for the well-posed drain's EMERGENT rate (docs/ideas/
// accretor.md §6). the retired KMK04 sink-weight / weighted-sum / mdot machinery
// was reaped when the drain replaced the mass-only sink.
//
// usage:
//   let lambda = accretion_coefficient(gamma);
// =============================================================================

use symbi_ir::algebra::Scalar;
use symbi_algebra::OrderedNumeric;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn accretion_coefficient_isothermal() {
        let lambda = accretion_coefficient(1.0_f64);
        let expected = std::f64::consts::E.powf(1.5) / 4.0;
        assert!(approx(lambda, expected));
    }

    #[test]
    fn accretion_coefficient_adiabatic() {
        let lambda = accretion_coefficient(5.0_f64 / 3.0);
        assert!(approx(lambda, 0.25));
    }

    #[test]
    fn accretion_coefficient_gamma_1_4() {
        let gamma = 1.4_f64;
        let lambda = accretion_coefficient(gamma);
        // general formula
        let num = 5.0 - 3.0 * gamma; // 0.8
        let den = 2.0 * gamma - 2.0; // 0.8
        let expected = 0.25 * (2.0 / num).powf(num / den);
        assert!(approx(lambda, expected));
    }

}
