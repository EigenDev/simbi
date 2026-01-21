// =============================================================================
// eos.rs
//
// equations of state for hydrodynamics.
// relates primitive variables (pressure, density, temperature) to
// conserved quantities (energy, momentum).
//
// current implementations:
//   - ideal gas eos: p = (γ - 1) * ρ * ε
//
// usage:
//   let pressure = ideal_gas_pressure(rho, specific_energy, gamma);
//   let sound_speed = ideal_gas_sound_speed(rho, pressure, gamma);
// =============================================================================

/// ideal gas equation of state: p = (γ - 1) * ρ * ε
///
/// # arguments
/// * `rho` - mass density
/// * `specific_energy` - internal energy per unit mass (ε = E/ρ - v²/2)
/// * `gamma` - adiabatic index (ratio of specific heats)
///
/// # returns
/// pressure
#[inline]
pub fn ideal_gas_pressure(rho: f64, specific_energy: f64, gamma: f64) -> f64 {
    (gamma - 1.0) * rho * specific_energy
}

/// sound speed for ideal gas: c = sqrt(γ * p / ρ)
///
/// # arguments
/// * `rho` - mass density
/// * `pressure` - gas pressure
/// * `gamma` - adiabatic index
///
/// # returns
/// adiabatic sound speed
#[inline]
pub fn ideal_gas_sound_speed(rho: f64, pressure: f64, gamma: f64) -> f64 {
    (gamma * pressure / rho).sqrt()
}

/// specific internal energy from total energy: ε = E/ρ - v²/2
///
/// # arguments
/// * `total_energy` - total energy (E)
/// * `rho` - mass density
/// * `velocity_squared` - |v|²
///
/// # returns
/// specific internal energy
#[inline]
pub fn specific_internal_energy(total_energy: f64, rho: f64, velocity_squared: f64) -> f64 {
    total_energy / rho - 0.5 * velocity_squared
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideal_gas_pressure() {
        let rho = 1.0;
        let epsilon = 2.5;
        let gamma = 1.4;

        let p = ideal_gas_pressure(rho, epsilon, gamma);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sound_speed() {
        let rho = 1.0;
        let p = 1.0;
        let gamma = 1.4;

        let c = ideal_gas_sound_speed(rho, p, gamma);
        assert!((c - 1.183).abs() < 0.001);
    }

    #[test]
    fn test_specific_internal_energy() {
        let e_total = 10.0;
        let rho = 2.0;
        let v_sq = 4.0;

        let epsilon = specific_internal_energy(e_total, rho, v_sq);
        assert_eq!(epsilon, 3.0);
    }
}
