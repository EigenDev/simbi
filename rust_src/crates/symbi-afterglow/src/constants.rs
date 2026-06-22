// =============================================================================
// constants.rs
//
// physical constants in CGS (Gaussian) units — the system the afterglow synchrotron
// physics is written in (ported from the legacy `constants.hpp`). each constant is a
// typed `Quantity` (src/units.rs) so it carries its dimension into the formulae and a
// dimensional mistake is a compile error. `PI` stays a bare f64 (it is dimensionless).
//
// usage:
//  let b: MagneticField = (8.0 * PI * eps_b * rho_e).sqrt(); // rho_e: EnergyDensity
//  let n_e: NumberDensity = rho_cgs / M_P;                    // rho_cgs: MassDensity
// =============================================================================

use crate::units::{Action, Area, Charge, Mass, Time, Velocity};

/// pi, as the synchrotron formulae spell it (CGS prefactors carry explicit 1/(4 pi) etc.).
pub const PI: f64 = std::f64::consts::PI;

/// speed of light [cm / s].
pub const C_LIGHT: Velocity = Velocity::new(2.997_924_58e10);

/// planck constant [erg s] — converts photon energy to frequency (nu = E / h).
pub const H_PLANCK: Action = Action::new(6.626_075_5e-27);

/// elementary charge [statC] (esu) — Gaussian units, so it appears bare in B-field formulae.
pub const E_CHARGE: Charge = Charge::new(4.803_206_8e-10);

/// electron mass [g].
pub const M_E: Mass = Mass::new(9.109_389e-28);

/// proton mass [g] — sets the upstream number density n = rho / m_p.
pub const M_P: Mass = Mass::new(1.672_623_1e-24);

/// thomson cross section [cm^2].
pub const SIGMA_THOMSON: Area = Area::new(6.6524e-25);

/// seconds per day — observer light curves are reported in days.
pub const SECONDS_PER_DAY: Time = Time::new(86_400.0);
