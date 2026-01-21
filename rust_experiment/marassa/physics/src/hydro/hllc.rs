// =============================================================================
// hllc.rs
//
// production hllc (harten-lax-van leer-contact) riemann solver.
// implements the exact algorithm from simbi c++ codebase with all fixes:
//   - proper contact wave resolution
//   - fleischmann et al. (2020) low-mach fix
//   - quirk's fix for odd-even decoupling
//   - accurate wave speed estimation
//
// references:
//   - toro (2009): riemann solvers and numerical methods for fluid dynamics
//   - fleischmann et al. (2020): low-mach hllc fix
//   - quirk (1994): odd-even decoupling problem
//
// usage:
//   let flux = hllc_newtonian(primL, primR, nhat, gamma);
// =============================================================================

use super::state::{Conserved, Newtonian, Primitive, Regime};
use std::ops::{Add, Mul, Sub};

// =============================================================================
// wave speeds structure
// =============================================================================

#[derive(Debug, Copy, Clone)]
pub struct WaveSpeeds {
    pub left: f64,
    pub right: f64,
}

impl WaveSpeeds {
    #[inline]
    pub fn new(left: f64, right: f64) -> Self {
        Self { left, right }
    }

    #[inline]
    pub fn min(&self) -> f64 {
        self.left.min(self.right)
    }

    #[inline]
    pub fn max(&self) -> f64 {
        self.left.max(self.right)
    }
}

// =============================================================================
// contact properties
// =============================================================================

#[derive(Debug, Copy, Clone)]
pub struct ContactProperties {
    pub speed: f64,
    pub pressure: f64,
}

impl ContactProperties {
    #[inline]
    pub fn new(speed: f64, pressure: f64) -> Self {
        Self { speed, pressure }
    }
}

// =============================================================================
// wave properties (speeds + contact)
// =============================================================================

#[derive(Debug, Copy, Clone)]
pub struct WaveProperties {
    pub speeds: WaveSpeeds,
    pub contact: ContactProperties,
}

// =============================================================================
// newtonian wave speed estimation (davis estimate)
// =============================================================================

/// computes wave speeds for newtonian hydrodynamics using davis estimate.
/// estimates fastest left and right-going waves based on sound speeds.
fn newtonian_wave_speeds<const RANK: usize>(
    primL: &Primitive<Newtonian, RANK>,
    primR: &Primitive<Newtonian, RANK>,
    nhat: &[f64; RANK],
    gamma: f64,
) -> WaveSpeeds {
    let csL = primL.sound_speed(gamma);
    let csR = primR.sound_speed(gamma);

    let vnL = dot_product(&primL.vel, nhat);
    let vnR = dot_product(&primR.vel, nhat);

    let aL = (vnL - csL).min(vnR - csR);
    let aR = (vnL + csL).max(vnR + csR);

    WaveSpeeds::new(aL, aR)
}

// =============================================================================
// newtonian contact wave properties
// =============================================================================

/// computes contact wave properties using pvrs estimate.
/// follows toro's exact riemann solver initialization.
fn newtonian_contact_properties<const RANK: usize>(
    primL: &Primitive<Newtonian, RANK>,
    primR: &Primitive<Newtonian, RANK>,
    nhat: &[f64; RANK],
    gamma: f64,
) -> ContactProperties {
    let rhoL = primL.rho;
    let rhoR = primR.rho;
    let pL = primL.p;
    let pR = primR.p;
    let csL = primL.sound_speed(gamma);
    let csR = primR.sound_speed(gamma);
    let vL = dot_product(&primL.vel, nhat);
    let vR = dot_product(&primR.vel, nhat);

    // pvrs pressure estimate
    let rho_bar = 0.5 * (rhoL + rhoR);
    let c_bar = 0.5 * (csL + csR);
    let pvrs = 0.5 * (pL + pR) - 0.5 * (vR - vL) * rho_bar * c_bar;
    let pmin = pL.min(pR);
    let pmax = pL.max(pR);

    // adaptive pressure guess with safety bounds
    let q_user = 2.0;
    let p0 = if pvrs < pmin {
        pvrs.max(1e-10)
    } else if pvrs > pmax && (pmax / pmin) < q_user {
        pvrs
    } else {
        // two-rarefaction approximation
        let gamma_factor = (gamma - 1.0) / (2.0 * gamma);
        let pL_pow = pL.powf(gamma_factor);
        let pR_pow = pR.powf(gamma_factor);
        let numerator = csL + csR - 0.5 * (gamma - 1.0) * (vR - vL);
        let denominator = csL / pL_pow + csR / pR_pow;
        (numerator / denominator).powf(1.0 / gamma_factor)
    };

    // compute auxiliary variables
    let alpha_L = 2.0 / ((gamma + 1.0) * rhoL);
    let alpha_R = 2.0 / ((gamma + 1.0) * rhoR);
    let beta_L = pL * (gamma - 1.0) / (gamma + 1.0);
    let beta_R = pR * (gamma - 1.0) / (gamma + 1.0);

    let gL = ((alpha_L) / (p0 + beta_L)).sqrt();
    let gR = ((alpha_R) / (p0 + beta_R)).sqrt();

    // iterative solution for p_star (newton-raphson)
    let mut p_star = p0;
    const MAX_ITER: usize = 10;
    const TOL: f64 = 1e-6;

    for _ in 0..MAX_ITER {
        let fL = (p_star - pL) * gL;
        let fR = (p_star - pR) * gR;
        let f = fL + fR + (vR - vL);

        let dfL = gL * (1.0 - 0.5 * (p_star - pL) / (beta_L + p_star));
        let dfR = gR * (1.0 - 0.5 * (p_star - pR) / (beta_R + p_star));
        let df = dfL + dfR;

        let dp = -f / df;
        p_star += dp;

        if dp.abs() < TOL * p_star {
            break;
        }
    }

    p_star = p_star.max(1e-10);

    // contact wave speed
    let aL = if p_star > pL {
        // shock
        (1.0 + ((gamma + 1.0) / (2.0 * gamma)) * (p_star / pL - 1.0)).sqrt()
    } else {
        // rarefaction
        1.0
    };

    let aR = if p_star > pR {
        (1.0 + ((gamma + 1.0) / (2.0 * gamma)) * (p_star / pR - 1.0)).sqrt()
    } else {
        1.0
    };

    let a_star = 0.5 * (vL + vR)
        + 0.5 * ((p_star - pR) / (aR * csR * rhoR) - (p_star - pL) / (aL * csL * rhoL));

    ContactProperties::new(a_star, p_star)
}

// =============================================================================
// newtonian wave properties (combined)
// =============================================================================

fn newtonian_wave_properties<const RANK: usize>(
    primL: &Primitive<Newtonian, RANK>,
    primR: &Primitive<Newtonian, RANK>,
    nhat: &[f64; RANK],
    gamma: f64,
) -> WaveProperties {
    let speeds = newtonian_wave_speeds(primL, primR, nhat, gamma);
    let contact = newtonian_contact_properties(primL, primR, nhat, gamma);
    WaveProperties { speeds, contact }
}

// =============================================================================
// low-mach correction (fleischmann et al. 2020)
// =============================================================================

/// computes adaptive dissipation parameter phi for low-mach correction.
/// phi = 1 for high-mach flows, phi < 1 for low-mach flows.
fn compute_adaptive_phi<const RANK: usize>(
    primL: &Primitive<Newtonian, RANK>,
    primR: &Primitive<Newtonian, RANK>,
    nhat: &[f64; RANK],
    gamma: f64,
    enable_lowmach_fix: bool,
) -> f64 {
    if !enable_lowmach_fix {
        return 1.0;
    }

    const MACH_LIM: f64 = 0.3;

    let csL = primL.sound_speed(gamma);
    let csR = primR.sound_speed(gamma);
    let vnL = dot_product(&primL.vel, nhat);
    let vnR = dot_product(&primR.vel, nhat);

    let mach_L = vnL.abs() / csL;
    let mach_R = vnR.abs() / csR;
    let ma_local = mach_L.max(mach_R);

    if ma_local >= MACH_LIM {
        return 1.0;
    }

    // smooth transition for low mach numbers
    let phi = (ma_local / MACH_LIM).min(1.0);
    phi * phi // quadratic cutoff
}

// =============================================================================
// quirk's fix for odd-even decoupling
// =============================================================================

/// detects if quirk's strong shock fix should be applied.
/// returns true if pressure ratio exceeds threshold.
fn quirk_strong_shock(pL: f64, pR: f64) -> bool {
    const QUIRK_THRESHOLD: f64 = 10.0;
    let pressure_ratio = (pL / pR).max(pR / pL);
    pressure_ratio > QUIRK_THRESHOLD
}

// =============================================================================
// hllc flux computation (newtonian)
// =============================================================================

/// computes hllc flux for newtonian hydrodynamics.
/// includes contact wave resolution, low-mach fix, and quirk's fix.
pub fn hllc_newtonian<const RANK: usize>(
    primL: &Primitive<Newtonian, RANK>,
    primR: &Primitive<Newtonian, RANK>,
    nhat: &[f64; RANK],
    gamma: f64,
    enable_lowmach_fix: bool,
) -> Conserved<Newtonian, RANK> {
    // check for quirk's strong shock (fallback to hlle if needed)
    if RANK > 1 && quirk_strong_shock(primL.p, primR.p) {
        return super::riemann::hlle_newtonian(primL, primR, nhat, gamma);
    }

    // convert to conserved and flux
    let uL = primL.to_conserved(gamma);
    let uR = primR.to_conserved(gamma);
    let fL = compute_flux(primL, nhat, gamma);
    let fR = compute_flux(primR, nhat, gamma);

    // compute wave speeds and contact properties
    let wave_info = newtonian_wave_properties(primL, primR, nhat, gamma);
    let aL = wave_info.speeds.min();
    let aR = wave_info.speeds.max();
    let a_star = wave_info.contact.speed;
    let p_star = wave_info.contact.pressure;

    // compute left star state
    let vnL = dot_product(&primL.vel, nhat);
    let vnR = dot_product(&primR.vel, nhat);

    let facL = 1.0 / (aL - a_star);
    let rhostarL = facL * (aL - vnL) * uL.den;
    let mom_term1 = scale_vec(&uL.mom, (aL - vnL) * facL);
    let mom_term2 = scale_vec(nhat, (p_star - primL.p) * facL);
    let mut mstarL = [0.0; RANK];
    for i in 0..RANK {
        mstarL[i] = mom_term1[i] + mom_term2[i];
    }
    let estarL = facL * (uL.nrg * (aL - vnL) + p_star * a_star - primL.p * vnL);

    let starStateL = Conserved::<Newtonian, RANK>::new(rhostarL, mstarL, estarL);

    // compute right star state
    let facR = 1.0 / (aR - a_star);
    let rhostarR = facR * (aR - vnR) * uR.den;
    let mom_term1 = scale_vec(&uR.mom, (aR - vnR) * facR);
    let mom_term2 = scale_vec(nhat, (p_star - primR.p) * facR);
    let mut mstarR = [0.0; RANK];
    for i in 0..RANK {
        mstarR[i] = mom_term1[i] + mom_term2[i];
    }
    let estarR = facR * (uR.nrg * (aR - vnR) + p_star * a_star - primR.p * vnR);

    let starStateR = Conserved::<Newtonian, RANK>::new(rhostarR, mstarR, estarR);

    // apply low-mach correction
    let phi = compute_adaptive_phi(primL, primR, nhat, gamma, enable_lowmach_fix);
    let aL_lm = phi * aL;
    let aR_lm = phi * aR;

    // select face star state based on contact wave position
    let face_starState = if a_star <= 0.0 {
        starStateR
    } else {
        starStateL
    };

    // compute net flux with hllc formula
    // F = 0.5*(fL + fR) + 0.5*[aL*(uL* - uL) + |a*|*(uL* - uR*) + aR*(uR* - uR)]
    let flux_avg = (fL + fR) * 0.5;

    let term1 = (starStateL - uL) * aL_lm;
    let term2 = (starStateL - starStateR) * a_star.abs();
    let term3 = (starStateR - uR) * aR_lm;

    let correction = (term1 + term2 + term3) * 0.5;

    flux_avg + correction
}

// =============================================================================
// helper functions
// =============================================================================

#[inline]
fn dot_product<const RANK: usize>(a: &[f64; RANK], b: &[f64; RANK]) -> f64 {
    let mut sum = 0.0;
    for i in 0..RANK {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn scale_vec<const RANK: usize>(v: &[f64; RANK], s: f64) -> [f64; RANK] {
    let mut result = [0.0; RANK];
    for i in 0..RANK {
        result[i] = v[i] * s;
    }
    result
}

// helper function for flux computation
fn compute_flux<const RANK: usize>(
    prim: &Primitive<Newtonian, RANK>,
    nhat: &[f64; RANK],
    gamma: f64,
) -> Conserved<Newtonian, RANK> {
    let vn = dot_product(&prim.vel, nhat);
    let mass_flux = prim.rho * vn;

    let mut mom_flux = [0.0; RANK];
    for i in 0..RANK {
        mom_flux[i] = prim.rho * prim.vel[i] * vn + prim.p * nhat[i];
    }

    let cons = prim.to_conserved(gamma);
    let energy_flux = (cons.nrg + prim.p) * vn;

    Conserved::new(mass_flux, mom_flux, energy_flux)
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const GAMMA: f64 = 1.4;

    #[test]
    fn test_wave_speeds() {
        let ws = WaveSpeeds::new(-1.5, 2.5);
        assert_eq!(ws.min(), -1.5);
        assert_eq!(ws.max(), 2.5);
    }

    #[test]
    fn test_newtonian_wave_speed_estimation() {
        let primL = Primitive::<Newtonian, 1>::new(1.0, [0.0], 1.0);
        let primR = Primitive::<Newtonian, 1>::new(0.125, [0.0], 0.1);
        let nhat = [1.0];

        let ws = newtonian_wave_speeds(&primL, &primR, &nhat, GAMMA);

        assert!(ws.left < 0.0);
        assert!(ws.right > 0.0);
    }

    #[test]
    fn test_hllc_stationary_state() {
        let state = Primitive::<Newtonian, 1>::new(1.0, [0.0], 1.0);
        let nhat = [1.0];

        let flux = hllc_newtonian(&state, &state, &nhat, GAMMA, false);

        // zero velocity -> zero mass flux
        assert!(flux.den.abs() < 1e-10);
    }

    #[test]
    fn test_hllc_sod_shock() {
        let primL = Primitive::<Newtonian, 1>::new(1.0, [0.0], 1.0);
        let primR = Primitive::<Newtonian, 1>::new(0.125, [0.0], 0.1);
        let nhat = [1.0];

        let flux = hllc_newtonian(&primL, &primR, &nhat, GAMMA, false);

        // should have positive mass flux (left to right)
        assert!(flux.den > 0.0);
        assert!(flux.den.is_finite());
        assert!(flux.nrg.is_finite());
    }

    #[test]
    fn test_lowmach_correction() {
        let primL = Primitive::<Newtonian, 1>::new(1.0, [0.01], 1.0);
        let primR = Primitive::<Newtonian, 1>::new(1.0, [-0.01], 1.0);
        let nhat = [1.0];

        let phi = compute_adaptive_phi(&primL, &primR, &nhat, GAMMA, true);

        // low mach number -> phi < 1
        assert!(phi < 1.0);
        assert!(phi > 0.0);
    }

    #[test]
    fn test_quirk_detection() {
        assert!(!quirk_strong_shock(1.0, 1.0));
        assert!(quirk_strong_shock(100.0, 1.0));
        assert!(quirk_strong_shock(1.0, 100.0));
    }

    #[test]
    fn test_conserved_arithmetic() {
        let u1 = Conserved::<Newtonian, 1>::new(1.0, [2.0], 3.0);
        let u2 = Conserved::<Newtonian, 1>::new(0.5, [1.0], 1.5);

        let sum = u1 + u2;
        assert_eq!(sum.den, 1.5);
        assert_eq!(sum.mom[0], 3.0);
        assert_eq!(sum.nrg, 4.5);

        let diff = u1 - u2;
        assert_eq!(diff.den, 0.5);
        assert_eq!(diff.mom[0], 1.0);
        assert_eq!(diff.nrg, 1.5);

        let scaled = u1 * 2.0;
        assert_eq!(scaled.den, 2.0);
        assert_eq!(scaled.mom[0], 4.0);
        assert_eq!(scaled.nrg, 6.0);
    }

    #[test]
    fn test_2d_hllc() {
        let primL = Primitive::<Newtonian, 2>::new(1.0, [0.5, 0.0], 1.0);
        let primR = Primitive::<Newtonian, 2>::new(0.8, [-0.5, 0.0], 0.8);
        let nhat = [1.0, 0.0];

        let flux = hllc_newtonian(&primL, &primR, &nhat, GAMMA, false);

        assert!(flux.den.is_finite());
        assert!(flux.mom[0].is_finite());
        assert!(flux.mom[1].is_finite());
        assert!(flux.nrg.is_finite());
    }
}
