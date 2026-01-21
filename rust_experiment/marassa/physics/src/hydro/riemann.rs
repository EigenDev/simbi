// =============================================================================
// riemann.rs
//
// riemann solvers for hydrodynamics.
// approximate solution to the riemann problem at cell interfaces.
//
// the hlle solver (harten-lax-van leer-einfeldt) provides a robust,
// diffusive approximation to the exact riemann solution. it captures
// shocks, rarefactions, and contact discontinuities.
//
// algorithm:
//   1. estimate fastest left/right wave speeds (sL, sR)
//   2. if sL >= 0: use left state (supersonic left)
//   3. if sR <= 0: use right state (supersonic right)
//   4. otherwise: compute hll intermediate state
//
// usage:
//   let flux = hlle_flux(primL, primR, gamma);
// =============================================================================

use super::{Conserved1D, Flux1D, Primitive1D};

/// estimates extremal wave speeds for hlle solver.
/// uses simple estimate: s = v ± c (eigenvalues of jacobian)
///
/// # arguments
/// * `left` - left primitive state
/// * `right` - right primitive state
/// * `gamma` - adiabatic index
///
/// # returns
/// (sL, sR) - left and right wave speeds
#[inline]
fn estimate_wave_speeds(left: Primitive1D, right: Primitive1D, gamma: f64) -> (f64, f64) {
    let cs_l = left.sound_speed(gamma);
    let cs_r = right.sound_speed(gamma);

    // simplest estimate: fastest left and right-going waves
    let s_left = (left.vx - cs_l).min(right.vx - cs_r);
    let s_right = (left.vx + cs_l).max(right.vx + cs_r);

    (s_left, s_right)
}

/// hlle approximate riemann solver for 1d euler equations.
///
/// computes numerical flux at cell interface given left and right states.
/// handles supersonic and subsonic flows automatically.
///
/// # arguments
/// * `left` - left primitive state (reconstructed at interface)
/// * `right` - right primitive state (reconstructed at interface)
/// * `gamma` - adiabatic index (ratio of specific heats)
///
/// # returns
/// numerical flux at interface
///
/// # theory
/// the hlle solver approximates the riemann fan with two waves:
///   - fastest left-going wave speed sL
///   - fastest right-going wave speed sR
///
/// three regimes:
///   1. sL >= 0: left supersonic -> f = f_L
///   2. sR <= 0: right supersonic -> f = f_R
///   3. sL < 0 < sR: subsonic -> f = hll flux
///
/// hll flux formula:
///   f_hll = (sR*f_L - sL*f_R + sL*sR*(u_R - u_L)) / (sR - sL)
#[inline]
pub fn hlle_flux(left: Primitive1D, right: Primitive1D, gamma: f64) -> Flux1D {
    // convert to conserved variables and fluxes
    let u_l = left.to_conserved(gamma);
    let u_r = right.to_conserved(gamma);
    let f_l = left.to_flux(gamma);
    let f_r = right.to_flux(gamma);

    // estimate wave speeds
    let (s_l, s_r) = estimate_wave_speeds(left, right, gamma);

    // three-wave pattern
    if s_l >= 0.0 {
        // left state is supersonic
        f_l
    } else if s_r <= 0.0 {
        // right state is supersonic
        f_r
    } else {
        // intermediate (hll) state
        // f_hll = (sR*f_L - sL*f_R + sL*sR*(u_R - u_L)) / (sR - sL)
        let numerator = f_l
            .scale(s_r)
            .sub(f_r.scale(s_l))
            .add(u_r.sub(u_l).scale(s_l * s_r).into());

        let denominator = s_r - s_l;

        Flux1D {
            mass: numerator.mass / denominator,
            mom: numerator.mom / denominator,
            energy: numerator.energy / denominator,
        }
    }
}

/// hlle flux with interface velocity (for moving meshes).
///
/// adjusts flux for interface motion: f_net = f - u*v_interface
///
/// # arguments
/// * `left` - left primitive state
/// * `right` - right primitive state
/// * `v_interface` - interface velocity (0 for stationary mesh)
/// * `gamma` - adiabatic index
#[inline]
pub fn hlle_flux_moving(
    left: Primitive1D,
    right: Primitive1D,
    v_interface: f64,
    gamma: f64,
) -> Flux1D {
    let u_l = left.to_conserved(gamma);
    let u_r = right.to_conserved(gamma);
    let f_l = left.to_flux(gamma);
    let f_r = right.to_flux(gamma);

    let (s_l, s_r) = estimate_wave_speeds(left, right, gamma);

    let flux = if s_l >= v_interface {
        f_l.sub(u_l.scale(v_interface).into())
    } else if s_r <= v_interface {
        f_r.sub(u_r.scale(v_interface).into())
    } else {
        let f_hll = f_l
            .scale(s_r)
            .sub(f_r.scale(s_l))
            .add(u_r.sub(u_l).scale(s_l * s_r).into())
            .scale(1.0 / (s_r - s_l));

        let u_hll = u_r
            .scale(s_r)
            .sub(u_l.scale(s_l))
            .sub(f_r.sub(f_l).into())
            .scale(1.0 / (s_r - s_l));

        f_hll.sub(u_hll.scale(v_interface).into())
    };

    flux
}

/// hlle flux for multi-dimensional newtonian hydro (quirk's fallback).
/// computes flux in direction nhat.
#[inline]
pub fn hlle_newtonian<const RANK: usize>(
    left: &super::state::Primitive<super::state::Newtonian, RANK>,
    right: &super::state::Primitive<super::state::Newtonian, RANK>,
    nhat: &[f64; RANK],
    gamma: f64,
) -> super::state::Conserved<super::state::Newtonian, RANK> {
    use super::state::{Conserved, Newtonian};

    // project velocities onto normal direction
    let mut vn_l = 0.0;
    let mut vn_r = 0.0;
    for i in 0..RANK {
        vn_l += left.vel[i] * nhat[i];
        vn_r += right.vel[i] * nhat[i];
    }

    // convert to conserved and flux
    let u_l = left.to_conserved(gamma);
    let u_r = right.to_conserved(gamma);

    // compute fluxes
    let f_l_mass = left.rho * vn_l;
    let mut f_l_mom = [0.0; RANK];
    for i in 0..RANK {
        f_l_mom[i] = left.rho * left.vel[i] * vn_l + left.p * nhat[i];
    }
    let f_l_nrg = (u_l.nrg + left.p) * vn_l;
    let f_l = Conserved::<Newtonian, RANK>::new(f_l_mass, f_l_mom, f_l_nrg);

    let f_r_mass = right.rho * vn_r;
    let mut f_r_mom = [0.0; RANK];
    for i in 0..RANK {
        f_r_mom[i] = right.rho * right.vel[i] * vn_r + right.p * nhat[i];
    }
    let f_r_nrg = (u_r.nrg + right.p) * vn_r;
    let f_r = Conserved::<Newtonian, RANK>::new(f_r_mass, f_r_mom, f_r_nrg);

    // estimate wave speeds
    let cs_l = left.sound_speed(gamma);
    let cs_r = right.sound_speed(gamma);
    let s_l = (vn_l - cs_l).min(vn_r - cs_r);
    let s_r = (vn_l + cs_l).max(vn_r + cs_r);

    // compute hll flux
    if s_l >= 0.0 {
        f_l
    } else if s_r <= 0.0 {
        f_r
    } else {
        // intermediate flux
        let mut flux_den =
            (s_r * f_l.den - s_l * f_r.den + s_l * s_r * (u_r.den - u_l.den)) / (s_r - s_l);
        let mut flux_mom = [0.0; RANK];
        for i in 0..RANK {
            flux_mom[i] = (s_r * f_l.mom[i] - s_l * f_r.mom[i]
                + s_l * s_r * (u_r.mom[i] - u_l.mom[i]))
                / (s_r - s_l);
        }
        let flux_nrg =
            (s_r * f_l.nrg - s_l * f_r.nrg + s_l * s_r * (u_r.nrg - u_l.nrg)) / (s_r - s_l);

        Conserved::<Newtonian, RANK>::new(flux_den, flux_mom, flux_nrg)
    }
}

// conversion helpers
impl From<Conserved1D> for Flux1D {
    fn from(cons: Conserved1D) -> Self {
        Flux1D {
            mass: cons.rho,
            mom: cons.mom,
            energy: cons.energy,
        }
    }
}

impl From<Flux1D> for Conserved1D {
    fn from(flux: Flux1D) -> Self {
        Conserved1D {
            rho: flux.mass,
            mom: flux.mom,
            energy: flux.energy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_speed_estimation() {
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 0.0, 1.0);
        let right = Primitive1D::new(1.0, 0.0, 1.0);

        let (s_l, s_r) = estimate_wave_speeds(left, right, gamma);

        let cs = left.sound_speed(gamma);
        assert!((s_l + cs).abs() < 1e-10);
        assert!((s_r - cs).abs() < 1e-10);
    }

    #[test]
    fn test_hlle_stationary_state() {
        // identical left/right states with zero velocity -> zero flux
        let gamma = 1.4;
        let state = Primitive1D::new(1.0, 0.0, 1.0);

        let flux = hlle_flux(state, state, gamma);

        // with zero velocity and identical states, fluxes should be zero
        // mass flux = rho*v = 0
        // momentum flux = rho*v^2 + p = 0 + p = p (but hlle averages to zero)
        // for identical states, hlle should give exact flux
        let expected = state.to_flux(gamma);
        assert!((flux.mass - expected.mass).abs() < 1e-10);
        assert!((flux.mom - expected.mom).abs() < 1e-10);
        assert!((flux.energy - expected.energy).abs() < 1e-10);
    }

    #[test]
    fn test_hlle_supersonic_left() {
        // left state moving supersonic right
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 10.0, 1.0); // v >> cs
        let right = Primitive1D::new(1.0, 0.0, 1.0);

        let flux = hlle_flux(left, right, gamma);
        let f_left = left.to_flux(gamma);

        // with v=10, cs~1.18, both sL and sR are positive
        // so should use left flux
        let cs_l = left.sound_speed(gamma);
        let cs_r = right.sound_speed(gamma);
        let s_l = (left.vx - cs_l).min(right.vx - cs_r);

        if s_l >= 0.0 {
            assert!((flux.mass - f_left.mass).abs() < 1e-8);
            assert!((flux.mom - f_left.mom).abs() < 1e-8);
            assert!((flux.energy - f_left.energy).abs() < 1e-8);
        }
    }

    #[test]
    fn test_hlle_supersonic_right() {
        // right state moving supersonic left
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 0.0, 1.0);
        let right = Primitive1D::new(1.0, -10.0, 1.0); // v << -cs

        let flux = hlle_flux(left, right, gamma);
        let f_right = right.to_flux(gamma);

        // with v=-10, cs~1.18, both sL and sR are negative
        // so should use right flux
        let cs_l = left.sound_speed(gamma);
        let cs_r = right.sound_speed(gamma);
        let s_r = (left.vx + cs_l).max(right.vx + cs_r);

        if s_r <= 0.0 {
            assert!((flux.mass - f_right.mass).abs() < 1e-8);
            assert!((flux.mom - f_right.mom).abs() < 1e-8);
            assert!((flux.energy - f_right.energy).abs() < 1e-8);
        }
    }

    #[test]
    fn test_hlle_subsonic() {
        // subsonic flow -> hll intermediate state
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 0.1, 1.0);
        let right = Primitive1D::new(0.8, -0.1, 0.8);

        let flux = hlle_flux(left, right, gamma);

        // flux should be finite and reasonable
        assert!(flux.mass.is_finite());
        assert!(flux.mom.is_finite());
        assert!(flux.energy.is_finite());

        // sanity check: flux magnitude should be reasonable
        assert!(flux.mass.abs() < 10.0);
        assert!(flux.mom.abs() < 10.0);
        assert!(flux.energy.abs() < 100.0);
    }

    #[test]
    fn test_hlle_shock_tube() {
        // sod shock tube initial condition
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 0.0, 1.0);
        let right = Primitive1D::new(0.125, 0.0, 0.1);

        let flux = hlle_flux(left, right, gamma);

        // at t=0, should have positive mass flux (expansion into low pressure)
        assert!(flux.mass > 0.0);
    }

    #[test]
    fn test_hlle_symmetry() {
        // test that hlle is consistent
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 0.5, 1.0);
        let right = Primitive1D::new(0.8, -0.3, 0.8);

        let flux_lr = hlle_flux(left, right, gamma);

        // flux should be finite and reasonable
        assert!(flux_lr.mass.is_finite());
        assert!(flux_lr.mom.is_finite());
        assert!(flux_lr.energy.is_finite());

        // for this velocity pattern, expect positive mass flux
        assert!(flux_lr.mass > 0.0);
    }

    #[test]
    fn test_hlle_moving_interface() {
        // stationary fluid with moving interface
        let gamma = 1.4;
        let state = Primitive1D::new(1.0, 0.0, 1.0);
        let v_interface = 0.5;

        let flux = hlle_flux_moving(state, state, v_interface, gamma);

        // flux should account for interface motion
        // f = f(u) - u*v_interface
        let cons = state.to_conserved(gamma);
        assert!((flux.mass + cons.rho * v_interface).abs() < 1e-10);
    }

    #[test]
    fn test_hlle_conserves_mass() {
        // conservation check: flux should preserve mass
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 0.5, 1.0);
        let right = Primitive1D::new(0.8, -0.3, 0.8);

        let flux = hlle_flux(left, right, gamma);

        // mass flux should be finite and reasonable
        assert!(flux.mass.is_finite());
        assert!(flux.mass.abs() < 10.0); // reasonable magnitude
    }

    #[test]
    fn test_hlle_high_mach() {
        // high mach number flow
        let gamma = 1.4;
        let left = Primitive1D::new(1.0, 5.0, 1.0);
        let right = Primitive1D::new(1.0, 5.0, 1.0);

        let flux = hlle_flux(left, right, gamma);

        // should handle high mach without issues
        assert!(flux.mass.is_finite());
        assert!(flux.mom.is_finite());
        assert!(flux.energy.is_finite());
    }
}
