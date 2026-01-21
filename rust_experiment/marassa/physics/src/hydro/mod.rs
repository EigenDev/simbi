// =============================================================================
// hydro/mod.rs
//
// newtonian hydrodynamics module.
// implements riemann solvers and equations of state for the euler equations.
//
// conservation law:
//   ∂u/∂t + ∇·f(u) = 0
//
// where:
//   u = [ρ, ρv, E]^T  (conserved variables)
//   f = riemann flux
//
// usage:
//   use physics::hydro::{hlle_flux, Primitive1D, Conserved1D};
// =============================================================================

pub mod boundary_kernel;
pub mod boundary_policy;
pub mod cons2prim_kernel;
pub mod eos;
pub mod euler1d_aos;
pub mod euler1d_parallel;
pub mod euler1d_simple;
pub mod flux_kernel;
pub mod hllc;
pub mod integrator;
pub mod riemann;
pub mod state;
pub mod timestepping;

pub use boundary_kernel::*;
pub use boundary_policy::*;
pub use cons2prim_kernel::*;
pub use eos::*;
pub use euler1d_aos::AoSEuler1D;
pub use euler1d_parallel::ParallelEuler1D;
pub use euler1d_simple::SimpleEuler1D;
pub use flux_kernel::*;
pub use hllc::*;
pub use integrator::*;
pub use riemann::*;
pub use state::*;
pub use timestepping::*;

// =============================================================================
// 1d euler state types
// =============================================================================

/// primitive variables for 1d euler equations.
/// these are the physically meaningful quantities.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Primitive1D {
    /// mass density
    pub rho: f64,
    /// velocity
    pub vx: f64,
    /// pressure
    pub p: f64,
}

impl Primitive1D {
    /// creates a new primitive state.
    #[inline]
    pub fn new(rho: f64, vx: f64, p: f64) -> Self {
        Self { rho, vx, p }
    }

    /// converts to conserved variables.
    #[inline]
    pub fn to_conserved(self, gamma: f64) -> Conserved1D {
        let rho = self.rho;
        let mom = self.rho * self.vx;
        let ke = 0.5 * self.rho * self.vx * self.vx;
        let ie = self.p / (gamma - 1.0);
        let energy = ke + ie;

        Conserved1D { rho, mom, energy }
    }

    /// computes flux vector f(u) for this primitive state.
    #[inline]
    pub fn to_flux(self, gamma: f64) -> Flux1D {
        let cons = self.to_conserved(gamma);
        let mass_flux = self.rho * self.vx;
        let mom_flux = self.rho * self.vx * self.vx + self.p;
        let energy_flux = (cons.energy + self.p) * self.vx;

        Flux1D {
            mass: mass_flux,
            mom: mom_flux,
            energy: energy_flux,
        }
    }

    /// sound speed for ideal gas.
    #[inline]
    pub fn sound_speed(self, gamma: f64) -> f64 {
        ideal_gas_sound_speed(self.rho, self.p, gamma)
    }

    /// maximum characteristic speed: |v| + c
    #[inline]
    pub fn max_wave_speed(self, gamma: f64) -> f64 {
        self.vx.abs() + self.sound_speed(gamma)
    }
}

/// conserved variables for 1d euler equations.
/// these are the quantities that satisfy the conservation law.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Conserved1D {
    /// mass density
    pub rho: f64,
    /// momentum density
    pub mom: f64,
    /// total energy density
    pub energy: f64,
}

impl Conserved1D {
    /// creates a new conserved state.
    #[inline]
    pub fn new(rho: f64, mom: f64, energy: f64) -> Self {
        Self { rho, mom, energy }
    }

    /// converts to primitive variables.
    #[inline]
    pub fn to_primitive(self, gamma: f64) -> Primitive1D {
        let rho = self.rho;
        let vx = self.mom / self.rho;
        let ke = 0.5 * self.mom * self.mom / self.rho;
        let ie = self.energy - ke;
        let p = (gamma - 1.0) * ie;

        Primitive1D { rho, vx, p }
    }

    /// scalar multiplication (for hll state computation).
    #[inline]
    pub fn scale(self, scalar: f64) -> Self {
        Self {
            rho: self.rho * scalar,
            mom: self.mom * scalar,
            energy: self.energy * scalar,
        }
    }

    /// vector addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            rho: self.rho + other.rho,
            mom: self.mom + other.mom,
            energy: self.energy + other.energy,
        }
    }

    /// vector subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            rho: self.rho - other.rho,
            mom: self.mom - other.mom,
            energy: self.energy - other.energy,
        }
    }
}

/// flux vector for 1d euler equations.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Flux1D {
    /// mass flux
    pub mass: f64,
    /// momentum flux
    pub mom: f64,
    /// energy flux
    pub energy: f64,
}

impl Flux1D {
    /// creates a new flux vector.
    #[inline]
    pub fn new(mass: f64, mom: f64, energy: f64) -> Self {
        Self { mass, mom, energy }
    }

    /// scalar multiplication.
    #[inline]
    pub fn scale(self, scalar: f64) -> Self {
        Self {
            mass: self.mass * scalar,
            mom: self.mom * scalar,
            energy: self.energy * scalar,
        }
    }

    /// vector addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            mass: self.mass + other.mass,
            mom: self.mom + other.mom,
            energy: self.energy + other.energy,
        }
    }

    /// vector subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            mass: self.mass - other.mass,
            mom: self.mom - other.mom,
            energy: self.energy - other.energy,
        }
    }

    /// converts flux to conserved (for hll computation).
    #[inline]
    pub fn to_conserved(self) -> Conserved1D {
        Conserved1D {
            rho: self.mass,
            mom: self.mom,
            energy: self.energy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_to_conserved() {
        let gamma = 1.4;
        let prim = Primitive1D::new(1.0, 0.5, 1.0);
        let cons = prim.to_conserved(gamma);

        assert_eq!(cons.rho, 1.0);
        assert_eq!(cons.mom, 0.5);
        // e = ke + ie = 0.5*rho*v^2 + p/(gamma-1)
        // e = 0.5*1.0*0.25 + 1.0/0.4 = 0.125 + 2.5 = 2.625
        assert!((cons.energy - 2.625).abs() < 1e-10);
    }

    #[test]
    fn test_conserved_to_primitive() {
        let gamma = 1.4;
        let cons = Conserved1D::new(1.0, 0.5, 2.625);
        let prim = cons.to_primitive(gamma);

        assert!((prim.rho - 1.0).abs() < 1e-10);
        assert!((prim.vx - 0.5).abs() < 1e-10);
        assert!((prim.p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let gamma = 1.4;
        let prim_orig = Primitive1D::new(2.0, -0.3, 5.0);
        let cons = prim_orig.to_conserved(gamma);
        let prim_back = cons.to_primitive(gamma);

        assert!((prim_back.rho - prim_orig.rho).abs() < 1e-10);
        assert!((prim_back.vx - prim_orig.vx).abs() < 1e-10);
        assert!((prim_back.p - prim_orig.p).abs() < 1e-10);
    }

    #[test]
    fn test_flux_computation() {
        let gamma = 1.4;
        let prim = Primitive1D::new(1.0, 2.0, 1.0);
        let flux = prim.to_flux(gamma);

        // mass flux = rho * v = 1.0 * 2.0 = 2.0
        assert_eq!(flux.mass, 2.0);

        // momentum flux = rho*v^2 + p = 1.0*4.0 + 1.0 = 5.0
        assert_eq!(flux.mom, 5.0);

        // energy flux = (e + p) * v
        let cons = prim.to_conserved(gamma);
        let expected_eflux = (cons.energy + prim.p) * prim.vx;
        assert!((flux.energy - expected_eflux).abs() < 1e-10);
    }

    #[test]
    fn test_sound_speed() {
        let gamma = 1.4;
        let prim = Primitive1D::new(1.0, 0.0, 1.0);
        let cs = prim.sound_speed(gamma);

        // cs = sqrt(gamma * p / rho) = sqrt(1.4)
        assert!((cs - 1.183).abs() < 0.001);
    }

    #[test]
    fn test_max_wave_speed() {
        let gamma = 1.4;
        let prim = Primitive1D::new(1.0, 0.5, 1.0);
        let lambda = prim.max_wave_speed(gamma);

        let cs = prim.sound_speed(gamma);
        let expected = prim.vx.abs() + cs;
        assert!((lambda - expected).abs() < 1e-10);
    }

    #[test]
    fn test_conserved_arithmetic() {
        let u1 = Conserved1D::new(1.0, 2.0, 3.0);
        let u2 = Conserved1D::new(0.5, 1.0, 1.5);

        let sum = u1.add(u2);
        assert_eq!(sum.rho, 1.5);
        assert_eq!(sum.mom, 3.0);
        assert_eq!(sum.energy, 4.5);

        let diff = u1.sub(u2);
        assert_eq!(diff.rho, 0.5);
        assert_eq!(diff.mom, 1.0);
        assert_eq!(diff.energy, 1.5);

        let scaled = u1.scale(2.0);
        assert_eq!(scaled.rho, 2.0);
        assert_eq!(scaled.mom, 4.0);
        assert_eq!(scaled.energy, 6.0);
    }

    #[test]
    fn test_flux_arithmetic() {
        let f1 = Flux1D::new(1.0, 2.0, 3.0);
        let f2 = Flux1D::new(0.5, 1.0, 1.5);

        let sum = f1.add(f2);
        assert_eq!(sum.mass, 1.5);

        let diff = f1.sub(f2);
        assert_eq!(diff.mass, 0.5);

        let scaled = f1.scale(2.0);
        assert_eq!(scaled.mass, 2.0);
    }
}
