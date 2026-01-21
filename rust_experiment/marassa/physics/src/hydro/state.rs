// =============================================================================
// state.rs
//
// physics state types for newtonian and special relativistic hydrodynamics.
// provides primitive and conserved variable representations with compile-time
// regime selection via marker traits (zero-sized types).
//
// the key difference between regimes:
//   newtonian: energy = kinetic + internal
//   srhd:      energy = total - rest mass (tau = rho*h*W^2 - p - rho*W)
//
// cons2prim for srhd requires iterative root-finding since the mapping is
// implicit. we use newton-raphson on the pressure.
//
// usage:
//   use physics::hydro::state::{Primitive, Conserved, Newtonian, Srhd};
//   let prim: Primitive<Srhd, 1> = Primitive::new(rho, [vx], p);
//   let cons = prim.to_conserved(gamma);
// =============================================================================

use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

// =============================================================================
// regime markers (zero-sized types for compile-time dispatch)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Newtonian;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Srhd;

// trait to mark valid regimes
pub trait Regime: Clone + Copy + Default {}
impl Regime for Newtonian {}
impl Regime for Srhd {}

// =============================================================================
// conversion result for fallible cons2prim
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    NegativePressure,
    MaxIterationsExceeded,
    NonFiniteValue,
    SuperluminalVelocity,
}

pub type ConversionResult<T> = Result<T, ConversionError>;

// =============================================================================
// primitive state
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primitive<R: Regime, const RANK: usize> {
    pub rho: f64,
    pub vel: [f64; RANK],
    pub p: f64,
    _regime: PhantomData<R>,
}

impl<R: Regime, const RANK: usize> Primitive<R, RANK> {
    #[inline]
    pub fn new(rho: f64, vel: [f64; RANK], p: f64) -> Self {
        Self {
            rho,
            vel,
            p,
            _regime: PhantomData,
        }
    }

    #[inline]
    pub fn vel_squared(&self) -> f64 {
        let mut vsq = 0.0;
        let mut ii = 0;
        while ii < RANK {
            vsq += self.vel[ii] * self.vel[ii];
            ii += 1;
        }
        vsq
    }
}

// newtonian-specific methods
impl<const RANK: usize> Primitive<Newtonian, RANK> {
    #[inline]
    pub fn lorentz_factor(&self) -> f64 {
        1.0
    }

    #[inline]
    pub fn sound_speed(&self, gamma: f64) -> f64 {
        (gamma * self.p / self.rho).sqrt()
    }

    #[inline]
    pub fn enthalpy(&self, gamma: f64) -> f64 {
        1.0 + gamma * self.p / ((gamma - 1.0) * self.rho)
    }

    #[inline]
    pub fn to_conserved(&self, gamma: f64) -> Conserved<Newtonian, RANK> {
        let den = self.rho;
        let mut mom = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            mom[ii] = self.rho * self.vel[ii];
            ii += 1;
        }
        let ke = 0.5 * self.rho * self.vel_squared();
        let ie = self.p / (gamma - 1.0);
        let nrg = ke + ie;

        Conserved::new(den, mom, nrg)
    }

    #[inline]
    pub fn to_flux(&self, gamma: f64, dir: usize) -> Conserved<Newtonian, RANK> {
        let cons = self.to_conserved(gamma);
        let vn = self.vel[dir];

        let den = cons.den * vn;
        let mut mom = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            mom[ii] = cons.mom[ii] * vn;
            ii += 1;
        }
        mom[dir] += self.p;
        let nrg = (cons.nrg + self.p) * vn;

        Conserved::new(den, mom, nrg)
    }

    #[inline]
    pub fn max_wave_speed(&self, gamma: f64) -> f64 {
        let cs = self.sound_speed(gamma);
        let mut max_speed = 0.0;
        let mut ii = 0;
        while ii < RANK {
            let speed = self.vel[ii].abs() + cs;
            if speed > max_speed {
                max_speed = speed;
            }
            ii += 1;
        }
        max_speed
    }
}

// srhd-specific methods
impl<const RANK: usize> Primitive<Srhd, RANK> {
    #[inline]
    pub fn lorentz_factor(&self) -> f64 {
        let vsq = self.vel_squared();
        1.0 / (1.0 - vsq).sqrt()
    }

    #[inline]
    pub fn lorentz_factor_squared(&self) -> f64 {
        let vsq = self.vel_squared();
        1.0 / (1.0 - vsq)
    }

    #[inline]
    pub fn sound_speed(&self, gamma: f64) -> f64 {
        // relativistic sound speed: cs^2 = gamma * p / (rho * h)
        let h = self.enthalpy(gamma);
        let cs_sq = gamma * self.p / (self.rho * h);
        cs_sq.sqrt()
    }

    #[inline]
    pub fn enthalpy(&self, gamma: f64) -> f64 {
        // h = 1 + epsilon + p/rho = 1 + gamma*p / ((gamma-1)*rho)
        1.0 + gamma * self.p / ((gamma - 1.0) * self.rho)
    }

    #[inline]
    pub fn to_conserved(&self, gamma: f64) -> Conserved<Srhd, RANK> {
        let w = self.lorentz_factor();
        let wsq = w * w;
        let h = self.enthalpy(gamma);

        // lab-frame density: D = rho * W
        let den = self.rho * w;

        // momentum: S = rho * h * W^2 * v
        let mut mom = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            mom[ii] = self.rho * h * wsq * self.vel[ii];
            ii += 1;
        }

        // energy (tau): tau = rho*h*W^2 - p - D
        let nrg = self.rho * h * wsq - self.p - den;

        Conserved::new(den, mom, nrg)
    }

    #[inline]
    pub fn to_flux(&self, gamma: f64, dir: usize) -> Conserved<Srhd, RANK> {
        let cons = self.to_conserved(gamma);
        let vn = self.vel[dir];

        // mass flux: D * v_n
        let den = cons.den * vn;

        // momentum flux: S_i * v_n + p * delta_{in}
        let mut mom = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            mom[ii] = cons.mom[ii] * vn;
            ii += 1;
        }
        mom[dir] += self.p;

        // energy flux: S_n - D * v_n (note: this is the tau flux)
        let nrg = cons.mom[dir] - cons.den * vn;

        Conserved::new(den, mom, nrg)
    }

    #[inline]
    pub fn max_wave_speed(&self, gamma: f64) -> f64 {
        // relativistic wave speeds: lambda_+/- = (v +/- cs) / (1 +/- v*cs)
        let cs = self.sound_speed(gamma);
        let mut max_speed = 0.0;
        let mut ii = 0;
        while ii < RANK {
            let v = self.vel[ii];
            let lambda_p = (v + cs) / (1.0 + v * cs);
            let lambda_m = (v - cs) / (1.0 - v * cs);
            let speed = lambda_p.abs().max(lambda_m.abs());
            if speed > max_speed {
                max_speed = speed;
            }
            ii += 1;
        }
        max_speed
    }
}

// =============================================================================
// conserved state
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conserved<R: Regime, const RANK: usize> {
    pub den: f64,
    pub mom: [f64; RANK],
    pub nrg: f64,
    _regime: PhantomData<R>,
}

impl<R: Regime, const RANK: usize> Conserved<R, RANK> {
    #[inline]
    pub fn new(den: f64, mom: [f64; RANK], nrg: f64) -> Self {
        Self {
            den,
            mom,
            nrg,
            _regime: PhantomData,
        }
    }

    #[inline]
    pub fn mom_squared(&self) -> f64 {
        let mut ssq = 0.0;
        let mut ii = 0;
        while ii < RANK {
            ssq += self.mom[ii] * self.mom[ii];
            ii += 1;
        }
        ssq
    }

    #[inline]
    pub fn mom_magnitude(&self) -> f64 {
        self.mom_squared().sqrt()
    }
}

// newtonian cons2prim (direct, no iteration)
impl<const RANK: usize> Conserved<Newtonian, RANK> {
    #[inline]
    pub fn to_primitive(&self, gamma: f64) -> ConversionResult<Primitive<Newtonian, RANK>> {
        let rho = self.den;
        let mut vel = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            vel[ii] = self.mom[ii] / self.den;
            ii += 1;
        }

        let ke = 0.5 * self.mom_squared() / self.den;
        let ie = self.nrg - ke;
        let p = (gamma - 1.0) * ie;

        if p <= 0.0 || !p.is_finite() {
            return Err(ConversionError::NegativePressure);
        }

        Ok(Primitive::new(rho, vel, p))
    }
}

// srhd cons2prim (newton-raphson on pressure)
impl<const RANK: usize> Conserved<Srhd, RANK> {
    const MAX_ITER: usize = 50;
    const EPSILON: f64 = 1e-12;

    #[inline]
    pub fn to_primitive(&self, gamma: f64) -> ConversionResult<Primitive<Srhd, RANK>> {
        let d = self.den;
        let tau = self.nrg;
        let smag = self.mom_magnitude();

        // initial guess: p = |S - D - tau| (from newtonian limit)
        let mut p = (smag - d - tau).abs();
        let tol = d * Self::EPSILON;

        // newton-raphson iteration
        // f(p) = (gamma - 1) * rho * eps - p = 0
        // where rho and eps depend on p through the lorentz factor
        let mut iter = 0;
        loop {
            let (f, g) = self.newton_fg(gamma, p);
            let dp = f / g;
            p -= dp;

            if !p.is_finite() {
                return Err(ConversionError::NonFiniteValue);
            }

            if iter >= Self::MAX_ITER {
                return Err(ConversionError::MaxIterationsExceeded);
            }

            if dp.abs() < tol {
                break;
            }

            iter += 1;
        }

        if p < 0.0 {
            return Err(ConversionError::NegativePressure);
        }

        // recover primitive variables from converged pressure
        let et = tau + d + p;
        let inv_et = 1.0 / et;

        // velocity: v = S / (tau + D + p)
        let mut vel = [0.0; RANK];
        let mut vsq = 0.0;
        let mut ii = 0;
        while ii < RANK {
            vel[ii] = self.mom[ii] * inv_et;
            vsq += vel[ii] * vel[ii];
            ii += 1;
        }

        if vsq >= 1.0 {
            return Err(ConversionError::SuperluminalVelocity);
        }

        let w = 1.0 / (1.0 - vsq).sqrt();
        let rho = d / w;

        Ok(Primitive::new(rho, vel, p))
    }

    // newton-raphson function and derivative for srhd cons2prim
    // returns (f, df/dp) where f(p) = (gamma-1)*rho*eps - p
    #[inline]
    fn newton_fg(&self, gamma: f64, p: f64) -> (f64, f64) {
        let d = self.den;
        let tau = self.nrg;
        let smag = self.mom_magnitude();

        let et = tau + d + p;
        let vsq = (smag * smag) / (et * et);
        let w = 1.0 / (1.0 - vsq).sqrt();
        let rho = d / w;

        // specific internal energy from tau definition
        // tau = rho*h*W^2 - p - D => eps = (tau + (1-W)*D + (1-W^2)*p) / (D*W)
        let wsq = w * w;
        let eps = (tau + (1.0 - w) * d + (1.0 - wsq) * p) / (d * w);

        // sound speed squared for derivative
        let cs_sq = (gamma - 1.0) * gamma * eps / (1.0 + gamma * eps);

        // f = (gamma - 1) * rho * eps - p
        let f = (gamma - 1.0) * rho * eps - p;

        // df/dp = cs^2 * v^2 - 1
        let g = cs_sq * vsq - 1.0;

        (f, g)
    }
}

// =============================================================================
// arithmetic operations (for riemann solvers)
// =============================================================================

impl<R: Regime, const RANK: usize> Add for Conserved<R, RANK> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        let mut mom = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            mom[ii] = self.mom[ii] + other.mom[ii];
            ii += 1;
        }
        Self::new(self.den + other.den, mom, self.nrg + other.nrg)
    }
}

impl<R: Regime, const RANK: usize> Sub for Conserved<R, RANK> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        let mut mom = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            mom[ii] = self.mom[ii] - other.mom[ii];
            ii += 1;
        }
        Self::new(self.den - other.den, mom, self.nrg - other.nrg)
    }
}

impl<R: Regime, const RANK: usize> Mul<f64> for Conserved<R, RANK> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self {
        let mut mom = [0.0; RANK];
        let mut ii = 0;
        while ii < RANK {
            mom[ii] = self.mom[ii] * scalar;
            ii += 1;
        }
        Self::new(self.den * scalar, mom, self.nrg * scalar)
    }
}

impl<R: Regime, const RANK: usize> Mul<Conserved<R, RANK>> for f64 {
    type Output = Conserved<R, RANK>;

    #[inline]
    fn mul(self, cons: Conserved<R, RANK>) -> Conserved<R, RANK> {
        cons * self
    }
}

// =============================================================================
// type aliases for convenience
// =============================================================================

pub type NewtonianPrimitive1d = Primitive<Newtonian, 1>;
pub type NewtonianConserved1d = Conserved<Newtonian, 1>;
pub type NewtonianPrimitive2d = Primitive<Newtonian, 2>;
pub type NewtonianConserved2d = Conserved<Newtonian, 2>;
pub type NewtonianPrimitive3d = Primitive<Newtonian, 3>;
pub type NewtonianConserved3d = Conserved<Newtonian, 3>;

pub type SrhdPrimitive1d = Primitive<Srhd, 1>;
pub type SrhdConserved1d = Conserved<Srhd, 1>;
pub type SrhdPrimitive2d = Primitive<Srhd, 2>;
pub type SrhdConserved2d = Conserved<Srhd, 2>;
pub type SrhdPrimitive3d = Primitive<Srhd, 3>;
pub type SrhdConserved3d = Conserved<Srhd, 3>;

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const GAMMA: f64 = 5.0 / 3.0;
    const TOL: f64 = 1e-10;

    // -------------------------------------------------------------------------
    // newtonian tests
    // -------------------------------------------------------------------------

    #[test]
    fn newtonian_1d_roundtrip() {
        let prim = NewtonianPrimitive1d::new(1.0, [0.5], 2.5);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < TOL);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < TOL);
        assert!((prim_back.p - prim.p).abs() < TOL);
    }

    #[test]
    fn newtonian_2d_roundtrip() {
        let prim = NewtonianPrimitive2d::new(2.0, [0.3, -0.4], 1.5);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < TOL);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < TOL);
        assert!((prim_back.vel[1] - prim.vel[1]).abs() < TOL);
        assert!((prim_back.p - prim.p).abs() < TOL);
    }

    #[test]
    fn newtonian_3d_roundtrip() {
        let prim = NewtonianPrimitive3d::new(1.5, [0.1, 0.2, 0.3], 3.0);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < TOL);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < TOL);
        assert!((prim_back.vel[1] - prim.vel[1]).abs() < TOL);
        assert!((prim_back.vel[2] - prim.vel[2]).abs() < TOL);
        assert!((prim_back.p - prim.p).abs() < TOL);
    }

    #[test]
    fn newtonian_sound_speed() {
        let prim = NewtonianPrimitive1d::new(1.0, [0.0], 1.0);
        let cs = prim.sound_speed(1.4);
        let expected = (1.4_f64).sqrt(); // sqrt(gamma * p / rho)
        assert!((cs - expected).abs() < TOL);
    }

    #[test]
    fn newtonian_flux_1d() {
        let prim = NewtonianPrimitive1d::new(1.0, [2.0], 1.0);
        let flux = prim.to_flux(1.4, 0);

        // mass flux = rho * v
        assert!((flux.den - 2.0).abs() < TOL);

        // momentum flux = rho*v^2 + p
        assert!((flux.mom[0] - 5.0).abs() < TOL);
    }

    // -------------------------------------------------------------------------
    // srhd tests
    // -------------------------------------------------------------------------

    #[test]
    fn srhd_lorentz_factor() {
        // v = 0.6c => W = 1.25
        let prim = SrhdPrimitive1d::new(1.0, [0.6], 1.0);
        let w = prim.lorentz_factor();
        let expected = 1.0 / (1.0 - 0.36_f64).sqrt();
        assert!((w - expected).abs() < TOL);
    }

    #[test]
    fn srhd_1d_roundtrip_slow() {
        // non-relativistic limit: v << c
        let prim = SrhdPrimitive1d::new(1.0, [0.01], 2.5);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < 1e-8);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < 1e-8);
        assert!((prim_back.p - prim.p).abs() < 1e-8);
    }

    #[test]
    fn srhd_1d_roundtrip_moderate() {
        // moderately relativistic: v = 0.5c
        let prim = SrhdPrimitive1d::new(1.0, [0.5], 2.5);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < 1e-8);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < 1e-8);
        assert!((prim_back.p - prim.p).abs() < 1e-8);
    }

    #[test]
    fn srhd_1d_roundtrip_fast() {
        // highly relativistic: v = 0.9c => W ~ 2.29
        let prim = SrhdPrimitive1d::new(1.0, [0.9], 2.5);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < 1e-8);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < 1e-8);
        assert!((prim_back.p - prim.p).abs() < 1e-8);
    }

    #[test]
    fn srhd_1d_roundtrip_ultrarelativistic() {
        // ultra-relativistic: v = 0.99c => W ~ 7.09
        let prim = SrhdPrimitive1d::new(1.0, [0.99], 2.5);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < 1e-6);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < 1e-6);
        assert!((prim_back.p - prim.p).abs() < 1e-6);
    }

    #[test]
    fn srhd_2d_roundtrip() {
        // 2d relativistic flow
        let prim = SrhdPrimitive2d::new(1.0, [0.4, 0.3], 2.5);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < 1e-8);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < 1e-8);
        assert!((prim_back.vel[1] - prim.vel[1]).abs() < 1e-8);
        assert!((prim_back.p - prim.p).abs() < 1e-8);
    }

    #[test]
    fn srhd_3d_roundtrip() {
        // 3d relativistic flow
        let prim = SrhdPrimitive3d::new(2.0, [0.3, 0.2, 0.1], 5.0);
        let cons = prim.to_conserved(GAMMA);
        let prim_back = cons.to_primitive(GAMMA).unwrap();

        assert!((prim_back.rho - prim.rho).abs() < 1e-8);
        assert!((prim_back.vel[0] - prim.vel[0]).abs() < 1e-8);
        assert!((prim_back.vel[1] - prim.vel[1]).abs() < 1e-8);
        assert!((prim_back.vel[2] - prim.vel[2]).abs() < 1e-8);
        assert!((prim_back.p - prim.p).abs() < 1e-8);
    }

    #[test]
    fn srhd_relativistic_sound_speed() {
        let prim = SrhdPrimitive1d::new(1.0, [0.0], 1.0);
        let cs = prim.sound_speed(GAMMA);

        // relativistic cs^2 = gamma * p / (rho * h)
        let h = prim.enthalpy(GAMMA);
        let expected = (GAMMA * prim.p / (prim.rho * h)).sqrt();
        assert!((cs - expected).abs() < TOL);
    }

    #[test]
    fn srhd_wave_speed_subluminal() {
        // wave speeds should always be < 1
        let prim = SrhdPrimitive1d::new(1.0, [0.9], 2.5);
        let lambda = prim.max_wave_speed(GAMMA);
        assert!(lambda < 1.0);
    }

    #[test]
    fn srhd_flux_mass_continuity() {
        // mass flux = D * v
        let prim = SrhdPrimitive1d::new(1.0, [0.5], 2.5);
        let cons = prim.to_conserved(GAMMA);
        let flux = prim.to_flux(GAMMA, 0);

        let expected_mass_flux = cons.den * prim.vel[0];
        assert!((flux.den - expected_mass_flux).abs() < TOL);
    }

    // -------------------------------------------------------------------------
    // arithmetic tests
    // -------------------------------------------------------------------------

    #[test]
    fn conserved_add() {
        let u1 = NewtonianConserved1d::new(1.0, [2.0], 3.0);
        let u2 = NewtonianConserved1d::new(0.5, [1.0], 1.5);
        let sum = u1 + u2;

        assert!((sum.den - 1.5).abs() < TOL);
        assert!((sum.mom[0] - 3.0).abs() < TOL);
        assert!((sum.nrg - 4.5).abs() < TOL);
    }

    #[test]
    fn conserved_sub() {
        let u1 = NewtonianConserved1d::new(1.0, [2.0], 3.0);
        let u2 = NewtonianConserved1d::new(0.5, [1.0], 1.5);
        let diff = u1 - u2;

        assert!((diff.den - 0.5).abs() < TOL);
        assert!((diff.mom[0] - 1.0).abs() < TOL);
        assert!((diff.nrg - 1.5).abs() < TOL);
    }

    #[test]
    fn conserved_scale() {
        let u = NewtonianConserved1d::new(1.0, [2.0], 3.0);
        let scaled = u * 2.0;

        assert!((scaled.den - 2.0).abs() < TOL);
        assert!((scaled.mom[0] - 4.0).abs() < TOL);
        assert!((scaled.nrg - 6.0).abs() < TOL);

        // commutative
        let scaled2 = 2.0 * u;
        assert!((scaled2.den - 2.0).abs() < TOL);
    }

    // -------------------------------------------------------------------------
    // error handling tests
    // -------------------------------------------------------------------------

    #[test]
    fn newtonian_negative_pressure_error() {
        // construct invalid conserved state (energy too low)
        let cons = NewtonianConserved1d::new(1.0, [10.0], 1.0);
        let result = cons.to_primitive(GAMMA);
        assert!(matches!(result, Err(ConversionError::NegativePressure)));
    }
}
