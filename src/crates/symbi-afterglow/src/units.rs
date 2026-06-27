// =============================================================================
// units.rs
//
// compile-time dimensional analysis for the afterglow physics (CGS gaussian).
// a `Quantity<M, L, T>` is an f64 tagged with type-level mass/length/time exponents:
// multiplying or dividing quantities tracks the resulting dimensions, and adding
// mismatched dimensions is a COMPILE error.
//
// exponents are HALF-INTEGER-ENCODED: each type parameter is twice the physical
// exponent (typenum `P2` => power 1), because gaussian electromagnetism is
// intrinsically half-integer (gauss = g^{1/2} cm^{-1/2} s^{-1}, so the equipartition
// relation B = sqrt(energy density) must type-check). this keeps every afterglow
// exponent an integer typenum constant while still supporting the one fractional
// operation the physics needs — sqrt (halve the exponents). charge and temperature
// are NOT separate base dimensions: gaussian charge is g^{1/2} cm^{3/2} s^{-1} (pure
// M,L,T) and temperature is unused here, so three dimensions suffice.
//
// why typenum: stable rust cannot evaluate `quantity<{M1+M2}>` exponent arithmetic in
// type position (that needs nightly `generic_const_exprs`), and `uom` supports only
// integer dimensions (no gaussian half-powers). typenum's type-level integer
// arithmetic is the portable, stable mechanism — the same one `uom` is built on.
//
// usage:
//  let b: MagneticField = (8.0 * PI * eps_b * rho_e).sqrt(); // rho_e: EnergyDensity
//  let nu_g: Frequency  = (E_CHARGE / (M_E * C_LIGHT)) * b;
//  let raw: f64         = b.value(); // exit the type system at a serialization boundary
// =============================================================================

use core::cmp::Ordering;
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use typenum::{Diff, Negate, PartialQuot, Prod, Sum};
use typenum::{N1, N2, N4, N6, P1, P2, P3, P4, P6, Z0};

/// a dimensionful value: an f64 carrying type-level mass/length/time exponents (each
/// twice the physical power, see the module header). the phantom carries no runtime cost.
///
/// mismatched dimensions do not compile — adding a length to a time is rejected:
/// ```compile_fail
/// use symbi_afterglow::units::{Length, Time};
/// let _ = Length::new(1.0) + Time::new(1.0);
/// ```
/// and sqrt of a non-perfect-square dimension is rejected (gauss has an odd half-exponent):
/// ```compile_fail
/// use symbi_afterglow::units::MagneticField;
/// let _ = MagneticField::new(1.0).sqrt();
/// ```
#[repr(transparent)]
pub struct Quantity<M, L, T>(f64, PhantomData<(M, L, T)>);

impl<M, L, T> Quantity<M, L, T> {
    /// wrap a raw CGS f64 in this dimension. the call site asserts the dimension.
    #[inline]
    pub const fn new(value: f64) -> Self {
        Quantity(value, PhantomData)
    }

    /// the raw CGS f64. drops the dimension — use only at a boundary (output, comparison
    /// against a raw threshold), never to sidestep a dimensional mismatch in the algebra.
    #[inline]
    pub const fn value(self) -> f64 {
        self.0
    }
}

// quantity is just an f64 in memory; clone/copy/debug/ordering are hand-written so they
// do NOT spuriously bound the phantom exponents (derive would require M,L,T: Trait).
impl<M, L, T> Clone for Quantity<M, L, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<M, L, T> Copy for Quantity<M, L, T> {}

impl<M, L, T> core::fmt::Debug for Quantity<M, L, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// comparisons are defined only between identical dimensions (different dimensions are
// different types, so a cross-dimension comparison is a compile error).
impl<M, L, T> PartialEq for Quantity<M, L, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<M, L, T> PartialOrd for Quantity<M, L, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

// addition / subtraction / negation preserve dimensions (same type in, same type out).
impl<M, L, T> Add for Quantity<M, L, T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Quantity(self.0 + rhs.0, PhantomData)
    }
}
impl<M, L, T> Sub for Quantity<M, L, T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Quantity(self.0 - rhs.0, PhantomData)
    }
}
impl<M, L, T> Neg for Quantity<M, L, T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Quantity(-self.0, PhantomData)
    }
}

// multiplication ADDS exponents, division SUBTRACTS them (the dimensional bookkeeping).
impl<M1, L1, T1, M2, L2, T2> Mul<Quantity<M2, L2, T2>> for Quantity<M1, L1, T1>
where
    M1: Add<M2>,
    L1: Add<L2>,
    T1: Add<T2>,
{
    type Output = Quantity<Sum<M1, M2>, Sum<L1, L2>, Sum<T1, T2>>;
    #[inline]
    fn mul(self, rhs: Quantity<M2, L2, T2>) -> Self::Output {
        Quantity(self.0 * rhs.0, PhantomData)
    }
}
impl<M1, L1, T1, M2, L2, T2> Div<Quantity<M2, L2, T2>> for Quantity<M1, L1, T1>
where
    M1: Sub<M2>,
    L1: Sub<L2>,
    T1: Sub<T2>,
{
    type Output = Quantity<Diff<M1, M2>, Diff<L1, L2>, Diff<T1, T2>>;
    #[inline]
    fn div(self, rhs: Quantity<M2, L2, T2>) -> Self::Output {
        Quantity(self.0 / rhs.0, PhantomData)
    }
}

// scalar (dimensionless f64) multiply/divide preserve dimensions.
impl<M, L, T> Mul<f64> for Quantity<M, L, T> {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Quantity(self.0 * s, PhantomData)
    }
}
impl<M, L, T> Div<f64> for Quantity<M, L, T> {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        Quantity(self.0 / s, PhantomData)
    }
}
// `f64 * quantity` preserves dimensions; `f64 / quantity` INVERTS them (negates exponents).
impl<M, L, T> Mul<Quantity<M, L, T>> for f64 {
    type Output = Quantity<M, L, T>;
    #[inline]
    fn mul(self, q: Quantity<M, L, T>) -> Quantity<M, L, T> {
        Quantity(self * q.0, PhantomData)
    }
}
impl<M, L, T> Div<Quantity<M, L, T>> for f64
where
    M: Neg,
    L: Neg,
    T: Neg,
{
    type Output = Quantity<Negate<M>, Negate<L>, Negate<T>>;
    #[inline]
    fn div(self, q: Quantity<M, L, T>) -> Self::Output {
        Quantity(self / q.0, PhantomData)
    }
}

impl<M, L, T> Quantity<M, L, T> {
    /// dimensional square root: HALVES every exponent. a compile error unless each
    /// exponent is even in half-integer encoding (i.e. the dimension is a perfect
    /// square) — `PartialDiv` has no impl otherwise. this is what makes
    /// `sqrt(energy density) -> magnetic field` legal and `sqrt(length)` illegal.
    #[inline]
    pub fn sqrt(self) -> Quantity<PartialQuot<M, P2>, PartialQuot<L, P2>, PartialQuot<T, P2>>
    where
        M: typenum::PartialDiv<P2>,
        L: typenum::PartialDiv<P2>,
        T: typenum::PartialDiv<P2>,
    {
        Quantity(self.0.sqrt(), PhantomData)
    }

    /// dimensional square: DOUBLES every exponent.
    #[inline]
    pub fn squared(self) -> Quantity<Prod<M, P2>, Prod<L, P2>, Prod<T, P2>>
    where
        M: Mul<P2>,
        L: Mul<P2>,
        T: Mul<P2>,
    {
        Quantity(self.0 * self.0, PhantomData)
    }

    /// dimensional cube: TRIPLES every exponent.
    #[inline]
    pub fn cubed(self) -> Quantity<Prod<M, P3>, Prod<L, P3>, Prod<T, P3>>
    where
        M: Mul<P3>,
        L: Mul<P3>,
        T: Mul<P3>,
    {
        Quantity(self.0 * self.0 * self.0, PhantomData)
    }
}

// =========================================================================
// named dimensions (half-integer-encoded: type param = 2 x physical power)
// =========================================================================

/// a pure number (no dimension); `.value()` is its f64.
pub type Dimensionless = Quantity<Z0, Z0, Z0>;
/// gram.
pub type Mass = Quantity<P2, Z0, Z0>;
/// centimeter.
pub type Length = Quantity<Z0, P2, Z0>;
/// second.
pub type Time = Quantity<Z0, Z0, P2>;
/// cm/s.
pub type Velocity = Quantity<Z0, P2, N2>;
/// cm^2.
pub type Area = Quantity<Z0, P4, Z0>;
/// cm^3.
pub type Volume = Quantity<Z0, P6, Z0>;
/// Hz (1/s).
pub type Frequency = Quantity<Z0, Z0, N2>;
/// erg (g cm^2 / s^2).
pub type Energy = Quantity<P2, P4, N4>;
/// action, erg s (energy x time) — the dimension of planck's constant.
pub type Action = Quantity<P2, P4, N2>;
/// erg/s.
pub type Power = Quantity<P2, P4, N6>;
/// statcoulomb / esu (g^{1/2} cm^{3/2} s^{-1}).
pub type Charge = Quantity<P1, P3, N2>;
/// g/cm^3.
pub type MassDensity = Quantity<P2, N6, Z0>;
/// erg/cm^3 (also pressure: dyne/cm^2).
pub type EnergyDensity = Quantity<P2, N2, N4>;
/// cm^-3.
pub type NumberDensity = Quantity<Z0, N6, Z0>;
/// gauss (g^{1/2} cm^{-1/2} s^{-1}).
pub type MagneticField = Quantity<P1, N1, N2>;

// spectral quantities. since Hz = 1/s, dividing a per-second power by a frequency cancels
// the time dimension — so these collapse onto energy / energy-flux / energy-density forms.
/// spectral power per unit frequency, erg/(s Hz) = erg (energy dimensions).
pub type SpectralPower = Energy;
/// spectral flux density, erg/(s cm^2 Hz) = erg/cm^2 (energy-per-area dimensions).
pub type SpectralFlux = Quantity<P2, Z0, N4>;
/// spectral emissivity, erg/(s cm^3 Hz) = erg/cm^3 (energy-density dimensions).
pub type SpectralEmissivity = EnergyDensity;

#[cfg(test)]
mod tests {
    use super::*;

    // multiplication adds dimensions: velocity * time == length (value and type).
    #[test]
    fn mul_adds_dimensions() {
        let v = Velocity::new(3.0e10);
        let t = Time::new(2.0);
        let r: Length = v * t;
        assert_eq!(r.value(), 6.0e10);
    }

    // sqrt halves dimensions: sqrt(energy density) is a magnetic field (the equipartition path).
    #[test]
    fn sqrt_halves_dimensions() {
        let u = EnergyDensity::new(16.0);
        let b: MagneticField = u.sqrt();
        assert_eq!(b.value(), 4.0);
    }

    // squared/cubed scale dimensions: charge^3 / energy gives the emissivity charge factor's dims.
    #[test]
    fn squared_and_cubed_scale_dimensions() {
        let c = Velocity::new(2.0);
        let area_like: Quantity<Z0, P4, N4> = c.squared();
        assert_eq!(area_like.value(), 4.0);
        let vol_like: Quantity<Z0, P6, N6> = c.cubed();
        assert_eq!(vol_like.value(), 8.0);
    }

    // f64 / quantity inverts dimensions: 1 / area has dimensions of inverse area (the flux denom).
    #[test]
    fn scalar_over_quantity_inverts_dimensions() {
        let a = Area::new(4.0);
        let inv: Quantity<Z0, N4, Z0> = 1.0 / a;
        assert_eq!(inv.value(), 0.25);
    }

    // same-dimension comparison works; the value ordering is the f64 ordering.
    #[test]
    fn same_dimension_comparison() {
        assert!(Frequency::new(1.0e14) > Frequency::new(1.0e10));
        assert!(Length::new(1.0) + Length::new(2.0) == Length::new(3.0));
    }
}
