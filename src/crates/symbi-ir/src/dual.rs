// =============================================================================
// dual.rs
//
// `Dual<S>` — a forward-mode automatic-differentiation carrier: a value paired with its
// tangent (derivative w.r.t. a single seeded input). it impls `Scalar` (over ANY inner
// `S: Scalar` — f64 for host validation, Gv for the trace), so ANY carrier-generic function is
// differentiable by evaluating it at `Dual` with the input of interest seeded via
// `Dual::variable`. the codebase anticipates this (algebra.rs: "future `Dual<C>` carries
// derivatives").
//
// the motivating use: evaluate a `Metric`'s lapse / shift / spatial_metric at `Dual` seeded on
// the radial coordinate to obtain the ANALYTIC radial derivatives (and, downstream, the
// christoffels) automatically — retiring hand-derived metric derivatives, which for spinning /
// non-diagonal metrics are dozens of error-prone terms.
//
// LIMITATION: `iterate` / `iterate_vec` (the c2p fixed-point machinery) are NOT differentiated
// (the derivative of a fixed point needs the implicit-function theorem, out of scope) — they
// panic. this is fine for STRAIGHT-LINE geometry (metrics use only arithmetic + sqrt + the
// transcendentals below); do not evaluate an iterative solver at `Dual`.
//
// usage:
//   let r = Dual::variable(3.0);          // seed d/dr = 1
//   let a = Schwarzschild { mass: 1.0 }.lapse(Tensor::new([r]));
//   // a.value == alpha(3),  a.tangent == d alpha/dr at r = 3
// =============================================================================

use crate::algebra::Scalar;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use symbi_algebra::algebra::Numeric;
use symbi_algebra::element::FieldElement;

/// a value + its tangent (forward-mode derivative w.r.t. one seeded input).
#[derive(Clone, Copy, Debug, Default)]
pub struct Dual<S> {
    pub value: S,
    pub tangent: S,
}

impl<S: Scalar> Dual<S> {
    /// a CONSTANT (derivative zero): a value that does not depend on the seeded input.
    #[inline]
    pub fn constant(value: S) -> Self {
        Self {
            value,
            tangent: S::ZERO,
        }
    }

    /// the SEEDED variable (derivative one): the single input differentiated with respect to.
    #[inline]
    pub fn variable(value: S) -> Self {
        Self {
            value,
            tangent: S::ONE,
        }
    }
}

// ── arithmetic (the dual rules) ──────────────────────────────────────────────
impl<S: Scalar> Add for Dual<S> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            value: self.value + o.value,
            tangent: self.tangent + o.tangent,
        }
    }
}
impl<S: Scalar> Sub for Dual<S> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            value: self.value - o.value,
            tangent: self.tangent - o.tangent,
        }
    }
}
impl<S: Scalar> Mul for Dual<S> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        // product rule: (a b)' = a' b + a b'.
        Self {
            value: self.value * o.value,
            tangent: self.tangent * o.value + self.value * o.tangent,
        }
    }
}
impl<S: Scalar> Div for Dual<S> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        // quotient rule: (a/b)' = (a' b - a b') / b^2.
        Self {
            value: self.value / o.value,
            tangent: (self.tangent * o.value - self.value * o.tangent) / (o.value * o.value),
        }
    }
}
impl<S: Scalar> Neg for Dual<S> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            tangent: -self.tangent,
        }
    }
}
impl<S: Scalar> AddAssign for Dual<S> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl<S: Scalar> SubAssign for Dual<S> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl<S: Scalar> MulAssign for Dual<S> {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl<S: Scalar> DivAssign for Dual<S> {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}
impl<S: Scalar> std::iter::Sum for Dual<S> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::constant(S::ZERO), |a, b| a + b)
    }
}
impl<S: Scalar> std::fmt::Display for Dual<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} +eps {}", self.value, self.tangent)
    }
}

// FieldElement is a marker (Copy + 'static) carrying the element's scalar type.
unsafe impl<S: Scalar> FieldElement for Dual<S> {
    type Scalar = Dual<S>;
}

// ── Numeric (constants + the closed sqrt/abs/min/max) ────────────────────────
impl<S: Scalar> Numeric for Dual<S> {
    const ZERO: Self = Dual {
        value: S::ZERO,
        tangent: S::ZERO,
    };
    const ONE: Self = Dual {
        value: S::ONE,
        tangent: S::ZERO,
    };

    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(S::from_f64(v)) // a literal is a constant -> tangent zero
    }
    #[inline]
    fn sqrt(self) -> Self {
        // (sqrt a)' = a' / (2 sqrt a).
        let sv = self.value.sqrt();
        Self {
            value: sv,
            tangent: self.tangent / (S::from_f64(2.0) * sv),
        }
    }
    #[inline]
    fn abs(self) -> Self {
        // |a|' = sign(a) a'  (undefined at 0; the geometry never hits it).
        let neg = self.value.cmp_lt(S::ZERO);
        Self {
            value: self.value.abs(),
            tangent: S::select(neg, -self.tangent, self.tangent),
        }
    }
    #[inline]
    fn min(self, o: Self) -> Self {
        // piecewise: carry the tangent of the selected branch.
        let take_self = self.value.cmp_le(o.value);
        Self {
            value: self.value.min(o.value),
            tangent: S::select(take_self, self.tangent, o.tangent),
        }
    }
    #[inline]
    fn max(self, o: Self) -> Self {
        let take_self = self.value.cmp_ge(o.value);
        Self {
            value: self.value.max(o.value),
            tangent: S::select(take_self, self.tangent, o.tangent),
        }
    }
}

// ── Scalar (comparisons on the value, piecewise select, the transcendental rules) ────
impl<S: Scalar> Scalar for Dual<S> {
    type Mask = S::Mask;

    const INFINITY: Self = Dual {
        value: S::INFINITY,
        tangent: S::ZERO,
    };
    const NEG_INFINITY: Self = Dual {
        value: S::NEG_INFINITY,
        tangent: S::ZERO,
    };
    const NAN: Self = Dual {
        value: S::NAN,
        tangent: S::NAN,
    };

    #[inline]
    fn to_f64(self) -> f64 {
        self.value.to_f64() // host-boundary read: the value (the tangent is separate diagnostics)
    }

    // comparisons decide on the VALUE (the tangent does not order points).
    #[inline]
    fn cmp_lt(self, b: Self) -> S::Mask {
        self.value.cmp_lt(b.value)
    }
    #[inline]
    fn cmp_le(self, b: Self) -> S::Mask {
        self.value.cmp_le(b.value)
    }
    #[inline]
    fn cmp_gt(self, b: Self) -> S::Mask {
        self.value.cmp_gt(b.value)
    }
    #[inline]
    fn cmp_ge(self, b: Self) -> S::Mask {
        self.value.cmp_ge(b.value)
    }
    #[inline]
    fn cmp_eq(self, b: Self) -> S::Mask {
        self.value.cmp_eq(b.value)
    }

    #[inline]
    fn select(m: S::Mask, t: Self, f: Self) -> Self {
        // piecewise-differentiable: pick the value AND the matching tangent branch.
        Self {
            value: S::select(m, t.value, f.value),
            tangent: S::select(m, t.tangent, f.tangent),
        }
    }
    #[inline]
    fn cond(m: S::Mask, t: impl FnOnce() -> Self, f: impl FnOnce() -> Self) -> Self {
        // eager (both arms) then select — loses the host laziness, but geometry never branches.
        Self::select(m, t(), f())
    }
    #[inline]
    fn cond_vec<const N: usize>(
        m: S::Mask,
        t: impl FnOnce() -> [Self; N],
        f: impl FnOnce() -> [Self; N],
    ) -> [Self; N] {
        let (tv, fv) = (t(), f());
        std::array::from_fn(|i| Self::select(m, tv[i], fv[i]))
    }

    #[inline]
    fn recip(self) -> Self {
        // (1/a)' = -a' / a^2.
        Self {
            value: self.value.recip(),
            tangent: -self.tangent / (self.value * self.value),
        }
    }

    #[inline]
    fn sin(self) -> Self {
        Self {
            value: self.value.sin(),
            tangent: self.tangent * self.value.cos(),
        }
    }
    #[inline]
    fn cos(self) -> Self {
        Self {
            value: self.value.cos(),
            tangent: -(self.tangent * self.value.sin()),
        }
    }
    #[inline]
    fn tan(self) -> Self {
        let tv = self.value.tan();
        Self {
            value: tv,
            tangent: self.tangent * (S::ONE + tv * tv),
        } // sec^2 = 1 + tan^2
    }
    #[inline]
    fn asin(self) -> Self {
        let d = (S::ONE - self.value * self.value).sqrt();
        Self {
            value: self.value.asin(),
            tangent: self.tangent / d,
        }
    }
    #[inline]
    fn acos(self) -> Self {
        let d = (S::ONE - self.value * self.value).sqrt();
        Self {
            value: self.value.acos(),
            tangent: -self.tangent / d,
        }
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        // d atan2(y, x) = (x dy - y dx) / (x^2 + y^2); self = y.
        let denom = x.value * x.value + self.value * self.value;
        Self {
            value: self.value.atan2(x.value),
            tangent: (x.value * self.tangent - self.value * x.tangent) / denom,
        }
    }
    #[inline]
    fn exp(self) -> Self {
        let ev = self.value.exp();
        Self {
            value: ev,
            tangent: self.tangent * ev,
        }
    }
    #[inline]
    fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            tangent: self.tangent / self.value,
        }
    }
    #[inline]
    fn log10(self) -> Self {
        Self {
            value: self.value.log10(),
            tangent: self.tangent / (self.value * S::from_f64(std::f64::consts::LN_10)),
        }
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        // (a^n)' = n a^{n-1} a'.
        Self {
            value: self.value.powi(n),
            tangent: S::from_f64(n as f64) * self.value.powi(n - 1) * self.tangent,
        }
    }
    #[inline]
    fn powf(self, e: Self) -> Self {
        // d(a^e) = a^e (e' ln a + e a'/a); handles a dual exponent (constant exponent -> e' = 0).
        let v = self.value.powf(e.value);
        Self {
            value: v,
            tangent: v * (e.tangent * self.value.ln() + e.value * self.tangent / self.value),
        }
    }
    #[inline]
    fn floor(self) -> Self {
        Self {
            value: self.value.floor(),
            tangent: S::ZERO,
        } // piecewise constant
    }
    #[inline]
    fn ceil(self) -> Self {
        Self {
            value: self.value.ceil(),
            tangent: S::ZERO,
        }
    }
    #[inline]
    fn sinh(self) -> Self {
        Self {
            value: self.value.sinh(),
            tangent: self.tangent * self.value.cosh(),
        }
    }
    #[inline]
    fn cosh(self) -> Self {
        Self {
            value: self.value.cosh(),
            tangent: self.tangent * self.value.sinh(),
        }
    }
    #[inline]
    fn tanh(self) -> Self {
        let tv = self.value.tanh();
        Self {
            value: tv,
            tangent: self.tangent * (S::ONE - tv * tv),
        }
    }
    #[inline]
    fn asinh(self) -> Self {
        let d = (self.value * self.value + S::ONE).sqrt();
        Self {
            value: self.value.asinh(),
            tangent: self.tangent / d,
        }
    }
    #[inline]
    fn acosh(self) -> Self {
        let d = (self.value * self.value - S::ONE).sqrt();
        Self {
            value: self.value.acosh(),
            tangent: self.tangent / d,
        }
    }
    #[inline]
    fn atanh(self) -> Self {
        Self {
            value: self.value.atanh(),
            tangent: self.tangent / (S::ONE - self.value * self.value),
        }
    }

    fn iterate(
        self,
        _n: usize,
        _body: impl Fn(Self) -> Self,
        _c: impl Fn(Self, Self) -> S::Mask,
    ) -> Self {
        unimplemented!(
            "Dual autodiff does not differentiate fixed-point iteration (metric geometry is straight-line)"
        )
    }
    fn iterate_vec<const N: usize>(
        _init: [Self; N],
        _n: usize,
        _body: impl Fn([Self; N]) -> [Self; N],
        _c: impl Fn([Self; N], [Self; N]) -> S::Mask,
        _result: usize,
    ) -> Self {
        unimplemented!("Dual autodiff does not differentiate fixed-point iteration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn dual_reproduces_analytic_derivatives() {
        // seed x, differentiate closed-form functions, compare tangent to the hand derivative.
        for &x in &[0.7_f64, 1.3, 2.5] {
            let d = Dual::variable(x);
            // d/dx (x^2) = 2x
            assert!(approx((d * d).tangent, 2.0 * x));
            // d/dx sqrt(x) = 1/(2 sqrt x)
            assert!(approx(Numeric::sqrt(d).tangent, 1.0 / (2.0 * x.sqrt())));
            // d/dx (1/x) = -1/x^2
            assert!(approx(d.recip().tangent, -1.0 / (x * x)));
            // product/quotient: d/dx (x^2 / (x+1)) = (x^2 + 2x)/(x+1)^2
            let g = (d * d) / (d + Dual::constant(1.0));
            assert!(approx(
                g.tangent,
                (x * x + 2.0 * x) / ((x + 1.0) * (x + 1.0))
            ));
            // chain: d/dx sqrt(1 + x^2) = x / sqrt(1 + x^2)
            let h = Numeric::sqrt(Dual::constant(1.0) + d * d);
            assert!(approx(h.tangent, x / (1.0 + x * x).sqrt()));
            // transcendental: d/dx sin(x^2) = 2x cos(x^2)
            assert!(approx((d * d).sin().tangent, 2.0 * x * (x * x).cos()));
        }
    }

    #[test]
    fn dual_constant_has_zero_tangent() {
        let c = Dual::<f64>::constant(3.0);
        assert_eq!(c.tangent, 0.0);
        assert_eq!((c * c).tangent, 0.0); // a constant expression stays constant
    }
}
