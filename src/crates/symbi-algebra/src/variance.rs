// =============================================================================
// variance.rs
//
// type-safe index variance for vectors on curved manifolds.
// prevents accidental mixing of contravariant (v^i) and covariant (v_i)
// quantities at compile time. zero runtime cost via #[repr(transparent)].
//
// contravariant (Upper): vectors, velocity, shift, B^i
// covariant (Lower): gradients, momentum density S_i, 1-forms
//
// same-variance arithmetic compiles:  Con + Con, Cov + Cov
// cross-variance arithmetic rejected: Con + Cov -> compile error
// natural pairing is metric-free:     contract(v^i, w_i) -> scalar
//
// usage:
//   let v = Contravariant::new(vec3(1.0, 0.0, 0.0));
//   let w = Covariant::new(vec3(1.0, 2.0, 3.0));
//   let s = v.contract(&w);  // v^i w_i = 1.0
//   // v + w;  // compile error: Con + Cov is rejected by the type system
// =============================================================================

use std::fmt;
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::Tensor;
use crate::algebra::Numeric as Scalar;

// ============================================================
// variance markers
// ============================================================

// the four inhabited (variance x frame) states. variance (Upper/Lower) is a
// coordinate-frame concept (the metric gamma != delta there); the orthonormal (Ortho) and global
// Cartesian (Cart) frames are Euclidean (gamma = delta), so variance collapses and they need no
// up/down tag. so the product collapses to four markers.

/// coordinate-frame contravariant index (upper, v^i along d/dx^i).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Upper {}

/// coordinate-frame covariant index (lower, v_i; 1-forms, gradients, gamma_{ij} v^j).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lower {}

/// orthonormal (physical) frame: components V_a along the unit vectors e_a = (1/h_a) d/dx^i.
/// V_a = h_a v^a; here the metric is delta, so |V|^2 = sum V_a^2 is metric-free and there is no
/// variance distinction. the frame a Riemann solver is locally flat in — where the substrate
/// stores velocity / momentum / B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ortho {}

/// global Cartesian embedding frame: components v^x in lab axes. what external/global couplings
/// use (immersed-body gravity, output, lab-frame IC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cart {}

// ============================================================
// indexed vector: tensor with compile-time variance tag
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Indexed<V, S, const D: usize> {
    data: Tensor<S, D>,
    _variance: PhantomData<V>,
}

/// contravariant vector: coordinate-frame components v^i (upper indices).
pub type Contravariant<S, const D: usize> = Indexed<Upper, S, D>;

/// covariant vector (1-form): coordinate-frame components v_i (lower indices).
pub type Covariant<S, const D: usize> = Indexed<Lower, S, D>;

/// physical vector: orthonormal-frame components V_a (= h_a v^a). the substrate's stored frame.
pub type Physical<S, const D: usize> = Indexed<Ortho, S, D>;

/// embedded vector: global-Cartesian components v^x.
pub type Embedded<S, const D: usize> = Indexed<Cart, S, D>;

// ============================================================
// constructors
// ============================================================

impl<V, S: Copy, const D: usize> Indexed<V, S, D> {
    pub fn new(data: Tensor<S, D>) -> Self {
        Self {
            data,
            _variance: PhantomData,
        }
    }

    pub fn from_array(data: [S; D]) -> Self {
        Self::new(Tensor::new(data))
    }

    /// access the underlying tensor (read-only).
    pub fn raw(&self) -> &Tensor<S, D> {
        &self.data
    }

    /// consume and return the underlying tensor.
    pub fn into_raw(self) -> Tensor<S, D> {
        self.data
    }
}

impl<V, S: Scalar, const D: usize> Indexed<V, S, D> {
    pub fn zeros() -> Self {
        Self::new(Tensor::zeros())
    }
}

// ============================================================
// indexing
// ============================================================

impl<V, S, const D: usize> Index<usize> for Indexed<V, S, D> {
    type Output = S;
    fn index(&self, idx: usize) -> &S {
        &self.data[idx]
    }
}

impl<V, S, const D: usize> IndexMut<usize> for Indexed<V, S, D> {
    fn index_mut(&mut self, idx: usize) -> &mut S {
        &mut self.data[idx]
    }
}

// ============================================================
// same-variance arithmetic: preserves variance, prevents mixing
// ============================================================

impl<V, S: Scalar, const D: usize> Add for Indexed<V, S, D> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.data + rhs.data)
    }
}

impl<V, S: Scalar, const D: usize> Sub for Indexed<V, S, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.data - rhs.data)
    }
}

impl<V, S: Scalar, const D: usize> Neg for Indexed<V, S, D> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.data)
    }
}

impl<V, S: Scalar, const D: usize> AddAssign for Indexed<V, S, D> {
    fn add_assign(&mut self, rhs: Self) {
        self.data += rhs.data;
    }
}

impl<V, S: Scalar, const D: usize> SubAssign for Indexed<V, S, D> {
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= rhs.data;
    }
}

// ============================================================
// scalar broadcast
// ============================================================

macro_rules! impl_indexed_scalar_ops {
    ($scalar:ty) => {
        impl<V, const D: usize> Mul<$scalar> for Indexed<V, $scalar, D> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                Self::new(self.data * rhs)
            }
        }

        impl<V, const D: usize> Mul<Indexed<V, $scalar, D>> for $scalar {
            type Output = Indexed<V, $scalar, D>;
            fn mul(self, rhs: Indexed<V, $scalar, D>) -> Indexed<V, $scalar, D> {
                Indexed::new(self * rhs.data)
            }
        }

        impl<V, const D: usize> Div<$scalar> for Indexed<V, $scalar, D> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                Self::new(self.data / rhs)
            }
        }

        impl<V, const D: usize> MulAssign<$scalar> for Indexed<V, $scalar, D> {
            fn mul_assign(&mut self, rhs: $scalar) {
                self.data *= rhs;
            }
        }

        impl<V, const D: usize> DivAssign<$scalar> for Indexed<V, $scalar, D> {
            fn div_assign(&mut self, rhs: $scalar) {
                self.data /= rhs;
            }
        }
    };
}

impl_indexed_scalar_ops!(f64);
impl_indexed_scalar_ops!(f32);

// ============================================================
// contraction: v^i w_i -> scalar (metric-free natural pairing)
// ============================================================

impl<S: Scalar, const D: usize> Contravariant<S, D> {
    /// contract with a covariant vector: v^i w_i.
    pub fn contract(&self, w: &Covariant<S, D>) -> S {
        self.data.dot(&w.data)
    }
}

impl<S: Scalar, const D: usize> Covariant<S, D> {
    /// contract with a contravariant vector: w_i v^i.
    pub fn contract(&self, v: &Contravariant<S, D>) -> S {
        self.data.dot(&v.data)
    }
}

impl<S: Scalar, const D: usize> Physical<S, D> {
    /// Euclidean squared norm |V|^2 = sum_a V_a^2 — metric-free, because the orthonormal frame has
    /// gamma = delta. (the coordinate-frame norm carries the metric instead: g_ij v^i v^j =
    /// v^i (lower v)_i = `Contravariant::contract(&metric.lower(v))`.)
    pub fn norm_sq(&self) -> S {
        self.data.dot(&self.data)
    }
}

/// free function: contract v^i w_i.
pub fn contract<S: Scalar, const D: usize>(v: &Contravariant<S, D>, w: &Covariant<S, D>) -> S {
    v.contract(w)
}

// ============================================================
// display
// ============================================================

impl<S: Copy + fmt::Display, const D: usize> fmt::Display for Contravariant<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "^{}", self.data)
    }
}

impl<S: Copy + fmt::Display, const D: usize> fmt::Display for Covariant<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.data)
    }
}

// ============================================================
// conversions
// ============================================================

impl<V, S: Copy, const D: usize> From<Tensor<S, D>> for Indexed<V, S, D> {
    fn from(t: Tensor<S, D>) -> Self {
        Self::new(t)
    }
}

impl<V, S: Copy, const D: usize> From<[S; D]> for Indexed<V, S, D> {
    fn from(data: [S; D]) -> Self {
        Self::from_array(data)
    }
}

impl<V, S: Copy + Default, const D: usize> Default for Indexed<V, S, D> {
    fn default() -> Self {
        Self::new(Tensor::default())
    }
}

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3;

    type Con3 = Contravariant<f64, 3>;
    type Cov3 = Covariant<f64, 3>;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    // ---- construction ----

    #[test]
    fn test_contravariant_new() {
        let v = Con3::new(vec3(1.0, 2.0, 3.0));
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_covariant_new() {
        let w = Cov3::from_array([4.0, 5.0, 6.0]);
        assert_eq!(w[0], 4.0);
        assert_eq!(w[1], 5.0);
        assert_eq!(w[2], 6.0);
    }

    #[test]
    fn test_zeros() {
        let v = Con3::zeros();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    // ---- same-variance arithmetic ----

    #[test]
    fn test_con_add_con() {
        let a = Con3::new(vec3(1.0, 2.0, 3.0));
        let b = Con3::new(vec3(4.0, 5.0, 6.0));
        let c = a + b;
        assert_eq!(c[0], 5.0);
        assert_eq!(c[1], 7.0);
        assert_eq!(c[2], 9.0);
    }

    #[test]
    fn test_cov_add_cov() {
        let a = Cov3::from_array([1.0, 2.0, 3.0]);
        let b = Cov3::from_array([4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c[0], 5.0);
    }

    #[test]
    fn test_con_sub_con() {
        let a = Con3::new(vec3(5.0, 7.0, 9.0));
        let b = Con3::new(vec3(1.0, 2.0, 3.0));
        let c = a - b;
        assert_eq!(c[0], 4.0);
        assert_eq!(c[1], 5.0);
        assert_eq!(c[2], 6.0);
    }

    #[test]
    fn test_neg() {
        let v = Con3::new(vec3(1.0, -2.0, 3.0));
        let n = -v;
        assert_eq!(n[0], -1.0);
        assert_eq!(n[1], 2.0);
        assert_eq!(n[2], -3.0);
    }

    // ---- scalar broadcast ----

    #[test]
    fn test_scalar_mul() {
        let v = Con3::new(vec3(1.0, 2.0, 3.0));
        let w = v * 2.0;
        assert_eq!(w[0], 2.0);
        assert_eq!(w[1], 4.0);
        assert_eq!(w[2], 6.0);
    }

    #[test]
    fn test_scalar_mul_reverse() {
        let v = Cov3::from_array([1.0, 2.0, 3.0]);
        let w = 3.0 * v;
        assert_eq!(w[0], 3.0);
        assert_eq!(w[1], 6.0);
        assert_eq!(w[2], 9.0);
    }

    #[test]
    fn test_scalar_div() {
        let v = Con3::new(vec3(2.0, 4.0, 6.0));
        let w = v / 2.0;
        assert_eq!(w[0], 1.0);
        assert_eq!(w[1], 2.0);
        assert_eq!(w[2], 3.0);
    }

    // ---- contraction ----

    #[test]
    fn test_contract_con_cov() {
        let v = Con3::new(vec3(1.0, 2.0, 3.0));
        let w = Cov3::from_array([4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 32
        assert!(approx(v.contract(&w), 32.0));
    }

    #[test]
    fn test_contract_cov_con() {
        let v = Con3::new(vec3(1.0, 2.0, 3.0));
        let w = Cov3::from_array([4.0, 5.0, 6.0]);
        // commutative
        assert!(approx(w.contract(&v), 32.0));
    }

    #[test]
    fn test_contract_free_fn() {
        let v = Con3::new(vec3(1.0, 0.0, 0.0));
        let w = Cov3::from_array([7.0, 8.0, 9.0]);
        assert!(approx(contract(&v, &w), 7.0));
    }

    #[test]
    fn test_contract_orthogonal() {
        let v = Con3::new(vec3(1.0, 0.0, 0.0));
        let w = Cov3::from_array([0.0, 1.0, 0.0]);
        assert!(approx(v.contract(&w), 0.0));
    }

    // ---- raw access ----

    #[test]
    fn test_raw_roundtrip() {
        let t = vec3(1.0, 2.0, 3.0);
        let v = Con3::new(t);
        assert_eq!(*v.raw(), t);
        assert_eq!(v.into_raw(), t);
    }

    // ---- index mutation ----

    #[test]
    fn test_index_mut() {
        let mut v = Con3::zeros();
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        assert_eq!(v[0], 1.0);
        assert_eq!(v[2], 3.0);
    }

    // ---- add assign ----

    #[test]
    fn test_add_assign() {
        let mut v = Con3::new(vec3(1.0, 2.0, 3.0));
        v += Con3::new(vec3(4.0, 5.0, 6.0));
        assert_eq!(v[0], 5.0);
    }

    // ---- display ----

    #[test]
    fn test_display_con() {
        let v = Contravariant::<f64, 2>::from_array([1.0, 2.0]);
        assert_eq!(format!("{}", v), "^(1, 2)");
    }

    #[test]
    fn test_display_cov() {
        let w = Covariant::<f64, 2>::from_array([3.0, 4.0]);
        assert_eq!(format!("{}", w), "_(3, 4)");
    }

    // ---- from conversions ----

    #[test]
    fn test_from_tensor() {
        let t = vec3(1.0, 2.0, 3.0);
        let v: Con3 = t.into();
        assert_eq!(v[0], 1.0);
    }

    #[test]
    fn test_from_array() {
        let v: Cov3 = [1.0, 2.0, 3.0].into();
        assert_eq!(v[0], 1.0);
    }

    // ---- f32 ----

    #[test]
    fn test_f32_variance() {
        let v = Contravariant::<f32, 2>::from_array([1.0, 2.0]);
        let w = Covariant::<f32, 2>::from_array([3.0, 4.0]);
        let s = v.contract(&w);
        assert!((s - 11.0f32).abs() < 1e-5);
    }

    // ---- compile-time safety (each of these is a type error) ----
    // uncomment any of these to verify it produces a type error:
    //
    // fn test_con_add_cov_fails() {
    //     let v = Con3::new(vec3(1.0, 0.0, 0.0));
    //     let w = Cov3::from_array([1.0, 0.0, 0.0]);
    //     let _ = v + w;  // error: mismatched types
    // }
    //
    // fn test_con_sub_cov_fails() {
    //     let v = Con3::new(vec3(1.0, 0.0, 0.0));
    //     let w = Cov3::from_array([1.0, 0.0, 0.0]);
    //     let _ = v - w;  // error: mismatched types
    // }
}
