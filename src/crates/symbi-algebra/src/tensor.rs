// =============================================================================
// tensor.rs
//
// fixed-size, stack-allocated tensor type with compile-time dimensions.
// wraps [T; N] with full arithmetic, dot/cross/norm, and field compatibility.
//
// the tensor type is generic over element type T and dimension N.
// f64-specific methods (dot, norm, normalize) live in a specialized impl block.
// cross product is only defined for N=3.
//
// usage:
//   let v = Tensor::<f64, 3>::new([1.0, 2.0, 3.0]);
//   let w = v * 2.0 + Tensor::splat(1.0);
//   let d = v.dot(&w);
//   let c = v.cross(&w);
// =============================================================================

use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Tensor<T, const N: usize> {
    pub data: [T; N],
}

// ============================================================
// constructors
// ============================================================

impl<T: Copy, const N: usize> Tensor<T, N> {
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// build a tensor from an index closure: `from_fn(|i| ...)`. the const-D bridge from a
    /// runtime slice to a fixed-rank tensor (e.g. the geometry source's `Metric<S, D>` call).
    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Self { data: std::array::from_fn(f) }
    }

    pub fn splat(val: T) -> Self {
        Self { data: [val; N] }
    }

    pub fn map<U: Copy>(self, f: impl Fn(T) -> U) -> Tensor<U, N> {
        Tensor {
            data: std::array::from_fn(|ii| f(self.data[ii])),
        }
    }

    pub fn zip_with<U: Copy, V: Copy>(
        self,
        other: Tensor<U, N>,
        f: impl Fn(T, U) -> V,
    ) -> Tensor<V, N> {
        Tensor {
            data: std::array::from_fn(|ii| f(self.data[ii], other.data[ii])),
        }
    }
}

impl<S: crate::algebra::Numeric, const N: usize> Tensor<S, N> {
    pub fn zeros() -> Self {
        Self { data: [S::ZERO; N] }
    }

    /// unit vector along axis `dir`: e_i = delta_{i,dir}.
    /// panics if dir >= N.
    pub fn unit(dir: usize) -> Self {
        let mut data = [S::ZERO; N];
        data[dir] = S::ONE;
        Self { data }
    }

    /// kronecker delta: delta(i, j) = 1 if i == j, 0 otherwise.
    pub fn kronecker(ii: usize, jj: usize) -> S {
        if ii == jj { S::ONE } else { S::ZERO }
    }

    /// scalar multiply: v * s. generic over S: Scalar.
    /// use this instead of the `*` operator when S is a generic type parameter,
    /// since `Tensor * S` operator is only implemented for concrete f64/f32.
    #[inline]
    pub fn scale(self, s: S) -> Self {
        self.map(|v| v * s)
    }
}

// the production `Selectable<S> for Tensor<S, N>` lives in `symbi_ir` so it can
// refer to `symbi_ir::algebra::Selectable` while keeping `Tensor` here.

// ============================================================
// indexing
// ============================================================

impl<T, const N: usize> Index<usize> for Tensor<T, N> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.data[idx]
    }
}

impl<T, const N: usize> IndexMut<usize> for Tensor<T, N> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.data[idx]
    }
}

// ============================================================
// element-wise arithmetic: tensor op tensor
// ============================================================

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Copy + $trait<Output = T>, const N: usize> $trait for Tensor<T, N> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                Tensor { data: std::array::from_fn(|ii| self.data[ii] $op rhs.data[ii]) }
            }
        }
    }
}

impl_binop!(Add, add, +);
impl_binop!(Sub, sub, -);
impl_binop!(Mul, mul, *);
impl_binop!(Div, div, /);

// ============================================================
// negation
// ============================================================

impl<T: Copy + Neg<Output = T>, const N: usize> Neg for Tensor<T, N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Tensor {
            data: std::array::from_fn(|ii| -self.data[ii]),
        }
    }
}

// ============================================================
// scalar broadcast: tensor op scalar, scalar op tensor
// ============================================================

macro_rules! impl_scalar_ops {
    ($scalar:ty) => {
        impl<const N: usize> Mul<$scalar> for Tensor<$scalar, N> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self::Output {
                Tensor {
                    data: std::array::from_fn(|ii| self.data[ii] * rhs),
                }
            }
        }

        impl<const N: usize> Mul<Tensor<$scalar, N>> for $scalar {
            type Output = Tensor<$scalar, N>;
            fn mul(self, rhs: Tensor<$scalar, N>) -> Self::Output {
                Tensor {
                    data: std::array::from_fn(|ii| self * rhs.data[ii]),
                }
            }
        }

        impl<const N: usize> Div<$scalar> for Tensor<$scalar, N> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self::Output {
                Tensor {
                    data: std::array::from_fn(|ii| self.data[ii] / rhs),
                }
            }
        }

        impl<const N: usize> Add<$scalar> for Tensor<$scalar, N> {
            type Output = Self;
            fn add(self, rhs: $scalar) -> Self::Output {
                Tensor {
                    data: std::array::from_fn(|ii| self.data[ii] + rhs),
                }
            }
        }

        impl<const N: usize> Sub<$scalar> for Tensor<$scalar, N> {
            type Output = Self;
            fn sub(self, rhs: $scalar) -> Self::Output {
                Tensor {
                    data: std::array::from_fn(|ii| self.data[ii] - rhs),
                }
            }
        }

        impl<const N: usize> MulAssign<$scalar> for Tensor<$scalar, N> {
            fn mul_assign(&mut self, rhs: $scalar) {
                for ii in 0..N {
                    self.data[ii] *= rhs;
                }
            }
        }

        impl<const N: usize> DivAssign<$scalar> for Tensor<$scalar, N> {
            fn div_assign(&mut self, rhs: $scalar) {
                for ii in 0..N {
                    self.data[ii] /= rhs;
                }
            }
        }
    };
}

impl_scalar_ops!(f64);
impl_scalar_ops!(f32);

// ============================================================
// compound assignment: tensor op= tensor
// ============================================================

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Copy + $trait, const N: usize> $trait for Tensor<T, N> {
            fn $method(&mut self, rhs: Self) {
                for ii in 0..N {
                    self.data[ii] $op rhs.data[ii];
                }
            }
        }
    }
}

impl_assign_op!(AddAssign, add_assign, +=);
impl_assign_op!(SubAssign, sub_assign, -=);
impl_assign_op!(MulAssign, mul_assign, *=);
impl_assign_op!(DivAssign, div_assign, /=);

// ============================================================
// dot, norm, normalize — generic over Scalar
// ============================================================

impl<S: crate::algebra::Numeric, const N: usize> Tensor<S, N> {
    pub fn dot(&self, other: &Self) -> S {
        let mut result = S::ZERO;
        for ii in 0..N {
            result += self.data[ii] * other.data[ii];
        }
        result
    }

    pub fn norm_squared(&self) -> S {
        self.dot(self)
    }

    pub fn norm(&self) -> S {
        self.norm_squared().sqrt()
    }

    pub fn component_sum(&self) -> S {
        let mut sum = S::ZERO;
        for ii in 0..N {
            sum += self.data[ii];
        }
        sum
    }

    pub fn component_min(&self) -> S {
        let mut val = self.data[0];
        for ii in 1..N {
            val = val.min(self.data[ii]);
        }
        val
    }

    pub fn component_max(&self) -> S {
        let mut val = self.data[0];
        for ii in 1..N {
            val = val.max(self.data[ii]);
        }
        val
    }
}

// cross product: only defined for 3-vectors
/// `normalize` requires host-side ordering (`if norm > 0`) — bound on
/// `OrderedNumeric`, which only host numeric types (f64, f32) impl. tracing
/// carriers (Gv) cannot be smuggled into `normalize`, preserving the A1
/// discipline ("no native `<` / `>` on a generic `S: Scalar`").
impl<S: crate::algebra::OrderedNumeric, const N: usize> Tensor<S, N> {
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > S::ZERO {
            Tensor {
                data: std::array::from_fn(|ii| self.data[ii] / n),
            }
        } else {
            *self
        }
    }
}

impl<S: crate::algebra::Numeric> Tensor<S, 3> {
    pub fn cross(&self, other: &Self) -> Self {
        Tensor::new([
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        ])
    }
}

// ============================================================
// free functions
// ============================================================

pub fn dot<const N: usize>(a: &Tensor<f64, N>, b: &Tensor<f64, N>) -> f64 {
    a.dot(b)
}

pub fn cross(a: &Tensor<f64, 3>, b: &Tensor<f64, 3>) -> Tensor<f64, 3> {
    a.cross(b)
}

pub fn norm<const N: usize>(v: &Tensor<f64, N>) -> f64 {
    v.norm()
}

pub fn normalize<const N: usize>(v: &Tensor<f64, N>) -> Tensor<f64, N> {
    v.normalize()
}

// ============================================================
// convenience constructors
// ============================================================

pub fn vec2(x: f64, y: f64) -> Tensor<f64, 2> {
    Tensor::new([x, y])
}

pub fn vec3(x: f64, y: f64, z: f64) -> Tensor<f64, 3> {
    Tensor::new([x, y, z])
}

pub fn vec4(x: f64, y: f64, z: f64, w: f64) -> Tensor<f64, 4> {
    Tensor::new([x, y, z, w])
}

// ============================================================
// conversions and formatting
// ============================================================

impl<T: Copy, const N: usize> From<[T; N]> for Tensor<T, N> {
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T: Copy, const N: usize> From<Tensor<T, N>> for [T; N] {
    fn from(t: Tensor<T, N>) -> Self {
        t.data
    }
}

impl<T: Copy + Default, const N: usize> Default for Tensor<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T: Copy + fmt::Display, const N: usize> fmt::Display for Tensor<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for ii in 0..N {
            if ii > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.data[ii])?;
        }
        write!(f, ")")
    }
}

// ============================================================
// type aliases
// ============================================================

pub type VecN<const N: usize> = Tensor<f64, N>;
pub type Vec2 = VecN<2>;
pub type Vec3 = VecN<3>;
pub type Vec4 = VecN<4>;

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    type V2 = Tensor<f64, 2>;
    type V3 = Tensor<f64, 3>;

    // ---- constructors ----

    #[test]
    fn test_new_and_index() {
        let v = V3::new([1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_splat() {
        assert_eq!(V3::splat(5.0), V3::new([5.0, 5.0, 5.0]));
    }

    #[test]
    fn test_zeros() {
        assert_eq!(V3::zeros(), V3::new([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_unit() {
        assert_eq!(V3::unit(0), V3::new([1.0, 0.0, 0.0]));
        assert_eq!(V3::unit(1), V3::new([0.0, 1.0, 0.0]));
        assert_eq!(V3::unit(2), V3::new([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_unit_1d() {
        assert_eq!(Tensor::<f64, 1>::unit(0), Tensor::new([1.0]));
    }

    #[test]
    fn test_kronecker() {
        assert_eq!(Tensor::<f64, 3>::kronecker(0, 0), 1.0);
        assert_eq!(Tensor::<f64, 3>::kronecker(0, 1), 0.0);
        assert_eq!(Tensor::<f64, 3>::kronecker(1, 1), 1.0);
        assert_eq!(Tensor::<f64, 3>::kronecker(2, 0), 0.0);
    }

    #[test]
    fn test_index_mut() {
        let mut v = V3::zeros();
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.0;
        assert_eq!(v, V3::new([1.0, 2.0, 3.0]));
    }

    // ---- element-wise arithmetic ----

    #[test]
    fn test_add() {
        let a = V3::new([1.0, 2.0, 3.0]);
        let b = V3::new([4.0, 5.0, 6.0]);
        assert_eq!(a + b, V3::new([5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_sub() {
        let a = V3::new([4.0, 5.0, 6.0]);
        let b = V3::new([1.0, 2.0, 3.0]);
        assert_eq!(a - b, V3::new([3.0, 3.0, 3.0]));
    }

    #[test]
    fn test_mul_elementwise() {
        let a = V3::new([1.0, 2.0, 3.0]);
        let b = V3::new([4.0, 5.0, 6.0]);
        assert_eq!(a * b, V3::new([4.0, 10.0, 18.0]));
    }

    #[test]
    fn test_div_elementwise() {
        let a = V3::new([4.0, 10.0, 18.0]);
        let b = V3::new([4.0, 5.0, 6.0]);
        assert_eq!(a / b, V3::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_neg() {
        let v = V3::new([1.0, -2.0, 3.0]);
        assert_eq!(-v, V3::new([-1.0, 2.0, -3.0]));
    }

    // ---- scalar broadcast ----

    #[test]
    fn test_scalar_mul() {
        let v = V3::new([1.0, 2.0, 3.0]);
        assert_eq!(v * 2.0, V3::new([2.0, 4.0, 6.0]));
        assert_eq!(2.0 * v, V3::new([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_scalar_div() {
        let v = V3::new([2.0, 4.0, 6.0]);
        assert_eq!(v / 2.0, V3::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_scalar_add() {
        let v = V3::new([1.0, 2.0, 3.0]);
        assert_eq!(v + 10.0, V3::new([11.0, 12.0, 13.0]));
    }

    #[test]
    fn test_scalar_sub() {
        let v = V3::new([1.0, 2.0, 3.0]);
        assert_eq!(v - 1.0, V3::new([0.0, 1.0, 2.0]));
    }

    // ---- compound assignment ----

    #[test]
    fn test_add_assign() {
        let mut v = V3::new([1.0, 2.0, 3.0]);
        v += V3::new([4.0, 5.0, 6.0]);
        assert_eq!(v, V3::new([5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_sub_assign() {
        let mut v = V3::new([5.0, 7.0, 9.0]);
        v -= V3::new([4.0, 5.0, 6.0]);
        assert_eq!(v, V3::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_mul_assign_scalar() {
        let mut v = V3::new([1.0, 2.0, 3.0]);
        v *= 2.0;
        assert_eq!(v, V3::new([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_div_assign_scalar() {
        let mut v = V3::new([2.0, 4.0, 6.0]);
        v /= 2.0;
        assert_eq!(v, V3::new([1.0, 2.0, 3.0]));
    }

    // ---- dot product ----

    #[test]
    fn test_dot() {
        let a = V3::new([1.0, 2.0, 3.0]);
        let b = V3::new([4.0, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_orthogonal() {
        let a = V3::new([1.0, 0.0, 0.0]);
        let b = V3::new([0.0, 1.0, 0.0]);
        assert!(a.dot(&b).abs() < 1e-10);
    }

    // ---- norm ----

    #[test]
    fn test_norm() {
        let v = V3::new([3.0, 4.0, 0.0]);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_squared() {
        let v = V3::new([3.0, 4.0, 0.0]);
        assert!((v.norm_squared() - 25.0).abs() < 1e-10);
    }

    // ---- normalize ----

    #[test]
    fn test_normalize() {
        let v = V3::new([3.0, 0.0, 0.0]);
        let n = v.normalize();
        assert!((n[0] - 1.0).abs() < 1e-10);
        assert!(n[1].abs() < 1e-10);
        assert!(n[2].abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v = V3::zeros();
        assert_eq!(v.normalize(), V3::zeros());
    }

    // ---- cross product ----

    #[test]
    fn test_cross_basis() {
        let x = V3::new([1.0, 0.0, 0.0]);
        let y = V3::new([0.0, 1.0, 0.0]);
        let z = x.cross(&y);
        assert!(z[0].abs() < 1e-10);
        assert!(z[1].abs() < 1e-10);
        assert!((z[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_anticommutative() {
        let a = V3::new([1.0, 2.0, 3.0]);
        let b = V3::new([4.0, 5.0, 6.0]);
        let ab = a.cross(&b);
        let ba = b.cross(&a);
        assert!((ab[0] + ba[0]).abs() < 1e-10);
        assert!((ab[1] + ba[1]).abs() < 1e-10);
        assert!((ab[2] + ba[2]).abs() < 1e-10);
    }

    #[test]
    fn test_cross_self_is_zero() {
        let v = V3::new([1.0, 2.0, 3.0]);
        let c = v.cross(&v);
        assert!(c[0].abs() < 1e-10);
        assert!(c[1].abs() < 1e-10);
        assert!(c[2].abs() < 1e-10);
    }

    // ---- map / zip_with ----

    #[test]
    fn test_map() {
        let v = V3::new([1.0, 4.0, 9.0]);
        let s = v.map(|x| x.sqrt());
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!((s[1] - 2.0).abs() < 1e-10);
        assert!((s[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_zip_with() {
        let a = V3::new([1.0, 2.0, 3.0]);
        let b = V3::new([4.0, 5.0, 6.0]);
        let c = a.zip_with(b, |x, y| x + y);
        assert_eq!(c, V3::new([5.0, 7.0, 9.0]));
    }

    // ---- conversions ----

    #[test]
    fn test_from_array() {
        let v: V3 = [1.0, 2.0, 3.0].into();
        assert_eq!(v, V3::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_into_array() {
        let v = V3::new([1.0, 2.0, 3.0]);
        let arr: [f64; 3] = v.into();
        assert_eq!(arr, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_destructure_via_data() {
        let v = V3::new([1.0, 2.0, 3.0]);
        let [x, y, z] = v.data;
        assert_eq!(x, 1.0);
        assert_eq!(y, 2.0);
        assert_eq!(z, 3.0);
    }

    // ---- display ----

    #[test]
    fn test_display() {
        let v = V2::new([1.5, -2.5]);
        assert_eq!(format!("{}", v), "(1.5, -2.5)");
    }

    // ---- free functions ----

    #[test]
    fn test_free_fn_dot() {
        let a = V3::new([1.0, 0.0, 0.0]);
        let b = V3::new([0.0, 1.0, 0.0]);
        assert!(dot(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_free_fn_cross() {
        let a = V3::new([1.0, 0.0, 0.0]);
        let b = V3::new([0.0, 1.0, 0.0]);
        let c = cross(&a, &b);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_fn_norm() {
        let v = V3::new([3.0, 4.0, 0.0]);
        assert!((norm(&v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_fn_normalize() {
        let v = V3::new([0.0, 4.0, 0.0]);
        let n = normalize(&v);
        assert!((n[1] - 1.0).abs() < 1e-10);
    }

    // ---- convenience constructors ----

    #[test]
    fn test_vec2_constructor() {
        assert_eq!(vec2(1.0, 2.0), Tensor::new([1.0, 2.0]));
    }

    #[test]
    fn test_vec3_constructor() {
        assert_eq!(vec3(1.0, 2.0, 3.0), Tensor::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_vec4_constructor() {
        assert_eq!(vec4(1.0, 2.0, 3.0, 4.0), Tensor::new([1.0, 2.0, 3.0, 4.0]));
    }

    // ---- complex expressions ----

    #[test]
    fn test_dream_api_expression() {
        // mirrors dream.md: ij * h - 0.5
        let ij = Tensor::<f64, 2>::new([50.0, 50.0]);
        let h = 0.01;
        let x = ij * h - 0.5;
        assert!((x[0] - 0.0).abs() < 1e-10);
        assert!((x[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_chained_ops() {
        // v = (1, 2, 3), w = v * 2 + splat(1) = (3, 5, 7)
        let v = V3::new([1.0, 2.0, 3.0]);
        let w = v * 2.0 + V3::splat(1.0);
        assert_eq!(w, V3::new([3.0, 5.0, 7.0]));
    }

    #[test]
    fn test_dot_via_ops() {
        // dot product via element-wise mul then sum
        let a = V3::new([1.0, 2.0, 3.0]);
        let b = V3::new([4.0, 5.0, 6.0]);
        let product = a * b;
        let sum: f64 = product.data.iter().sum();
        assert!((sum - 32.0).abs() < 1e-10);
    }

    // ---- generic over element type ----

    #[test]
    fn test_integer_tensor() {
        let a = Tensor::<i32, 3>::new([1, 2, 3]);
        let b = Tensor::<i32, 3>::new([4, 5, 6]);
        assert_eq!(a + b, Tensor::new([5, 7, 9]));
        assert_eq!(a * b, Tensor::new([4, 10, 18]));
        assert_eq!(-a, Tensor::new([-1, -2, -3]));
    }

    #[test]
    fn test_isize_tensor() {
        let a = Tensor::<isize, 2>::new([10, 20]);
        let b = Tensor::<isize, 2>::new([3, 7]);
        assert_eq!(a + b, Tensor::new([13, 27]));
        assert_eq!(a - b, Tensor::new([7, 13]));
    }
}
