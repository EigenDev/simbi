// =============================================================================
// matrix.rs
//
// fixed-size N x N square matrix with compile-time dimensions.
// wraps [[S; N]; N] with matrix-vector product, determinant, inverse,
// and standard arithmetic. row-major storage: data[i][j] = M_{ij}.
//
// used for metric tensors, Jacobians, and rank-2 objects in
// differential geometry.
//
// usage:
//   let g = Matrix::<f64, 3>::diag(Tensor::new([1.0, r*r, r*r*st*st]));
//   let v_lower = g * v_upper;  // index lowering: v_i = g_{ij} v^j
//   let det = g.det();
//   let g_inv = g.inv();
// =============================================================================

use std::fmt;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::algebra::Numeric as Scalar;
use crate::algebra::OrderedNumeric;
use crate::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<S, const N: usize> {
    pub data: [[S; N]; N],
}

// ============================================================
// constructors (generic over Copy)
// ============================================================

impl<S: Copy, const N: usize> Matrix<S, N> {
    pub const fn new(data: [[S; N]; N]) -> Self {
        Self { data }
    }

    pub fn from_fn(f: impl Fn(usize, usize) -> S) -> Self {
        Self {
            data: std::array::from_fn(|ii| std::array::from_fn(|jj| f(ii, jj))),
        }
    }

    pub fn row(&self, ii: usize) -> Tensor<S, N> {
        Tensor::new(self.data[ii])
    }

    pub fn col(&self, jj: usize) -> Tensor<S, N> {
        Tensor::new(std::array::from_fn(|ii| self.data[ii][jj]))
    }

    pub fn map<U: Copy>(self, f: impl Fn(S) -> U) -> Matrix<U, N> {
        Matrix {
            data: std::array::from_fn(|ii| std::array::from_fn(|jj| f(self.data[ii][jj]))),
        }
    }

    pub fn zip_with<U: Copy, V: Copy>(
        self,
        other: Matrix<U, N>,
        f: impl Fn(S, U) -> V,
    ) -> Matrix<V, N> {
        Matrix {
            data: std::array::from_fn(|ii| {
                std::array::from_fn(|jj| f(self.data[ii][jj], other.data[ii][jj]))
            }),
        }
    }
}

// ============================================================
// constructors requiring Scalar
// ============================================================

impl<S: Scalar, const N: usize> Matrix<S, N> {
    pub fn zeros() -> Self {
        Self {
            data: [[S::ZERO; N]; N],
        }
    }

    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for ii in 0..N {
            m.data[ii][ii] = S::ONE;
        }
        m
    }

    pub fn diag(d: Tensor<S, N>) -> Self {
        let mut m = Self::zeros();
        for ii in 0..N {
            m.data[ii][ii] = d[ii];
        }
        m
    }

    pub fn trace(&self) -> S {
        let mut sum = S::ZERO;
        for ii in 0..N {
            sum += self.data[ii][ii];
        }
        sum
    }

    pub fn transpose(&self) -> Self {
        Self::from_fn(|ii, jj| self.data[jj][ii])
    }

    pub fn symmetrize(&self) -> Self {
        let half = S::from_f64(0.5);
        Self::from_fn(|ii, jj| (self.data[ii][jj] + self.data[jj][ii]) * half)
    }

    pub fn frobenius_norm_sq(&self) -> S {
        let mut sum = S::ZERO;
        for ii in 0..N {
            for jj in 0..N {
                sum += self.data[ii][jj] * self.data[ii][jj];
            }
        }
        sum
    }

    /// bilinear contraction: a^T M b = M_{ij} a^i b^j.
    /// for a metric g, this computes the inner product g(u, v).
    pub fn contract(&self, a: &Tensor<S, N>, b: &Tensor<S, N>) -> S {
        let mut sum = S::ZERO;
        for ii in 0..N {
            for jj in 0..N {
                sum += self.data[ii][jj] * a[ii] * b[jj];
            }
        }
        sum
    }

    /// quadratic form: v^T M v = M_{ij} v^i v^j.
    /// for a metric g, this computes the squared norm g(v, v).
    pub fn quadratic(&self, v: &Tensor<S, N>) -> S {
        self.contract(v, v)
    }
}

// ============================================================
// outer product: v^i w^j -> M_{ij}
// ============================================================

pub fn outer<S: Scalar, const N: usize>(a: &Tensor<S, N>, b: &Tensor<S, N>) -> Matrix<S, N> {
    Matrix::from_fn(|ii, jj| a[ii] * b[jj])
}

// ============================================================
// indexing: m[(i, j)]
// ============================================================

impl<S, const N: usize> Index<(usize, usize)> for Matrix<S, N> {
    type Output = S;
    fn index(&self, (ii, jj): (usize, usize)) -> &S {
        &self.data[ii][jj]
    }
}

impl<S, const N: usize> IndexMut<(usize, usize)> for Matrix<S, N> {
    fn index_mut(&mut self, (ii, jj): (usize, usize)) -> &mut S {
        &mut self.data[ii][jj]
    }
}

// ============================================================
// matrix + matrix, matrix - matrix, -matrix
// ============================================================

impl<S: Scalar, const N: usize> Add for Matrix<S, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_fn(|ii, jj| self.data[ii][jj] + rhs.data[ii][jj])
    }
}

impl<S: Scalar, const N: usize> Sub for Matrix<S, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_fn(|ii, jj| self.data[ii][jj] - rhs.data[ii][jj])
    }
}

impl<S: Scalar, const N: usize> Neg for Matrix<S, N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_fn(|ii, jj| -self.data[ii][jj])
    }
}

impl<S: Scalar, const N: usize> AddAssign for Matrix<S, N> {
    fn add_assign(&mut self, rhs: Self) {
        for ii in 0..N {
            for jj in 0..N {
                self.data[ii][jj] += rhs.data[ii][jj];
            }
        }
    }
}

impl<S: Scalar, const N: usize> SubAssign for Matrix<S, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for ii in 0..N {
            for jj in 0..N {
                self.data[ii][jj] -= rhs.data[ii][jj];
            }
        }
    }
}

// ============================================================
// scalar broadcast: matrix * scalar, scalar * matrix
// ============================================================

macro_rules! impl_matrix_scalar_ops {
    ($scalar:ty) => {
        impl<const N: usize> Mul<$scalar> for Matrix<$scalar, N> {
            type Output = Self;
            fn mul(self, rhs: $scalar) -> Self {
                Self::from_fn(|ii, jj| self.data[ii][jj] * rhs)
            }
        }

        impl<const N: usize> Mul<Matrix<$scalar, N>> for $scalar {
            type Output = Matrix<$scalar, N>;
            fn mul(self, rhs: Matrix<$scalar, N>) -> Matrix<$scalar, N> {
                rhs * self
            }
        }

        impl<const N: usize> Div<$scalar> for Matrix<$scalar, N> {
            type Output = Self;
            fn div(self, rhs: $scalar) -> Self {
                Self::from_fn(|ii, jj| self.data[ii][jj] / rhs)
            }
        }

        impl<const N: usize> MulAssign<$scalar> for Matrix<$scalar, N> {
            fn mul_assign(&mut self, rhs: $scalar) {
                for ii in 0..N {
                    for jj in 0..N {
                        self.data[ii][jj] *= rhs;
                    }
                }
            }
        }
    };
}

impl_matrix_scalar_ops!(f64);
impl_matrix_scalar_ops!(f32);

// ============================================================
// matrix * matrix (N x N)
// ============================================================

impl<S: Scalar, const N: usize> Matrix<S, N> {
    pub fn matmul(&self, rhs: &Self) -> Self {
        let mut result = Self::zeros();
        for ii in 0..N {
            for kk in 0..N {
                let a_ik = self.data[ii][kk];
                for jj in 0..N {
                    result.data[ii][jj] += a_ik * rhs.data[kk][jj];
                }
            }
        }
        result
    }

    /// matrix-vector product: (M v)_i = M_{ij} v_j.
    /// generic over Scalar (unlike the Mul<Tensor> trait impls).
    pub fn mul_vec(&self, v: &Tensor<S, N>) -> Tensor<S, N> {
        Tensor::new(std::array::from_fn(|ii| {
            let mut sum = S::ZERO;
            for jj in 0..N {
                sum += self.data[ii][jj] * v.data[jj];
            }
            sum
        }))
    }
}

// ============================================================
// matrix * vector -> vector (contraction: (Mv)_i = M_{ij} v^j)
// ============================================================

macro_rules! impl_matrix_vec_mul {
    ($scalar:ty) => {
        impl<const N: usize> Mul<Tensor<$scalar, N>> for Matrix<$scalar, N> {
            type Output = Tensor<$scalar, N>;
            fn mul(self, rhs: Tensor<$scalar, N>) -> Tensor<$scalar, N> {
                Tensor::new(std::array::from_fn(|ii| {
                    let mut sum: $scalar = 0.0;
                    for jj in 0..N {
                        sum += self.data[ii][jj] * rhs.data[jj];
                    }
                    sum
                }))
            }
        }
    };
}

impl_matrix_vec_mul!(f64);
impl_matrix_vec_mul!(f32);

// ============================================================
// determinant and inverse: specialized for N = 1, 2, 3
// ============================================================

/// `is_symmetric` requires host-side ordering (`if diff.abs() > tol`) — bound
/// on `OrderedNumeric`, impl'd only by host numeric types (f64, f32). tracing
/// carriers (Gv) cannot be smuggled in, preserving the A1 discipline.
impl<S: OrderedNumeric, const N: usize> Matrix<S, N> {
    pub fn is_symmetric(&self, tol: S) -> bool {
        for ii in 0..N {
            for jj in (ii + 1)..N {
                if (self.data[ii][jj] - self.data[jj][ii]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

impl<S: Scalar> Matrix<S, 1> {
    pub fn det(&self) -> S {
        self.data[0][0]
    }

    pub fn inv(&self) -> Self {
        Self::new([[S::ONE / self.data[0][0]]])
    }
}

impl<S: Scalar> Matrix<S, 2> {
    pub fn det(&self) -> S {
        let [[a, b], [c, d]] = self.data;
        a * d - b * c
    }

    pub fn inv(&self) -> Self {
        let [[a, b], [c, d]] = self.data;
        let inv_det = S::ONE / (a * d - b * c);
        Self::new([[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]])
    }
}

impl<S: Scalar> Matrix<S, 3> {
    pub fn det(&self) -> S {
        let [[a, b, c], [d, e, f], [g, h, k]] = self.data;
        a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
    }

    pub fn inv(&self) -> Self {
        let [[a, b, c], [d, e, f], [g, h, k]] = self.data;
        let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
        let inv_det = S::ONE / det;
        Self::new([
            [
                (e * k - f * h) * inv_det,
                (c * h - b * k) * inv_det,
                (b * f - c * e) * inv_det,
            ],
            [
                (f * g - d * k) * inv_det,
                (a * k - c * g) * inv_det,
                (c * d - a * f) * inv_det,
            ],
            [
                (d * h - e * g) * inv_det,
                (b * g - a * h) * inv_det,
                (a * e - b * d) * inv_det,
            ],
        ])
    }
}

// ============================================================
// conversions and formatting
// ============================================================

impl<S: Copy, const N: usize> From<[[S; N]; N]> for Matrix<S, N> {
    fn from(data: [[S; N]; N]) -> Self {
        Self { data }
    }
}

impl<S: Copy, const N: usize> From<Matrix<S, N>> for [[S; N]; N] {
    fn from(m: Matrix<S, N>) -> Self {
        m.data
    }
}

impl<S: Copy + Default, const N: usize> Default for Matrix<S, N> {
    fn default() -> Self {
        Self {
            data: [[S::default(); N]; N],
        }
    }
}

impl<S: Copy + fmt::Display, const N: usize> fmt::Display for Matrix<S, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for ii in 0..N {
            if ii > 0 {
                write!(f, "; ")?;
            }
            for jj in 0..N {
                if jj > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[ii][jj])?;
            }
        }
        write!(f, "]")
    }
}

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3;

    type M2 = Matrix<f64, 2>;
    type M3 = Matrix<f64, 3>;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    fn mat_approx(a: &M3, b: &M3) -> bool {
        for ii in 0..3 {
            for jj in 0..3 {
                if !approx(a[(ii, jj)], b[(ii, jj)]) {
                    return false;
                }
            }
        }
        true
    }

    // ---- constructors ----

    #[test]
    fn test_zeros() {
        let m = M3::zeros();
        for ii in 0..3 {
            for jj in 0..3 {
                assert_eq!(m[(ii, jj)], 0.0);
            }
        }
    }

    #[test]
    fn test_identity() {
        let m = M3::identity();
        for ii in 0..3 {
            for jj in 0..3 {
                let expected = if ii == jj { 1.0 } else { 0.0 };
                assert_eq!(m[(ii, jj)], expected);
            }
        }
    }

    #[test]
    fn test_diag() {
        let m = M3::diag(vec3(2.0, 3.0, 5.0));
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(1, 1)], 3.0);
        assert_eq!(m[(2, 2)], 5.0);
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(1, 0)], 0.0);
    }

    #[test]
    fn test_from_fn() {
        let m = M3::from_fn(|ii, jj| (ii * 3 + jj) as f64);
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(0, 2)], 2.0);
        assert_eq!(m[(2, 1)], 7.0);
    }

    #[test]
    fn test_from_rows() {
        let m = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    // ---- row / col ----

    #[test]
    fn test_row_col() {
        let m = M3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(m.row(1), vec3(4.0, 5.0, 6.0));
        assert_eq!(m.col(2), vec3(3.0, 6.0, 9.0));
    }

    // ---- arithmetic ----

    #[test]
    fn test_add() {
        let a = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = M2::new([[5.0, 6.0], [7.0, 8.0]]);
        let c = a + b;
        assert_eq!(c, M2::new([[6.0, 8.0], [10.0, 12.0]]));
    }

    #[test]
    fn test_sub() {
        let a = M2::new([[5.0, 6.0], [7.0, 8.0]]);
        let b = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(a - b, M2::new([[4.0, 4.0], [4.0, 4.0]]));
    }

    #[test]
    fn test_neg() {
        let a = M2::new([[1.0, -2.0], [3.0, -4.0]]);
        assert_eq!(-a, M2::new([[-1.0, 2.0], [-3.0, 4.0]]));
    }

    #[test]
    fn test_scalar_mul() {
        let m = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m * 2.0, M2::new([[2.0, 4.0], [6.0, 8.0]]));
        assert_eq!(2.0 * m, M2::new([[2.0, 4.0], [6.0, 8.0]]));
    }

    #[test]
    fn test_scalar_div() {
        let m = M2::new([[2.0, 4.0], [6.0, 8.0]]);
        assert_eq!(m / 2.0, M2::new([[1.0, 2.0], [3.0, 4.0]]));
    }

    // ---- matrix * vector ----

    #[test]
    fn test_mat_vec_identity() {
        let v = vec3(1.0, 2.0, 3.0);
        let result = M3::identity() * v;
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat_vec_diag() {
        let g = M3::diag(vec3(2.0, 3.0, 5.0));
        let v = vec3(1.0, 1.0, 1.0);
        assert_eq!(g * v, vec3(2.0, 3.0, 5.0));
    }

    #[test]
    fn test_mat_vec_general() {
        let m = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Tensor::<f64, 2>::new([5.0, 6.0]);
        let result = m * v;
        // [1*5+2*6, 3*5+4*6] = [17, 39]
        assert_eq!(result, Tensor::new([17.0, 39.0]));
    }

    // ---- matrix * matrix ----

    #[test]
    fn test_matmul_identity() {
        let m = M3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert!(mat_approx(&m.matmul(&M3::identity()), &m));
        assert!(mat_approx(&M3::identity().matmul(&m), &m));
    }

    #[test]
    fn test_matmul_2x2() {
        let a = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = M2::new([[5.0, 6.0], [7.0, 8.0]]);
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(a.matmul(&b), M2::new([[19.0, 22.0], [43.0, 50.0]]));
    }

    // ---- trace ----

    #[test]
    fn test_trace() {
        let m = M3::new([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 9.0]]);
        assert!(approx(m.trace(), 15.0));
    }

    #[test]
    fn test_trace_identity() {
        assert!(approx(M3::identity().trace(), 3.0));
    }

    // ---- transpose ----

    #[test]
    fn test_transpose() {
        let m = M3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let mt = m.transpose();
        assert_eq!(mt[(0, 1)], 4.0);
        assert_eq!(mt[(1, 0)], 2.0);
        assert_eq!(mt[(2, 0)], 3.0);
    }

    #[test]
    fn test_transpose_symmetric() {
        let m = M2::new([[1.0, 2.0], [2.0, 4.0]]);
        assert_eq!(m.transpose(), m);
    }

    // ---- symmetrize ----

    #[test]
    fn test_symmetrize() {
        let m = M2::new([[1.0, 3.0], [1.0, 4.0]]);
        let s = m.symmetrize();
        assert_eq!(s, M2::new([[1.0, 2.0], [2.0, 4.0]]));
    }

    #[test]
    fn test_is_symmetric() {
        let sym = M2::new([[1.0, 2.0], [2.0, 4.0]]);
        let asym = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        assert!(sym.is_symmetric(1e-15));
        assert!(!asym.is_symmetric(1e-15));
    }

    // ---- determinant ----

    #[test]
    fn test_det_1x1() {
        let m = Matrix::<f64, 1>::new([[7.0]]);
        assert!(approx(m.det(), 7.0));
    }

    #[test]
    fn test_det_2x2() {
        let m = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        // 1*4 - 2*3 = -2
        assert!(approx(m.det(), -2.0));
    }

    #[test]
    fn test_det_2x2_identity() {
        assert!(approx(M2::identity().det(), 1.0));
    }

    #[test]
    fn test_det_3x3() {
        let m = M3::new([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]]);
        // cofactor expansion: 6*(-2*7-5*8) - 1*(4*7-5*2) + 1*(4*8-(-2)*2)
        // = 6*(-54) - 1*(18) + 1*(36) = -324 - 18 + 36 = -306
        assert!(approx(m.det(), -306.0));
    }

    #[test]
    fn test_det_3x3_identity() {
        assert!(approx(M3::identity().det(), 1.0));
    }

    #[test]
    fn test_det_3x3_diagonal() {
        let m = M3::diag(vec3(2.0, 3.0, 5.0));
        assert!(approx(m.det(), 30.0));
    }

    // ---- inverse ----

    #[test]
    fn test_inv_1x1() {
        let m = Matrix::<f64, 1>::new([[4.0]]);
        let mi = m.inv();
        assert!(approx(mi[(0, 0)], 0.25));
    }

    #[test]
    fn test_inv_2x2() {
        let m = M2::new([[4.0, 7.0], [2.0, 6.0]]);
        let mi = m.inv();
        let product = m.matmul(&mi);
        assert!(approx(product[(0, 0)], 1.0));
        assert!(approx(product[(0, 1)], 0.0));
        assert!(approx(product[(1, 0)], 0.0));
        assert!(approx(product[(1, 1)], 1.0));
    }

    #[test]
    fn test_inv_3x3() {
        let m = M3::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        let mi = m.inv();
        let product = m.matmul(&mi);
        assert!(mat_approx(&product, &M3::identity()));
    }

    #[test]
    fn test_inv_3x3_diagonal() {
        let m = M3::diag(vec3(2.0, 4.0, 8.0));
        let mi = m.inv();
        assert!(approx(mi[(0, 0)], 0.5));
        assert!(approx(mi[(1, 1)], 0.25));
        assert!(approx(mi[(2, 2)], 0.125));
        assert!(approx(mi[(0, 1)], 0.0));
    }

    // ---- bilinear contraction / quadratic form ----

    #[test]
    fn test_contract_identity() {
        let a = vec3(1.0, 2.0, 3.0);
        let b = vec3(4.0, 5.0, 6.0);
        // identity contraction = euclidean dot product
        let result = M3::identity().contract(&a, &b);
        assert!(approx(result, 32.0));
    }

    #[test]
    fn test_contract_diagonal_metric() {
        // spherical metric at r=2, theta=pi/4:
        // gamma = diag(1, r^2, r^2 sin^2(theta)) = diag(1, 4, 2)
        let r = 2.0;
        let sin_theta = std::f64::consts::FRAC_PI_4.sin();
        let g = M3::diag(vec3(1.0, r * r, r * r * sin_theta * sin_theta));
        let v = vec3(1.0, 1.0, 1.0);
        // |v|^2 = 1 + 4 + 2 = 7
        assert!(approx(g.quadratic(&v), 1.0 + r * r + r * r * sin_theta * sin_theta));
    }

    #[test]
    fn test_contract_symmetric() {
        // g(u, v) = g(v, u) for symmetric metric
        let g = M3::new([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]);
        let u = vec3(1.0, 2.0, 3.0);
        let v = vec3(4.0, 5.0, 6.0);
        assert!(approx(g.contract(&u, &v), g.contract(&v, &u)));
    }

    // ---- outer product ----

    #[test]
    fn test_outer_product() {
        let a = Tensor::<f64, 2>::new([1.0, 2.0]);
        let b = Tensor::<f64, 2>::new([3.0, 4.0]);
        let m = outer(&a, &b);
        assert_eq!(m, M2::new([[3.0, 4.0], [6.0, 8.0]]));
    }

    #[test]
    fn test_outer_product_trace() {
        // tr(a (x) b) = a . b
        let a = vec3(1.0, 2.0, 3.0);
        let b = vec3(4.0, 5.0, 6.0);
        assert!(approx(outer(&a, &b).trace(), a.dot(&b)));
    }

    // ---- index lowering / raising ----

    #[test]
    fn test_index_lowering() {
        // v_i = g_{ij} v^j
        let g = M3::diag(vec3(1.0, 4.0, 9.0));
        let v_up = vec3(1.0, 1.0, 1.0);
        let v_down = g * v_up;
        assert_eq!(v_down, vec3(1.0, 4.0, 9.0));
    }

    #[test]
    fn test_index_raising() {
        // v^i = g^{ij} v_j
        let g = M3::diag(vec3(1.0, 4.0, 9.0));
        let g_inv = g.inv();
        let v_down = vec3(1.0, 4.0, 9.0);
        let v_up = g_inv * v_down;
        assert_eq!(v_up, vec3(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_lower_raise_roundtrip() {
        let g = M3::new([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]);
        let g_inv = g.inv();
        let v = vec3(1.0, 2.0, 3.0);
        let v_lowered = g * v;
        let v_raised = g_inv * v_lowered;
        for ii in 0..3 {
            assert!(approx(v_raised[ii], v[ii]));
        }
    }

    // ---- display ----

    #[test]
    fn test_display() {
        let m = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(format!("{}", m), "[1, 2; 3, 4]");
    }

    // ---- map / zip_with ----

    #[test]
    fn test_map() {
        let m = M2::new([[1.0, 4.0], [9.0, 16.0]]);
        let s = m.map(|x| x.sqrt());
        assert_eq!(s, M2::new([[1.0, 2.0], [3.0, 4.0]]));
    }

    #[test]
    fn test_zip_with() {
        let a = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = M2::new([[5.0, 6.0], [7.0, 8.0]]);
        let c = a.zip_with(b, |x, y| x * y);
        assert_eq!(c, M2::new([[5.0, 12.0], [21.0, 32.0]]));
    }

    // ---- compound assignment ----

    #[test]
    fn test_add_assign() {
        let mut a = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        a += M2::new([[5.0, 6.0], [7.0, 8.0]]);
        assert_eq!(a, M2::new([[6.0, 8.0], [10.0, 12.0]]));
    }

    #[test]
    fn test_mul_assign_scalar() {
        let mut m = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        m *= 3.0;
        assert_eq!(m, M2::new([[3.0, 6.0], [9.0, 12.0]]));
    }

    // ---- frobenius norm ----

    #[test]
    fn test_frobenius_norm_identity() {
        // ||I_3||_F^2 = 3
        assert!(approx(M3::identity().frobenius_norm_sq(), 3.0));
    }

    // ---- conversions ----

    #[test]
    fn test_from_array() {
        let m: M2 = [[1.0, 2.0], [3.0, 4.0]].into();
        assert_eq!(m[(1, 0)], 3.0);
    }

    #[test]
    fn test_into_array() {
        let m = M2::new([[1.0, 2.0], [3.0, 4.0]]);
        let arr: [[f64; 2]; 2] = m.into();
        assert_eq!(arr, [[1.0, 2.0], [3.0, 4.0]]);
    }

    // ---- spherical metric integration test ----

    #[test]
    fn test_spherical_metric_det() {
        // gamma = diag(1, r^2, r^2 sin^2 theta)
        // det(gamma) = r^4 sin^2 theta
        // sqrt(det) = r^2 |sin theta|
        let r = 3.0;
        let theta = std::f64::consts::FRAC_PI_3;
        let st = theta.sin();
        let g = M3::diag(vec3(1.0, r * r, r * r * st * st));
        assert!(approx(g.det(), r * r * r * r * st * st));
        assert!(approx(g.det().sqrt(), r * r * st));
    }

    #[test]
    fn test_spherical_metric_inv() {
        let r = 2.5;
        let theta = std::f64::consts::FRAC_PI_4;
        let st = theta.sin();
        let g = M3::diag(vec3(1.0, r * r, r * r * st * st));
        let gi = g.inv();
        assert!(approx(gi[(0, 0)], 1.0));
        assert!(approx(gi[(1, 1)], 1.0 / (r * r)));
        assert!(approx(gi[(2, 2)], 1.0 / (r * r * st * st)));
    }

    // ---- f32 ----

    #[test]
    fn test_f32_matrix() {
        let m = Matrix::<f32, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Tensor::<f32, 2>::new([1.0, 1.0]);
        let result = m * v;
        assert_eq!(result, Tensor::new([3.0f32, 7.0f32]));
    }

    #[test]
    fn test_f32_det_inv() {
        let m = Matrix::<f32, 2>::new([[4.0, 7.0], [2.0, 6.0]]);
        let det = m.det();
        assert!((det - 10.0f32).abs() < 1e-5);
        let mi = m.inv();
        let product = m.matmul(&mi);
        assert!((product[(0, 0)] - 1.0f32).abs() < 1e-5);
        assert!((product[(0, 1)]).abs() < 1e-5);
    }
}
