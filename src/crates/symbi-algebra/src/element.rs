// =============================================================================
// element.rs
//
// marker trait for types that can be stored as field elements.
// the associated Scalar type is the underlying float type used for
// arithmetic scaling. for scalar fields it is Self; for vector fields
// it is the component type.
//
// safety: the type must be contiguous, fixed-size, and zero-initialized
// bytes must produce a valid value (Buffer<T> relies on alloc_zeroed).
//
// usage:
//   fn process<T: FieldElement>(val: T) { ... }
// =============================================================================

/// marker for types that can be stored as field elements.
///
/// # safety
/// the type must be contiguous, fixed-size, and zero-initialized bytes
/// must produce a valid value. Buffer<T> relies on alloc_zeroed.
pub unsafe trait FieldElement: Copy {
    type Scalar: Copy + Send + Sync;
}

unsafe impl FieldElement for f64 {
    type Scalar = f64;
}
unsafe impl FieldElement for f32 {
    type Scalar = f32;
}
unsafe impl FieldElement for u8 {
    type Scalar = f64;
} // for ErrorCode (repr(u8)) in c2p fields
unsafe impl<const N: usize> FieldElement for [f64; N] {
    type Scalar = f64;
}
unsafe impl<const N: usize> FieldElement for [f32; N] {
    type Scalar = f32;
}

// tensor<S, N> has #[repr(transparent)] over [S; N] — same layout.
unsafe impl<const N: usize> FieldElement for crate::Tensor<f64, N> {
    type Scalar = f64;
}
unsafe impl<const N: usize> FieldElement for crate::Tensor<f32, N> {
    type Scalar = f32;
}

// matrix<S, N> wraps [[S; N]; N] — contiguous, fixed-size, zero-valid.
unsafe impl<const N: usize> FieldElement for crate::matrix::Matrix<f64, N> {
    type Scalar = f64;
}
unsafe impl<const N: usize> FieldElement for crate::matrix::Matrix<f32, N> {
    type Scalar = f32;
}

// indexed<V, S, D> has #[repr(transparent)] over Tensor<S, D> — same layout.
unsafe impl<V: Copy + 'static, const D: usize> FieldElement
    for crate::variance::Indexed<V, f64, D>
{
    type Scalar = f64;
}
unsafe impl<V: Copy + 'static, const D: usize> FieldElement
    for crate::variance::Indexed<V, f32, D>
{
    type Scalar = f32;
}
