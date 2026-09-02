// =============================================================================
// normalized.rs
//
// the unit-normal witness: a `Normalized<T>` holds a vector whose components
// form a Euclidean unit vector, obtainable only through constructors that
// make it so. the face-normal use is the axis basis vector `e_k` — one-hot,
// exactly unit — which is what a directional sweep passes to a flux or
// wave-speed evaluation. the frame marker rides inside (`Normalized<Physical>`
// for a locally-flat solver, `Normalized<Covariant>` for a coordinate-frame
// flux), so a regime states which frame its normal is lawful in.
//
// usage:
//  let nhat: Normalized<Physical<f64, 3>> = Normalized::axis(0);
//  let vn = vel.dot(nhat.components());
// =============================================================================

use crate::Tensor;
use crate::algebra::Numeric;
use crate::variance::Indexed;

/// a vector whose components form a Euclidean unit vector. the payload is
/// reachable read-only, so the unit claim survives the value's lifetime, and
/// the claim is obtainable only through the witnessing constructors —
///
/// ```compile_fail
/// use symbi_algebra::{Normalized, Physical, vec3};
/// let v = Physical::<f64, 3>::new(vec3(2.0, 0.0, 0.0));
/// let _ = Normalized(v); // private field: no unit claim without a constructor
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Normalized<T>(T);

impl<T> Normalized<T> {
    /// read-only view of the witnessed vector.
    pub fn get(&self) -> &T {
        &self.0
    }
}

/// the face-normal contract a regime\'s flux and wave-speed evaluations
/// consume: an axis basis constructor (one-hot, exactly unit) and a
/// components view for the formula interior.
pub trait FaceNormal<S, const D: usize>: Copy {
    /// the axis basis normal `e_k` along grid axis `k`.
    fn axis(k: usize) -> Self;
    /// the component view the formula interior contracts against.
    fn components(&self) -> &Tensor<S, D>;
}

impl<V: Copy, S: Numeric, const D: usize> FaceNormal<S, D> for Normalized<Indexed<V, S, D>> {
    fn axis(k: usize) -> Self {
        Normalized(Indexed::new(Tensor::unit(k)))
    }
    fn components(&self) -> &Tensor<S, D> {
        self.0.raw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variance::Physical;

    #[test]
    fn axis_normal_is_one_hot() {
        let n: Normalized<Physical<f64, 3>> = Normalized::axis(1);
        assert_eq!(n.components()[0], 0.0);
        assert_eq!(n.components()[1], 1.0);
        assert_eq!(n.components()[2], 0.0);
    }

    #[test]
    fn witness_is_layout_neutral() {
        assert_eq!(
            std::mem::size_of::<Normalized<Physical<f64, 3>>>(),
            std::mem::size_of::<[f64; 3]>()
        );
    }
}
