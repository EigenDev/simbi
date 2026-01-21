// =============================================================================
// lib.rs
//
// rusti-math: functional math layer for device-agnostic computation.
// builds on rusti-xpu to provide fields, lazy computations, and expression graphs.
//
// architecture:
//   - domain: pure topology (index spaces)
//   - computation: lazy expression graphs (pure functions)
//   - field: data containers (upcoming)
//   - execution: binding computations to devices (upcoming)
//
// design philosophy:
//   - separation of concerns: what vs where vs when
//   - pure functional: computations are immutable, composable
//   - zero-cost: abstractions compile away via monomorphization
//   - type-safe: lifetimes prevent use-after-free
// =============================================================================

pub mod computation;
pub mod domain;
pub mod execution;
pub mod field;
pub mod reconstruction;
pub mod stencil;

pub use computation::{constant, from_fn, identity, Computation};
pub use domain::{Domain, Domain1, Domain2, Domain3};
pub use execution::{evaluate, evaluate_into, parallel_evaluate, parallel_evaluate_into};
pub use field::{Field, FieldView, FieldViewMut};
pub use reconstruction::{
    pcm_left, pcm_right, plm_left, plm_left_vector, plm_right, plm_right_vector, Limiter,
    Reconstructible,
};
pub use stencil::{
    left_pattern, reconstruction_order, right_pattern, stencil_computation, stencil_size,
    Reconstruction, StencilView,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // create domain
        let domain = Domain::from_shape([100, 100]);

        // create lazy computations
        let x_coord = from_fn(domain, |coord| coord[0] as f64);
        let y_coord = from_fn(domain, |coord| coord[1] as f64);

        // compose: r^2 = x^2 + y^2
        let x_sq = x_coord.clone().mul(x_coord);
        let y_sq = y_coord.clone().mul(y_coord);
        let r_squared = x_sq.add(y_sq);

        // evaluate at a point
        let value = r_squared.eval([3, 4]);
        assert_eq!(value, 25.0); // 3^2 + 4^2 = 25

        // domain is preserved through operations
        assert_eq!(r_squared.domain(), domain);
    }

    #[test]
    fn test_expression_building() {
        let domain = Domain::from_shape([10]);

        // build: 2*x + 5
        let x = from_fn(domain, |coord| coord[0] as f64);
        let expr = x.scale(2.0).add_scalar(5.0);

        assert_eq!(expr.eval([0]), 5.0);
        assert_eq!(expr.eval([1]), 7.0);
        assert_eq!(expr.eval([5]), 15.0);
    }

    #[test]
    fn test_domain_operations() {
        let d1 = Domain::new([0, 0], [10, 10]);
        let d2 = Domain::new([5, 5], [15, 15]);

        let intersection = d1.intersect(&d2);
        assert_eq!(intersection.start, [5, 5]);
        assert_eq!(intersection.end, [10, 10]);

        let contracted = d1.contract(1);
        assert_eq!(contracted.shape(), [8, 8]);
    }

    #[test]
    fn test_lazy_evaluation() {
        let domain = Domain::from_shape([1000, 1000]);

        // these operations build the graph but don't execute
        let comp1 = constant(domain, 1.0);
        let comp2 = constant(domain, 2.0);
        let sum = comp1.add(comp2);

        // only when we eval() does computation happen
        assert_eq!(sum.eval([0, 0]), 3.0);
    }
}
