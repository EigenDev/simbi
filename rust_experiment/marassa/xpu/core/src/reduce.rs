// =============================================================================
// reduce.rs
//
// reduction operations for device buffers.
// provides generic reduction framework with built-in operations (sum, max, min)
// and support for custom user-defined reductions.
//
// design:
//   - `Reduce<T>` trait defines reduction operation (identity + combine)
//   - built-in operations: Sum, Max, Min
//   - device-specific optimization (cpu loop, gpu kernels)
//   - type-safe: works on any T implementing the reduction
//
// usage:
//   let total = device.reduce(&buffer, Sum)?;
//   let maximum = device.reduce(&buffer, Max)?;
//   let custom = device.reduce(&buffer, MyReduction)?;
// =============================================================================

/// trait defining a reduction operation.
/// implementations provide identity element and combining function.
pub trait Reduce<T> {
    /// returns the identity element for this reduction.
    /// for sum: 0, for max: -infinity, for min: +infinity
    fn identity() -> T;

    /// combines two values according to the reduction operation.
    /// must be associative: combine(combine(a, b), c) == combine(a, combine(b, c))
    fn combine(a: T, b: T) -> T;
}

/// summation reduction: computes sum of all elements.
pub struct Sum;

/// maximum reduction: finds largest element.
pub struct Max;

/// minimum reduction: finds smallest element.
pub struct Min;

/// product reduction: computes product of all elements.
pub struct Product;

// =============================================================================
// sum implementations
// =============================================================================

impl Reduce<i32> for Sum {
    fn identity() -> i32 {
        0
    }
    fn combine(a: i32, b: i32) -> i32 {
        a + b
    }
}

impl Reduce<i64> for Sum {
    fn identity() -> i64 {
        0
    }
    fn combine(a: i64, b: i64) -> i64 {
        a + b
    }
}

impl Reduce<f32> for Sum {
    fn identity() -> f32 {
        0.0
    }
    fn combine(a: f32, b: f32) -> f32 {
        a + b
    }
}

impl Reduce<f64> for Sum {
    fn identity() -> f64 {
        0.0
    }
    fn combine(a: f64, b: f64) -> f64 {
        a + b
    }
}

impl Reduce<usize> for Sum {
    fn identity() -> usize {
        0
    }
    fn combine(a: usize, b: usize) -> usize {
        a + b
    }
}

// =============================================================================
// max implementations
// =============================================================================

impl Reduce<i32> for Max {
    fn identity() -> i32 {
        i32::MIN
    }
    fn combine(a: i32, b: i32) -> i32 {
        a.max(b)
    }
}

impl Reduce<i64> for Max {
    fn identity() -> i64 {
        i64::MIN
    }
    fn combine(a: i64, b: i64) -> i64 {
        a.max(b)
    }
}

impl Reduce<f32> for Max {
    fn identity() -> f32 {
        f32::NEG_INFINITY
    }
    fn combine(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

impl Reduce<f64> for Max {
    fn identity() -> f64 {
        f64::NEG_INFINITY
    }
    fn combine(a: f64, b: f64) -> f64 {
        a.max(b)
    }
}

impl Reduce<usize> for Max {
    fn identity() -> usize {
        usize::MIN
    }
    fn combine(a: usize, b: usize) -> usize {
        a.max(b)
    }
}

// =============================================================================
// min implementations
// =============================================================================

impl Reduce<i32> for Min {
    fn identity() -> i32 {
        i32::MAX
    }
    fn combine(a: i32, b: i32) -> i32 {
        a.min(b)
    }
}

impl Reduce<i64> for Min {
    fn identity() -> i64 {
        i64::MAX
    }
    fn combine(a: i64, b: i64) -> i64 {
        a.min(b)
    }
}

impl Reduce<f32> for Min {
    fn identity() -> f32 {
        f32::INFINITY
    }
    fn combine(a: f32, b: f32) -> f32 {
        a.min(b)
    }
}

impl Reduce<f64> for Min {
    fn identity() -> f64 {
        f64::INFINITY
    }
    fn combine(a: f64, b: f64) -> f64 {
        a.min(b)
    }
}

impl Reduce<usize> for Min {
    fn identity() -> usize {
        usize::MAX
    }
    fn combine(a: usize, b: usize) -> usize {
        a.min(b)
    }
}

// =============================================================================
// product implementations
// =============================================================================

impl Reduce<i32> for Product {
    fn identity() -> i32 {
        1
    }
    fn combine(a: i32, b: i32) -> i32 {
        a * b
    }
}

impl Reduce<i64> for Product {
    fn identity() -> i64 {
        1
    }
    fn combine(a: i64, b: i64) -> i64 {
        a * b
    }
}

impl Reduce<f32> for Product {
    fn identity() -> f32 {
        1.0
    }
    fn combine(a: f32, b: f32) -> f32 {
        a * b
    }
}

impl Reduce<f64> for Product {
    fn identity() -> f64 {
        1.0
    }
    fn combine(a: f64, b: f64) -> f64 {
        a * b
    }
}

impl Reduce<usize> for Product {
    fn identity() -> usize {
        1
    }
    fn combine(a: usize, b: usize) -> usize {
        a * b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_identity() {
        assert_eq!(<Sum as Reduce<i32>>::identity(), 0i32);
        assert_eq!(<Sum as Reduce<f64>>::identity(), 0.0f64);
    }

    #[test]
    fn test_sum_combine() {
        assert_eq!(<Sum as Reduce<i32>>::combine(2, 3), 5);
        assert_eq!(<Sum as Reduce<f64>>::combine(1.5, 2.5), 4.0);
    }

    #[test]
    fn test_max_identity() {
        assert_eq!(<Max as Reduce<i32>>::identity(), i32::MIN);
        assert_eq!(<Max as Reduce<f64>>::identity(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_max_combine() {
        assert_eq!(<Max as Reduce<i32>>::combine(2, 5), 5);
        assert_eq!(<Max as Reduce<f64>>::combine(3.0, 1.5), 3.0);
    }

    #[test]
    fn test_min_identity() {
        assert_eq!(<Min as Reduce<i32>>::identity(), i32::MAX);
        assert_eq!(<Min as Reduce<f64>>::identity(), f64::INFINITY);
    }

    #[test]
    fn test_min_combine() {
        assert_eq!(<Min as Reduce<i32>>::combine(2, 5), 2);
        assert_eq!(<Min as Reduce<f64>>::combine(3.0, 1.5), 1.5);
    }

    #[test]
    fn test_product_identity() {
        assert_eq!(<Product as Reduce<i32>>::identity(), 1i32);
        assert_eq!(<Product as Reduce<f64>>::identity(), 1.0f64);
    }

    #[test]
    fn test_product_combine() {
        assert_eq!(<Product as Reduce<i32>>::combine(2, 3), 6);
        assert_eq!(<Product as Reduce<f64>>::combine(2.0, 3.0), 6.0);
    }

    #[test]
    fn test_associativity() {
        // sum is associative
        let a = 1;
        let b = 2;
        let c = 3;
        assert_eq!(
            <Sum as Reduce<i32>>::combine(<Sum as Reduce<i32>>::combine(a, b), c),
            <Sum as Reduce<i32>>::combine(a, <Sum as Reduce<i32>>::combine(b, c))
        );

        // max is associative
        assert_eq!(
            <Max as Reduce<i32>>::combine(<Max as Reduce<i32>>::combine(a, b), c),
            <Max as Reduce<i32>>::combine(a, <Max as Reduce<i32>>::combine(b, c))
        );
    }
}
