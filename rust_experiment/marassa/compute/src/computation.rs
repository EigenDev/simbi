// =============================================================================
// computation.rs
//
// lazy expression graphs for device-agnostic computation.
// pure functional design - computations are immutable, composable expressions
// that describe what to compute without specifying where or when.
//
// design:
//   - computation<T, const N: usize, F> wraps any callable F
//   - purely lazy: no execution, no memory allocation
//   - composable via methods (add, mul, etc) instead of operators
//   - domain-aware: tracks valid index space
//
// usage:
//   let comp1 = Computation::new(domain, |coord| coord[0] as f64);
//   let comp2 = Computation::new(domain, |coord| coord[1] as f64);
//   let result = comp1.add(comp2);  // builds expression graph, no execution
// =============================================================================

use crate::domain::Domain;
use core::marker::PhantomData;

/// lazy computation graph.
/// wraps a function F that maps coordinates to values of type T.
/// purely functional - no side effects, no execution.
#[derive(Clone)]
pub struct Computation<T, const N: usize, F> {
    func: F,
    domain: Domain<N>,
    _phantom: PhantomData<T>,
}

impl<T, const N: usize, F> Computation<T, N, F> {
    /// creates a new computation from function and domain.
    pub fn new(func: F, domain: Domain<N>) -> Self {
        Self {
            func,
            domain,
            _phantom: PhantomData,
        }
    }

    /// returns the domain this computation is defined over.
    pub fn domain(&self) -> Domain<N> {
        self.domain
    }

    /// evaluates the computation at a single coordinate.
    pub fn eval(&self, coord: [i64; N]) -> T
    where
        F: Fn([i64; N]) -> T,
    {
        (self.func)(coord)
    }

    /// maps the result of this computation through a function.
    /// composition: (f ∘ g)(x) = f(g(x))
    pub fn map<U, G>(self, op: G) -> Computation<U, N, impl Fn([i64; N]) -> U>
    where
        F: Fn([i64; N]) -> T,
        G: Fn(T) -> U,
    {
        let func = self.func;
        let combined = move |coord| op(func(coord));
        Computation::new(combined, self.domain)
    }

    /// combines with another computation element-wise.
    pub fn zip<U, V, G, H>(
        self,
        other: Computation<U, N, G>,
        op: H,
    ) -> Computation<V, N, impl Fn([i64; N]) -> V>
    where
        F: Fn([i64; N]) -> T,
        G: Fn([i64; N]) -> U,
        H: Fn(T, U) -> V,
    {
        let f = self.func;
        let g = other.func;
        let combined = move |coord| op(f(coord), g(coord));
        let domain = self.domain.intersect(&other.domain);
        Computation::new(combined, domain)
    }

    /// restricts computation to a subdomain.
    pub fn at(self, subdomain: Domain<N>) -> Self
    where
        F: Fn([i64; N]) -> T,
    {
        let domain = self.domain.intersect(&subdomain);
        Self {
            func: self.func,
            domain,
            _phantom: PhantomData,
        }
    }

    /// transforms coordinates before evaluation.
    /// useful for boundary conditions, stencils, shifts.
    /// remap: f(x) -> f(op(x))
    pub fn remap<G>(self, coord_transform: G) -> Computation<T, N, impl Fn([i64; N]) -> T>
    where
        F: Fn([i64; N]) -> T,
        G: Fn([i64; N]) -> [i64; N],
    {
        let func = self.func;
        let combined = move |coord| func(coord_transform(coord));
        Computation::new(combined, self.domain)
    }

    /// scalar addition
    pub fn add_scalar(self, scalar: T) -> Computation<T, N, impl Fn([i64; N]) -> T>
    where
        F: Fn([i64; N]) -> T,
        T: std::ops::Add<Output = T> + Copy,
    {
        self.map(move |x| x + scalar)
    }

    /// scalar multiplication
    pub fn scale(self, scalar: T) -> Computation<T, N, impl Fn([i64; N]) -> T>
    where
        F: Fn([i64; N]) -> T,
        T: std::ops::Mul<Output = T> + Copy,
    {
        self.map(move |x| x * scalar)
    }

    /// add two computations element-wise
    pub fn add<G>(self, other: Computation<T, N, G>) -> Computation<T, N, impl Fn([i64; N]) -> T>
    where
        F: Fn([i64; N]) -> T,
        G: Fn([i64; N]) -> T,
        T: std::ops::Add<Output = T>,
    {
        self.zip(other, |a, b| a + b)
    }

    /// subtract two computations element-wise
    pub fn sub<G>(self, other: Computation<T, N, G>) -> Computation<T, N, impl Fn([i64; N]) -> T>
    where
        F: Fn([i64; N]) -> T,
        G: Fn([i64; N]) -> T,
        T: std::ops::Sub<Output = T>,
    {
        self.zip(other, |a, b| a - b)
    }

    /// multiply two computations element-wise
    pub fn mul<G>(self, other: Computation<T, N, G>) -> Computation<T, N, impl Fn([i64; N]) -> T>
    where
        F: Fn([i64; N]) -> T,
        G: Fn([i64; N]) -> T,
        T: std::ops::Mul<Output = T>,
    {
        self.zip(other, |a, b| a * b)
    }

    /// divide two computations element-wise
    pub fn div<G>(self, other: Computation<T, N, G>) -> Computation<T, N, impl Fn([i64; N]) -> T>
    where
        F: Fn([i64; N]) -> T,
        G: Fn([i64; N]) -> T,
        T: std::ops::Div<Output = T>,
    {
        self.zip(other, |a, b| a / b)
    }
}

/// factory function for creating computation from function
pub fn from_fn<T, const N: usize, F>(domain: Domain<N>, func: F) -> Computation<T, N, F>
where
    F: Fn([i64; N]) -> T,
{
    Computation::new(func, domain)
}

/// creates a constant computation (same value everywhere)
pub fn constant<T: Clone, const N: usize>(
    domain: Domain<N>,
    value: T,
) -> Computation<T, N, impl Fn([i64; N]) -> T> {
    Computation::new(move |_| value.clone(), domain)
}

/// creates identity computation (returns coordinates as f64 array)
pub fn identity<const N: usize>(
    domain: Domain<N>,
) -> Computation<[f64; N], N, impl Fn([i64; N]) -> [f64; N]> {
    Computation::new(
        |coord: [i64; N]| -> [f64; N] {
            let mut result = [0.0; N];
            for i in 0..N {
                result[i] = coord[i] as f64;
            }
            result
        },
        domain,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_computation() {
        let domain = Domain::from_shape([10]);
        let comp = from_fn(domain, |coord| coord[0] as f64);

        assert_eq!(comp.eval([0]), 0.0);
        assert_eq!(comp.eval([5]), 5.0);
    }

    #[test]
    fn test_map() {
        let domain = Domain::from_shape([10]);
        let comp = from_fn(domain, |coord| coord[0] as f64);
        let doubled = comp.map(|x| x * 2.0);

        assert_eq!(doubled.eval([3]), 6.0);
    }

    #[test]
    fn test_zip() {
        let domain = Domain::from_shape([10]);
        let comp1 = from_fn(domain, |coord| coord[0] as f64);
        let comp2 = from_fn(domain, |coord| coord[0] as f64 + 1.0);
        let sum = comp1.zip(comp2, |a, b| a + b);

        assert_eq!(sum.eval([5]), 11.0);
    }

    #[test]
    fn test_arithmetic() {
        let domain = Domain::from_shape([10]);
        let comp1 = from_fn(domain, |coord| coord[0] as f64);
        let comp2 = from_fn(domain, |_| 2.0);

        let sum = comp1.clone().add(comp2.clone());
        assert_eq!(sum.eval([3]), 5.0);

        let prod = comp1.clone().mul(comp2.clone());
        assert_eq!(prod.eval([3]), 6.0);
    }

    #[test]
    fn test_constant() {
        let domain = Domain::from_shape([10]);
        let comp = constant(domain, 42.0);

        assert_eq!(comp.eval([0]), 42.0);
        assert_eq!(comp.eval([5]), 42.0);
    }

    #[test]
    fn test_composition() {
        let domain = Domain::from_shape([10]);
        let x = from_fn(domain, |coord| coord[0] as f64);

        let expr = x.scale(2.0).add_scalar(5.0);

        assert_eq!(expr.eval([0]), 5.0);
        assert_eq!(expr.eval([3]), 11.0);
    }

    #[test]
    fn test_remap() {
        let domain = Domain::from_shape([10]);
        let comp = from_fn(domain, |coord| coord[0] as f64);

        let shifted = comp.remap(|coord| [coord[0] - 1]);

        assert_eq!(shifted.eval([5]), 4.0);
    }
}
