// =============================================================================
// domain.rs
//
// pure topology for multi-dimensional index spaces.
// represents half-open interval [start, end) with integer coordinates.
//
// design:
//   - const generic rank N (compile-time dimensionality)
//   - pure value type (copy, no allocation)
//   - iterator support for range-based loops
//   - geometric operations (intersection, slicing, containment)
//
// usage:
//   let domain = Domain::new([0, 0], [100, 200]);
//   for coord in domain.iter() {
//       // process coord
//   }
// =============================================================================

use core::ops::Range;

/// multi-dimensional index space representing [start, end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Domain<const N: usize> {
    /// starting coordinate (inclusive)
    pub start: [i64; N],
    /// ending coordinate (exclusive)
    pub end: [i64; N],
}

impl<const N: usize> Domain<N> {
    /// creates a new domain from start and end coordinates.
    pub const fn new(start: [i64; N], end: [i64; N]) -> Self {
        Self { start, end }
    }

    /// creates a domain from origin to given shape.
    pub const fn from_shape(shape: [i64; N]) -> Self {
        Self {
            start: [0; N],
            end: shape,
        }
    }

    /// returns the shape (size in each dimension).
    pub const fn shape(&self) -> [i64; N] {
        let mut result = [0; N];
        let mut ii = 0;
        while ii < N {
            result[ii] = self.end[ii] - self.start[ii];
            ii += 1;
        }
        result
    }

    /// returns the total number of elements.
    pub const fn size(&self) -> usize {
        let shape = self.shape();
        let mut result = 1usize;
        let mut ii = 0;
        while ii < N {
            result *= shape[ii] as usize;
            ii += 1;
        }
        result
    }

    /// returns true if domain is empty.
    pub const fn is_empty(&self) -> bool {
        let mut ii = 0;
        while ii < N {
            if self.end[ii] <= self.start[ii] {
                return true;
            }
            ii += 1;
        }
        false
    }

    /// checks if coordinate is within domain bounds.
    pub const fn contains(&self, coord: [i64; N]) -> bool {
        let mut ii = 0;
        while ii < N {
            if coord[ii] < self.start[ii] || coord[ii] >= self.end[ii] {
                return false;
            }
            ii += 1;
        }
        true
    }

    /// computes intersection with another domain.
    pub const fn intersect(&self, other: &Domain<N>) -> Domain<N> {
        let mut start = [0; N];
        let mut end = [0; N];
        let mut ii = 0;
        while ii < N {
            start[ii] = if self.start[ii] > other.start[ii] {
                self.start[ii]
            } else {
                other.start[ii]
            };
            end[ii] = if self.end[ii] < other.end[ii] {
                self.end[ii]
            } else {
                other.end[ii]
            };
            ii += 1;
        }
        Domain { start, end }
    }

    /// contracts domain by given width on all sides.
    pub const fn contract(&self, width: i64) -> Domain<N> {
        let mut start = [0; N];
        let mut end = [0; N];
        let mut ii = 0;
        while ii < N {
            start[ii] = self.start[ii] + width;
            end[ii] = self.end[ii] - width;
            ii += 1;
        }
        Domain { start, end }
    }

    /// expands domain by given width on all sides.
    pub const fn expand(&self, width: i64) -> Domain<N> {
        self.contract(-width)
    }

    /// slices domain along given axis.
    pub const fn slice(&self, axis: usize, range: Range<i64>) -> Domain<N> {
        let mut result = *self;
        result.start[axis] = if self.start[axis] > range.start {
            self.start[axis]
        } else {
            range.start
        };
        result.end[axis] = if self.end[axis] < range.end {
            self.end[axis]
        } else {
            range.end
        };
        result
    }

    /// converts linear index to coordinate.
    pub fn linear_to_coord(&self, mut linear: usize) -> [i64; N] {
        let mut coord = [0; N];
        let shape = self.shape();

        for ii in (0..N).rev() {
            let dim_size = shape[ii] as usize;
            coord[ii] = self.start[ii] + (linear % dim_size) as i64;
            linear /= dim_size;
        }

        coord
    }

    /// converts coordinate to linear index.
    pub fn coord_to_linear(&self, coord: [i64; N]) -> usize {
        let mut linear = 0usize;
        let shape = self.shape();

        for ii in 0..N {
            linear *= shape[ii] as usize;
            linear += (coord[ii] - self.start[ii]) as usize;
        }

        linear
    }

    /// returns an iterator over all coordinates in the domain.
    pub fn iter(&self) -> DomainIter<N> {
        DomainIter {
            domain: *self,
            current: self.start,
            done: self.is_empty(),
        }
    }
}

/// iterator over domain coordinates in row-major order.
#[derive(Debug, Clone)]
pub struct DomainIter<const N: usize> {
    domain: Domain<N>,
    current: [i64; N],
    done: bool,
}

impl<const N: usize> Iterator for DomainIter<N> {
    type Item = [i64; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current;

        // increment in row-major order
        for dim in (0..N).rev() {
            self.current[dim] += 1;
            if self.current[dim] < self.domain.end[dim] {
                return Some(result);
            }
            if dim > 0 {
                self.current[dim] = self.domain.start[dim];
            } else {
                self.done = true;
            }
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            let size = self.domain.size();
            (size, Some(size))
        }
    }
}

impl<const N: usize> ExactSizeIterator for DomainIter<N> {}

// convenience type aliases
pub type Domain1 = Domain<1>;
pub type Domain2 = Domain<2>;
pub type Domain3 = Domain<3>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_creation() {
        let domain = Domain::new([0, 0], [10, 20]);
        assert_eq!(domain.start, [0, 0]);
        assert_eq!(domain.end, [10, 20]);
    }

    #[test]
    fn test_domain_from_shape() {
        let domain = Domain::from_shape([10, 20]);
        assert_eq!(domain.start, [0, 0]);
        assert_eq!(domain.end, [10, 20]);
    }

    #[test]
    fn test_shape() {
        let domain = Domain::new([5, 10], [15, 30]);
        assert_eq!(domain.shape(), [10, 20]);
    }

    #[test]
    fn test_size() {
        let domain = Domain::new([0, 0], [10, 20]);
        assert_eq!(domain.size(), 200);
    }

    #[test]
    fn test_contains() {
        let domain = Domain::new([0, 0], [10, 20]);
        assert!(domain.contains([5, 10]));
        assert!(!domain.contains([10, 10])); // end is exclusive
        assert!(!domain.contains([-1, 5]));
    }

    #[test]
    fn test_intersect() {
        let d1 = Domain::new([0, 0], [10, 10]);
        let d2 = Domain::new([5, 5], [15, 15]);
        let intersection = d1.intersect(&d2);
        assert_eq!(intersection.start, [5, 5]);
        assert_eq!(intersection.end, [10, 10]);
    }

    #[test]
    fn test_contract() {
        let domain = Domain::new([0, 0], [10, 10]);
        let contracted = domain.contract(1);
        assert_eq!(contracted.start, [1, 1]);
        assert_eq!(contracted.end, [9, 9]);
    }

    #[test]
    fn test_expand() {
        let domain = Domain::new([1, 1], [9, 9]);
        let expanded = domain.expand(1);
        assert_eq!(expanded.start, [0, 0]);
        assert_eq!(expanded.end, [10, 10]);
    }

    #[test]
    fn test_slice() {
        let domain = Domain::new([0, 0, 0], [10, 20, 30]);
        let sliced = domain.slice(1, 5..15);
        assert_eq!(sliced.start, [0, 5, 0]);
        assert_eq!(sliced.end, [10, 15, 30]);
    }

    #[test]
    fn test_iterator() {
        let domain = Domain::new([0, 0], [2, 3]);
        let coords: Vec<_> = domain.iter().collect();

        assert_eq!(coords.len(), 6);
        assert_eq!(coords[0], [0, 0]);
        assert_eq!(coords[1], [0, 1]);
        assert_eq!(coords[2], [0, 2]);
        assert_eq!(coords[3], [1, 0]);
        assert_eq!(coords[4], [1, 1]);
        assert_eq!(coords[5], [1, 2]);
    }

    #[test]
    fn test_linear_conversion() {
        let domain = Domain::new([0, 0], [3, 4]);

        for linear in 0..domain.size() {
            let coord = domain.linear_to_coord(linear);
            let back = domain.coord_to_linear(coord);
            assert_eq!(linear, back);
        }
    }

    #[test]
    fn test_empty_domain() {
        let domain = Domain::new([5, 5], [5, 10]);
        assert!(domain.is_empty());
        assert_eq!(domain.iter().count(), 0);
    }
}
