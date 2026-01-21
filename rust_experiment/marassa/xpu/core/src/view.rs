// =============================================================================
// view.rs
//
// multi-dimensional non-owning views into device memory.
// provides strided access to buffer regions for zero-copy slicing.
//
// design:
//   - generic over element type T and rank N (1d, 2d, 3d)
//   - lifetime-bound to source buffer (rust ownership)
//   - explicit stride support for arbitrary memory layouts
//   - immutable and mutable variants
//
// usage:
//   let view = View::new(&buffer, shape, start, strides);
//   let val = view[[i, j, k]];  // multi-dimensional indexing
//   let slice = view.slice([0, 0], [10, 10]);  // create subview
// =============================================================================

use core::ops::{Index, IndexMut};

/// multi-dimensional shape/index/stride representation.
/// const generic N allows compile-time rank checking.
pub type Shape<const N: usize> = [usize; N];

/// immutable view into device memory with multi-dimensional indexing.
/// lifetime 'a ties the view to the underlying buffer.
#[derive(Debug)]
pub struct View<'a, T, const N: usize> {
    data: *const T,
    shape: Shape<N>,
    start: Shape<N>,
    strides: Shape<N>,
    _phantom: core::marker::PhantomData<&'a [T]>,
}

/// mutable view into device memory with multi-dimensional indexing.
#[derive(Debug)]
pub struct ViewMut<'a, T, const N: usize> {
    data: *mut T,
    shape: Shape<N>,
    start: Shape<N>,
    strides: Shape<N>,
    _phantom: core::marker::PhantomData<&'a mut [T]>,
}

// views are safe to send across threads if T is
unsafe impl<'a, T: Send, const N: usize> Send for View<'a, T, N> {}
unsafe impl<'a, T: Sync, const N: usize> Sync for View<'a, T, N> {}
unsafe impl<'a, T: Send, const N: usize> Send for ViewMut<'a, T, N> {}
unsafe impl<'a, T: Sync, const N: usize> Sync for ViewMut<'a, T, N> {}

impl<'a, T, const N: usize> View<'a, T, N> {
    /// creates a new immutable view.
    ///
    /// # safety
    /// caller must ensure:
    /// - data pointer is valid for the computed index range
    /// - data remains valid for lifetime 'a
    /// - shape, start, strides describe a valid region
    pub unsafe fn new(data: *const T, shape: Shape<N>, start: Shape<N>, strides: Shape<N>) -> Self {
        Self {
            data,
            shape,
            start,
            strides,
            _phantom: core::marker::PhantomData,
        }
    }

    /// creates a view from a slice with default layout (row-major).
    pub fn from_slice(slice: &'a [T], shape: Shape<N>) -> Option<Self> {
        let total: usize = shape.iter().product();
        if total != slice.len() {
            return None;
        }

        let strides = Self::compute_row_major_strides(&shape);
        let start = [0; N];

        Some(unsafe { Self::new(slice.as_ptr(), shape, start, strides) })
    }

    /// computes row-major strides for given shape.
    /// for 3d [nz, ny, nx]: strides = [ny*nx, nx, 1]
    fn compute_row_major_strides(shape: &Shape<N>) -> Shape<N> {
        let mut strides = [0; N];
        let mut stride = 1;

        for ii in (0..N).rev() {
            strides[ii] = stride;
            stride *= shape[ii];
        }

        strides
    }

    /// returns the shape of the view.
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// returns the strides of the view.
    pub fn strides(&self) -> Shape<N> {
        self.strides
    }

    /// returns the start offset.
    pub fn start(&self) -> Shape<N> {
        self.start
    }

    /// returns the total number of elements in the view.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// returns true if the view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// computes the linear index from multi-dimensional coordinates.
    fn compute_index(&self, coord: Shape<N>) -> usize {
        let mut index = 0;
        for ii in 0..N {
            index += (coord[ii] - self.start[ii]) * self.strides[ii];
        }
        index
    }

    /// returns a reference to the element at the given coordinate.
    ///
    /// # safety
    /// caller must ensure coordinate is within bounds.
    pub unsafe fn get_unchecked(&self, coord: Shape<N>) -> &T {
        let index = self.compute_index(coord);
        unsafe { &*self.data.add(index) }
    }

    /// returns a reference to the element at the given coordinate.
    /// returns none if out of bounds.
    pub fn get(&self, coord: Shape<N>) -> Option<&T> {
        if !self.in_bounds(coord) {
            return None;
        }
        Some(unsafe { self.get_unchecked(coord) })
    }

    /// checks if a coordinate is within the view bounds.
    pub fn in_bounds(&self, coord: Shape<N>) -> bool {
        for ii in 0..N {
            if coord[ii] < self.start[ii] || coord[ii] >= self.start[ii] + self.shape[ii] {
                return false;
            }
        }
        true
    }

    /// returns a raw pointer to the underlying data.
    pub fn as_ptr(&self) -> *const T {
        self.data
    }
}

impl<'a, T, const N: usize> ViewMut<'a, T, N> {
    /// creates a new mutable view.
    ///
    /// # safety
    /// caller must ensure:
    /// - data pointer is valid and uniquely accessible
    /// - data remains valid for lifetime 'a
    /// - shape, start, strides describe a valid region
    pub unsafe fn new(data: *mut T, shape: Shape<N>, start: Shape<N>, strides: Shape<N>) -> Self {
        Self {
            data,
            shape,
            start,
            strides,
            _phantom: core::marker::PhantomData,
        }
    }

    /// creates a mutable view from a slice with default layout (row-major).
    pub fn from_slice_mut(slice: &'a mut [T], shape: Shape<N>) -> Option<Self> {
        let total: usize = shape.iter().product();
        if total != slice.len() {
            return None;
        }

        let strides = Self::compute_row_major_strides(&shape);
        let start = [0; N];

        Some(unsafe { Self::new(slice.as_mut_ptr(), shape, start, strides) })
    }

    /// computes row-major strides for given shape.
    fn compute_row_major_strides(shape: &Shape<N>) -> Shape<N> {
        let mut strides = [0; N];
        let mut stride = 1;

        for ii in (0..N).rev() {
            strides[ii] = stride;
            stride *= shape[ii];
        }

        strides
    }

    /// returns the shape of the view.
    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    /// returns the strides of the view.
    pub fn strides(&self) -> Shape<N> {
        self.strides
    }

    /// returns the start offset.
    pub fn start(&self) -> Shape<N> {
        self.start
    }

    /// returns the total number of elements in the view.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// returns true if the view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// computes the linear index from multi-dimensional coordinates.
    fn compute_index(&self, coord: Shape<N>) -> usize {
        let mut index = 0;
        for ii in 0..N {
            index += (coord[ii] - self.start[ii]) * self.strides[ii];
        }
        index
    }

    /// returns a reference to the element at the given coordinate.
    ///
    /// # safety
    /// caller must ensure coordinate is within bounds.
    pub unsafe fn get_unchecked(&self, coord: Shape<N>) -> &T {
        let index = self.compute_index(coord);
        unsafe { &*self.data.add(index) }
    }

    /// returns a mutable reference to the element at the given coordinate.
    ///
    /// # safety
    /// caller must ensure coordinate is within bounds.
    pub unsafe fn get_unchecked_mut(&mut self, coord: Shape<N>) -> &mut T {
        let index = self.compute_index(coord);
        unsafe { &mut *self.data.add(index) }
    }

    /// returns a reference to the element at the given coordinate.
    /// returns none if out of bounds.
    pub fn get(&self, coord: Shape<N>) -> Option<&T> {
        if !self.in_bounds(coord) {
            return None;
        }
        Some(unsafe { self.get_unchecked(coord) })
    }

    /// returns a mutable reference to the element at the given coordinate.
    /// returns none if out of bounds.
    pub fn get_mut(&mut self, coord: Shape<N>) -> Option<&mut T> {
        if !self.in_bounds(coord) {
            return None;
        }
        Some(unsafe { self.get_unchecked_mut(coord) })
    }

    /// checks if a coordinate is within the view bounds.
    pub fn in_bounds(&self, coord: Shape<N>) -> bool {
        for ii in 0..N {
            if coord[ii] < self.start[ii] || coord[ii] >= self.start[ii] + self.shape[ii] {
                return false;
            }
        }
        true
    }

    /// creates a subview (slice) of this view.

    /// returns a raw pointer to the underlying data.
    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    /// returns a raw mutable pointer to the underlying data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data
    }

    /// borrows the view as immutable.
    pub fn as_view(&self) -> View<'_, T, N> {
        unsafe { View::new(self.data, self.shape, self.start, self.strides) }
    }
}

// implement indexing for convenient access
impl<'a, T, const N: usize> Index<Shape<N>> for View<'a, T, N> {
    type Output = T;

    fn index(&self, coord: Shape<N>) -> &Self::Output {
        self.get(coord).expect("index out of bounds")
    }
}

impl<'a, T, const N: usize> Index<Shape<N>> for ViewMut<'a, T, N> {
    type Output = T;

    fn index(&self, coord: Shape<N>) -> &Self::Output {
        self.get(coord).expect("index out of bounds")
    }
}

impl<'a, T, const N: usize> IndexMut<Shape<N>> for ViewMut<'a, T, N> {
    fn index_mut(&mut self, coord: Shape<N>) -> &mut Self::Output {
        self.get_mut(coord).expect("index out of bounds")
    }
}

// convenience type aliases for common dimensions
pub type View1<'a, T> = View<'a, T, 1>;
pub type View2<'a, T> = View<'a, T, 2>;
pub type View3<'a, T> = View<'a, T, 3>;

pub type ViewMut1<'a, T> = ViewMut<'a, T, 1>;
pub type ViewMut2<'a, T> = ViewMut<'a, T, 2>;
pub type ViewMut3<'a, T> = ViewMut<'a, T, 3>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view1_from_slice() {
        let data = vec![1, 2, 3, 4, 5];
        let view = View1::from_slice(&data, [5]).unwrap();

        assert_eq!(view.len(), 5);
        assert_eq!(view[[0]], 1);
        assert_eq!(view[[4]], 5);
    }

    #[test]
    fn test_view2_row_major() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let view = View2::from_slice(&data, [2, 3]).unwrap();

        assert_eq!(view.shape(), [2, 3]);
        assert_eq!(view.strides(), [3, 1]); // row-major: [ny, 1]

        // [0,0]=1, [0,1]=2, [0,2]=3
        // [1,0]=4, [1,1]=5, [1,2]=6
        assert_eq!(view[[0, 0]], 1);
        assert_eq!(view[[0, 2]], 3);
        assert_eq!(view[[1, 0]], 4);
        assert_eq!(view[[1, 2]], 6);
    }

    #[test]
    fn test_view3_row_major() {
        // 2x2x2 cube
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let view = View3::from_slice(&data, [2, 2, 2]).unwrap();

        assert_eq!(view.shape(), [2, 2, 2]);
        assert_eq!(view.strides(), [4, 2, 1]); // [ny*nx, nx, 1]

        assert_eq!(view[[0, 0, 0]], 1);
        assert_eq!(view[[0, 0, 1]], 2);
        assert_eq!(view[[0, 1, 0]], 3);
        assert_eq!(view[[1, 1, 1]], 8);
    }

    #[test]
    fn test_view_mut() {
        let mut data = vec![0; 6];
        let mut view = ViewMut2::from_slice_mut(&mut data, [2, 3]).unwrap();

        view[[0, 0]] = 1;
        view[[1, 2]] = 6;

        assert_eq!(view[[0, 0]], 1);
        assert_eq!(view[[1, 2]], 6);
        assert_eq!(data[0], 1);
        assert_eq!(data[5], 6);
    }

    #[test]
    fn test_view_bounds_check() {
        let data = vec![1, 2, 3, 4];
        let view = View2::from_slice(&data, [2, 2]).unwrap();

        assert!(view.in_bounds([0, 0]));
        assert!(view.in_bounds([1, 1]));
        assert!(!view.in_bounds([2, 0]));
        assert!(!view.in_bounds([0, 2]));
    }

    #[test]
    fn test_view_slice() {
        // todo: slicing needs better api design
        // for now, just test that basic view indexing works
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = View2::from_slice(&data, [3, 3]).unwrap();

        // verify we can access all elements
        assert_eq!(view[[0, 0]], 1);
        assert_eq!(view[[1, 1]], 5);
        assert_eq!(view[[2, 2]], 9);
    }

    #[test]
    fn test_view_get_none() {
        let data = vec![1, 2, 3, 4];
        let view = View2::from_slice(&data, [2, 2]).unwrap();

        assert!(view.get([0, 0]).is_some());
        assert!(view.get([2, 0]).is_none());
        assert!(view.get([0, 2]).is_none());
    }

    #[test]
    fn test_strided_access() {
        // manual stride setup for column-major
        let data = vec![1, 2, 3, 4, 5, 6];
        let view = unsafe { View2::new(data.as_ptr(), [2, 3], [0, 0], [1, 2]) };

        // column-major: strides [1, 2] means columns are contiguous
        assert_eq!(view[[0, 0]], 1);
        assert_eq!(view[[1, 0]], 2);
        assert_eq!(view[[0, 1]], 3);
        assert_eq!(view[[1, 1]], 4);
    }
}
