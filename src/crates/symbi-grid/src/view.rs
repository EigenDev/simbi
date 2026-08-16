// =============================================================================
// view.rs
//
// non-owning, strided, multi-dimensional view into contiguous memory.
// the functor interface: view.at(coord) -> &T.
//
// a view is Copy — a borrowed pointer + strides. the field owns the
// storage and guarantees the pointer is valid for the view's lifetime. a view acts
// as a function from coordinates to values: view.at(coord) -> &T.
//
// usage:
//   let view = field.view();
//   let val = view.at([2, 3]);
// =============================================================================

use symbi_algebra::Domain;

/// non-owning, strided view into contiguous memory.
/// `T` is the element type. `D` is the dimensionality.
///
/// Copy — views are lightweight handles.
#[derive(Clone, Copy)]
pub struct View<T, const D: usize> {
    ptr: *const T,
    start: [isize; D],
    strides: [usize; D],
    len: usize,
}

unsafe impl<T: Send, const D: usize> Send for View<T, D> {}
unsafe impl<T: Sync, const D: usize> Sync for View<T, D> {}

impl<T, const D: usize> View<T, D> {
    /// create a view from raw parts.
    ///
    /// # safety
    /// `ptr` must point to a valid allocation of at least `len` elements.
    /// the strides must be correct for the domain's shape.
    pub unsafe fn from_raw(
        ptr: *const T,
        start: [isize; D],
        strides: [usize; D],
        len: usize,
    ) -> Self {
        View {
            ptr,
            start,
            strides,
            len,
        }
    }

    /// create a view from a domain and a pointer to the base of the allocation.
    /// computes row-major strides from the domain's shape.
    pub fn from_domain(ptr: *const T, domain: &Domain<D>) -> Self {
        let shape = domain.shape();
        let strides = Self::row_major_strides(&shape);
        let start = std::array::from_fn(|ax| domain.spaces[ax].lo);
        let len = domain.volume();
        View {
            ptr,
            start,
            strides,
            len,
        }
    }

    /// read the value at the given coordinate.
    #[inline]
    pub fn at(&self, coord: [isize; D]) -> &T {
        let idx = self.flat_index(coord);
        debug_assert!(
            idx < self.len,
            "view: index {} out of bounds (len {})",
            idx,
            self.len
        );
        unsafe { &*self.ptr.add(idx) }
    }

    /// flat index from coordinate.
    #[inline]
    fn flat_index(&self, coord: [isize; D]) -> usize {
        symbi_algebra::flat_offset(coord, self.start, self.strides)
    }

    /// strides from shape under the **physical-x-fastest convention**:
    /// `strides[0] = 1`, `strides[ax] = strides[ax-1] * shape[ax-1]`. axis 0
    /// is the fastest-varying in memory, axis `D-1` the slowest. matches
    /// `Domain::compute_strides` — see the comment there for why this
    /// convention (vs the prior axis-`D-1`-fastest) matters for GPU coalescing.
    fn row_major_strides(shape: &[usize; D]) -> [usize; D] {
        // the formula lives in one place: `symbi_algebra::strides_from_extent`.
        let mut strides = [0usize; D];
        symbi_algebra::strides_from_extent(shape, &mut strides);
        strides
    }

    /// raw pointer to the base of the allocation.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }
}

/// mutable view — same layout, but allows writes.
#[derive(Clone, Copy)]
pub struct ViewMut<T, const D: usize> {
    ptr: *mut T,
    start: [isize; D],
    strides: [usize; D],
    len: usize,
}

unsafe impl<T: Send, const D: usize> Send for ViewMut<T, D> {}
unsafe impl<T: Sync, const D: usize> Sync for ViewMut<T, D> {}

impl<T, const D: usize> ViewMut<T, D> {
    pub fn from_domain(ptr: *mut T, domain: &Domain<D>) -> Self {
        let shape = domain.shape();
        let strides = View::<T, D>::row_major_strides(&shape);
        let start = std::array::from_fn(|ax| domain.spaces[ax].lo);
        let len = domain.volume();
        ViewMut {
            ptr,
            start,
            strides,
            len,
        }
    }

    #[inline]
    pub fn at(&self, coord: [isize; D]) -> &T {
        let idx = self.flat_index(coord);
        debug_assert!(idx < self.len);
        unsafe { &*self.ptr.add(idx) }
    }

    #[inline]
    pub fn at_mut(&mut self, coord: [isize; D]) -> &mut T {
        let idx = self.flat_index(coord);
        debug_assert!(idx < self.len);
        unsafe { &mut *self.ptr.add(idx) }
    }

    #[inline]
    pub fn set(&self, coord: [isize; D], val: T) {
        let idx = self.flat_index(coord);
        debug_assert!(idx < self.len);
        unsafe {
            *self.ptr.add(idx) = val;
        }
    }

    #[inline]
    fn flat_index(&self, coord: [isize; D]) -> usize {
        symbi_algebra::flat_offset(coord, self.start, self.strides)
    }

    pub fn as_view(&self) -> View<T, D> {
        View {
            ptr: self.ptr as *const T,
            start: self.start,
            strides: self.strides,
            len: self.len,
        }
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::{Domain, Space};

    #[test]
    fn view_1d() {
        let data: Vec<f64> = (0..10).map(|ii| ii as f64).collect();
        let dom = Domain::new([Space {
            name: "x",
            lo: 0,
            hi: 10,
        }]);
        let view = View::from_domain(data.as_ptr(), &dom);
        assert_eq!(*view.at([0]), 0.0);
        assert_eq!(*view.at([5]), 5.0);
        assert_eq!(*view.at([9]), 9.0);
    }

    #[test]
    fn view_2d() {
        // 3x4 axis-0-fastest: strides [1, 3], data[i + j*3]
        let data: Vec<f64> = (0..12).map(|ii| ii as f64).collect();
        let dom = Domain::new([
            Space {
                name: "i",
                lo: 0,
                hi: 3,
            },
            Space {
                name: "j",
                lo: 0,
                hi: 4,
            },
        ]);
        let view = View::from_domain(data.as_ptr(), &dom);
        assert_eq!(*view.at([0, 0]), 0.0);
        assert_eq!(*view.at([1, 0]), 1.0); // step axis 0 -> stride 1
        assert_eq!(*view.at([0, 3]), 9.0); // step axis 1 -> 3*3 = 9
        assert_eq!(*view.at([2, 3]), 11.0);
    }

    #[test]
    fn view_2d_with_offset() {
        // 4x4 axis-0-fastest: strides [1, 4], data[(i+2) + (j+2)*4]
        let data: Vec<f64> = (0..16).map(|ii| ii as f64).collect();
        let dom = Domain::new([
            Space {
                name: "i",
                lo: -2,
                hi: 2,
            },
            Space {
                name: "j",
                lo: -2,
                hi: 2,
            },
        ]);
        let view = View::from_domain(data.as_ptr(), &dom);
        assert_eq!(*view.at([-2, -2]), 0.0);
        assert_eq!(*view.at([-2, 1]), 12.0); // (0) + (3)*4 = 12
        assert_eq!(*view.at([1, 1]), 15.0); // (3) + (3)*4 = 15
    }

    #[test]
    fn view_mut_write_read() {
        let mut data = vec![0.0f64; 10];
        let dom = Domain::new([Space {
            name: "x",
            lo: 0,
            hi: 10,
        }]);
        let view = ViewMut::from_domain(data.as_mut_ptr(), &dom);
        for ii in 0..10 {
            view.set([ii as isize], ii as f64 * ii as f64);
        }
        assert_eq!(*view.at([3]), 9.0);
        assert_eq!(*view.at([7]), 49.0);
    }

    #[test]
    fn view_3d() {
        // 2x3x4 axis-0-fastest: strides [1, 2, 6], data[i + j*2 + k*6]
        let data: Vec<f64> = (0..24).map(|ii| ii as f64).collect();
        let dom = Domain::new([
            Space {
                name: "i",
                lo: 0,
                hi: 2,
            },
            Space {
                name: "j",
                lo: 0,
                hi: 3,
            },
            Space {
                name: "k",
                lo: 0,
                hi: 4,
            },
        ]);
        let view = View::from_domain(data.as_ptr(), &dom);
        assert_eq!(*view.at([0, 0, 0]), 0.0);
        assert_eq!(*view.at([1, 0, 0]), 1.0); // step axis 0
        assert_eq!(*view.at([0, 1, 0]), 2.0); // step axis 1
        assert_eq!(*view.at([0, 0, 3]), 18.0); // step axis 2: 3*6
        assert_eq!(*view.at([0, 2, 3]), 22.0); // 2*2 + 3*6
        assert_eq!(*view.at([1, 2, 3]), 23.0); // 1 + 2*2 + 3*6
    }

    #[test]
    fn view_3d_with_ghost_zones() {
        // 3D domain with ghost zones: [-2, 6) x [-2, 6) x [-2, 6)
        // shape 8x8x8 = 512 elements
        let ng = 2_isize;
        let nn = 4_isize;
        let lo = -ng;
        let hi = nn + ng;
        let sz = (hi - lo) as usize; // 8
        let data: Vec<f64> = (0..(sz * sz * sz)).map(|ii| ii as f64).collect();
        let dom = Domain::new([
            Space { name: "i", lo, hi },
            Space { name: "j", lo, hi },
            Space { name: "k", lo, hi },
        ]);
        let view = View::from_domain(data.as_ptr(), &dom);

        // corner: [-2, -2, -2] -> flat index 0
        assert_eq!(*view.at([-2, -2, -2]), 0.0);
        // first interior cell: [0, 0, 0] -> flat (0-(-2))*64 + (0-(-2))*8 + (0-(-2)) = 2*64 + 2*8 + 2 = 146
        assert_eq!(*view.at([0, 0, 0]), 146.0);
        // last ghost cell: [5, 5, 5] -> (5-(-2))*64 + (5-(-2))*8 + (5-(-2)) = 7*64 + 7*8 + 7 = 511
        assert_eq!(*view.at([5, 5, 5]), 511.0);
    }

    #[test]
    fn view_3d_stencil_access() {
        // verify that stencil-like access patterns work with coordinate offsets
        let dom = Domain::new([
            Space {
                name: "i",
                lo: 0,
                hi: 10,
            },
            Space {
                name: "j",
                lo: 0,
                hi: 10,
            },
            Space {
                name: "k",
                lo: 0,
                hi: 10,
            },
        ]);
        let data: Vec<f64> = (0..1000).map(|ii| ii as f64).collect();
        let view = View::from_domain(data.as_ptr(), &dom);

        // the 6-point stencil about the center cell [5, 5, 5]
        let xm = *view.at([4, 5, 5]);
        let xp = *view.at([6, 5, 5]);
        let ym = *view.at([5, 4, 5]);
        let yp = *view.at([5, 6, 5]);
        let zm = *view.at([5, 5, 4]);
        let zp = *view.at([5, 5, 6]);

        // axis-0-fastest: strides [1, 10, 100]. neighbor differences scale
        // by 2 x the per-axis stride.
        assert_eq!(xp - xm, 2.0); // 2 * stride_i = 2
        assert_eq!(yp - ym, 20.0); // 2 * stride_j = 20
        assert_eq!(zp - zm, 200.0); // 2 * stride_k = 200
    }
}
