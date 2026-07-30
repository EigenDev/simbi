// =============================================================================
// field.rs
//
// the field: owns memory, bound to a domain. the simulation's data container.
//
// clone = shallow (shared handle).
// reads/writes go through views (view/view_mut) or coord-indexed at/set,
// which the #[symbi::kernel(coord)] macro lowers to.
//
// usage:
//   let f = Field::<f64, 2>::zeros(&domain)?;
//   let v = f.view();
//   let x = *v.at([i, j]);   // read
//   f.set([i, j], x);        // write (interior mutability)
// =============================================================================

use crate::centering::{Cell, Centering};
use crate::view::{View, ViewMut};
use symbi_algebra::Domain;
use symbi_xpu::{DefaultMemory, MemoryBlock, MemorySpace, SharedHandle};

// =============================================================================
// field
// =============================================================================

/// memory locality indicator for field dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Locality {
    Cpu,
    Gpu,
}

impl Locality {
    pub fn is_gpu(self) -> bool {
        self == Locality::Gpu
    }
    pub fn is_cpu(self) -> bool {
        self == Locality::Cpu
    }
}

/// memory-backed field over a domain. clone = shallow.
///
/// the `C` parameter tags the field's centering on a staggered grid
/// (cell / face-d / edge-d). default is `Cell`, so 3-arg uses
/// `Field<T, D, M>` keep working unchanged. the centering is phantom —
/// zero runtime cost — and exists for compile-time type safety in
/// staggered-grid morphisms (Curl, FaceAvg, Diff<AX>).
pub struct Field<T, const D: usize, M: MemorySpace = DefaultMemory, C: Centering = Cell> {
    storage: SharedHandle<MemoryBlock<M>>,
    pub domain: Domain<D>,
    pub locality: Locality,
    _element: std::marker::PhantomData<T>,
    _centering: std::marker::PhantomData<C>,
}

// clone = shallow (shared handle)
impl<T, const D: usize, M: MemorySpace, C: Centering> Clone for Field<T, D, M, C> {
    fn clone(&self) -> Self {
        Field {
            storage: self.storage.clone(),
            domain: self.domain.clone(),
            locality: self.locality,
            _element: std::marker::PhantomData,
            _centering: std::marker::PhantomData,
        }
    }
}

impl<T: Copy + Default + 'static, const D: usize, M: MemorySpace, C: Centering> Field<T, D, M, C> {
    /// allocate a field over the given domain, zero-initialized.
    pub fn zeros(domain: &Domain<D>) -> symbi_xpu::Result<Self> {
        let block = MemoryBlock::<M>::for_elements::<T>(domain.volume())?;
        let locality = if M::IS_DEVICE_ACCESSIBLE {
            Locality::Gpu
        } else {
            Locality::Cpu
        };
        Ok(Field {
            storage: SharedHandle::new(block),
            domain: domain.clone(),
            locality,
            _element: std::marker::PhantomData,
            _centering: std::marker::PhantomData,
        })
    }

    /// the domain this field is defined over.
    pub fn domain(&self) -> &Domain<D> {
        &self.domain
    }

    // ---- views ----

    /// immutable view over the entire domain.
    pub fn view(&self) -> View<T, D> {
        View::from_domain(self.storage.get().as_ptr::<T>(), &self.domain)
    }

    /// mutable view over the entire domain.
    pub fn view_mut(&self) -> ViewMut<T, D> {
        // safety: shared handle + unified memory. the caller is responsible
        // for not aliasing mutable views; writers run sequentially between
        // sync points.
        let ptr = self.storage.get().as_ptr::<T>() as *mut T;
        ViewMut::from_domain(ptr, &self.domain)
    }

    // ---- coord-indexed access (used by #[symbi::kernel(coord)] macro) ----
    // the macro rewrites field[coord] -> field.at(coord) for reads,
    // and field[coord] = val -> field.set(coord, val) for writes.
    // uses interior mutability: &Field can both read and write.

    /// flat index from coordinate. delegates to `Domain::flat_index` so
    /// `Field` and `View` agree on the storage convention (axis 0 fastest).
    /// this must not hand-roll an independent formula: the kernel dispatch
    /// writes via `View` using `Domain`'s strides, while host
    /// `Field::at`/`Field::set` read via this method. any divergence would
    /// land writes and reads at different addresses and silently corrupt
    /// code that mixes kernel writes with `Field::at` reads.
    #[inline]
    fn flat_index(&self, coord: [isize; D]) -> usize {
        self.domain.flat_index(coord)
    }

    /// read a value at a coordinate.
    #[inline]
    pub fn at(&self, coord: [isize; D]) -> &T {
        let idx = self.flat_index(coord);
        unsafe { &*self.storage.get().as_ptr::<T>().add(idx) }
    }

    /// write a value at a coordinate (interior mutability).
    #[inline]
    pub fn set(&self, coord: [isize; D], val: T) {
        let idx = self.flat_index(coord);
        unsafe {
            *(self.storage.get().as_ptr::<T>() as *mut T).add(idx) = val;
        }
    }

    /// accumulate `delta` into the value at `coord`, with a RELEASE-ACTIVE bounds check.
    /// the single audited write path for pointwise per-cell ops (`op` computes `delta` purely,
    /// this performs the only field mutation). unlike `set`/`ViewMut::set` (which guard with
    /// `debug_assert!`, compiled out in release) the assert here is ALWAYS on, so a bad index is a
    /// LOUD PANIC; a bad index would otherwise silently corrupt the heap. sound to call concurrently from many threads
    /// PROVIDED each `coord` is written by exactly one thread (disjoint cells) — `T` is a Copy
    /// scalar so the read-modify-write touches only this cell.
    #[inline]
    pub fn add_assign_checked(&self, coord: [isize; D], delta: T)
    where
        T: Copy + core::ops::Add<Output = T>,
    {
        let idx = self.flat_index(coord);
        let volume = self.domain.volume();
        assert!(
            idx < volume,
            "Field::add_assign_checked: flat index {idx} out of bounds (volume {volume}) at coord {coord:?}",
        );
        unsafe {
            let p = (self.storage.get().as_ptr::<T>() as *mut T).add(idx);
            *p = *p + delta;
        }
    }

    // ---- direct access for initialization ----

    /// raw mutable pointer to the underlying storage.
    /// use only for initialization (e.g., filling initial data).
    pub fn as_mut_ptr(&self) -> *mut T {
        self.storage.get().as_ptr::<T>() as *mut T
    }

    /// raw pointer to the underlying storage.
    pub fn as_ptr(&self) -> *const T {
        self.storage.get().as_ptr::<T>()
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::{Domain, Space};
    use symbi_xpu::HostMemory;

    fn dom_1d(n: isize) -> Domain<1> {
        Domain::new([Space {
            name: "x",
            lo: 0,
            hi: n,
        }])
    }

    #[test]
    fn field_zeros() {
        let dom = dom_1d(10);
        let field = Field::<f64, 1, HostMemory>::zeros(&dom).unwrap();
        let view = field.view();
        for ii in 0..10 {
            assert_eq!(*view.at([ii]), 0.0);
        }
    }

    #[test]
    fn field_clone_is_shallow() {
        let dom = dom_1d(5);
        let f1 = Field::<f64, 1, HostMemory>::zeros(&dom).unwrap();
        let f2 = f1.clone();
        // writing through f1 should be visible through f2
        f1.view_mut().set([2], 42.0);
        assert_eq!(*f2.view().at([2]), 42.0);
    }
}
