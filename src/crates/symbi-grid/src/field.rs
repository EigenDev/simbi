// =============================================================================
// field.rs
//
// the field: owns memory, bound to a domain. the simulation's data container.
//
// clone = shallow (shared handle).
// on a host-dereferenceable space (`MemorySpace`: host or managed memory) reads and
// writes go through views (view/view_mut) or coord-indexed at/set, which the
// #[symbi::kernel(coord)] macro lowers to. on every space (`MemoryStorage`, device-
// local memory included) host code moves values only through the synchronized
// transfers download/upload/read_one/write_one; kernels bind the raw pointers.
//
// usage:
//   let f = Field::<f64, 2>::zeros(&domain)?;
//   let v = f.view();
//   let x = *v.at([i, j]);   // read
//   f.set([i, j], x);        // write (interior mutability)
//   let host: Vec<f64> = f.download()?;   // any space
// =============================================================================

use crate::centering::{Cell, Centering};
use crate::view::{View, ViewMut};
use symbi_algebra::Domain;
use symbi_xpu::{DefaultMemory, MemoryBlock, MemorySpace, MemoryStorage, SharedHandle};

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
/// staggered-grid morphisms (Curl, FaceAvg, Diff<ax>).
pub struct Field<T, const D: usize, M: MemoryStorage = DefaultMemory, C: Centering = Cell> {
    storage: SharedHandle<MemoryBlock<M>>,
    pub domain: Domain<D>,
    pub locality: Locality,
    _element: std::marker::PhantomData<T>,
    _centering: std::marker::PhantomData<C>,
}

// clone = shallow (shared handle)
impl<T, const D: usize, M: MemoryStorage, C: Centering> Clone for Field<T, D, M, C> {
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

impl<T: Copy + Default + 'static, const D: usize, M: MemoryStorage, C: Centering> Field<T, D, M, C> {
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

    // ---- raw storage, for kernel binding ----

    /// raw mutable pointer to the underlying storage, for binding a kernel argument. host code
    /// dereferences it only on a `MemorySpace`.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.storage.get().as_ptr::<T>() as *mut T
    }

    /// raw pointer to the underlying storage, for binding a kernel argument.
    pub fn as_ptr(&self) -> *const T {
        self.storage.get().as_ptr::<T>()
    }

    // ---- synchronized host transfers, every space ----

    /// the whole allocation (ghosts included, storage order) copied to the host, after every
    /// queued kernel writing it has finished
    pub fn download(&self) -> symbi_xpu::Result<Vec<T>> {
        let mut out = vec![T::default(); self.domain.volume()];
        unsafe {
            M::copy_to_host(out.as_mut_ptr() as *mut u8, self.as_ptr() as *const u8, out.len() * std::mem::size_of::<T>())?;
        }
        Ok(out)
    }

    /// overwrite the whole allocation (ghosts included, storage order) from the host, ordered
    /// after every queued kernel and visible to every later one
    pub fn upload(&self, values: &[T]) -> symbi_xpu::Result<()> {
        assert_eq!(
            values.len(),
            self.domain.volume(),
            "Field::upload: {} values for a field of {} cells",
            values.len(),
            self.domain.volume()
        );
        unsafe { M::copy_from_host(self.as_mut_ptr() as *mut u8, values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
    }

    /// one value copied to the host, synchronized like `download`
    pub fn read_one(&self, coord: [isize; D]) -> symbi_xpu::Result<T> {
        let idx = self.checked_index(coord);
        let mut out = T::default();
        unsafe {
            M::copy_to_host(&mut out as *mut T as *mut u8, self.as_ptr().add(idx) as *const u8, std::mem::size_of::<T>())?;
        }
        Ok(out)
    }

    /// one value written from the host, synchronized like `upload`
    pub fn write_one(&self, coord: [isize; D], value: T) -> symbi_xpu::Result<()> {
        let idx = self.checked_index(coord);
        unsafe {
            M::copy_from_host(self.as_mut_ptr().add(idx) as *mut u8, &value as *const T as *const u8, std::mem::size_of::<T>())
        }
    }

    /// this field re-typed as host-dereferenceable storage: `Some` on host and managed memory,
    /// `None` on device-local storage
    pub fn host(&self) -> Option<&Field<T, D, M::HostAccess, C>> {
        if symbi_xpu::host_alias_is_identity::<M>() {
            // the host alias is this very type, so the cast is an identity
            Some(unsafe { &*(self as *const Self as *const Field<T, D, M::HostAccess, C>) })
        } else {
            None
        }
    }

    fn checked_index(&self, coord: [isize; D]) -> usize {
        let idx = self.domain.flat_index(coord);
        assert!(idx < self.domain.volume(), "Field: coord {coord:?} lies outside the allocation");
        idx
    }
}

/// host access to a field whose storage host code may dereference. a field in storage outside
/// `MemorySpace` (device-local memory) has none of these methods, so host code that reads it
/// fails to compile:
///
/// ```compile_fail,E0599
/// use symbi_algebra::{Domain, Space};
/// use symbi_grid::Field;
/// use symbi_xpu::HostOpaque;
/// let dom = Domain::new([Space { name: "x", lo: 0, hi: 4 }]);
/// let f = Field::<f64, 1, HostOpaque>::zeros(&dom).unwrap();
/// let _ = f.view(); // HostOpaque is not a MemorySpace
/// ```
///
/// the same field passes through the synchronized transfers:
///
/// ```
/// use symbi_algebra::{Domain, Space};
/// use symbi_grid::Field;
/// use symbi_xpu::HostOpaque;
/// let dom = Domain::new([Space { name: "x", lo: 0, hi: 4 }]);
/// let f = Field::<f64, 1, HostOpaque>::zeros(&dom).unwrap();
/// f.upload(&[1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(f.read_one([2]).unwrap(), 3.0);
/// ```
impl<T: Copy + Default + 'static, const D: usize, M: MemorySpace, C: Centering> Field<T, D, M, C> {
    // ---- views ----

    /// immutable view over the entire domain.
    pub fn view(&self) -> View<T, D> {
        // safety: a `MemorySpace` allocation is host-dereferenceable and spans the domain
        unsafe { View::from_domain(self.storage.get().as_ptr::<T>(), &self.domain) }
    }

    /// mutable view over the entire domain.
    pub fn view_mut(&self) -> ViewMut<T, D> {
        // safety: shared handle + unified memory. the caller must keep
        // mutable views disjoint; writers run sequentially between
        // sync points.
        let ptr = self.storage.get().as_ptr::<T>() as *mut T;
        unsafe { ViewMut::from_domain(ptr, &self.domain) }
    }

    // ---- coord-indexed access (used by #[symbi::kernel(coord)] macro) ----
    // the macro rewrites field[coord] -> field.at(coord) for reads,
    // and field[coord] = val -> field.set(coord, val) for writes.
    // uses interior mutability: &Field can both read and write.

    /// flat index from coordinate. delegates to `Domain::flat_index` so
    /// `Field` and `View` agree on the storage convention (axis 0 fastest)
    /// through one shared formula: the kernel dispatch writes via `View`
    /// using `Domain`'s strides, while host `Field::at`/`Field::set` read
    /// via this method. any divergence would land writes and reads at
    /// different addresses and silently corrupt code that mixes kernel
    /// writes with `Field::at` reads.
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

    /// accumulate `delta` into the value at `coord`, with a release-active bounds check.
    /// the single audited write path for pointwise per-cell ops (`op` computes `delta` purely,
    /// this performs the only field mutation). unlike `set`/`ViewMut::set` (which guard with
    /// `debug_assert!`, compiled out in release) the assert here stays compiled in for every
    /// build, so a bad index panics loudly; leaving it unchecked would corrupt the heap silently.
    /// sound to call concurrently from many threads provided each `coord` is written by one thread
    /// (disjoint cells) — `T` is a Copy scalar so the read-modify-write touches only this cell.
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
    fn opaque_storage_round_trips_through_the_transfers() {
        use symbi_xpu::HostOpaque;
        let dom = dom_1d(6);
        let field = Field::<f64, 1, HostOpaque>::zeros(&dom).unwrap();
        assert_eq!(field.download().unwrap(), vec![0.0; 6]);
        field.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        field.write_one([4], -7.5).unwrap();
        assert_eq!(field.read_one([4]).unwrap(), -7.5);
        assert_eq!(field.download().unwrap(), vec![1.0, 2.0, 3.0, 4.0, -7.5, 6.0]);
        // a clone shares the storage, as on every space
        assert_eq!(field.clone().read_one([0]).unwrap(), 1.0);
    }

    #[test]
    fn the_host_alias_exists_on_host_storage_alone() {
        use symbi_xpu::{HostMemory, HostOpaque};
        let host = Field::<f64, 1, HostMemory>::zeros(&dom_1d(3)).unwrap();
        host.set([1], 2.5);
        let alias = host.host().expect("host memory is its own host alias");
        assert_eq!(*alias.view().at([1]), 2.5);
        assert!(Field::<f64, 1, HostOpaque>::zeros(&dom_1d(3)).unwrap().host().is_none());
    }

    #[test]
    #[should_panic(expected = "lies outside the allocation")]
    fn a_transfer_outside_the_allocation_is_refused() {
        use symbi_xpu::HostOpaque;
        let field = Field::<f64, 1, HostOpaque>::zeros(&dom_1d(3)).unwrap();
        let _ = field.read_one([3]);
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
