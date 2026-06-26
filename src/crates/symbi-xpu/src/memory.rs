// =============================================================================
// memory.rs
//
// memory space trait and memory block. the memory space defines WHERE data
// lives (host, device, unified). the memory block provides RAII ownership
// of an allocation in a given space.
//
// design:
//   - memory spaces are types, not values. compile-time dispatch.
//   - memory blocks are move-only. drop frees the allocation.
//   - no layout knowledge here — strides, shapes, multi-dim indexing
//     belong in the grid/field layer.
//
// usage:
//   let block = MemoryBlock::<UnifiedMemory>::new(1024);
//   let ptr = block.as_ptr::<f64>();
//   // block dropped -> memory freed
// =============================================================================

use crate::error;
use std::marker::PhantomData;

// =============================================================================
// memory space trait
// =============================================================================

/// defines where data lives. implemented by HostMemory, DeviceMemory,
/// UnifiedMemory. all methods are static — the space itself is stateless.
pub trait MemorySpace: 'static + Send + Sync + Sized {
    const IS_HOST_ACCESSIBLE: bool;
    const IS_DEVICE_ACCESSIBLE: bool;
    const IS_UNIFIED: bool;
    const PREFERRED_ALIGNMENT: usize;

    /// allocate `bytes` of memory in this space. returns a raw pointer.
    /// the pointer is valid for both host and device access if the space
    /// is unified.
    fn allocate(bytes: usize) -> error::Result<*mut u8>;

    /// free a previous allocation.
    fn deallocate(ptr: *mut u8, bytes: usize);
}

// =============================================================================
// memory block
// =============================================================================

/// RAII memory ownership. move-only. drop frees the allocation.
/// the block knows its size and memory space but not its layout.
pub struct MemoryBlock<M: MemorySpace> {
    ptr: *mut u8,
    bytes: usize,
    _space: PhantomData<M>,
}

unsafe impl<M: MemorySpace> Send for MemoryBlock<M> {}
unsafe impl<M: MemorySpace> Sync for MemoryBlock<M> {}

impl<M: MemorySpace> MemoryBlock<M> {
    /// allocate a new block of `bytes` in the given memory space.
    pub fn new(bytes: usize) -> error::Result<Self> {
        if bytes == 0 {
            return Ok(MemoryBlock { ptr: std::ptr::null_mut(), bytes: 0, _space: PhantomData });
        }
        let ptr = M::allocate(bytes)?;
        Ok(MemoryBlock { ptr, bytes, _space: PhantomData })
    }

    /// allocate storage for `count` elements of type T.
    pub fn for_elements<T>(count: usize) -> error::Result<Self> {
        Self::new(count * std::mem::size_of::<T>())
    }

    /// raw pointer, reinterpreted as *const T.
    pub fn as_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }

    /// raw mutable pointer, reinterpreted as *mut T.
    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.ptr as *mut T
    }

    /// raw byte pointer.
    pub fn raw(&self) -> *mut u8 {
        self.ptr
    }

    /// size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// true if the block has no allocation.
    pub fn is_empty(&self) -> bool {
        self.ptr.is_null() || self.bytes == 0
    }

    /// number of elements of type T that fit in this block.
    pub fn count<T>(&self) -> usize {
        self.bytes / std::mem::size_of::<T>()
    }
}

impl<M: MemorySpace> Drop for MemoryBlock<M> {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.bytes > 0 {
            M::deallocate(self.ptr, self.bytes);
            self.ptr = std::ptr::null_mut();
        }
    }
}

// move-only: no Copy, no Clone

// =============================================================================
// host memory space
// =============================================================================

/// host memory. cache-line aligned. accessible from CPU only.
pub struct HostMemory;

impl MemorySpace for HostMemory {
    const IS_HOST_ACCESSIBLE: bool = true;
    const IS_DEVICE_ACCESSIBLE: bool = false;
    const IS_UNIFIED: bool = false;
    const PREFERRED_ALIGNMENT: usize = 64; // cache line

    fn allocate(bytes: usize) -> error::Result<*mut u8> {
        let layout = std::alloc::Layout::from_size_align(bytes, Self::PREFERRED_ALIGNMENT)
            .map_err(|_| error::XpuError { operation: "host_alloc", code: -1, detail: "invalid layout".into() })?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(error::XpuError { operation: "host_alloc", code: -2, detail: format!("allocation failed for {} bytes", bytes) });
        }
        Ok(ptr)
    }

    fn deallocate(ptr: *mut u8, bytes: usize) {
        let layout = std::alloc::Layout::from_size_align(bytes, Self::PREFERRED_ALIGNMENT)
            .expect("invalid layout");
        unsafe { std::alloc::dealloc(ptr, layout); }
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_block_lifecycle() {
        let block = MemoryBlock::<HostMemory>::new(1024).unwrap();
        assert!(!block.is_empty());
        assert_eq!(block.bytes(), 1024);
    }

    #[test]
    fn host_block_for_elements() {
        let block = MemoryBlock::<HostMemory>::for_elements::<f64>(100).unwrap();
        assert_eq!(block.bytes(), 800);
        assert_eq!(block.count::<f64>(), 100);
    }

    #[test]
    fn host_block_write_read() {
        let mut block = MemoryBlock::<HostMemory>::for_elements::<f64>(10).unwrap();
        let ptr = block.as_mut_ptr::<f64>();
        unsafe {
            for ii in 0..10 {
                *ptr.add(ii) = ii as f64;
            }
            for ii in 0..10 {
                assert_eq!(*ptr.add(ii), ii as f64);
            }
        }
    }

    #[test]
    fn empty_block() {
        let block = MemoryBlock::<HostMemory>::new(0).unwrap();
        assert!(block.is_empty());
    }
}
