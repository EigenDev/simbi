// =============================================================================
// memory.rs
//
// memory storage traits and memory block. a storage type defines where data
// lives (host, managed, device-local) and how bytes cross to and from the host.
// the memory block provides raii ownership of an allocation in a given space.
//
// design:
//   - storage spaces are encoded as types, so the space is fixed at compile time.
//   - `MemoryStorage` is every space: allocation, the space constants, and the
//     synchronized host transfers. `MemorySpace` marks the spaces host code may
//     dereference directly (host and managed memory); a device-local space
//     implements `MemoryStorage` alone, so host code reaches its contents only
//     through the explicit transfers.
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

/// where data lives and how its bytes reach the host. implemented by every space; all methods
/// are static, the space itself is stateless.
pub trait MemoryStorage: 'static + Send + Sync + Sized {
    /// the space host code reaches this storage as: the space itself where host code may
    /// dereference it (host and managed memory), `NoHostAccess` where it may not. a container in
    /// this space is re-typed to its host alias only when the two are the same type, so the
    /// re-typing is an identity and device-local storage never gains one.
    type HostAccess: MemorySpace + Sync;
    /// host code may dereference the storage directly (host and managed memory)
    const IS_HOST_ACCESSIBLE: bool;
    /// the storage is mapped in host memory, so a host reference into it is valid: host and
    /// managed memory, and the opaque cpu stand-in. kernel bindings of host-resident storage are
    /// host slices; every other space binds as a device pointer and length, and the cpu executor
    /// refuses it
    const IS_HOST_RESIDENT: bool;
    /// kernels in this space run on a gpu
    const IS_DEVICE_ACCESSIBLE: bool;
    const IS_UNIFIED: bool;
    const PREFERRED_ALIGNMENT: usize;

    /// allocate `bytes` of zeroed memory in this space. returns a raw pointer.
    /// the pointer is valid for both host and device access if the space
    /// is unified.
    fn allocate(bytes: usize) -> error::Result<*mut u8>;

    /// free a previous allocation.
    fn deallocate(ptr: *mut u8, bytes: usize);

    /// copy `bytes` from storage `src` in this space to host memory `dst`, after every queued
    /// kernel that writes the source has finished; on return the host copy is complete.
    ///
    /// safety: `src` holds `bytes` of this space's storage and `dst` `bytes` of host memory.
    unsafe fn copy_to_host(dst: *mut u8, src: *const u8, bytes: usize) -> error::Result<()>;

    /// copy `bytes` from host memory `src` into storage `dst` in this space, ordered after every
    /// queued kernel; on return the storage holds the bytes for any later kernel or transfer.
    ///
    /// safety: `dst` holds `bytes` of this space's storage and `src` `bytes` of host memory.
    unsafe fn copy_from_host(dst: *mut u8, src: *const u8, bytes: usize) -> error::Result<()>;
}

/// a space host code may dereference directly: host memory and managed (unified) memory. the
/// field layer's views and coordinate access require it, so storage outside it (device-local
/// memory) cannot be read or written by host code at compile time.
pub trait MemorySpace: MemoryStorage {}

/// whether storage in space `M` is host-dereferenceable as itself: its host alias is the same
/// type. the one test every host re-typing of a container passes before casting.
pub fn host_alias_is_identity<M: MemoryStorage>() -> bool {
    std::any::TypeId::of::<M>() == std::any::TypeId::of::<M::HostAccess>()
}

/// the host alias of storage host code cannot dereference. it allocates nothing; a container is
/// never re-typed to it, because it differs from the storage it stands for.
pub struct NoHostAccess;

impl MemoryStorage for NoHostAccess {
    type HostAccess = NoHostAccess;
    const IS_HOST_ACCESSIBLE: bool = false;
    const IS_HOST_RESIDENT: bool = false;
    const IS_DEVICE_ACCESSIBLE: bool = false;
    const IS_UNIFIED: bool = false;
    const PREFERRED_ALIGNMENT: usize = 1;

    fn allocate(_bytes: usize) -> error::Result<*mut u8> {
        Err(error::XpuError {
            operation: "no_host_access_alloc",
            code: -1,
            detail: "NoHostAccess names the absence of host access and holds no storage".into(),
        })
    }

    fn deallocate(_ptr: *mut u8, _bytes: usize) {}

    unsafe fn copy_to_host(_dst: *mut u8, _src: *const u8, _bytes: usize) -> error::Result<()> {
        unreachable!("NoHostAccess holds no storage")
    }

    unsafe fn copy_from_host(_dst: *mut u8, _src: *const u8, _bytes: usize) -> error::Result<()> {
        unreachable!("NoHostAccess holds no storage")
    }
}

impl MemorySpace for NoHostAccess {}

// =============================================================================
// memory block
// =============================================================================

/// raii memory ownership. move-only. drop frees the allocation.
/// the block knows its size and memory space but not its layout.
pub struct MemoryBlock<M: MemoryStorage> {
    ptr: *mut u8,
    bytes: usize,
    _space: PhantomData<M>,
}

unsafe impl<M: MemoryStorage> Send for MemoryBlock<M> {}
unsafe impl<M: MemoryStorage> Sync for MemoryBlock<M> {}

impl<M: MemoryStorage> MemoryBlock<M> {
    /// allocate a new block of `bytes` in the given memory space.
    pub fn new(bytes: usize) -> error::Result<Self> {
        if bytes == 0 {
            return Ok(MemoryBlock {
                ptr: std::ptr::null_mut(),
                bytes: 0,
                _space: PhantomData,
            });
        }
        let ptr = M::allocate(bytes)?;
        Ok(MemoryBlock {
            ptr,
            bytes,
            _space: PhantomData,
        })
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

impl<M: MemoryStorage> Drop for MemoryBlock<M> {
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

const HOST_ALIGNMENT: usize = 64; // cache line

fn host_alloc_zeroed(bytes: usize) -> error::Result<*mut u8> {
    let layout = std::alloc::Layout::from_size_align(bytes, HOST_ALIGNMENT).map_err(|_| error::XpuError {
        operation: "host_alloc",
        code: -1,
        detail: "invalid layout".into(),
    })?;
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(error::XpuError {
            operation: "host_alloc",
            code: -2,
            detail: format!("allocation failed for {} bytes", bytes),
        });
    }
    Ok(ptr)
}

fn host_dealloc(ptr: *mut u8, bytes: usize) {
    let layout = std::alloc::Layout::from_size_align(bytes, HOST_ALIGNMENT).expect("invalid layout");
    unsafe {
        std::alloc::dealloc(ptr, layout);
    }
}

impl MemoryStorage for HostMemory {
    type HostAccess = HostMemory;
    const IS_HOST_ACCESSIBLE: bool = true;
    const IS_HOST_RESIDENT: bool = true;
    const IS_DEVICE_ACCESSIBLE: bool = false;
    const IS_UNIFIED: bool = false;
    const PREFERRED_ALIGNMENT: usize = HOST_ALIGNMENT;

    fn allocate(bytes: usize) -> error::Result<*mut u8> {
        host_alloc_zeroed(bytes)
    }

    fn deallocate(ptr: *mut u8, bytes: usize) {
        host_dealloc(ptr, bytes)
    }

    unsafe fn copy_to_host(dst: *mut u8, src: *const u8, bytes: usize) -> error::Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, bytes) };
        Ok(())
    }

    unsafe fn copy_from_host(dst: *mut u8, src: *const u8, bytes: usize) -> error::Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, bytes) };
        Ok(())
    }
}

impl MemorySpace for HostMemory {}

// =============================================================================
// opaque host storage
// =============================================================================

/// host-resident storage that host code cannot dereference: the cpu stand-in for device-local
/// memory. kernels run on it through the cpu executor, and host code reaches its contents only
/// through `copy_to_host`/`copy_from_host`, so a host build exercises the device-local discipline,
/// and the compiler rejects any field view, coordinate read or write on it.
pub struct HostOpaque;

impl MemoryStorage for HostOpaque {
    type HostAccess = NoHostAccess;
    const IS_HOST_ACCESSIBLE: bool = false;
    const IS_HOST_RESIDENT: bool = true;
    const IS_DEVICE_ACCESSIBLE: bool = false;
    const IS_UNIFIED: bool = false;
    const PREFERRED_ALIGNMENT: usize = HOST_ALIGNMENT;

    fn allocate(bytes: usize) -> error::Result<*mut u8> {
        host_alloc_zeroed(bytes)
    }

    fn deallocate(ptr: *mut u8, bytes: usize) {
        host_dealloc(ptr, bytes)
    }

    unsafe fn copy_to_host(dst: *mut u8, src: *const u8, bytes: usize) -> error::Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, bytes) };
        Ok(())
    }

    unsafe fn copy_from_host(dst: *mut u8, src: *const u8, bytes: usize) -> error::Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, bytes) };
        Ok(())
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
