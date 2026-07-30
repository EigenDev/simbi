// =============================================================================
// handle.rs
//
// reference-counted shared ownership with explicit coherency tracking.
// wraps a value (typically a MemoryBlock) with atomic dirty flags for
// host/device synchronization.
//
// design:
//   - clone = shallow (increments ref count)
//   - coherency is explicit: mark_host_dirty(), needs_device_sync(), etc.
//   - no implicit transfers — the caller decides when to sync
//
// usage:
//   let handle = SharedHandle::new(block);
//   handle.mark_host_dirty();
//   if handle.needs_device_sync() { /* transfer */ }
// =============================================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// =============================================================================
// shared handle
// =============================================================================

/// reference-counted shared ownership with coherency tracking.
/// clone is shallow (increments Arc refcount).
pub struct SharedHandle<T> {
    inner: Arc<ControlBlock<T>>,
}

struct ControlBlock<T> {
    data: T,
    host_dirty: AtomicBool,
    device_dirty: AtomicBool,
}

impl<T> SharedHandle<T> {
    /// create a new handle wrapping the given data.
    pub fn new(data: T) -> Self {
        SharedHandle {
            inner: Arc::new(ControlBlock {
                data,
                host_dirty: AtomicBool::new(false),
                device_dirty: AtomicBool::new(false),
            }),
        }
    }

    /// immutable access to the wrapped data.
    pub fn get(&self) -> &T {
        &self.inner.data
    }

    /// mark that the host has written to this data.
    /// the device copy is now stale.
    pub fn mark_host_dirty(&self) {
        self.inner.host_dirty.store(true, Ordering::Relaxed);
    }

    /// mark that the device has written to this data.
    /// the host copy is now stale.
    pub fn mark_device_dirty(&self) {
        self.inner.device_dirty.store(true, Ordering::Relaxed);
    }

    /// mark both copies as synchronized.
    pub fn mark_synchronized(&self) {
        self.inner.host_dirty.store(false, Ordering::Relaxed);
        self.inner.device_dirty.store(false, Ordering::Relaxed);
    }

    /// true if the host wrote and the device hasn't been updated.
    pub fn needs_device_sync(&self) -> bool {
        self.inner.host_dirty.load(Ordering::Relaxed)
    }

    /// true if the device wrote and the host hasn't been updated.
    pub fn needs_host_sync(&self) -> bool {
        self.inner.device_dirty.load(Ordering::Relaxed)
    }

    /// reference count.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

// clone = shallow (Arc increment)
impl<T> Clone for SharedHandle<T> {
    fn clone(&self) -> Self {
        SharedHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_lifecycle() {
        let handle = SharedHandle::new(42u64);
        assert_eq!(*handle.get(), 42);
        assert_eq!(handle.ref_count(), 1);
    }

    #[test]
    fn clone_is_shallow() {
        let h1 = SharedHandle::new(42u64);
        let h2 = h1.clone();
        assert_eq!(h1.ref_count(), 2);
        assert_eq!(h2.ref_count(), 2);
        assert_eq!(*h1.get(), *h2.get());
    }

    #[test]
    fn coherency_tracking() {
        let handle = SharedHandle::new(0u64);
        assert!(!handle.needs_device_sync());
        assert!(!handle.needs_host_sync());

        handle.mark_host_dirty();
        assert!(handle.needs_device_sync());
        assert!(!handle.needs_host_sync());

        handle.mark_synchronized();
        assert!(!handle.needs_device_sync());
        assert!(!handle.needs_host_sync());

        handle.mark_device_dirty();
        assert!(!handle.needs_device_sync());
        assert!(handle.needs_host_sync());
    }

    #[test]
    fn clone_shares_coherency() {
        let h1 = SharedHandle::new(0u64);
        let h2 = h1.clone();
        h1.mark_host_dirty();
        assert!(h2.needs_device_sync());
    }
}
