// =============================================================================
// lib.rs
//
// core traits for compile-time device abstraction in xpu.
// defines device, buffer, and kernel interfaces that allow zero-cost
// generic programming across cpu, cuda, rocm, etc.
//
// key design:
//   - device type chosen at compile time (monomorphization)
//   - device instances managed at runtime (multi-gpu support)
//   - no trait objects, no vtables, no runtime overhead
//
// usage:
//   fn my_algorithm<D: Device>(device: &D) {
//       let buf = device.alloc::<f64>(1000);
//       device.launch(my_kernel, grid, block, &buf);
//   }
// =============================================================================

#![cfg_attr(not(test), no_std)]

pub mod reduce;
pub mod token;
pub mod view;

pub use reduce::{Max, Min, Product, Reduce, Sum};
pub use token::Token;
pub use view::{Shape, View, View1, View2, View3, ViewMut, ViewMut1, ViewMut2, ViewMut3};

/// launch configuration for parallel kernels.
/// specifies grid and block dimensions for execution.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LaunchConfig {
    /// number of blocks in each dimension (x, y, z)
    pub grid: (u32, u32, u32),
    /// number of threads per block in each dimension (x, y, z)
    pub block: (u32, u32, u32),
}

impl LaunchConfig {
    /// creates a 1d launch configuration.
    pub fn new_1d(num_blocks: u32, threads_per_block: u32) -> Self {
        Self {
            grid: (num_blocks, 1, 1),
            block: (threads_per_block, 1, 1),
        }
    }

    /// creates a 2d launch configuration.
    pub fn new_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
        }
    }
}

/// trait for device-resident memory buffers.
/// each device type has its own buffer implementation.
pub trait DeviceBuffer<T>: Sized {
    /// the device type this buffer belongs to.
    type Device: Device;

    /// returns the number of elements in the buffer.
    fn len(&self) -> usize;

    /// returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// returns a raw const pointer to device memory.
    /// warning: this pointer is only valid on the device.
    fn as_ptr(&self) -> *const T;

    /// returns a raw mutable pointer to device memory.
    /// warning: this pointer is only valid on the device.
    fn as_mut_ptr(&mut self) -> *mut T;
}

/// trait for kernel functions that can execute on a device.
/// kernels are stateless and operate on buffers passed as arguments.
pub trait Kernel<Args> {
    /// executes the kernel with the given arguments.
    /// args typically contain device buffers and scalar parameters.
    fn run(args: Args);
}

/// trait for execution devices (cpu, cuda gpu, etc.).
/// this is the core abstraction - device type is compile-time,
/// but device instances (gpu 0, gpu 1) are runtime.
pub trait Device: Sized {
    /// the buffer type for this device.
    type Buffer<T>: DeviceBuffer<T, Device = Self>;

    /// the error type for operations on this device.
    type Error: core::fmt::Debug + core::fmt::Display;

    /// the event type for tracking async operations.
    type Event: Send;

    /// returns the device instance id (e.g., gpu 0, gpu 1).
    fn id(&self) -> usize;

    /// returns the total number of devices of this type available.
    fn device_count() -> usize;

    /// creates a new device handle for the given id.
    /// returns error if id is out of range.
    fn new(id: usize) -> Result<Self, Self::Error>;

    /// allocates a buffer of n elements on this device.
    /// elements are uninitialized.
    fn alloc<T>(&self, n: usize) -> Result<Self::Buffer<T>, Self::Error>
    where
        T: Default + Clone;

    /// allocates a buffer and initializes all elements to the given value.
    fn alloc_init<T: Clone>(&self, n: usize, value: T) -> Result<Self::Buffer<T>, Self::Error>;

    /// copies data from host to device buffer.
    fn copy_to_device<T>(
        &self,
        host_data: &[T],
        device_buf: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error>
    where
        T: Clone;

    /// copies data from device buffer to host.
    fn copy_to_host<T>(
        &self,
        device_buf: &Self::Buffer<T>,
        host_data: &mut [T],
    ) -> Result<(), Self::Error>
    where
        T: Clone;

    /// launches a kernel on this device with the given configuration.
    fn launch<K, Args>(
        &self,
        kernel: K,
        config: LaunchConfig,
        args: Args,
    ) -> Result<(), Self::Error>
    where
        K: Kernel<Args>;

    /// synchronizes the device, blocking until all operations complete.
    fn synchronize(&self) -> Result<(), Self::Error>;

    /// fills a buffer with the given value.
    fn fill<T: Clone>(&self, buf: &mut Self::Buffer<T>, value: T) -> Result<(), Self::Error>;

    /// zeros out a buffer (fills with default value).
    fn zero<T: Default + Clone>(&self, buf: &mut Self::Buffer<T>) -> Result<(), Self::Error> {
        self.fill(buf, T::default())
    }

    /// copies data from one buffer to another on the same device.
    /// buffers must have the same length.
    fn copy_buffer<T: Clone>(
        &self,
        src: &Self::Buffer<T>,
        dst: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error>;

    /// queries if an event has completed.
    /// non-blocking poll.
    fn event_query(event: &Self::Event) -> Result<bool, Self::Error>;

    /// waits for an event to complete.
    /// blocking synchronization.
    fn event_synchronize(event: &Self::Event) -> Result<(), Self::Error>;

    /// records an event on this device's stream.
    /// returns a token tracking the recorded event.
    fn record_event(&self) -> Result<crate::token::Token<Self>, Self::Error>;

    /// reduces a buffer using the given reduction operation.
    /// performs a parallel reduction to combine all elements.
    fn reduce<T, R>(&self, buf: &Self::Buffer<T>, op: R) -> Result<T, Self::Error>
    where
        T: Clone,
        R: crate::reduce::Reduce<T>;

    /// computes the sum of all elements in the buffer.
    fn sum<T>(&self, buf: &Self::Buffer<T>) -> Result<T, Self::Error>
    where
        T: Clone,
        crate::reduce::Sum: crate::reduce::Reduce<T>,
    {
        self.reduce(buf, crate::reduce::Sum)
    }

    /// finds the maximum element in the buffer.
    fn max<T>(&self, buf: &Self::Buffer<T>) -> Result<T, Self::Error>
    where
        T: Clone,
        crate::reduce::Max: crate::reduce::Reduce<T>,
    {
        self.reduce(buf, crate::reduce::Max)
    }

    /// finds the minimum element in the buffer.
    fn min<T>(&self, buf: &Self::Buffer<T>) -> Result<T, Self::Error>
    where
        T: Clone,
        crate::reduce::Min: crate::reduce::Reduce<T>,
    {
        self.reduce(buf, crate::reduce::Min)
    }
}

/// multi-device pool for managing multiple instances of the same device type.
/// provides runtime access to multiple gpus while maintaining compile-time
/// device type selection.
#[derive(Debug)]
pub struct DevicePool<D: Device> {
    devices: alloc::vec::Vec<D>,
}

impl<D: Device> DevicePool<D> {
    /// creates a pool containing all available devices of type d.
    pub fn new() -> Result<Self, D::Error> {
        let count = D::device_count();
        let mut devices = alloc::vec::Vec::with_capacity(count);
        for id in 0..count {
            devices.push(D::new(id)?);
        }
        Ok(DevicePool { devices })
    }

    /// creates a pool with specific device ids.
    pub fn from_ids(ids: &[usize]) -> Result<Self, D::Error> {
        let mut devices = alloc::vec::Vec::with_capacity(ids.len());
        for &id in ids {
            devices.push(D::new(id)?);
        }
        Ok(DevicePool { devices })
    }

    /// returns the number of devices in the pool.
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// returns a reference to the device at the given index.
    pub fn get(&self, index: usize) -> Option<&D> {
        self.devices.get(index)
    }

    /// returns an iterator over all devices in the pool.
    pub fn iter(&self) -> core::slice::Iter<'_, D> {
        self.devices.iter()
    }
}

// no_std compatibility: need alloc for Vec
extern crate alloc;

// =============================================================================
// view creation helpers for device buffers
// =============================================================================

/// extension trait for creating views from device buffers.
pub trait DeviceBufferExt<T>: DeviceBuffer<T> {
    /// creates a 1d view of the entire buffer.
    fn view_1d(&self) -> view::View1<'_, T> {
        let shape = [self.len()];
        let start = [0];
        let strides = [1];
        unsafe { view::View1::new(self.as_ptr(), shape, start, strides) }
    }

    /// creates a mutable 1d view of the entire buffer.
    fn view_mut_1d(&mut self) -> view::ViewMut1<'_, T> {
        let shape = [self.len()];
        let start = [0];
        let strides = [1];
        unsafe { view::ViewMut1::new(self.as_mut_ptr(), shape, start, strides) }
    }

    /// creates a 2d row-major view of the buffer.
    /// returns none if shape doesn't match buffer size.
    fn view_2d(&self, shape: [usize; 2]) -> Option<view::View2<'_, T>> {
        let total: usize = shape.iter().product();
        if total != self.len() {
            return None;
        }

        let strides = [shape[1], 1]; // row-major: [nx, 1]
        let start = [0, 0];
        Some(unsafe { view::View2::new(self.as_ptr(), shape, start, strides) })
    }

    /// creates a mutable 2d row-major view of the buffer.
    fn view_mut_2d(&mut self, shape: [usize; 2]) -> Option<view::ViewMut2<'_, T>> {
        let total: usize = shape.iter().product();
        if total != self.len() {
            return None;
        }

        let strides = [shape[1], 1];
        let start = [0, 0];
        Some(unsafe { view::ViewMut2::new(self.as_mut_ptr(), shape, start, strides) })
    }

    /// creates a 3d row-major view of the buffer.
    /// returns none if shape doesn't match buffer size.
    fn view_3d(&self, shape: [usize; 3]) -> Option<view::View3<'_, T>> {
        let total: usize = shape.iter().product();
        if total != self.len() {
            return None;
        }

        let strides = [shape[1] * shape[2], shape[2], 1]; // row-major: [ny*nx, nx, 1]
        let start = [0, 0, 0];
        Some(unsafe { view::View3::new(self.as_ptr(), shape, start, strides) })
    }

    /// creates a mutable 3d row-major view of the buffer.
    fn view_mut_3d(&mut self, shape: [usize; 3]) -> Option<view::ViewMut3<'_, T>> {
        let total: usize = shape.iter().product();
        if total != self.len() {
            return None;
        }

        let strides = [shape[1] * shape[2], shape[2], 1];
        let start = [0, 0, 0];
        Some(unsafe { view::ViewMut3::new(self.as_mut_ptr(), shape, start, strides) })
    }
}

// blanket implementation for all device buffers
impl<T, B: DeviceBuffer<T>> DeviceBufferExt<T> for B {}
