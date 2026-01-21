// =============================================================================
// lib.rs
//
// metal gpu device implementation for xpu abstraction.
// provides gpu device management and metal buffers for apple silicon/amd gpus.
//
// uses metal-rs for safe bindings to apple's metal framework.
// supports m1/m2/m3 and intel mac gpus.
//
// usage:
//   let metal = MetalDevice::new(0)?;
//   let mut buf = metal.alloc::<f64>(1024)?;
//   metal.copy_to_device(&host_data, &mut buf)?;
//   metal.launch(my_kernel, config, &buf)?;
// =============================================================================

use metal::{Buffer, CommandBuffer, CommandQueue, Device as MTLDevice, MTLResourceOptions};
use std::sync::{Arc, Mutex};
use xpu_core::{Device, DeviceBuffer, Kernel, LaunchConfig, Token};

/// metal gpu device handle.
#[derive(Debug, Clone)]
pub struct MetalDevice {
    id: usize,
    device: MTLDevice,
    queue: Arc<CommandQueue>,
}

/// device memory buffer on metal gpu.
pub struct MetalBuffer<T> {
    buffer: Buffer,
    len: usize,
    device_id: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for MetalBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("len", &self.len)
            .field("device_id", &self.device_id)
            .field("size_bytes", &self.buffer.length())
            .finish()
    }
}

// metal buffers are thread-safe via metal's internal synchronization
unsafe impl<T> Send for MetalBuffer<T> {}
unsafe impl<T> Sync for MetalBuffer<T> {}

/// error types for metal operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// no metal devices available
    NoDevicesAvailable,
    /// requested device id is out of range
    InvalidDeviceId { requested: usize, available: usize },
    /// memory allocation failed
    AllocationFailed { size: usize },
    /// memory copy operation failed
    CopyFailed { reason: &'static str },
    /// kernel compilation failed
    CompilationFailed { reason: String },
    /// kernel launch failed
    LaunchFailed { reason: &'static str },
    /// command buffer execution failed
    ExecutionFailed { reason: String },
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::NoDevicesAvailable => {
                write!(f, "no metal devices available on this system")
            }
            MetalError::InvalidDeviceId {
                requested,
                available,
            } => {
                write!(
                    f,
                    "invalid metal device id {}, only {} devices available",
                    requested, available
                )
            }
            MetalError::AllocationFailed { size } => {
                write!(f, "failed to allocate {} bytes on metal device", size)
            }
            MetalError::CopyFailed { reason } => {
                write!(f, "metal memory copy failed: {}", reason)
            }
            MetalError::CompilationFailed { reason } => {
                write!(f, "metal shader compilation failed: {}", reason)
            }
            MetalError::LaunchFailed { reason } => {
                write!(f, "metal kernel launch failed: {}", reason)
            }
            MetalError::ExecutionFailed { reason } => {
                write!(f, "metal command execution failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for MetalError {}

/// metal event type - tracks command buffer completion.
/// wraps a command buffer and provides completion polling.
#[derive(Clone)]
pub struct MetalEvent {
    // command buffer is optional - immediate events don't have one
    command_buffer: Option<Arc<Mutex<CommandBuffer>>>,
}

impl MetalEvent {
    fn new(command_buffer: CommandBuffer) -> Self {
        Self {
            command_buffer: Some(Arc::new(Mutex::new(command_buffer))),
        }
    }

    fn is_complete(&self) -> bool {
        match &self.command_buffer {
            Some(cb) => {
                let cb = cb.lock().unwrap();
                cb.status() == metal::MTLCommandBufferStatus::Completed
                    || cb.status() == metal::MTLCommandBufferStatus::Error
            }
            None => true, // immediate events are always complete
        }
    }

    fn wait(&self) {
        if let Some(cb) = &self.command_buffer {
            let cb = cb.lock().unwrap();
            cb.wait_until_completed();
        }
    }
}

impl<T> DeviceBuffer<T> for MetalBuffer<T> {
    type Device = MetalDevice;

    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const T {
        self.buffer.contents() as *const T
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.buffer.contents() as *mut T
    }
}

impl<T> MetalBuffer<T> {
    /// returns a reference to the underlying metal buffer.
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// provides a slice view of the buffer.
    /// unsafe: caller must ensure no concurrent gpu access.
    pub unsafe fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        std::slice::from_raw_parts(self.as_ptr(), self.len)
    }

    /// provides a mutable slice view of the buffer.
    /// unsafe: caller must ensure no concurrent gpu access.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len)
    }
}

impl Device for MetalDevice {
    type Buffer<T> = MetalBuffer<T>;
    type Error = MetalError;
    type Event = MetalEvent;

    fn id(&self) -> usize {
        self.id
    }

    fn device_count() -> usize {
        MTLDevice::all().len()
    }

    fn new(id: usize) -> Result<Self, Self::Error> {
        let devices = MTLDevice::all();
        let count = devices.len();

        if count == 0 {
            return Err(MetalError::NoDevicesAvailable);
        }

        if id >= count {
            return Err(MetalError::InvalidDeviceId {
                requested: id,
                available: count,
            });
        }

        let device = devices[id].clone();
        let queue = device.new_command_queue();

        Ok(MetalDevice {
            id,
            device,
            queue: Arc::new(queue),
        })
    }

    fn alloc<T>(&self, n: usize) -> Result<Self::Buffer<T>, Self::Error>
    where
        T: Default + Clone,
    {
        let size = n * std::mem::size_of::<T>();

        // metal doesn't support zero-size allocations, use minimum size of 1 byte
        let alloc_size = if size == 0 { 1 } else { size };

        let buffer = self
            .device
            .new_buffer(alloc_size as u64, MTLResourceOptions::StorageModeShared);

        Ok(MetalBuffer {
            buffer,
            len: n,
            device_id: self.id,
            _phantom: std::marker::PhantomData,
        })
    }

    fn alloc_init<T: Clone>(&self, n: usize, value: T) -> Result<Self::Buffer<T>, Self::Error> {
        let size = n * std::mem::size_of::<T>();
        let buffer = self
            .device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

        let mut buf = MetalBuffer {
            buffer,
            len: n,
            device_id: self.id,
            _phantom: std::marker::PhantomData,
        };

        // initialize on host side
        unsafe {
            let slice = buf.as_mut_slice();
            for elem in slice.iter_mut() {
                *elem = value.clone();
            }
        }

        Ok(buf)
    }

    fn copy_to_device<T>(
        &self,
        host_data: &[T],
        device_buf: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error>
    where
        T: Clone,
    {
        if host_data.len() != device_buf.len() {
            return Err(MetalError::CopyFailed {
                reason: "size mismatch between source and destination",
            });
        }

        // manual clone to respect Clone bound (not Copy)
        unsafe {
            let dst = device_buf.as_mut_slice();
            for (ii, elem) in host_data.iter().enumerate() {
                dst[ii] = elem.clone();
            }
        }

        Ok(())
    }

    fn copy_to_host<T>(
        &self,
        device_buf: &Self::Buffer<T>,
        host_data: &mut [T],
    ) -> Result<(), Self::Error>
    where
        T: Clone,
    {
        if host_data.len() != device_buf.len() {
            return Err(MetalError::CopyFailed {
                reason: "size mismatch between source and destination",
            });
        }

        // manual clone to respect Clone bound (not Copy)
        unsafe {
            let src = device_buf.as_slice();
            for (ii, elem) in src.iter().enumerate() {
                host_data[ii] = elem.clone();
            }
        }

        Ok(())
    }

    fn launch<K, Args>(
        &self,
        _kernel: K,
        _config: LaunchConfig,
        args: Args,
    ) -> Result<(), Self::Error>
    where
        K: Kernel<Args>,
    {
        // for cpu-style kernels on metal with shared memory,
        // we can just run them directly since buffers are cpu-accessible.
        // this is a fallback for simple cases.
        //
        // for real gpu kernels, you'd need to:
        // 1. compile metal shader language (msl) source
        // 2. create compute pipeline state
        // 3. encode compute commands
        // 4. commit and wait
        //
        // since kernels are currently cpu-style, run directly
        K::run(args);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        // with shared memory and direct kernel execution,
        // synchronization is implicit.
        //
        // for real gpu compute, you'd wait on command buffer completion.
        Ok(())
    }

    fn fill<T: Clone>(&self, buf: &mut Self::Buffer<T>, value: T) -> Result<(), Self::Error> {
        // with shared memory mode, we can directly access and fill
        unsafe {
            let slice = buf.as_mut_slice();
            for elem in slice.iter_mut() {
                *elem = value.clone();
            }
        }
        Ok(())
    }

    fn copy_buffer<T: Clone>(
        &self,
        src: &Self::Buffer<T>,
        dst: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error> {
        if src.len() != dst.len() {
            return Err(MetalError::CopyFailed {
                reason: "buffer size mismatch",
            });
        }

        // with shared memory mode, direct copy
        unsafe {
            let src_slice = src.as_slice();
            let dst_slice = dst.as_mut_slice();
            for (ii, elem) in src_slice.iter().enumerate() {
                dst_slice[ii] = elem.clone();
            }
        }
        Ok(())
    }

    fn event_query(event: &Self::Event) -> Result<bool, Self::Error> {
        Ok(event.is_complete())
    }

    fn event_synchronize(event: &Self::Event) -> Result<(), Self::Error> {
        event.wait();
        Ok(())
    }

    fn record_event(&self) -> Result<Token<Self>, Self::Error> {
        // create a command buffer and commit it
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.commit();

        let event = MetalEvent::new(command_buffer.to_owned());
        Ok(Token::from_event(event))
    }

    fn reduce<T, R>(&self, buf: &Self::Buffer<T>, _op: R) -> Result<T, Self::Error>
    where
        T: Clone,
        R: xpu_core::reduce::Reduce<T>,
    {
        // check for empty buffer before accessing memory
        if buf.len() == 0 {
            return Ok(R::identity());
        }

        // fallback to cpu-side reduction using shared memory
        // todo: implement metal compute shader for parallel reduction
        let mut result = R::identity();
        unsafe {
            for elem in buf.as_slice() {
                result = R::combine(result, elem.clone());
            }
        }
        Ok(result)
    }
}

impl MetalDevice {
    /// returns a reference to the underlying mtldevice.
    pub fn metal_device(&self) -> &MTLDevice {
        &self.device
    }

    /// returns a reference to the command queue.
    pub fn command_queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// returns the device name.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// returns whether this is a low-power device.
    pub fn is_low_power(&self) -> bool {
        self.device.is_low_power()
    }

    /// returns whether this is a headless device (no display).
    pub fn is_headless(&self) -> bool {
        self.device.is_headless()
    }
}

// =============================================================================
// implementation notes for gpu compute kernels:
//
// current implementation uses shared memory and cpu-style kernel execution.
// this works but doesn't utilize gpu parallelism.
//
// for real gpu compute:
//
// 1. write kernels in metal shading language (msl):
//    ```metal
//    kernel void vector_add(
//        device const float* a [[buffer(0)]],
//        device const float* b [[buffer(1)]],
//        device float* c [[buffer(2)]],
//        uint id [[thread_position_in_grid]]
//    ) {
//        c[id] = a[id] + b[id];
//    }
//    ```
//
// 2. compile at runtime:
//    let library = device.new_library_with_source(msl_source, &options)?;
//    let function = library.get_function("vector_add", None)?;
//    let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
//
// 3. encode and dispatch:
//    let cmd_buffer = queue.new_command_buffer();
//    let encoder = cmd_buffer.new_compute_command_encoder();
//    encoder.set_compute_pipeline_state(&pipeline);
//    encoder.set_buffer(0, Some(&a.buffer), 0);
//    encoder.set_buffer(1, Some(&b.buffer), 0);
//    encoder.set_buffer(2, Some(&c.buffer), 0);
//    encoder.dispatch_thread_groups(grid, threads_per_group);
//    encoder.end_encoding();
//    cmd_buffer.commit();
//    cmd_buffer.wait_until_completed();
//
// 4. handle errors via command buffer status
//
// for now, the abstraction works with cpu-style kernels.
// gpu compute requires msl compilation infrastructure.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_count() {
        let count = MetalDevice::device_count();
        println!("metal devices available: {}", count);
        // should have at least 1 on a mac
        assert!(count > 0, "no metal devices found");
    }

    #[test]
    fn test_device_creation() {
        let metal = MetalDevice::new(0).unwrap();
        assert_eq!(metal.id(), 0);
        println!("device name: {}", metal.name());
        println!("low power: {}", metal.is_low_power());
        println!("headless: {}", metal.is_headless());
    }

    #[test]
    fn test_invalid_device_id() {
        let count = MetalDevice::device_count();
        let result = MetalDevice::new(count);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_allocation() {
        let metal = MetalDevice::new(0).unwrap();
        let buf = metal.alloc::<f64>(100).unwrap();
        assert_eq!(buf.len(), 100);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_init() {
        let metal = MetalDevice::new(0).unwrap();
        let buf = metal.alloc_init(50, 3.14).unwrap();
        assert_eq!(buf.len(), 50);
        unsafe {
            assert!(buf.as_slice().iter().all(|&x| x == 3.14));
        }
    }

    #[test]
    fn test_copy_to_device() {
        let metal = MetalDevice::new(0).unwrap();
        let host_data: Vec<i32> = (0..10).collect();
        let mut device_buf = metal.alloc::<i32>(10).unwrap();

        metal.copy_to_device(&host_data, &mut device_buf).unwrap();
        unsafe {
            assert_eq!(device_buf.as_slice(), host_data.as_slice());
        }
    }

    #[test]
    fn test_copy_to_host() {
        let metal = MetalDevice::new(0).unwrap();
        let device_buf = metal.alloc_init(10, 42).unwrap();
        let mut host_data = vec![0; 10];

        metal.copy_to_host(&device_buf, &mut host_data).unwrap();
        assert!(host_data.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_copy_size_mismatch() {
        let metal = MetalDevice::new(0).unwrap();
        let host_data = vec![1, 2, 3];
        let mut device_buf = metal.alloc::<i32>(5).unwrap();

        let result = metal.copy_to_device(&host_data, &mut device_buf);
        assert!(result.is_err());
    }

    struct TestKernel;

    impl<T> Kernel<&mut MetalBuffer<T>> for TestKernel
    where
        T: std::ops::MulAssign + Copy,
    {
        fn run(buf: &mut MetalBuffer<T>) {
            unsafe {
                for val in buf.as_mut_slice() {
                    *val *= *val;
                }
            }
        }
    }

    #[test]
    fn test_kernel_launch() {
        let metal = MetalDevice::new(0).unwrap();
        let mut buf = metal.alloc_init(5, 3).unwrap();

        let config = LaunchConfig::new_1d(1, 1);
        metal.launch(TestKernel, config, &mut buf).unwrap();

        let mut result = vec![0; 5];
        metal.copy_to_host(&buf, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 9));
    }

    #[test]
    fn test_synchronize() {
        let metal = MetalDevice::new(0).unwrap();
        assert!(metal.synchronize().is_ok());
    }

    #[test]
    fn test_multiple_devices() {
        let count = MetalDevice::device_count();
        println!("testing {} metal device(s)", count);

        for id in 0..count {
            let device = MetalDevice::new(id).unwrap();
            println!("device {}: {}", id, device.name());

            let buf = device.alloc_init(100, 1.0f32).unwrap();
            assert_eq!(buf.len(), 100);
        }
    }

    #[test]
    fn test_fill_buffer() {
        let metal = MetalDevice::new(0).unwrap();
        let mut buf = metal.alloc::<i32>(10).unwrap();

        metal.fill(&mut buf, 42).unwrap();

        let mut result = vec![0; 10];
        metal.copy_to_host(&buf, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_zero_buffer() {
        let metal = MetalDevice::new(0).unwrap();
        let mut buf = metal.alloc_init(10, 99).unwrap();

        metal.zero(&mut buf).unwrap();

        let mut result = vec![99; 10];
        metal.copy_to_host(&buf, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_copy_buffer() {
        let metal = MetalDevice::new(0).unwrap();
        let src = metal.alloc_init(5, 10).unwrap();
        let mut dst = metal.alloc::<i32>(5).unwrap();

        metal.copy_buffer(&src, &mut dst).unwrap();

        let mut result = vec![0; 5];
        metal.copy_to_host(&dst, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 10));
    }

    #[test]
    fn test_copy_buffer_size_mismatch() {
        let metal = MetalDevice::new(0).unwrap();
        let src = metal.alloc_init(5, 10).unwrap();
        let mut dst = metal.alloc::<i32>(3).unwrap();

        let result = metal.copy_buffer(&src, &mut dst);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_sum() {
        let metal = MetalDevice::new(0).unwrap();
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut buf = metal.alloc::<i32>(5).unwrap();
        metal.copy_to_device(&data, &mut buf).unwrap();

        let total = metal.sum(&buf).unwrap();
        assert_eq!(total, 15);
    }

    #[test]
    fn test_reduce_max() {
        let metal = MetalDevice::new(0).unwrap();
        let data: Vec<f64> = vec![1.5, 9.2, 3.7, 2.1, 8.4];
        let mut buf = metal.alloc::<f64>(5).unwrap();
        metal.copy_to_device(&data, &mut buf).unwrap();

        let maximum = metal.max(&buf).unwrap();
        assert_eq!(maximum, 9.2);
    }

    #[test]
    fn test_reduce_min() {
        let metal = MetalDevice::new(0).unwrap();
        let data: Vec<f64> = vec![1.5, 9.2, 3.7, 0.3, 8.4];
        let mut buf = metal.alloc::<f64>(5).unwrap();
        metal.copy_to_device(&data, &mut buf).unwrap();

        let minimum = metal.min(&buf).unwrap();
        assert_eq!(minimum, 0.3);
    }

    #[test]
    fn test_reduce_empty_buffer() {
        let metal = MetalDevice::new(0).unwrap();
        let buf = metal.alloc::<i32>(0).unwrap();

        let total = metal.sum(&buf).unwrap();
        assert_eq!(total, 0); // identity for sum
    }

    #[test]
    fn test_reduce_custom_operation() {
        use xpu_core::reduce::Product;

        let metal = MetalDevice::new(0).unwrap();
        let data: Vec<i32> = vec![2, 3, 4];
        let mut buf = metal.alloc::<i32>(3).unwrap();
        metal.copy_to_device(&data, &mut buf).unwrap();

        let product = metal.reduce(&buf, Product).unwrap();
        assert_eq!(product, 24); // 2 * 3 * 4
    }

    #[test]
    fn test_record_event() {
        let metal = MetalDevice::new(0).unwrap();
        let token = metal.record_event().unwrap();

        // metal tokens wrap command buffers
        assert!(!token.is_immediate());
    }

    #[test]
    fn test_token_wait() {
        let metal = MetalDevice::new(0).unwrap();
        let token = metal.record_event().unwrap();

        // wait for command buffer completion
        token.wait().unwrap();
    }

    #[test]
    fn test_async_fill_with_token() {
        let metal = MetalDevice::new(0).unwrap();
        let mut buf = metal.alloc::<i32>(10).unwrap();

        // fill buffer
        metal.fill(&mut buf, 42).unwrap();

        // record completion event
        let token = metal.record_event().unwrap();

        // wait for completion
        token.wait().unwrap();

        // verify
        let mut result = vec![0; 10];
        metal.copy_to_host(&buf, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_token_ready_polling() {
        let metal = MetalDevice::new(0).unwrap();
        let token = metal.record_event().unwrap();

        // poll until ready
        while !token.ready().unwrap() {
            // busy wait (not recommended in production)
        }

        // should be complete now
        assert!(token.ready().unwrap());
    }
}
