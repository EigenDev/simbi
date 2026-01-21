// =============================================================================
// lib.rs
//
// cpu implementation of the xpu device abstraction.
// provides single-threaded and parallel cpu "devices" and host memory buffers.
//
// usage:
//   let cpu = CpuDevice::new(0)?;          // single-threaded
//   let par_cpu = ParCpuDevice::new(0)?;   // multi-threaded (rayon)
//   let mut buf = cpu.alloc::<f64>(1024)?;
//   cpu.copy_to_device(&host_data, &mut buf)?;
//   cpu.launch(my_kernel, config, &buf)?;
// =============================================================================

use rayon::prelude::*;
use xpu_core::{Device, DeviceBuffer, Kernel, LaunchConfig, Token};

/// cpu device implementation.
/// represents the host processor as a "device" for uniform api.
#[derive(Debug, Clone)]
pub struct CpuDevice {
    id: usize,
}

/// host memory buffer backed by a vec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBuffer<T> {
    data: Vec<T>,
}

/// error type for cpu operations.
/// cpu operations are generally infallible, but we maintain
/// the error type for api consistency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuError {
    InvalidDeviceId { requested: usize, max: usize },
    AllocationFailed { size: usize },
    CopyFailed { reason: &'static str },
}

impl core::fmt::Display for CpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CpuError::InvalidDeviceId { requested, max } => {
                write!(f, "invalid cpu device id {}, max is {}", requested, max)
            }
            CpuError::AllocationFailed { size } => {
                write!(f, "failed to allocate {} bytes on cpu", size)
            }
            CpuError::CopyFailed { reason } => {
                write!(f, "memory copy failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for CpuError {}

/// cpu event type - cpu execution is synchronous, so events are trivial.
/// always immediately complete.
#[derive(Debug, Clone)]
pub struct CpuEvent;

impl<T> DeviceBuffer<T> for HostBuffer<T> {
    type Device = CpuDevice;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<T> HostBuffer<T> {
    /// creates a new uninitialized buffer.
    /// elements will have undefined values.
    pub fn new_uninit(size: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            data: vec![T::default(); size],
        }
    }

    /// creates a buffer from an existing vector.
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// provides direct access to the underlying vector.
    pub fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// provides mutable access to the underlying vector.
    pub fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// provides a slice view of the buffer.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// provides a mutable slice view of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl Device for CpuDevice {
    type Buffer<T> = HostBuffer<T>;
    type Error = CpuError;
    type Event = CpuEvent;

    fn id(&self) -> usize {
        self.id
    }

    fn device_count() -> usize {
        // cpu is a single logical device for api uniformity
        1
    }

    fn new(id: usize) -> Result<Self, Self::Error> {
        if id >= Self::device_count() {
            return Err(CpuError::InvalidDeviceId {
                requested: id,
                max: Self::device_count() - 1,
            });
        }
        Ok(CpuDevice { id })
    }

    fn alloc<T>(&self, n: usize) -> Result<Self::Buffer<T>, Self::Error>
    where
        T: Default + Clone,
    {
        Ok(HostBuffer::new_uninit(n))
    }

    fn alloc_init<T: Clone>(&self, n: usize, value: T) -> Result<Self::Buffer<T>, Self::Error> {
        Ok(HostBuffer {
            data: vec![value; n],
        })
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
            return Err(CpuError::CopyFailed {
                reason: "size mismatch between source and destination",
            });
        }
        device_buf.data.clone_from_slice(host_data);
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
            return Err(CpuError::CopyFailed {
                reason: "size mismatch between source and destination",
            });
        }
        host_data.clone_from_slice(&device_buf.data);
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
        // on cpu, "launching" a kernel is just a function call.
        // launch config is ignored for cpu execution.
        K::run(args);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        // cpu execution is synchronous, nothing to wait for
        Ok(())
    }

    fn fill<T: Clone>(&self, buf: &mut Self::Buffer<T>, value: T) -> Result<(), Self::Error> {
        for elem in buf.as_mut_slice() {
            *elem = value.clone();
        }
        Ok(())
    }

    fn copy_buffer<T: Clone>(
        &self,
        src: &Self::Buffer<T>,
        dst: &mut Self::Buffer<T>,
    ) -> Result<(), Self::Error> {
        if src.len() != dst.len() {
            return Err(CpuError::CopyFailed {
                reason: "buffer size mismatch",
            });
        }

        let src_slice = src.as_slice();
        let dst_slice = dst.as_mut_slice();
        for (ii, elem) in src_slice.iter().enumerate() {
            dst_slice[ii] = elem.clone();
        }
        Ok(())
    }

    fn event_query(_event: &Self::Event) -> Result<bool, Self::Error> {
        // cpu events are always complete (synchronous execution)
        Ok(true)
    }

    fn event_synchronize(_event: &Self::Event) -> Result<(), Self::Error> {
        // cpu is synchronous, nothing to wait for
        Ok(())
    }

    fn record_event(&self) -> Result<Token<Self>, Self::Error> {
        // cpu execution is immediate, return immediate token
        Ok(Token::immediate())
    }

    fn reduce<T, R>(&self, buf: &Self::Buffer<T>, _op: R) -> Result<T, Self::Error>
    where
        T: Clone,
        R: xpu_core::reduce::Reduce<T>,
    {
        if buf.is_empty() {
            return Ok(R::identity());
        }

        // serial reduction on cpu
        let mut result = R::identity();
        for elem in buf.as_slice() {
            result = R::combine(result, elem.clone());
        }
        Ok(result)
    }
}

// =============================================================================
// parallel cpu device using rayon for multi-threaded execution
// note: only works efficiently with Send + Sync types
// =============================================================================

/// parallel cpu device implementation using rayon.
/// represents the host processor with multi-threaded parallelism.
/// works best with types that implement Send + Sync.
#[derive(Debug, Clone)]
pub struct ParCpuDevice {
    id: usize,
}

impl ParCpuDevice {
    /// returns the device id.
    pub fn id(&self) -> usize {
        self.id
    }

    /// creates a parallel cpu device.
    /// note: parallel operations only work efficiently with Send + Sync types.
    pub fn new_parallel(id: usize) -> Result<Self, CpuError> {
        if id >= 1 {
            return Err(CpuError::InvalidDeviceId {
                requested: id,
                max: 0,
            });
        }
        Ok(ParCpuDevice { id })
    }

    /// allocates a buffer using parallel initialization where beneficial.
    pub fn alloc_par<T>(&self, n: usize) -> Result<HostBuffer<T>, CpuError>
    where
        T: Default + Clone + Send + Sync,
    {
        let data = if n > 10000 {
            // parallel initialization for large buffers
            (0..n).into_par_iter().map(|_| T::default()).collect()
        } else {
            vec![T::default(); n]
        };
        Ok(HostBuffer::from_vec(data))
    }

    /// allocates and fills buffer using parallel operations.
    pub fn alloc_init_par<T>(&self, n: usize, value: T) -> Result<HostBuffer<T>, CpuError>
    where
        T: Clone + Send + Sync,
    {
        Ok(HostBuffer::from_vec(vec![value; n]))
    }

    /// parallel copy to device buffer.
    pub fn copy_to_device_par<T>(
        &self,
        host_data: &[T],
        device_buf: &mut HostBuffer<T>,
    ) -> Result<(), CpuError>
    where
        T: Clone + Send + Sync,
    {
        if host_data.len() != device_buf.data.len() {
            return Err(CpuError::CopyFailed {
                reason: "size mismatch",
            });
        }

        if host_data.len() > 1024 {
            device_buf
                .data
                .par_iter_mut()
                .zip(host_data.par_iter())
                .for_each(|(d, s)| *d = s.clone());
        } else {
            device_buf.data.clone_from_slice(host_data);
        }
        Ok(())
    }

    /// parallel copy from device buffer.
    pub fn copy_to_host_par<T>(
        &self,
        device_buf: &HostBuffer<T>,
        host_data: &mut [T],
    ) -> Result<(), CpuError>
    where
        T: Clone + Send + Sync,
    {
        if host_data.len() != device_buf.data.len() {
            return Err(CpuError::CopyFailed {
                reason: "size mismatch",
            });
        }

        if host_data.len() > 1024 {
            host_data
                .par_iter_mut()
                .zip(device_buf.data.par_iter())
                .for_each(|(d, s)| *d = s.clone());
        } else {
            host_data.clone_from_slice(&device_buf.data);
        }
        Ok(())
    }

    /// parallel buffer fill.
    pub fn fill_par<T>(&self, buf: &mut HostBuffer<T>, value: T) -> Result<(), CpuError>
    where
        T: Clone + Send + Sync,
    {
        if buf.len() > 1024 {
            buf.data
                .par_iter_mut()
                .for_each(|elem| *elem = value.clone());
        } else {
            buf.data.fill(value);
        }
        Ok(())
    }

    /// parallel buffer copy.
    pub fn copy_buffer_par<T>(
        &self,
        src: &HostBuffer<T>,
        dst: &mut HostBuffer<T>,
    ) -> Result<(), CpuError>
    where
        T: Clone + Send + Sync,
    {
        if src.len() != dst.len() {
            return Err(CpuError::CopyFailed {
                reason: "buffer size mismatch",
            });
        }

        if src.len() > 1024 {
            dst.data
                .par_iter_mut()
                .zip(src.data.par_iter())
                .for_each(|(d, s)| *d = s.clone());
        } else {
            dst.data.clone_from_slice(&src.data);
        }
        Ok(())
    }

    /// parallel reduction.
    pub fn reduce_par<T, R>(&self, buf: &HostBuffer<T>, _op: R) -> Result<T, CpuError>
    where
        T: Clone + Send + Sync,
        R: xpu_core::reduce::Reduce<T>,
    {
        if buf.is_empty() {
            return Ok(R::identity());
        }

        let result = buf
            .data
            .par_iter()
            .cloned()
            .reduce(|| R::identity(), |a, b| R::combine(a, b));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let cpu = CpuDevice::new(0).unwrap();
        assert_eq!(cpu.id(), 0);
    }

    #[test]
    fn test_invalid_device_id() {
        let result = CpuDevice::new(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_allocation() {
        let cpu = CpuDevice::new(0).unwrap();
        let buf = cpu.alloc::<f64>(100).unwrap();
        assert_eq!(buf.len(), 100);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_init() {
        let cpu = CpuDevice::new(0).unwrap();
        let buf = cpu.alloc_init(50, 3.14).unwrap();
        assert_eq!(buf.len(), 50);
        assert!(buf.as_slice().iter().all(|&x| x == 3.14));
    }

    #[test]
    fn test_copy_to_device() {
        let cpu = CpuDevice::new(0).unwrap();
        let host_data: Vec<i32> = (0..10).collect();
        let mut device_buf = cpu.alloc::<i32>(10).unwrap();

        cpu.copy_to_device(&host_data, &mut device_buf).unwrap();
        assert_eq!(device_buf.as_slice(), host_data.as_slice());
    }

    #[test]
    fn test_copy_to_host() {
        let cpu = CpuDevice::new(0).unwrap();
        let device_buf = cpu.alloc_init(10, 42).unwrap();
        let mut host_data = vec![0; 10];

        cpu.copy_to_host(&device_buf, &mut host_data).unwrap();
        assert!(host_data.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_copy_size_mismatch() {
        let cpu = CpuDevice::new(0).unwrap();
        let host_data = vec![1, 2, 3];
        let mut device_buf = cpu.alloc::<i32>(5).unwrap();

        let result = cpu.copy_to_device(&host_data, &mut device_buf);
        assert!(result.is_err());
    }

    struct TestKernel;

    impl Kernel<&mut HostBuffer<i32>> for TestKernel {
        fn run(buf: &mut HostBuffer<i32>) {
            for val in buf.as_mut_slice() {
                *val *= 2;
            }
        }
    }

    #[test]
    fn test_kernel_launch() {
        let cpu = CpuDevice::new(0).unwrap();
        let mut buf = cpu.alloc_init(5, 10).unwrap();

        let config = LaunchConfig::new_1d(1, 1);
        cpu.launch(TestKernel, config, &mut buf).unwrap();

        assert!(buf.as_slice().iter().all(|&x| x == 20));
    }

    #[test]
    fn test_synchronize() {
        let cpu = CpuDevice::new(0).unwrap();
        assert!(cpu.synchronize().is_ok());
    }

    #[test]
    fn test_fill_buffer() {
        let cpu = CpuDevice::new(0).unwrap();
        let mut buf = cpu.alloc::<i32>(10).unwrap();

        cpu.fill(&mut buf, 42).unwrap();

        let mut result = vec![0; 10];
        cpu.copy_to_host(&buf, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_zero_buffer() {
        let cpu = CpuDevice::new(0).unwrap();
        let mut buf = cpu.alloc_init(10, 99).unwrap();

        cpu.zero(&mut buf).unwrap();

        let mut result = vec![99; 10];
        cpu.copy_to_host(&buf, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_copy_buffer() {
        let cpu = CpuDevice::new(0).unwrap();
        let src = cpu.alloc_init(5, 10).unwrap();
        let mut dst = cpu.alloc::<i32>(5).unwrap();

        cpu.copy_buffer(&src, &mut dst).unwrap();

        let mut result = vec![0; 5];
        cpu.copy_to_host(&dst, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 10));
    }

    #[test]
    fn test_copy_buffer_size_mismatch() {
        let cpu = CpuDevice::new(0).unwrap();
        let src = cpu.alloc_init(5, 10).unwrap();
        let mut dst = cpu.alloc::<i32>(3).unwrap();

        let result = cpu.copy_buffer(&src, &mut dst);
        assert!(result.is_err());
    }

    // =============================================================================
    // parallel cpu device tests
    // =============================================================================

    #[test]
    fn test_par_device_creation() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        assert_eq!(par_cpu.id, 0);
    }

    #[test]
    fn test_par_invalid_device_id() {
        let result = ParCpuDevice::new_parallel(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_par_buffer_allocation() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let buf = par_cpu.alloc_par::<f64>(1024).unwrap();
        assert_eq!(buf.len(), 1024);
    }

    #[test]
    fn test_par_buffer_init() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let buf = par_cpu.alloc_init_par(100, 42).unwrap();
        assert_eq!(buf.len(), 100);
        assert!(buf.as_slice().iter().all(|&x| x == 42));
    }

    #[test]
    fn test_par_copy_to_device() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let mut buf = par_cpu.alloc_par::<i32>(5).unwrap();
        let data = vec![1, 2, 3, 4, 5];

        par_cpu.copy_to_device_par(&data, &mut buf).unwrap();

        assert_eq!(buf.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_par_copy_to_host() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let buf = par_cpu.alloc_init_par(5, 99).unwrap();
        let mut result = vec![0; 5];

        par_cpu.copy_to_host_par(&buf, &mut result).unwrap();

        assert!(result.iter().all(|&x| x == 99));
    }

    #[test]
    fn test_par_copy_size_mismatch() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let data = vec![1, 2, 3];
        let mut buf = par_cpu.alloc_par::<i32>(5).unwrap();

        let result = par_cpu.copy_to_device_par(&data, &mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_par_fill_buffer() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let mut buf = par_cpu.alloc_par::<i32>(100).unwrap();

        par_cpu.fill_par(&mut buf, 42).unwrap();

        assert!(buf.as_slice().iter().all(|&x| x == 42));
    }

    #[test]
    fn test_par_copy_buffer() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let src = par_cpu.alloc_init_par(5, 10).unwrap();
        let mut dst = par_cpu.alloc_par::<i32>(5).unwrap();

        par_cpu.copy_buffer_par(&src, &mut dst).unwrap();

        let mut result = vec![0; 5];
        par_cpu.copy_to_host_par(&dst, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 10));
    }

    #[test]
    fn test_par_copy_buffer_size_mismatch() {
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let src = par_cpu.alloc_init_par(5, 10).unwrap();
        let mut dst = par_cpu.alloc_par::<i32>(3).unwrap();

        let result = par_cpu.copy_buffer_par(&src, &mut dst);
        assert!(result.is_err());
    }

    #[test]
    fn test_par_reduce_sum() {
        use xpu_core::reduce::Sum;

        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let data: Vec<i32> = (1..=100).collect();
        let mut buf = par_cpu.alloc_par::<i32>(100).unwrap();
        par_cpu.copy_to_device_par(&data, &mut buf).unwrap();

        let total = par_cpu.reduce_par(&buf, Sum).unwrap();
        assert_eq!(total, 5050); // 1+2+...+100 = 5050
    }

    #[test]
    fn test_par_reduce_max() {
        use xpu_core::reduce::Max;

        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let data = vec![3, 7, 2, 9, 1, 5];
        let mut buf = par_cpu.alloc_par::<i32>(data.len()).unwrap();
        par_cpu.copy_to_device_par(&data, &mut buf).unwrap();

        let max_val = par_cpu.reduce_par(&buf, Max).unwrap();
        assert_eq!(max_val, 9);
    }

    #[test]
    fn test_par_reduce_min() {
        use xpu_core::reduce::Min;

        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let data = vec![3, 7, 2, 9, 1, 5];
        let mut buf = par_cpu.alloc_par::<i32>(data.len()).unwrap();
        par_cpu.copy_to_device_par(&data, &mut buf).unwrap();

        let min_val = par_cpu.reduce_par(&buf, Min).unwrap();
        assert_eq!(min_val, 1);
    }

    #[test]
    fn test_par_reduce_empty_buffer() {
        use xpu_core::reduce::Sum;

        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let buf = par_cpu.alloc_par::<i32>(0).unwrap();

        let total = par_cpu.reduce_par(&buf, Sum).unwrap();
        assert_eq!(total, 0); // identity for sum
    }

    #[test]
    fn test_par_reduce_custom_operation() {
        use xpu_core::reduce::Product;

        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let data: Vec<i32> = vec![2, 3, 4];
        let mut buf = par_cpu.alloc_par::<i32>(data.len()).unwrap();
        par_cpu.copy_to_device_par(&data, &mut buf).unwrap();

        let product = par_cpu.reduce_par(&buf, Product).unwrap();
        assert_eq!(product, 24); // 2*3*4 = 24
    }

    #[test]
    fn test_par_large_buffer_performance() {
        // test that parallel operations work on large buffers
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let size = 100_000;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let mut buf = par_cpu.alloc_par::<f64>(size).unwrap();

        par_cpu.copy_to_device_par(&data, &mut buf).unwrap();
        par_cpu.fill_par(&mut buf, 1.0).unwrap();

        let mut result = vec![0.0; size];
        par_cpu.copy_to_host_par(&buf, &mut result).unwrap();

        assert!(result.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_buffer_view_1d() {
        use xpu_core::DeviceBufferExt;

        let cpu = CpuDevice::new(0).unwrap();
        let buf = cpu.alloc_init(10, 42).unwrap();

        let view = buf.view_1d();
        assert_eq!(view.len(), 10);
        assert_eq!(view.shape(), [10]);
        unsafe {
            assert_eq!(*view.get_unchecked([0]), 42);
            assert_eq!(*view.get_unchecked([9]), 42);
        }
    }

    #[test]
    fn test_buffer_view_2d() {
        use xpu_core::DeviceBufferExt;

        let cpu = CpuDevice::new(0).unwrap();
        let data: Vec<i32> = (0..12).collect();
        let mut buf = cpu.alloc::<i32>(12).unwrap();
        cpu.copy_to_device(&data, &mut buf).unwrap();

        // create 3x4 view (3 rows, 4 columns)
        let view = buf.view_2d([3, 4]).unwrap();
        assert_eq!(view.shape(), [3, 4]);

        // row-major layout: [row*ncols + col]
        unsafe {
            assert_eq!(*view.get_unchecked([0, 0]), 0);
            assert_eq!(*view.get_unchecked([0, 3]), 3);
            assert_eq!(*view.get_unchecked([1, 0]), 4);
            assert_eq!(*view.get_unchecked([2, 3]), 11);
        }
    }

    #[test]
    fn test_buffer_view_3d() {
        use xpu_core::DeviceBufferExt;

        let cpu = CpuDevice::new(0).unwrap();
        let data: Vec<i32> = (1..=24).collect(); // 2x3x4 cube
        let mut buf = cpu.alloc::<i32>(24).unwrap();
        cpu.copy_to_device(&data, &mut buf).unwrap();

        let view = buf.view_3d([2, 3, 4]).unwrap();
        assert_eq!(view.shape(), [2, 3, 4]);
        assert_eq!(view.strides(), [12, 4, 1]); // row-major: [ny*nx, nx, 1]

        // verify corner elements
        unsafe {
            assert_eq!(*view.get_unchecked([0, 0, 0]), 1);
            assert_eq!(*view.get_unchecked([0, 0, 3]), 4);
            assert_eq!(*view.get_unchecked([1, 2, 3]), 24);
        }
    }

    #[test]
    fn test_buffer_view_mut() {
        use xpu_core::DeviceBufferExt;

        let cpu = CpuDevice::new(0).unwrap();
        let mut buf = cpu.alloc::<i32>(6).unwrap();

        {
            let mut view = buf.view_mut_2d([2, 3]).unwrap();
            unsafe {
                *view.get_unchecked_mut([0, 0]) = 10;
                *view.get_unchecked_mut([1, 2]) = 20;
            }
        }

        // verify mutations persisted
        let mut result = vec![0; 6];
        cpu.copy_to_host(&buf, &mut result).unwrap();
        assert_eq!(result[0], 10);
        assert_eq!(result[5], 20);
    }

    #[test]
    fn test_record_event() {
        let cpu = CpuDevice::new(0).unwrap();
        let token = cpu.record_event().unwrap();

        // cpu tokens are always immediate
        assert!(token.is_immediate());
        assert!(token.ready().unwrap());
    }

    #[test]
    fn test_token_wait() {
        let cpu = CpuDevice::new(0).unwrap();
        let token = cpu.record_event().unwrap();

        // waiting on cpu token is a no-op
        token.wait().unwrap();
    }

    #[test]
    fn test_async_fill_with_token() {
        let cpu = CpuDevice::new(0).unwrap();
        let mut buf = cpu.alloc::<i32>(10).unwrap();

        // fill buffer
        cpu.fill(&mut buf, 42).unwrap();

        // record completion
        let token = cpu.record_event().unwrap();

        // wait for completion
        token.wait().unwrap();

        // verify
        let mut result = vec![0; 10];
        cpu.copy_to_host(&buf, &mut result).unwrap();
        assert!(result.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_reduce_sum() {
        let cpu = CpuDevice::new(0).unwrap();
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut buf = cpu.alloc::<i32>(5).unwrap();
        cpu.copy_to_device(&data, &mut buf).unwrap();

        let total = cpu.sum(&buf).unwrap();
        assert_eq!(total, 15);
    }

    #[test]
    fn test_reduce_max() {
        let cpu = CpuDevice::new(0).unwrap();
        let data: Vec<f64> = vec![1.5, 9.2, 3.7, 2.1, 8.4];
        let mut buf = cpu.alloc::<f64>(5).unwrap();
        cpu.copy_to_device(&data, &mut buf).unwrap();

        let maximum = cpu.max(&buf).unwrap();
        assert_eq!(maximum, 9.2);
    }

    #[test]
    fn test_reduce_min() {
        let cpu = CpuDevice::new(0).unwrap();
        let data: Vec<f64> = vec![1.5, 9.2, 3.7, 0.3, 8.4];
        let mut buf = cpu.alloc::<f64>(5).unwrap();
        cpu.copy_to_device(&data, &mut buf).unwrap();

        let minimum = cpu.min(&buf).unwrap();
        assert_eq!(minimum, 0.3);
    }

    #[test]
    fn test_reduce_empty_buffer() {
        let cpu = CpuDevice::new(0).unwrap();
        let buf = cpu.alloc::<i32>(0).unwrap();

        let total = cpu.sum(&buf).unwrap();
        assert_eq!(total, 0); // identity for sum
    }

    #[test]
    fn test_reduce_custom_operation() {
        use xpu_core::reduce::Product;

        let cpu = CpuDevice::new(0).unwrap();
        let data: Vec<i32> = vec![2, 3, 4];
        let mut buf = cpu.alloc::<i32>(3).unwrap();
        cpu.copy_to_device(&data, &mut buf).unwrap();

        let product = cpu.reduce(&buf, Product).unwrap();
        assert_eq!(product, 24); // 2 * 3 * 4
    }
}
