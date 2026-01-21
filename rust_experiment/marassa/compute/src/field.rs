// =============================================================================
// field.rs
//
// field and field view types for grid-based data storage.
// field owns device buffers, fieldview provides non-owning references.
//
// design:
//   - field<'d, T, D, const N: usize> owns data on device D
//   - fieldview<'a, T, const N: usize> borrows from field (lifetime 'a)
//   - views can be converted to computations (lazy evaluation)
//   - fields can be created from computations via evaluation
//
// usage:
//   let field = Field::new(device, domain)?;
//   let view = field.view();
//   let computation = view.as_computation();
// =============================================================================

use crate::computation::Computation;
use crate::domain::Domain;
use xpu_core::{Device, DeviceBufferExt};

/// field: owns grid data on a device.
/// generic over device type D and element type T.
pub struct Field<'d, T, D: Device, const N: usize> {
    buffer: D::Buffer<T>,
    domain: Domain<N>,
    device: &'d D,
}

impl<'d, T, D: Device, const N: usize> Field<'d, T, D, N>
where
    T: Default + Clone,
{
    /// creates a new field with uninitialized data.
    pub fn new(device: &'d D, domain: Domain<N>) -> Result<Self, D::Error> {
        let buffer = device.alloc(domain.size())?;
        Ok(Self {
            buffer,
            domain,
            device,
        })
    }

    /// creates a field initialized to a constant value.
    pub fn filled(device: &'d D, domain: Domain<N>, value: T) -> Result<Self, D::Error> {
        let mut field = Self::new(device, domain)?;
        device.fill(&mut field.buffer, value)?;
        Ok(field)
    }

    /// creates a field initialized to zero/default value.
    pub fn zeros(device: &'d D, domain: Domain<N>) -> Result<Self, D::Error> {
        let mut field = Self::new(device, domain)?;
        device.zero(&mut field.buffer)?;
        Ok(field)
    }

    /// returns the domain this field is defined over.
    pub fn domain(&self) -> Domain<N> {
        self.domain
    }

    /// returns a reference to the underlying device buffer.
    pub fn buffer(&self) -> &D::Buffer<T> {
        &self.buffer
    }

    /// returns a mutable reference to the underlying device buffer.
    pub fn buffer_mut(&mut self) -> &mut D::Buffer<T> {
        &mut self.buffer
    }

    /// returns a reference to the device.
    pub fn device(&self) -> &D {
        self.device
    }

    /// creates a non-owning view of this field.
    pub fn view(&self) -> FieldView<'_, T, N> {
        FieldView {
            view: self.buffer.view_1d(),
            domain: self.domain,
        }
    }

    /// creates a mutable non-owning view of this field.
    pub fn view_mut(&mut self) -> FieldViewMut<'_, T, N> {
        FieldViewMut {
            view: self.buffer.view_mut_1d(),
            domain: self.domain,
        }
    }

    /// clones the field data (deep copy on device).
    pub fn clone_data(&self) -> Result<Self, D::Error> {
        let mut new_field = Self::new(self.device, self.domain)?;
        self.device
            .copy_buffer(&self.buffer, &mut new_field.buffer)?;
        Ok(new_field)
    }

    /// copies data to host memory.
    pub fn to_host(&self) -> Result<Vec<T>, D::Error> {
        let mut host_data = vec![T::default(); self.domain.size()];
        self.device.copy_to_host(&self.buffer, &mut host_data)?;
        Ok(host_data)
    }

    /// copies data from host memory.
    pub fn from_host(&mut self, host_data: &[T]) -> Result<(), D::Error> {
        self.device.copy_to_device(host_data, &mut self.buffer)
    }
}

/// fieldview: non-owning reference to field data.
/// lifetime 'a ties view to the field it borrows from.
pub struct FieldView<'a, T, const N: usize> {
    view: xpu_core::View1<'a, T>,
    domain: Domain<N>,
}

impl<'a, T, const N: usize> FieldView<'a, T, N> {
    /// returns the domain this view is defined over.
    pub fn domain(&self) -> Domain<N> {
        self.domain
    }

    /// evaluates the view at a coordinate.
    pub fn eval(&self, coord: [i64; N]) -> &T {
        let linear = self.domain.coord_to_linear(coord);
        unsafe { self.view.get_unchecked([linear]) }
    }

    /// converts view to a lazy computation.
    /// note: captures raw pointer, so computation is only valid while field lives.
    pub fn as_computation(&self) -> Computation<T, N, impl Fn([i64; N]) -> T + 'a>
    where
        T: Clone,
    {
        let ptr = self.view.as_ptr();
        let domain = self.domain;

        let func = move |coord: [i64; N]| {
            let linear = domain.coord_to_linear(coord);
            unsafe { (*ptr.add(linear)).clone() }
        };

        Computation::new(func, self.domain)
    }
}

/// mutable fieldview: mutable non-owning reference to field data.
pub struct FieldViewMut<'a, T, const N: usize> {
    view: xpu_core::ViewMut1<'a, T>,
    domain: Domain<N>,
}

impl<'a, T, const N: usize> FieldViewMut<'a, T, N> {
    /// returns the domain this view is defined over.
    pub fn domain(&self) -> Domain<N> {
        self.domain
    }

    /// evaluates the view at a coordinate (immutable).
    pub fn eval(&self, coord: [i64; N]) -> &T {
        let linear = self.domain.coord_to_linear(coord);
        unsafe { self.view.get_unchecked([linear]) }
    }

    /// evaluates the view at a coordinate (mutable).
    pub fn eval_mut(&mut self, coord: [i64; N]) -> &mut T {
        let linear = self.domain.coord_to_linear(coord);
        unsafe { self.view.get_unchecked_mut([linear]) }
    }

    /// sets value at a coordinate.
    pub fn set(&mut self, coord: [i64; N], value: T)
    where
        T: Clone,
    {
        let linear = self.domain.coord_to_linear(coord);
        unsafe {
            *self.view.get_unchecked_mut([linear]) = value;
        }
    }

    /// converts to immutable view.
    pub fn as_view(&self) -> FieldView<'_, T, N> {
        FieldView {
            view: self.view.as_view(),
            domain: self.domain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xpu_host::CpuDevice;

    #[test]
    fn test_field_creation() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10, 10]);
        let field = Field::<f64, _, 2>::new(&device, domain).unwrap();

        assert_eq!(field.domain(), domain);
    }

    #[test]
    fn test_field_filled() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([5, 5]);
        let field = Field::<f64, _, 2>::filled(&device, domain, 3.14).unwrap();

        let data = field.to_host().unwrap();
        assert!(data.iter().all(|&x| x == 3.14));
    }

    #[test]
    fn test_field_zeros() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([5, 5]);
        let field = Field::<f64, _, 2>::zeros(&device, domain).unwrap();

        let data = field.to_host().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_field_view() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([3, 3]);
        let mut field = Field::<i32, _, 2>::zeros(&device, domain).unwrap();

        {
            let mut view = field.view_mut();
            view.set([0, 0], 42);
            view.set([1, 1], 99);
        }

        let view = field.view();
        assert_eq!(*view.eval([0, 0]), 42);
        assert_eq!(*view.eval([1, 1]), 99);
    }

    #[test]
    fn test_field_to_host() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([2, 2]);
        let mut field = Field::<i32, _, 2>::zeros(&device, domain).unwrap();

        let data = vec![1, 2, 3, 4];
        field.from_host(&data).unwrap();

        let result = field.to_host().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_field_clone() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([5]);
        let field = Field::<f64, _, 1>::filled(&device, domain, 1.5).unwrap();

        let cloned = field.clone_data().unwrap();
        let data = cloned.to_host().unwrap();
        assert!(data.iter().all(|&x| x == 1.5));
    }

    #[test]
    fn test_view_as_computation() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([3]);
        let mut field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        field.from_host(&[1.0, 2.0, 3.0]).unwrap();

        let view = field.view();
        let comp = view.as_computation();

        assert_eq!(comp.eval([0]), 1.0);
        assert_eq!(comp.eval([1]), 2.0);
        assert_eq!(comp.eval([2]), 3.0);
    }
}
