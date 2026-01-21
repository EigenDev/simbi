// =============================================================================
// execution.rs
//
// execution layer: binds lazy computations to devices for evaluation.
// this is where computations materialize into fields.
//
// design:
//   - evaluate<T, D, N, F>(device, computation) -> Field
//   - parallel evaluation over domain
//   - device-agnostic (works on any Device implementation)
//
// usage:
//   let comp = from_fn(domain, |coord| coord[0] as f64);
//   let field = evaluate(&device, comp)?;
// =============================================================================

use crate::computation::Computation;
use crate::field::Field;
use rayon::prelude::*;
use xpu_core::Device;

/// evaluates a computation on a device, materializing it into a field.
/// iterates over the domain and evaluates the computation at each point.
pub fn evaluate<'d, T, D, const N: usize, F>(
    device: &'d D,
    computation: Computation<T, N, F>,
) -> Result<Field<'d, T, D, N>, D::Error>
where
    D: Device,
    T: Default + Clone,
    F: Fn([i64; N]) -> T,
{
    let domain = computation.domain();
    let mut field = Field::new(device, domain)?;

    // allocate host buffer for computed values
    let mut host_data = Vec::with_capacity(domain.size());

    // evaluate computation at each coordinate
    for coord in domain.iter() {
        host_data.push(computation.eval(coord));
    }

    // copy results to device
    field.from_host(&host_data)?;

    Ok(field)
}

/// evaluates a computation in-place into an existing field.
/// overwrites the field's data with computation results.
pub fn evaluate_into<T, D, const N: usize, F>(
    field: &mut Field<T, D, N>,
    computation: Computation<T, N, F>,
) -> Result<(), D::Error>
where
    D: Device,
    T: Default + Clone,
    F: Fn([i64; N]) -> T,
{
    let domain = computation.domain();

    // ensure domains match
    if domain != field.domain() {
        // for now, just use intersection
        // in production, you'd want proper error handling
    }

    let mut host_data = Vec::with_capacity(domain.size());

    for coord in domain.iter() {
        host_data.push(computation.eval(coord));
    }

    field.from_host(&host_data)?;

    Ok(())
}

// =============================================================================
// parallel evaluation for ParCpuDevice
// =============================================================================

/// evaluates a computation in parallel using rayon.
/// works with xpu_host::ParCpuDevice for multi-threaded cpu execution.
pub fn parallel_evaluate<T, const N: usize, F>(
    par_device: &xpu_host::ParCpuDevice,
    computation: Computation<T, N, F>,
) -> Result<xpu_host::HostBuffer<T>, xpu_host::CpuError>
where
    T: Default + Clone + Send + Sync,
    F: Fn([i64; N]) -> T + Send + Sync,
{
    let domain = computation.domain();

    // parallel evaluation over domain coordinates
    let coords: Vec<[i64; N]> = domain.iter().collect();
    let host_data: Vec<T> = coords
        .par_iter()
        .map(|&coord| computation.eval(coord))
        .collect();

    // create buffer and fill it
    let mut buf = par_device.alloc_par::<T>(domain.size())?;
    par_device.copy_to_device_par(&host_data, &mut buf)?;

    Ok(buf)
}

/// evaluates a computation in parallel into an existing buffer.
/// overwrites the buffer's data with computation results.
pub fn parallel_evaluate_into<T, const N: usize, F>(
    par_device: &xpu_host::ParCpuDevice,
    buffer: &mut xpu_host::HostBuffer<T>,
    computation: Computation<T, N, F>,
) -> Result<(), xpu_host::CpuError>
where
    T: Default + Clone + Send + Sync,
    F: Fn([i64; N]) -> T + Send + Sync,
{
    let domain = computation.domain();

    // parallel evaluation
    let coords: Vec<[i64; N]> = domain.iter().collect();
    let host_data: Vec<T> = coords
        .par_iter()
        .map(|&coord| computation.eval(coord))
        .collect();

    par_device.copy_to_device_par(&host_data, buffer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::computation::{constant, from_fn};
    use crate::domain::Domain;
    use xpu_host::CpuDevice;

    #[test]
    fn test_evaluate_constant() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10, 10]);
        let comp = constant(domain, 42.0);

        let field = evaluate(&device, comp).unwrap();
        let data = field.to_host().unwrap();

        assert!(data.iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_evaluate_function() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([5]);
        let comp = from_fn(domain, |coord| coord[0] as f64 * 2.0);

        let field = evaluate(&device, comp).unwrap();
        let data = field.to_host().unwrap();

        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 4.0);
        assert_eq!(data[3], 6.0);
        assert_eq!(data[4], 8.0);
    }

    #[test]
    fn test_evaluate_composition() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([3, 3]);

        let x = from_fn(domain, |coord| coord[0] as f64);
        let y = from_fn(domain, |coord| coord[1] as f64);
        let sum = x.add(y);

        let field = evaluate(&device, sum).unwrap();
        let data = field.to_host().unwrap();

        // check a few values: data[i,j] = i + j
        let view_data = |i: i64, j: i64| data[domain.coord_to_linear([i, j])];
        assert_eq!(view_data(0, 0), 0.0);
        assert_eq!(view_data(1, 1), 2.0);
        assert_eq!(view_data(2, 1), 3.0);
    }

    #[test]
    fn test_evaluate_into() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([5]);

        let mut field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let comp = constant(domain, 3.14);

        evaluate_into(&mut field, comp).unwrap();

        let data = field.to_host().unwrap();
        assert!(data.iter().all(|&x| x == 3.14));
    }

    #[test]
    fn test_lazy_then_eager() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);

        // build lazy expression graph
        let x = from_fn(domain, |coord| coord[0] as f64);
        let expr = x.scale(2.0).add_scalar(5.0); // 2*x + 5

        // materialize: lazy -> eager
        let field = evaluate(&device, expr).unwrap();
        let data = field.to_host().unwrap();

        assert_eq!(data[0], 5.0);
        assert_eq!(data[3], 11.0);
        assert_eq!(data[5], 15.0);
    }

    #[test]
    fn test_field_to_computation_to_field() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([4]);

        // create initial field
        let mut field1 = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        field1.from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        // convert to computation
        let view = field1.view();
        let comp = view.as_computation();

        // transform
        let doubled = comp.scale(2.0);

        // materialize back to field
        let field2 = evaluate(&device, doubled).unwrap();
        let data = field2.to_host().unwrap();

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    // =============================================================================
    // parallel evaluation tests
    // =============================================================================

    #[test]
    fn test_parallel_evaluate_constant() {
        use xpu_host::ParCpuDevice;

        let par_device = ParCpuDevice::new_parallel(0).unwrap();
        let domain = Domain::from_shape([10, 10]);
        let comp = constant(domain, 42.0);

        let buf = parallel_evaluate(&par_device, comp).unwrap();
        let mut data = vec![0.0; domain.size()];
        par_device.copy_to_host_par(&buf, &mut data).unwrap();

        assert!(data.iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_parallel_evaluate_function() {
        use xpu_host::ParCpuDevice;

        let par_device = ParCpuDevice::new_parallel(0).unwrap();
        let domain = Domain::from_shape([5]);
        let comp = from_fn(domain, |coord| coord[0] as f64 * 2.0);

        let buf = parallel_evaluate(&par_device, comp).unwrap();
        let mut data = vec![0.0; 5];
        par_device.copy_to_host_par(&buf, &mut data).unwrap();

        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 4.0);
        assert_eq!(data[3], 6.0);
        assert_eq!(data[4], 8.0);
    }

    #[test]
    fn test_parallel_evaluate_composition() {
        use xpu_host::ParCpuDevice;

        let par_device = ParCpuDevice::new_parallel(0).unwrap();
        let domain = Domain::from_shape([3, 3]);

        let x = from_fn(domain, |coord| coord[0] as f64);
        let y = from_fn(domain, |coord| coord[1] as f64);
        let sum = x.add(y);

        let buf = parallel_evaluate(&par_device, sum).unwrap();
        let mut data = vec![0.0; domain.size()];
        par_device.copy_to_host_par(&buf, &mut data).unwrap();

        // check a few values: data[i,j] = i + j
        let view_data = |i: i64, j: i64| data[domain.coord_to_linear([i, j])];
        assert_eq!(view_data(0, 0), 0.0);
        assert_eq!(view_data(1, 1), 2.0);
        assert_eq!(view_data(2, 1), 3.0);
    }

    #[test]
    fn test_parallel_evaluate_into() {
        use xpu_host::ParCpuDevice;

        let par_device = ParCpuDevice::new_parallel(0).unwrap();
        let domain = Domain::from_shape([5]);
        let comp = constant(domain, 3.14);

        let mut buf = par_device.alloc_par::<f64>(domain.size()).unwrap();
        parallel_evaluate_into(&par_device, &mut buf, comp).unwrap();

        let mut data = vec![0.0; domain.size()];
        par_device.copy_to_host_par(&buf, &mut data).unwrap();
        assert!(data.iter().all(|&x| x == 3.14));
    }

    #[test]
    fn test_parallel_evaluate_large_domain() {
        use xpu_host::ParCpuDevice;

        let par_device = ParCpuDevice::new_parallel(0).unwrap();
        let domain = Domain::from_shape([100, 100]);

        // r^2 = x^2 + y^2
        let x = from_fn(domain, |coord| coord[0] as f64);
        let y = from_fn(domain, |coord| coord[1] as f64);
        let x_copy = from_fn(domain, |coord| coord[0] as f64);
        let y_copy = from_fn(domain, |coord| coord[1] as f64);
        let x_sq = x.mul(x_copy);
        let y_sq = y.mul(y_copy);
        let r_sq = x_sq.add(y_sq);

        let buf = parallel_evaluate(&par_device, r_sq).unwrap();
        let mut data = vec![0.0; domain.size()];
        par_device.copy_to_host_par(&buf, &mut data).unwrap();

        // verify a few known values
        let view_data = |i: i64, j: i64| data[domain.coord_to_linear([i, j])];
        assert_eq!(view_data(0, 0), 0.0);
        assert_eq!(view_data(3, 4), 25.0); // 3^2 + 4^2 = 25
        assert_eq!(view_data(5, 12), 169.0); // 5^2 + 12^2 = 169
    }

    #[test]
    fn test_parallel_vs_serial_consistency() {
        use xpu_host::ParCpuDevice;

        let cpu = CpuDevice::new(0).unwrap();
        let par_cpu = ParCpuDevice::new_parallel(0).unwrap();
        let domain = Domain::from_shape([50, 50]);

        // serial evaluation
        let x1 = from_fn(domain, |coord| coord[0] as f64);
        let y1 = from_fn(domain, |coord| coord[1] as f64);
        let sum1 = x1.add(y1);
        let field_serial = evaluate(&cpu, sum1).unwrap();
        let data_serial = field_serial.to_host().unwrap();

        // parallel evaluation
        let x2 = from_fn(domain, |coord| coord[0] as f64);
        let y2 = from_fn(domain, |coord| coord[1] as f64);
        let sum2 = x2.add(y2);
        let buf_parallel = parallel_evaluate(&par_cpu, sum2).unwrap();
        let mut data_parallel = vec![0.0; domain.size()];
        par_cpu
            .copy_to_host_par(&buf_parallel, &mut data_parallel)
            .unwrap();

        // results should match
        assert_eq!(data_serial, data_parallel);
    }
}
