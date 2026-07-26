// =============================================================================
// tracer_device.rs
//
// gpu execution of the continuous-tracer affine cic/ito update on uniform,
// logarithmic, and geometrically graded logical grids. cuda and hip consume
// the same runtime-compiled kernel and soa argument layout.
// =============================================================================

use symbi_xpu::runtime::{GpuRuntime, current_dispatcher};
use symbi_xpu::{KernelArgs, LaunchConfig, MemorySpace};

use crate::tracers::{ContinuousTracerSet, ItoCoefficientFields};

const CONTINUOUS_TRACER_KERNEL: &str = r#"
extern "C" __device__ unsigned long long tracer_mix64(unsigned long long x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

extern "C" __device__ double tracer_unit(
    unsigned long long seed,
    unsigned long long id,
    unsigned long long counter,
    unsigned int axis)
{
    unsigned long long bits = tracer_mix64(
        seed ^ tracer_mix64(id) ^ tracer_mix64(counter) ^
        tracer_mix64((unsigned long long)axis) ^ 0x243f6a8885a308d3ULL);
    return (double)(bits >> 11) * 0x1.0p-53;
}

extern "C" __global__ void continuous_tracer_advance(
    double* x0, double* x1, double* x2,
    double* step0, double* step1, double* step2,
    const unsigned long long* id,
    const unsigned char* escaped,
    const unsigned char* crossed,
    unsigned long long* counter,
    const double* drift0, const double* drift1, const double* drift2,
    const double* variance0, const double* variance1, const double* variance2,
    const double* third0, const double* third1, const double* third2,
    unsigned int n, unsigned int ndim, unsigned int order,
    unsigned long long seed,
    int alo0, int alo1, int alo2,
    unsigned int shape0, unsigned int shape1, unsigned int shape2,
    unsigned int map_kind0, unsigned int map_kind1, unsigned int map_kind2,
    double map_p00, double map_p01, double map_p02,
    double map_p10, double map_p11, double map_p12,
    double map_p20, double map_p21, double map_p22,
    double xlo0, double xlo1, double xlo2,
    double dx0, double dx1, double dx2,
    double scale_start, double scale_end,
    double offset_start0, double offset_start1, double offset_start2,
    double offset_end0, double offset_end1, double offset_end2,
    double dt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || escaped[i] || crossed[i]) return;
    double* x[3] = {x0, x1, x2};
    double* step[3] = {step0, step1, step2};
    const double* drift[3] = {drift0, drift1, drift2};
    const double* variance[3] = {variance0, variance1, variance2};
    const double* third[3] = {third0, third1, third2};
    int alo[3] = {alo0, alo1, alo2};
    unsigned int shape[3] = {shape0, shape1, shape2};
    unsigned int map_kind[3] = {map_kind0, map_kind1, map_kind2};
    double map_p0[3] = {map_p00, map_p01, map_p02};
    double map_p1[3] = {map_p10, map_p11, map_p12};
    double map_p2[3] = {map_p20, map_p21, map_p22};
    double xlo[3] = {xlo0, xlo1, xlo2};
    double dx[3] = {dx0, dx1, dx2};
    double offset_start[3] = {offset_start0, offset_start1, offset_start2};
    double offset_end[3] = {offset_end0, offset_end1, offset_end2};
    double logical[3] = {0.0, 0.0, 0.0};
    int lower[3] = {0, 0, 0};
    double upper_weight[3] = {0.0, 0.0, 0.0};
    for (unsigned int axis = 0; axis < ndim; ++axis) {
        step[axis][i] = x[axis][i];
        logical[axis] = (x[axis][i] - offset_start[axis]) / scale_start;
        int containing;
        double containing_center;
        if (map_kind[axis] == 1u) {
            containing = (int)floor(log10(logical[axis] / map_p0[axis]) / map_p1[axis]);
            containing_center = map_p0[axis] *
                pow(10.0, ((double)containing + 0.5) * map_p1[axis]);
        } else if (map_kind[axis] == 2u) {
            double ratio = map_p2[axis];
            if (fabs(ratio - 1.0) < 1.0e-12) {
                containing = (int)floor((logical[axis] - map_p0[axis]) / map_p1[axis]);
            } else {
                containing = (int)floor(
                    log(1.0 + (logical[axis] - map_p0[axis]) *
                        (ratio - 1.0) / map_p1[axis]) / log(ratio));
            }
            double face0 = fabs(ratio - 1.0) < 1.0e-12
                ? map_p0[axis] + (double)containing * map_p1[axis]
                : map_p0[axis] + map_p1[axis] *
                    (pow(ratio, (double)containing) - 1.0) / (ratio - 1.0);
            double face1 = fabs(ratio - 1.0) < 1.0e-12
                ? face0 + map_p1[axis]
                : map_p0[axis] + map_p1[axis] *
                    (pow(ratio, (double)containing + 1.0) - 1.0) / (ratio - 1.0);
            containing_center = 0.5 * (face0 + face1);
        } else {
            containing = (int)floor((logical[axis] - xlo[axis]) / dx[axis]);
            containing_center = xlo[axis] + ((double)containing + 0.5) * dx[axis];
        }
        int base = logical[axis] < containing_center ? containing - 1 : containing;
        int high_base = alo[axis] + (int)shape[axis] - 2;
        base = base < alo[axis] ? alo[axis] : base;
        base = base > high_base ? high_base : base;
        lower[axis] = base;
        double center0;
        double center1;
        if (map_kind[axis] == 1u) {
            center0 = map_p0[axis] * pow(10.0, ((double)base + 0.5) * map_p1[axis]);
            center1 = map_p0[axis] * pow(10.0, ((double)base + 1.5) * map_p1[axis]);
        } else if (map_kind[axis] == 2u) {
            double ratio = map_p2[axis];
            if (fabs(ratio - 1.0) < 1.0e-12) {
                center0 = map_p0[axis] + ((double)base + 0.5) * map_p1[axis];
                center1 = center0 + map_p1[axis];
            } else {
                double face0 = map_p0[axis] + map_p1[axis] *
                    (pow(ratio, (double)base) - 1.0) / (ratio - 1.0);
                double face1 = map_p0[axis] + map_p1[axis] *
                    (pow(ratio, (double)base + 1.0) - 1.0) / (ratio - 1.0);
                double face2 = map_p0[axis] + map_p1[axis] *
                    (pow(ratio, (double)base + 2.0) - 1.0) / (ratio - 1.0);
                center0 = 0.5 * (face0 + face1);
                center1 = 0.5 * (face1 + face2);
            }
        } else {
            center0 = xlo[axis] + ((double)base + 0.5) * dx[axis];
            center1 = center0 + dx[axis];
        }
        upper_weight[axis] = fmin(
            1.0,
            fmax(0.0, (logical[axis] - center0) / (center1 - center0)));
    }
    double rates[3][3] = {{0.0}};
    unsigned int corners = 1u << ndim;
    for (unsigned int corner = 0; corner < corners; ++corner) {
        unsigned int flat = 0;
        unsigned int stride = 1;
        double weight = 1.0;
        for (unsigned int axis = 0; axis < ndim; ++axis) {
            unsigned int upper = (corner >> axis) & 1u;
            int coord = lower[axis] + (int)upper;
            flat += (unsigned int)(coord - alo[axis]) * stride;
            stride *= shape[axis];
            weight *= upper ? upper_weight[axis] : 1.0 - upper_weight[axis];
        }
        for (unsigned int axis = 0; axis < ndim; ++axis) {
            rates[axis][0] += weight * drift[axis][flat];
            rates[axis][1] += weight * variance[axis][flat];
            rates[axis][2] += weight * third[axis][flat];
        }
    }
    for (unsigned int axis = 0; axis < ndim; ++axis) {
        double unit = tracer_unit(seed, id[i], counter[i], axis);
        double variance_dt = fmax(0.0, rates[axis][1] * dt);
        double standardized;
        if (order == 2u) {
            standardized = sqrt(12.0) * (unit - 0.5);
        } else {
            double skewness = variance_dt == 0.0
                ? 0.0
                : rates[axis][2] * dt / pow(variance_dt, 1.5);
            double root = hypot(sqrt(27.0), 2.0 * skewness);
            double left;
            double right;
            if (skewness >= 0.0) {
                right = (root + 2.0 * skewness) / 3.0;
                left = 3.0 / right;
            } else {
                left = (root - 2.0 * skewness) / 3.0;
                right = 3.0 / left;
            }
            double sum = left + right;
            double left_density = right / (left * sum);
            double right_density = left / (right * sum);
            double left_mass = right / sum;
            standardized = unit < left_mass
                ? -left + unit / left_density
                : (unit - left_mass) / right_density;
        }
        double displacement = rates[axis][0] * dt + sqrt(variance_dt) * standardized;
        x[axis][i] = scale_end * logical[axis] + offset_end[axis] + displacement;
    }
    counter[i] += 1ULL;
}
"#;

pub(crate) fn advance_device<const D: usize, Mem: MemorySpace>(
    tracers: &mut ContinuousTracerSet<D, Mem>,
    coefficients: &ItoCoefficientFields<D, Mem>,
    geometry: &crate::state::PartitionGeometry<D>,
    scale_start: f64,
    scale_end: f64,
    offset_start: [f64; D],
    offset_end: [f64; D],
    dt: f64,
) -> Result<(), String> {
    if !(1..=3).contains(&D) {
        return Err(format!(
            "continuous tracer device update supports one to three dimensions, got {D}"
        ));
    }
    if tracers.len == 0 {
        return Ok(());
    }
    let x_ptr: [u64; 3] =
        std::array::from_fn(|dd| tracers.x[dd.min(D - 1)].as_mut_ptr::<f64>() as u64);
    let step_ptr: [u64; 3] =
        std::array::from_fn(|dd| tracers.step_x[dd.min(D - 1)].as_mut_ptr::<f64>() as u64);
    let coefficient_ptr = |fields: &[symbi_grid::Field<f64, D, Mem>; D]| -> [u64; 3] {
        std::array::from_fn(|dd| fields[dd.min(D - 1)].as_ptr() as u64)
    };
    let drift_ptr = coefficient_ptr(&coefficients.drift);
    let variance_ptr = coefficient_ptr(&coefficients.variance);
    let third_ptr = coefficient_ptr(&coefficients.third);
    let id_ptr = tracers.id.as_ptr::<u64>() as u64;
    let escaped_ptr = tracers.escaped.as_ptr::<u8>() as u64;
    let crossed_ptr = tracers.crossed_sink.as_ptr::<u8>() as u64;
    let counter_ptr = tracers.random_counter.as_mut_ptr::<u64>() as u64;
    let n = tracers.len as u32;
    let ndim = D as u32;
    let order = tracers.order as u32;
    let seed = tracers.run_seed;
    let allocated_lo: [i32; 3] = std::array::from_fn(|dd| {
        if dd < D {
            geometry.allocated.spaces[dd].lo as i32
        } else {
            0
        }
    });
    let shape: [u32; 3] = std::array::from_fn(|dd| {
        if dd < D {
            geometry.allocated.spaces[dd].size() as u32
        } else {
            1
        }
    });
    let mut map_kind = [0u32; 3];
    let mut map_p0 = [0.0; 3];
    let mut map_p1 = [0.0; 3];
    let mut map_p2 = [0.0; 3];
    if let Some(maps) = geometry.maps {
        for dd in 0..D {
            match maps[dd] {
                symbi_geometry::AxisMap::Uniform { start, dx } => {
                    map_p0[dd] = start;
                    map_p1[dd] = dx;
                }
                symbi_geometry::AxisMap::Log { start, log_slope } => {
                    map_kind[dd] = 1;
                    map_p0[dd] = start;
                    map_p1[dd] = log_slope;
                }
                symbi_geometry::AxisMap::Geometric {
                    start,
                    width,
                    ratio,
                } => {
                    map_kind[dd] = 2;
                    map_p0[dd] = start;
                    map_p1[dd] = width;
                    map_p2[dd] = ratio;
                }
            }
        }
    }
    let x_lo: [f64; 3] = std::array::from_fn(|dd| if dd < D { geometry.x_lo[dd] } else { 0.0 });
    let dx: [f64; 3] = std::array::from_fn(|dd| if dd < D { geometry.dx[dd] } else { 1.0 });
    let start: [f64; 3] = std::array::from_fn(|dd| if dd < D { offset_start[dd] } else { 0.0 });
    let end: [f64; 3] = std::array::from_fn(|dd| if dd < D { offset_end[dd] } else { 0.0 });
    let kernel = current_dispatcher().jit_kernel_keyed(
        CONTINUOUS_TRACER_KERNEL,
        "tracers/continuous_advance",
        "continuous_tracer_advance",
    );
    let mut args = KernelArgs::with_capacity_bytes(640, 59);
    for pointer in x_ptr
        .into_iter()
        .chain(step_ptr)
        .chain([id_ptr, escaped_ptr, crossed_ptr, counter_ptr])
        .chain(drift_ptr)
        .chain(variance_ptr)
        .chain(third_ptr)
    {
        args.push(&pointer);
    }
    for value in [n, ndim, order] {
        args.push(&value);
    }
    args.push(&seed);
    for value in allocated_lo {
        args.push(&value);
    }
    for value in shape {
        args.push(&value);
    }
    for value in map_kind {
        args.push(&value);
    }
    for value in map_p0.into_iter().chain(map_p1).chain(map_p2) {
        args.push(&value);
    }
    for value in x_lo
        .into_iter()
        .chain(dx)
        .chain([scale_start, scale_end])
        .chain(start)
        .chain(end)
        .chain([dt])
    {
        args.push(&value);
    }
    unsafe {
        current_dispatcher()
            .runtime()
            .launch(&kernel, LaunchConfig::for_1d(n, 128), args.as_mut_slice())
            .map_err(|error| format!("continuous tracer device launch failed: {error:?}"))?;
    }
    symbi_xpu::ctx_sync();
    Ok(())
}
