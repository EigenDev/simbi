// =============================================================================
// boundary_kernel.rs
//
// boundary kernel for applying boundary conditions to ghost zones.
// uses policy-based design for flexibility and zero-cost abstractions.
//
// design:
//   - operates on partition fields directly
//   - applies policies to ghost zones only
//   - supports periodic, outflow, reflecting boundaries
//   - dimension-by-dimension application
//
// algorithm:
//   for each dimension:
//     for left ghost zone:
//       apply policy(edge_value) -> ghost_value
//     for right ghost zone:
//       apply policy(edge_value) -> ghost_value
//
// usage:
//   apply_boundaries(&mut fields, nghosts, boundary_types, policies)?;
// =============================================================================

use super::boundary_policy::{BoundaryContext, BoundaryPolicy, BoundaryType, Side};
use compute::{Domain, Field};
use xpu_core::Device;

// =============================================================================
// boundary specification
// =============================================================================

#[derive(Debug, Clone)]
pub struct BoundarySpec<const RANK: usize> {
    /// boundary types per dimension, per side (left/right)
    /// layout: [dim0_left, dim0_right, dim1_left, dim1_right, ...]
    pub types: Vec<BoundaryType>,
    /// number of ghost zones per dimension
    pub nghosts: [usize; RANK],
}

impl<const RANK: usize> BoundarySpec<RANK> {
    /// creates boundary spec with same type on all faces
    pub fn uniform(boundary_type: BoundaryType, nghosts: [usize; RANK]) -> Self {
        Self {
            types: vec![boundary_type; RANK * 2],
            nghosts,
        }
    }

    /// creates outflow boundaries on all faces
    pub fn outflow(nghosts: [usize; RANK]) -> Self {
        Self::uniform(BoundaryType::Outflow, nghosts)
    }

    /// creates reflecting boundaries on all faces
    pub fn reflecting(nghosts: [usize; RANK]) -> Self {
        Self::uniform(BoundaryType::Reflect, nghosts)
    }

    /// gets boundary type for specific dimension and side
    pub fn get(&self, dim: usize, side: Side) -> BoundaryType {
        let idx = match side {
            Side::Left => dim * 2,
            Side::Right => dim * 2 + 1,
        };
        self.types[idx]
    }

    /// sets boundary type for specific dimension and side
    pub fn set(&mut self, dim: usize, side: Side, boundary_type: BoundaryType) {
        let idx = match side {
            Side::Left => dim * 2,
            Side::Right => dim * 2 + 1,
        };
        self.types[idx] = boundary_type;
    }
}

// =============================================================================
// 1d boundary application
// =============================================================================

/// applies boundary condition to 1d field
pub fn apply_boundary_1d<'d, D, P>(
    field: &mut Field<'d, f64, D, 1>,
    spec: &BoundarySpec<1>,
    policy: &P,
    device: &'d D,
) -> Result<(), D::Error>
where
    D: Device,
    P: BoundaryPolicy<f64, 1>,
{
    let mut data = field.to_host()?;
    let n = data.len();
    let ng = spec.nghosts[0];

    // left boundary
    let bc_left = spec.get(0, Side::Left);
    if bc_left != BoundaryType::Periodic {
        let context = BoundaryContext::simple(0, Side::Left, bc_left);
        let edge_value = data[ng];
        for i in 0..ng {
            data[i] = policy.apply(edge_value, &context);
        }
    }

    // right boundary
    let bc_right = spec.get(0, Side::Right);
    if bc_right != BoundaryType::Periodic {
        let context = BoundaryContext::simple(0, Side::Right, bc_right);
        let edge_value = data[n - ng - 1];
        for i in (n - ng)..n {
            data[i] = policy.apply(edge_value, &context);
        }
    }

    field.from_host(&data)?;
    Ok(())
}

// =============================================================================
// multi-dimensional boundary application
// =============================================================================

/// applies boundary conditions to 2d field
pub fn apply_boundary_2d<'d, D, P>(
    field: &mut Field<'d, f64, D, 2>,
    spec: &BoundarySpec<2>,
    policy: &P,
    device: &'d D,
) -> Result<(), D::Error>
where
    D: Device,
    P: BoundaryPolicy<f64, 2>,
{
    let mut data = field.to_host()?;
    let domain = field.domain();
    let shape = domain.shape();
    let nx = shape[0] as usize;
    let ny = shape[1] as usize;

    let ngx = spec.nghosts[0];
    let ngy = spec.nghosts[1];

    // apply x-direction boundaries (left and right)
    for dim in 0..2 {
        match dim {
            0 => {
                // left x boundary
                let bc_left = spec.get(0, Side::Left);
                if bc_left != BoundaryType::Periodic {
                    let context = BoundaryContext::simple(0, Side::Left, bc_left);
                    for j in 0..ny {
                        let edge_idx = j * nx + ngx;
                        let edge_value = data[edge_idx];
                        for i in 0..ngx {
                            let idx = j * nx + i;
                            data[idx] = policy.apply(edge_value, &context);
                        }
                    }
                }

                // right x boundary
                let bc_right = spec.get(0, Side::Right);
                if bc_right != BoundaryType::Periodic {
                    let context = BoundaryContext::simple(0, Side::Right, bc_right);
                    for j in 0..ny {
                        let edge_idx = j * nx + (nx - ngx - 1);
                        let edge_value = data[edge_idx];
                        for i in (nx - ngx)..nx {
                            let idx = j * nx + i;
                            data[idx] = policy.apply(edge_value, &context);
                        }
                    }
                }
            }
            1 => {
                // left y boundary (bottom)
                let bc_left = spec.get(1, Side::Left);
                if bc_left != BoundaryType::Periodic {
                    let context = BoundaryContext::simple(1, Side::Left, bc_left);
                    for i in 0..nx {
                        let edge_idx = ngy * nx + i;
                        let edge_value = data[edge_idx];
                        for j in 0..ngy {
                            let idx = j * nx + i;
                            data[idx] = policy.apply(edge_value, &context);
                        }
                    }
                }

                // right y boundary (top)
                let bc_right = spec.get(1, Side::Right);
                if bc_right != BoundaryType::Periodic {
                    let context = BoundaryContext::simple(1, Side::Right, bc_right);
                    for i in 0..nx {
                        let edge_idx = (ny - ngy - 1) * nx + i;
                        let edge_value = data[edge_idx];
                        for j in (ny - ngy)..ny {
                            let idx = j * nx + i;
                            data[idx] = policy.apply(edge_value, &context);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    field.from_host(&data)?;
    Ok(())
}

/// applies boundary conditions to 3d field
pub fn apply_boundary_3d<'d, D, P>(
    field: &mut Field<'d, f64, D, 3>,
    spec: &BoundarySpec<3>,
    policy: &P,
    device: &'d D,
) -> Result<(), D::Error>
where
    D: Device,
    P: BoundaryPolicy<f64, 3>,
{
    let mut data = field.to_host()?;
    let domain = field.domain();
    let shape = domain.shape();
    let nx = shape[0] as usize;
    let ny = shape[1] as usize;
    let nz = shape[2] as usize;

    let ngx = spec.nghosts[0];
    let ngy = spec.nghosts[1];
    let ngz = spec.nghosts[2];

    // x-direction boundaries
    let bc_x_left = spec.get(0, Side::Left);
    if bc_x_left != BoundaryType::Periodic {
        let context = BoundaryContext::simple(0, Side::Left, bc_x_left);
        for k in 0..nz {
            for j in 0..ny {
                let edge_idx = k * ny * nx + j * nx + ngx;
                let edge_value = data[edge_idx];
                for i in 0..ngx {
                    let idx = k * ny * nx + j * nx + i;
                    data[idx] = policy.apply(edge_value, &context);
                }
            }
        }
    }

    let bc_x_right = spec.get(0, Side::Right);
    if bc_x_right != BoundaryType::Periodic {
        let context = BoundaryContext::simple(0, Side::Right, bc_x_right);
        for k in 0..nz {
            for j in 0..ny {
                let edge_idx = k * ny * nx + j * nx + (nx - ngx - 1);
                let edge_value = data[edge_idx];
                for i in (nx - ngx)..nx {
                    let idx = k * ny * nx + j * nx + i;
                    data[idx] = policy.apply(edge_value, &context);
                }
            }
        }
    }

    // y-direction boundaries
    let bc_y_left = spec.get(1, Side::Left);
    if bc_y_left != BoundaryType::Periodic {
        let context = BoundaryContext::simple(1, Side::Left, bc_y_left);
        for k in 0..nz {
            for i in 0..nx {
                let edge_idx = k * ny * nx + ngy * nx + i;
                let edge_value = data[edge_idx];
                for j in 0..ngy {
                    let idx = k * ny * nx + j * nx + i;
                    data[idx] = policy.apply(edge_value, &context);
                }
            }
        }
    }

    let bc_y_right = spec.get(1, Side::Right);
    if bc_y_right != BoundaryType::Periodic {
        let context = BoundaryContext::simple(1, Side::Right, bc_y_right);
        for k in 0..nz {
            for i in 0..nx {
                let edge_idx = k * ny * nx + (ny - ngy - 1) * nx + i;
                let edge_value = data[edge_idx];
                for j in (ny - ngy)..ny {
                    let idx = k * ny * nx + j * nx + i;
                    data[idx] = policy.apply(edge_value, &context);
                }
            }
        }
    }

    // z-direction boundaries
    let bc_z_left = spec.get(2, Side::Left);
    if bc_z_left != BoundaryType::Periodic {
        let context = BoundaryContext::simple(2, Side::Left, bc_z_left);
        for j in 0..ny {
            for i in 0..nx {
                let edge_idx = ngz * ny * nx + j * nx + i;
                let edge_value = data[edge_idx];
                for k in 0..ngz {
                    let idx = k * ny * nx + j * nx + i;
                    data[idx] = policy.apply(edge_value, &context);
                }
            }
        }
    }

    let bc_z_right = spec.get(2, Side::Right);
    if bc_z_right != BoundaryType::Periodic {
        let context = BoundaryContext::simple(2, Side::Right, bc_z_right);
        for j in 0..ny {
            for i in 0..nx {
                let edge_idx = (nz - ngz - 1) * ny * nx + j * nx + i;
                let edge_value = data[edge_idx];
                for k in (nz - ngz)..nz {
                    let idx = k * ny * nx + j * nx + i;
                    data[idx] = policy.apply(edge_value, &context);
                }
            }
        }
    }

    field.from_host(&data)?;
    Ok(())
}

// =============================================================================
// convenience wrappers for all fields in partition
// =============================================================================

/// applies boundaries to all conserved fields (1d)
pub fn apply_conserved_boundaries_1d<'d, D>(
    den: &mut Field<'d, f64, D, 1>,
    mom: &mut [Field<'d, f64, D, 1>; 1],
    nrg: &mut Field<'d, f64, D, 1>,
    spec: &BoundarySpec<1>,
    device: &'d D,
) -> Result<(), D::Error>
where
    D: Device,
{
    let outflow = super::boundary_policy::outflow::<1>();

    apply_boundary_1d(den, spec, &outflow, device)?;
    apply_boundary_1d(&mut mom[0], spec, &outflow, device)?;
    apply_boundary_1d(nrg, spec, &outflow, device)?;

    Ok(())
}

/// applies boundaries to all conserved fields (2d)
pub fn apply_conserved_boundaries_2d<'d, D>(
    den: &mut Field<'d, f64, D, 2>,
    mom: &mut [Field<'d, f64, D, 2>; 2],
    nrg: &mut Field<'d, f64, D, 2>,
    spec: &BoundarySpec<2>,
    device: &'d D,
) -> Result<(), D::Error>
where
    D: Device,
{
    let outflow = super::boundary_policy::outflow::<2>();

    apply_boundary_2d(den, spec, &outflow, device)?;
    apply_boundary_2d(&mut mom[0], spec, &outflow, device)?;
    apply_boundary_2d(&mut mom[1], spec, &outflow, device)?;
    apply_boundary_2d(nrg, spec, &outflow, device)?;

    Ok(())
}

/// applies boundaries to all conserved fields (3d)
pub fn apply_conserved_boundaries_3d<'d, D>(
    den: &mut Field<'d, f64, D, 3>,
    mom: &mut [Field<'d, f64, D, 3>; 3],
    nrg: &mut Field<'d, f64, D, 3>,
    spec: &BoundarySpec<3>,
    device: &'d D,
) -> Result<(), D::Error>
where
    D: Device,
{
    let outflow = super::boundary_policy::outflow::<3>();

    apply_boundary_3d(den, spec, &outflow, device)?;
    apply_boundary_3d(&mut mom[0], spec, &outflow, device)?;
    apply_boundary_3d(&mut mom[1], spec, &outflow, device)?;
    apply_boundary_3d(&mut mom[2], spec, &outflow, device)?;
    apply_boundary_3d(nrg, spec, &outflow, device)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute::Domain;
    use xpu_host::CpuDevice;

    #[test]
    fn test_boundary_spec_uniform() {
        let spec = BoundarySpec::<2>::uniform(BoundaryType::Outflow, [2, 2]);

        assert_eq!(spec.get(0, Side::Left), BoundaryType::Outflow);
        assert_eq!(spec.get(0, Side::Right), BoundaryType::Outflow);
        assert_eq!(spec.get(1, Side::Left), BoundaryType::Outflow);
        assert_eq!(spec.get(1, Side::Right), BoundaryType::Outflow);
    }

    #[test]
    fn test_boundary_spec_mixed() {
        let mut spec = BoundarySpec::<2>::outflow([2, 2]);
        spec.set(0, Side::Left, BoundaryType::Reflect);
        spec.set(1, Side::Right, BoundaryType::Periodic);

        assert_eq!(spec.get(0, Side::Left), BoundaryType::Reflect);
        assert_eq!(spec.get(0, Side::Right), BoundaryType::Outflow);
        assert_eq!(spec.get(1, Side::Left), BoundaryType::Outflow);
        assert_eq!(spec.get(1, Side::Right), BoundaryType::Periodic);
    }

    #[test]
    fn test_apply_boundary_1d_outflow() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut field = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // interior values
        let mut data = vec![0.0; 10];
        for i in 2..8 {
            data[i] = 5.0;
        }
        field.from_host(&data).unwrap();

        let spec = BoundarySpec::<1>::outflow([2]);
        let policy = super::super::boundary_policy::outflow::<1>();

        apply_boundary_1d(&mut field, &spec, &policy, &device).unwrap();

        let result = field.to_host().unwrap();
        assert_eq!(result[0], 5.0); // ghost = edge
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 5.0); // edge
        assert_eq!(result[7], 5.0); // edge
        assert_eq!(result[8], 5.0); // ghost = edge
        assert_eq!(result[9], 5.0);
    }

    #[test]
    fn test_apply_boundary_2d_outflow() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([6, 6]);
        let mut field = Field::<f64, _, 2>::zeros(&device, domain).unwrap();

        // set interior to 1.0
        let mut data = vec![0.0; 36];
        for j in 2..4 {
            for i in 2..4 {
                data[j * 6 + i] = 1.0;
            }
        }
        field.from_host(&data).unwrap();

        let spec = BoundarySpec::<2>::outflow([2, 2]);
        let policy = super::super::boundary_policy::outflow::<2>();

        apply_boundary_2d(&mut field, &spec, &policy, &device).unwrap();

        let result = field.to_host().unwrap();

        // check left boundary
        assert_eq!(result[2 * 6 + 0], 1.0);
        assert_eq!(result[2 * 6 + 1], 1.0);
        assert_eq!(result[2 * 6 + 2], 1.0); // edge
    }
}
