// =============================================================================
// boundary.rs
//
// boundary condition operators for ghost zone filling.
// provides trait-based extensibility for different bc types.
//
// design:
//   - BoundaryCondition trait defines interface
//   - each bc type is zero-sized marker type
//   - apply method launches device kernel
//   - outflow bc is simplest (copy interior to ghost)
//
// mathematical structure:
//   B: State -> State (endomorphism on state space)
//   B is idempotent: B(B(x)) = B(x)
//
// usage:
//   let bc = OutflowBC;
//   bc.apply(&mut partition, &config)?;
// =============================================================================

use compute::Domain;

use xpu_core::Device;

// =============================================================================
// boundary condition trait
// =============================================================================

/// boundary condition operator.
/// fills ghost zones based on interior values and bc policy.
pub trait BoundaryCondition: Send + Sync {
    /// applies boundary condition to all ghost zones in partition.
    fn apply<D: Device, const RANK: usize>(
        &self,
        den: &mut compute::Field<f64, D, RANK>,
        mom: &mut [compute::Field<f64, D, RANK>; RANK],
        nrg: &mut compute::Field<f64, D, RANK>,
        domain: Domain<RANK>,
        nghosts: [usize; RANK],
    ) -> Result<(), D::Error>;
}

// =============================================================================
// outflow boundary condition
// =============================================================================

/// outflow boundary condition (zero-gradient extrapolation).
/// copies interior values to ghost zones.
///
/// mathematical definition:
///   for ghost cell g with nearest interior cell i:
///   u(g) = u(i)
///
/// this is first-order accurate and preserves monotonicity.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutflowBC;

impl BoundaryCondition for OutflowBC {
    fn apply<D: Device, const RANK: usize>(
        &self,
        den: &mut compute::Field<f64, D, RANK>,
        mom: &mut [compute::Field<f64, D, RANK>; RANK],
        nrg: &mut compute::Field<f64, D, RANK>,
        domain: Domain<RANK>,
        nghosts: [usize; RANK],
    ) -> Result<(), D::Error> {
        // for each dimension, fill lower and upper ghost zones
        for axis in 0..RANK {
            // lower ghost zone: copy from first interior cell
            self.fill_lower_ghost(axis, den, domain, nghosts)?;
            for d in 0..RANK {
                self.fill_lower_ghost(axis, &mut mom[d], domain, nghosts)?;
            }
            self.fill_lower_ghost(axis, nrg, domain, nghosts)?;

            // upper ghost zone: copy from last interior cell
            self.fill_upper_ghost(axis, den, domain, nghosts)?;
            for d in 0..RANK {
                self.fill_upper_ghost(axis, &mut mom[d], domain, nghosts)?;
            }
            self.fill_upper_ghost(axis, nrg, domain, nghosts)?;
        }

        Ok(())
    }
}

impl OutflowBC {
    /// fills lower ghost zone along given axis.
    /// copies first interior cell to all ghost cells on lower boundary.
    fn fill_lower_ghost<D: Device, const RANK: usize>(
        &self,
        axis: usize,
        field: &mut compute::Field<f64, D, RANK>,
        domain: Domain<RANK>,
        nghosts: [usize; RANK],
    ) -> Result<(), D::Error> {
        let ng = nghosts[axis] as i64;
        let interior_start = domain.start[axis];

        // ghost region: [start - ng, start)
        for offset in 0..ng {
            let ghost_idx = interior_start - ng + offset;
            let interior_idx = interior_start;

            // copy along this axis, iterate over perpendicular directions
            self.copy_slice(axis, ghost_idx, interior_idx, field, domain)?;
        }

        Ok(())
    }

    /// fills upper ghost zone along given axis.
    /// copies last interior cell to all ghost cells on upper boundary.
    fn fill_upper_ghost<D: Device, const RANK: usize>(
        &self,
        axis: usize,
        field: &mut compute::Field<f64, D, RANK>,
        domain: Domain<RANK>,
        nghosts: [usize; RANK],
    ) -> Result<(), D::Error> {
        let ng = nghosts[axis] as i64;
        let interior_end = domain.end[axis];

        // ghost region: [end, end + ng)
        for offset in 0..ng {
            let ghost_idx = interior_end + offset;
            let interior_idx = interior_end - 1;

            // copy along this axis, iterate over perpendicular directions
            self.copy_slice(axis, ghost_idx, interior_idx, field, domain)?;
        }

        Ok(())
    }

    /// copies a slice from interior_idx to ghost_idx along given axis.
    /// iterates over all perpendicular directions.
    fn copy_slice<D: Device, const RANK: usize>(
        &self,
        axis: usize,
        ghost_idx: i64,
        interior_idx: i64,
        field: &mut compute::Field<f64, D, RANK>,
        domain: Domain<RANK>,
    ) -> Result<(), D::Error> {
        // copy entire field to host, modify, copy back
        // todo: optimize to single device kernel
        let field_domain = field.domain();
        let size = field_domain.size();
        let mut host_data = vec![0.0; size];

        // copy from device to host
        field
            .device()
            .copy_to_host(field.buffer(), &mut host_data)?;

        // iterate over perpendicular slice
        // for 1d: single iteration (perp domain is empty conceptually)
        if RANK == 1 {
            let mut ghost_coord = [0i64; RANK];
            let mut interior_coord = [0i64; RANK];

            ghost_coord[axis] = ghost_idx;
            interior_coord[axis] = interior_idx;

            let interior_linear = field_domain.coord_to_linear(interior_coord);
            let ghost_linear = field_domain.coord_to_linear(ghost_coord);

            host_data[ghost_linear] = host_data[interior_linear];
        } else {
            // for 2d/3d: iterate over all points in domain
            for coord in field_domain.iter() {
                let mut ghost_coord = coord;
                let mut interior_coord = coord;

                ghost_coord[axis] = ghost_idx;
                interior_coord[axis] = interior_idx;

                let interior_linear = field_domain.coord_to_linear(interior_coord);
                let ghost_linear = field_domain.coord_to_linear(ghost_coord);

                host_data[ghost_linear] = host_data[interior_linear];
            }
        }

        // copy modified data back to device
        field.from_host(&host_data)?;

        Ok(())
    }
}

// =============================================================================
// periodic boundary condition
// =============================================================================

/// periodic boundary condition (wraps around).
/// copies from opposite side of domain.
///
/// mathematical definition:
///   u(x_min - dx) = u(x_max - dx)
///   u(x_max + dx) = u(x_min + dx)
#[derive(Debug, Clone, Copy, Default)]
pub struct PeriodicBC;

impl BoundaryCondition for PeriodicBC {
    fn apply<D: Device, const RANK: usize>(
        &self,
        _den: &mut compute::Field<f64, D, RANK>,
        _mom: &mut [compute::Field<f64, D, RANK>; RANK],
        _nrg: &mut compute::Field<f64, D, RANK>,
        _domain: Domain<RANK>,
        _nghosts: [usize; RANK],
    ) -> Result<(), D::Error> {
        // todo: implement periodic bc
        // for now: no-op
        Ok(())
    }
}

// =============================================================================
// reflecting boundary condition
// =============================================================================

/// reflecting boundary condition (mirror symmetry).
/// copies values with velocity sign flip.
///
/// mathematical definition:
///   rho(ghost) = rho(interior)
///   v_normal(ghost) = -v_normal(interior)
///   v_tangent(ghost) = v_tangent(interior)
///   p(ghost) = p(interior)
#[derive(Debug, Clone, Copy, Default)]
pub struct ReflectingBC;

impl BoundaryCondition for ReflectingBC {
    fn apply<D: Device, const RANK: usize>(
        &self,
        _den: &mut compute::Field<f64, D, RANK>,
        _mom: &mut [compute::Field<f64, D, RANK>; RANK],
        _nrg: &mut compute::Field<f64, D, RANK>,
        _domain: Domain<RANK>,
        _nghosts: [usize; RANK],
    ) -> Result<(), D::Error> {
        // todo: implement reflecting bc
        // for now: no-op
        Ok(())
    }
}

// =============================================================================
// user-defined boundary condition
// =============================================================================

/// user-defined boundary condition.
/// placeholder for future extensibility.
pub struct UserDefinedBC;

#[cfg(test)]
mod tests {
    use super::*;
    use xpu_host::CpuDevice;

    #[test]
    fn test_outflow_1d() {
        let device = CpuDevice::new(0).unwrap();

        // domain: [2, 8] interior, [0, 10] with ghosts
        let interior = Domain::new([2], [8]);
        let total = Domain::new([0], [10]);
        let nghosts = [2];

        let mut field = compute::Field::zeros(&device, total).unwrap();

        // set interior values
        for i in 2..8 {
            field.view_mut().set([i], (i * 10) as f64);
        }

        // apply outflow bc
        let bc = OutflowBC;
        let mut mom_dummy = [compute::Field::zeros(&device, total).unwrap()];
        let mut nrg_dummy = compute::Field::zeros(&device, total).unwrap();

        bc.apply(
            &mut field,
            &mut mom_dummy,
            &mut nrg_dummy,
            interior,
            nghosts,
        )
        .unwrap();

        // check lower ghost zone (should copy from i=2)
        assert_eq!(*field.view().eval([0]), 20.0);
        assert_eq!(*field.view().eval([1]), 20.0);

        // check upper ghost zone (should copy from i=7)
        assert_eq!(*field.view().eval([8]), 70.0);
        assert_eq!(*field.view().eval([9]), 70.0);
    }

    #[test]
    fn test_outflow_2d() {
        let device = CpuDevice::new(0).unwrap();

        // domain: [1,1] to [3,3] interior, [0,0] to [4,4] with ghosts
        let interior = Domain::new([1, 1], [3, 3]);
        let total = Domain::new([0, 0], [4, 4]);
        let nghosts = [1, 1];

        let mut field = compute::Field::zeros(&device, total).unwrap();

        // set interior values: value = 10*i + j
        for i in 1..3 {
            for j in 1..3 {
                field.view_mut().set([i, j], (10 * i + j) as f64);
            }
        }

        let bc = OutflowBC;
        let mut mom_dummy = [
            compute::Field::zeros(&device, total).unwrap(),
            compute::Field::zeros(&device, total).unwrap(),
        ];
        let mut nrg_dummy = compute::Field::zeros(&device, total).unwrap();

        bc.apply(
            &mut field,
            &mut mom_dummy,
            &mut nrg_dummy,
            interior,
            nghosts,
        )
        .unwrap();

        // check lower x ghost (i=0, should copy from i=1)
        assert_eq!(*field.view().eval([0, 1]), 11.0);
        assert_eq!(*field.view().eval([0, 2]), 12.0);

        // check upper x ghost (i=3, should copy from i=2)
        assert_eq!(*field.view().eval([3, 1]), 21.0);
        assert_eq!(*field.view().eval([3, 2]), 22.0);

        // check lower y ghost (j=0, should copy from j=1)
        assert_eq!(*field.view().eval([1, 0]), 11.0);
        assert_eq!(*field.view().eval([2, 0]), 21.0);

        // check upper y ghost (j=3, should copy from j=2)
        assert_eq!(*field.view().eval([1, 3]), 12.0);
        assert_eq!(*field.view().eval([2, 3]), 22.0);
    }

    #[test]
    fn test_boundary_polymorphism() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::new([1], [3]);
        let total = Domain::new([0], [4]);
        let nghosts = [1];

        let mut field = compute::Field::zeros(&device, total).unwrap();
        let mut mom = [compute::Field::zeros(&device, total).unwrap()];
        let mut nrg = compute::Field::zeros(&device, total).unwrap();

        // test that different bc types work
        let bc = OutflowBC;
        bc.apply(&mut field, &mut mom, &mut nrg, domain, nghosts)
            .unwrap();

        let bc2 = PeriodicBC;
        bc2.apply(&mut field, &mut mom, &mut nrg, domain, nghosts)
            .unwrap();
    }
}
