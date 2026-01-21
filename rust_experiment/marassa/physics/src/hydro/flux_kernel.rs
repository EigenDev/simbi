// =============================================================================
// flux_kernel.rs
//
// godunov flux computation kernel for multi-dimensional euler equations.
// computes numerical fluxes at all cell interfaces using riemann solvers.
//
// design:
//   - directional splitting: solve 1d riemann problems per axis
//   - face-centered output: flux[axis] lives on faces normal to axis
//   - lazy computation: builds expression graph, evaluates to fields
//   - reconstruction: plm with minmod limiter (default)
//
// algorithm:
//   for each axis:
//     for each interface i+½:
//       1. reconstruct left state at i+½ from cell i
//       2. reconstruct right state at i+½ from cell i+1
//       3. solve riemann problem (hlle)
//       4. store flux at interface
//
// usage:
//   compute_fluxes(&prim_fields, &mut flux_fields, gamma, dx, device, domains)?;
// =============================================================================

use super::riemann::hlle_flux;
use super::state::{Newtonian, Primitive, Regime, Srhd};
use compute::reconstruction::{plm_left, plm_right, Limiter};
use compute::{Domain, Field};
use xpu_core::Device;

// =============================================================================
// flux kernel interface
// =============================================================================

/// computes fluxes at all cell faces using godunov method with plm reconstruction.
///
/// # type parameters
/// * `R` - regime (newtonian or srhd)
/// * `D` - device type
/// * `RANK` - spatial dimensionality
///
/// # arguments
/// * `rho` - primitive density field (cell-centered)
/// * `vel` - primitive velocity fields (cell-centered)
/// * `pre` - primitive pressure field (cell-centered)
/// * `flux_den` - output density fluxes (face-centered per axis)
/// * `flux_mom` - output momentum fluxes (face-centered per axis)
/// * `flux_nrg` - output energy fluxes (face-centered per axis)
/// * `gamma` - adiabatic index
/// * `dx` - cell spacing per dimension
/// * `limiter` - slope limiter for plm reconstruction
/// * `device` - device for computation
/// * `cell_domain` - cell-centered domain
/// * `face_domains` - face-centered domains per axis
///
/// # returns
/// ok on success
pub fn compute_fluxes<'d, R, D, const RANK: usize>(
    rho: &Field<'d, f64, D, RANK>,
    vel: &[Field<'d, f64, D, RANK>; RANK],
    pre: &Field<'d, f64, D, RANK>,
    flux_den: &mut [Field<'d, f64, D, RANK>; RANK],
    flux_mom: &mut [[Field<'d, f64, D, RANK>; RANK]; RANK],
    flux_nrg: &mut [Field<'d, f64, D, RANK>; RANK],
    gamma: f64,
    dx: [f64; RANK],
    limiter: Limiter,
    device: &'d D,
    cell_domain: Domain<RANK>,
    face_domains: &[Domain<RANK>; RANK],
) -> Result<(), D::Error>
where
    R: Regime,
    D: Device,
{
    // compute flux for each axis independently (dimensional splitting)
    for axis in 0..RANK {
        compute_flux_axis::<R, D, RANK>(
            rho,
            vel,
            pre,
            &mut flux_den[axis],
            &mut flux_mom[axis],
            &mut flux_nrg[axis],
            gamma,
            dx[axis],
            limiter,
            device,
            cell_domain,
            face_domains[axis],
            axis,
        )?;
    }

    Ok(())
}

/// computes fluxes along single axis with plm reconstruction.
fn compute_flux_axis<'d, R, D, const RANK: usize>(
    rho: &Field<'d, f64, D, RANK>,
    vel: &[Field<'d, f64, D, RANK>; RANK],
    pre: &Field<'d, f64, D, RANK>,
    flux_den: &mut Field<'d, f64, D, RANK>,
    flux_mom: &mut [Field<'d, f64, D, RANK>; RANK],
    flux_nrg: &mut Field<'d, f64, D, RANK>,
    gamma: f64,
    _dx: f64,
    limiter: Limiter,
    _device: &'d D,
    _cell_domain: Domain<RANK>,
    face_domain: Domain<RANK>,
    axis: usize,
) -> Result<(), D::Error>
where
    R: Regime,
    D: Device,
{
    // allocate temporary storage for reconstructed states
    let mut flux_den_data = vec![0.0; face_domain.size()];
    let mut flux_mom_data: [Vec<f64>; RANK] =
        std::array::from_fn(|_| vec![0.0; face_domain.size()]);
    let mut flux_nrg_data = vec![0.0; face_domain.size()];

    // get cell data on host (for now - will be device kernel later)
    let rho_data = rho.to_host()?;
    let vel_data: [Vec<f64>; RANK] = std::array::from_fn(|d| vel[d].to_host().unwrap());
    let pre_data = pre.to_host()?;

    // iterate over face domain
    let strides = compute_strides::<RANK>(rho.domain());

    for face_idx in 0..face_domain.size() {
        let face_coord = index_to_coord(face_idx, face_domain, &strides);

        // face i is between cells i-1 and i
        // get left and right cell indices
        let mut left_coord = face_coord;
        left_coord[axis] = left_coord[axis].saturating_sub(1);
        let right_coord = face_coord;

        // check if indices are valid
        if !is_valid_coord(left_coord, rho.domain()) || !is_valid_coord(right_coord, rho.domain()) {
            continue;
        }

        // check if interface is valid
        if !is_valid_coord(left_coord, rho.domain()) || !is_valid_coord(right_coord, rho.domain()) {
            continue;
        }

        let left_idx = coord_to_index(left_coord, rho.domain(), &strides);
        let right_idx = coord_to_index(right_coord, rho.domain(), &strides);

        // plm reconstruction: need left-left and right-right neighbors
        let mut left_left_coord = left_coord;
        left_left_coord[axis] = left_left_coord[axis].saturating_sub(1);
        let mut right_right_coord = right_coord;
        right_right_coord[axis] += 1;

        let left_left_idx = coord_to_index(left_left_coord, rho.domain(), &strides);
        let right_right_idx = coord_to_index(right_right_coord, rho.domain(), &strides);

        // check bounds for plm stencil
        let has_left_stencil = is_valid_coord(left_left_coord, rho.domain());
        let has_right_stencil = is_valid_coord(right_right_coord, rho.domain());

        // reconstruct density
        let (rho_l, rho_r) = if has_left_stencil && has_right_stencil {
            let rho_ll = rho_data[left_left_idx];
            let rho_l_cell = rho_data[left_idx];
            let rho_r_cell = rho_data[right_idx];
            let rho_rr = rho_data[right_right_idx];

            let rho_l_recon = plm_left([&rho_ll, &rho_l_cell, &rho_r_cell], limiter);
            let rho_r_recon = plm_right([&rho_l_cell, &rho_r_cell, &rho_rr], limiter);
            (rho_l_recon, rho_r_recon)
        } else {
            // fallback to piecewise constant at boundaries
            (rho_data[left_idx], rho_data[right_idx])
        };

        // reconstruct velocities
        let mut vel_l = [0.0; RANK];
        let mut vel_r = [0.0; RANK];
        for d in 0..RANK {
            if has_left_stencil && has_right_stencil {
                let v_ll = vel_data[d][left_left_idx];
                let v_l = vel_data[d][left_idx];
                let v_r = vel_data[d][right_idx];
                let v_rr = vel_data[d][right_right_idx];

                vel_l[d] = plm_left([&v_ll, &v_l, &v_r], limiter);
                vel_r[d] = plm_right([&v_l, &v_r, &v_rr], limiter);
            } else {
                vel_l[d] = vel_data[d][left_idx];
                vel_r[d] = vel_data[d][right_idx];
            }
        }

        // reconstruct pressure
        let (pre_l, pre_r) = if has_left_stencil && has_right_stencil {
            let p_ll = pre_data[left_left_idx];
            let p_l = pre_data[left_idx];
            let p_r = pre_data[right_idx];
            let p_rr = pre_data[right_right_idx];

            let pre_l_recon = plm_left([&p_ll, &p_l, &p_r], limiter);
            let pre_r_recon = plm_right([&p_l, &p_r, &p_rr], limiter);
            (pre_l_recon, pre_r_recon)
        } else {
            (pre_data[left_idx], pre_data[right_idx])
        };

        // construct primitive states (rotate velocity to face-normal frame)
        let prim_l = Primitive::<R, 1>::new(rho_l, [vel_l[axis]], pre_l);
        let prim_r = Primitive::<R, 1>::new(rho_r, [vel_r[axis]], pre_r);

        // solve riemann problem
        let flux_1d = solve_riemann_1d::<R>(prim_l, prim_r, gamma);

        // store fluxes
        flux_den_data[face_idx] = flux_1d.den;
        flux_mom_data[axis][face_idx] = flux_1d.mom_normal;
        flux_nrg_data[face_idx] = flux_1d.nrg;

        // tangential momentum fluxes: F(ρv_t) = ρv_n * v_t
        for d in 0..RANK {
            if d != axis {
                let v_tang_l = vel_l[d];
                let v_tang_r = vel_r[d];
                let v_tang_avg = 0.5 * (v_tang_l + v_tang_r);
                flux_mom_data[d][face_idx] = flux_1d.den * v_tang_avg;
            }
        }
    }

    // copy back to device
    flux_den.from_host(&flux_den_data)?;
    for d in 0..RANK {
        flux_mom[d].from_host(&flux_mom_data[d])?;
    }
    flux_nrg.from_host(&flux_nrg_data)?;

    Ok(())
}

// =============================================================================
// riemann solver wrapper
// =============================================================================

/// flux result from 1d riemann problem
struct Flux1DResult {
    den: f64,
    mom_normal: f64,
    nrg: f64,
}

/// solves 1d riemann problem and returns flux components
fn solve_riemann_1d<R: Regime>(
    left: Primitive<R, 1>,
    right: Primitive<R, 1>,
    gamma: f64,
) -> Flux1DResult {
    // convert to 1d states for riemann solver
    let prim_l = super::Primitive1D::new(left.rho, left.vel[0], left.p);
    let prim_r = super::Primitive1D::new(right.rho, right.vel[0], right.p);

    // solve with hlle
    let flux = hlle_flux(prim_l, prim_r, gamma);

    Flux1DResult {
        den: flux.mass,
        mom_normal: flux.mom,
        nrg: flux.energy,
    }
}

// =============================================================================
// reconstruction (piecewise constant for now)
// =============================================================================

/// reconstructs state at interface using plm with minmod limiter.
/// for now: piecewise constant (first order).
#[allow(dead_code)]
fn reconstruct_left<const RANK: usize>(
    data: &[f64],
    cell_idx: usize,
    axis: usize,
    domain: Domain<RANK>,
    strides: &[usize; RANK],
) -> f64 {
    // first order: just return cell value
    data[cell_idx]
}

#[allow(dead_code)]
fn reconstruct_right<const RANK: usize>(
    data: &[f64],
    cell_idx: usize,
    axis: usize,
    domain: Domain<RANK>,
    strides: &[usize; RANK],
) -> f64 {
    // first order: just return cell value
    data[cell_idx]
}

/// minmod limiter for plm reconstruction
#[allow(dead_code)]
fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

// =============================================================================
// index arithmetic utilities
// =============================================================================

/// computes memory strides for row-major layout
fn compute_strides<const RANK: usize>(domain: Domain<RANK>) -> [usize; RANK] {
    let shape = domain.shape();
    let mut strides = [1; RANK];

    for d in (0..RANK - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as usize;
    }

    strides
}

/// converts linear index to coordinate
fn index_to_coord<const RANK: usize>(
    idx: usize,
    domain: Domain<RANK>,
    strides: &[usize; RANK],
) -> [i64; RANK] {
    let mut coord = domain.start;
    let mut remainder = idx;

    for d in 0..RANK {
        coord[d] += (remainder / strides[d]) as i64;
        remainder %= strides[d];
    }

    coord
}

/// converts coordinate to linear index
/// returns 0 if coordinate is out of bounds
fn coord_to_index<const RANK: usize>(
    coord: [i64; RANK],
    domain: Domain<RANK>,
    strides: &[usize; RANK],
) -> usize {
    // check bounds first to avoid overflow
    if !is_valid_coord(coord, domain) {
        return 0;
    }

    let mut idx = 0;
    for d in 0..RANK {
        let offset = (coord[d] - domain.start[d]) as usize;
        idx += offset * strides[d];
    }
    idx
}

/// checks if coordinate is within domain
fn is_valid_coord<const RANK: usize>(coord: [i64; RANK], domain: Domain<RANK>) -> bool {
    for d in 0..RANK {
        if coord[d] < domain.start[d] || coord[d] >= domain.end[d] {
            return false;
        }
    }
    true
}

// =============================================================================
// newtonian specialization
// =============================================================================

/// computes fluxes for newtonian regime
pub fn compute_fluxes_newtonian<'d, D, const RANK: usize>(
    rho: &Field<'d, f64, D, RANK>,
    vel: &[Field<'d, f64, D, RANK>; RANK],
    pre: &Field<'d, f64, D, RANK>,
    flux_den: &mut [Field<'d, f64, D, RANK>; RANK],
    flux_mom: &mut [[Field<'d, f64, D, RANK>; RANK]; RANK],
    flux_nrg: &mut [Field<'d, f64, D, RANK>; RANK],
    gamma: f64,
    dx: [f64; RANK],
    limiter: Limiter,
    device: &'d D,
    cell_domain: Domain<RANK>,
    face_domains: &[Domain<RANK>; RANK],
) -> Result<(), D::Error>
where
    D: Device,
{
    compute_fluxes::<Newtonian, D, RANK>(
        rho,
        vel,
        pre,
        flux_den,
        flux_mom,
        flux_nrg,
        gamma,
        dx,
        limiter,
        device,
        cell_domain,
        face_domains,
    )
}

/// computes fluxes for srhd regime
pub fn compute_fluxes_srhd<'d, D, const RANK: usize>(
    rho: &Field<'d, f64, D, RANK>,
    vel: &[Field<'d, f64, D, RANK>; RANK],
    pre: &Field<'d, f64, D, RANK>,
    flux_den: &mut [Field<'d, f64, D, RANK>; RANK],
    flux_mom: &mut [[Field<'d, f64, D, RANK>; RANK]; RANK],
    flux_nrg: &mut [Field<'d, f64, D, RANK>; RANK],
    gamma: f64,
    dx: [f64; RANK],
    limiter: Limiter,
    device: &'d D,
    cell_domain: Domain<RANK>,
    face_domains: &[Domain<RANK>; RANK],
) -> Result<(), D::Error>
where
    D: Device,
{
    compute_fluxes::<Srhd, D, RANK>(
        rho,
        vel,
        pre,
        flux_den,
        flux_mom,
        flux_nrg,
        gamma,
        dx,
        limiter,
        device,
        cell_domain,
        face_domains,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute::Domain;
    use xpu_host::CpuDevice;

    const GAMMA: f64 = 1.4;

    #[test]
    fn test_index_arithmetic() {
        let domain = Domain::from_shape([10, 20]);
        let strides = compute_strides(domain);

        assert_eq!(strides, [20, 1]);

        let coord = [3, 5];
        let idx = coord_to_index(coord, domain, &strides);
        assert_eq!(idx, 3 * 20 + 5);

        let recovered = index_to_coord(idx, domain, &strides);
        assert_eq!(recovered, coord);
    }

    #[test]
    fn test_valid_coord() {
        let domain = Domain::new([0, 0], [10, 10]);

        assert!(is_valid_coord([5, 5], domain));
        assert!(is_valid_coord([0, 0], domain));
        assert!(is_valid_coord([9, 9], domain));

        assert!(!is_valid_coord([10, 5], domain));
        assert!(!is_valid_coord([-1, 5], domain));
    }

    #[test]
    fn test_minmod() {
        assert_eq!(minmod(1.0, 2.0), 1.0);
        assert_eq!(minmod(2.0, 1.0), 1.0);
        assert_eq!(minmod(1.0, -2.0), 0.0);
        assert_eq!(minmod(-1.0, 2.0), 0.0);
    }

    #[test]
    fn test_riemann_solver_wrapper() {
        let left = Primitive::<Newtonian, 1>::new(1.0, [0.0], 1.0);
        let right = Primitive::<Newtonian, 1>::new(0.125, [0.0], 0.1);

        let flux = solve_riemann_1d::<Newtonian>(left, right, GAMMA);

        // sod shock tube should have positive mass flux
        assert!(flux.den > 0.0);
        assert!(flux.den.is_finite());
        assert!(flux.mom_normal.is_finite());
        assert!(flux.nrg.is_finite());
    }

    #[test]
    fn test_flux_computation_1d() {
        let device = CpuDevice::new(0).unwrap();
        let cell_domain = Domain::from_shape([10]);
        let face_domain = Domain::from_shape([11]); // n+1 faces

        let mut rho = Field::<f64, _, 1>::zeros(&device, cell_domain).unwrap();
        let mut vel = [Field::<f64, _, 1>::zeros(&device, cell_domain).unwrap()];
        let mut pre = Field::<f64, _, 1>::zeros(&device, cell_domain).unwrap();

        // uniform state: rho=1, v=0, p=1
        rho.from_host(&vec![1.0; 10]).unwrap();
        vel[0].from_host(&vec![0.0; 10]).unwrap();
        pre.from_host(&vec![1.0; 10]).unwrap();

        let mut flux_den = [Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()];
        let mut flux_mom = [[Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()]];
        let mut flux_nrg = [Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()];

        let result = compute_fluxes_newtonian(
            &rho,
            &vel,
            &pre,
            &mut flux_den,
            &mut flux_mom,
            &mut flux_nrg,
            GAMMA,
            [0.1],
            Limiter::MinMod,
            &device,
            cell_domain,
            &[face_domain],
        );

        assert!(result.is_ok());

        let flux_den_data = flux_den[0].to_host().unwrap();

        // uniform state with zero velocity -> zero mass flux
        for &f in &flux_den_data {
            assert!(f.abs() < 1e-10);
        }
    }

    #[test]
    fn test_flux_computation_2d() {
        let device = CpuDevice::new(0).unwrap();
        let cell_domain = Domain::from_shape([5, 5]);
        let face_domains = [
            Domain::new([0, 0], [6, 5]), // x-faces
            Domain::new([0, 0], [5, 6]), // y-faces
        ];

        let mut rho = Field::<f64, _, 2>::zeros(&device, cell_domain).unwrap();
        let mut vel = [
            Field::<f64, _, 2>::zeros(&device, cell_domain).unwrap(),
            Field::<f64, _, 2>::zeros(&device, cell_domain).unwrap(),
        ];
        let mut pre = Field::<f64, _, 2>::zeros(&device, cell_domain).unwrap();

        let ncells = 25;
        rho.from_host(&vec![1.0; ncells]).unwrap();
        vel[0].from_host(&vec![0.1; ncells]).unwrap();
        vel[1].from_host(&vec![0.2; ncells]).unwrap();
        pre.from_host(&vec![1.0; ncells]).unwrap();

        let mut flux_den = [
            Field::<f64, _, 2>::zeros(&device, face_domains[0]).unwrap(),
            Field::<f64, _, 2>::zeros(&device, face_domains[1]).unwrap(),
        ];
        let mut flux_mom = [
            [
                Field::<f64, _, 2>::zeros(&device, face_domains[0]).unwrap(),
                Field::<f64, _, 2>::zeros(&device, face_domains[0]).unwrap(),
            ],
            [
                Field::<f64, _, 2>::zeros(&device, face_domains[1]).unwrap(),
                Field::<f64, _, 2>::zeros(&device, face_domains[1]).unwrap(),
            ],
        ];
        let mut flux_nrg = [
            Field::<f64, _, 2>::zeros(&device, face_domains[0]).unwrap(),
            Field::<f64, _, 2>::zeros(&device, face_domains[1]).unwrap(),
        ];

        let result = compute_fluxes_newtonian(
            &rho,
            &vel,
            &pre,
            &mut flux_den,
            &mut flux_mom,
            &mut flux_nrg,
            GAMMA,
            [0.1, 0.1],
            Limiter::MinMod,
            &device,
            cell_domain,
            &face_domains,
        );

        assert!(result.is_ok());

        let flux_x = flux_den[0].to_host().unwrap();
        let flux_y = flux_den[1].to_host().unwrap();

        // uniform flow: flux should be uniform
        // mass flux = rho * v
        for &f in &flux_x {
            if f != 0.0 {
                assert!((f - 0.1).abs() < 0.1); // approximately rho*vx
            }
        }

        for &f in &flux_y {
            if f != 0.0 {
                assert!((f - 0.2).abs() < 0.1); // approximately rho*vy
            }
        }
    }

    #[test]
    fn test_shock_tube_flux() {
        let device = CpuDevice::new(0).unwrap();
        let cell_domain = Domain::from_shape([10]);
        let face_domain = Domain::new([0], [11]); // faces at i=0..11

        let mut rho = Field::<f64, _, 1>::zeros(&device, cell_domain).unwrap();
        let mut vel = [Field::<f64, _, 1>::zeros(&device, cell_domain).unwrap()];
        let mut pre = Field::<f64, _, 1>::zeros(&device, cell_domain).unwrap();

        // sod shock tube: left high pressure, right low pressure
        let mut rho_data = vec![0.125; 10];
        let mut pre_data = vec![0.1; 10];
        for i in 0..5 {
            rho_data[i] = 1.0;
            pre_data[i] = 1.0;
        }

        rho.from_host(&rho_data).unwrap();
        vel[0].from_host(&vec![0.0; 10]).unwrap();
        pre.from_host(&pre_data).unwrap();

        let mut flux_den = [Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()];
        let mut flux_mom = [[Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()]];
        let mut flux_nrg = [Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()];

        let result = compute_fluxes_newtonian(
            &rho,
            &vel,
            &pre,
            &mut flux_den,
            &mut flux_mom,
            &mut flux_nrg,
            GAMMA,
            [0.1],
            Limiter::MinMod,
            &device,
            cell_domain,
            &[face_domain],
        );

        assert!(result.is_ok());

        let flux_data = flux_den[0].to_host().unwrap();

        // at interface between high/low pressure (face 5 is between cells 4 and 5), expect positive flux
        let interface_flux = flux_data[5];
        assert!(interface_flux > 0.0);
    }
}
