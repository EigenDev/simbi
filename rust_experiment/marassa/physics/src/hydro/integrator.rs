// =============================================================================
// integrator.rs
//
// time integration kernels for godunov-type solvers.
// implements forward euler, rk2 (heun), and rk3 (shu-osher) methods.
//
// design:
//   - operates on conserved fields directly
//   - flux divergence: ∂u/∂t = -∇·F
//   - cfl-based timestep computation
//   - regime-agnostic (works with newtonian and srhd)
//
// algorithm (forward euler):
//   u^{n+1} = u^n - (dt/dx) * (F_{i+½} - F_{i-½})
//
// algorithm (rk2):
//   u* = u^n + dt * L(u^n)
//   u^{n+1} = ½u^n + ½u* + ½dt * L(u*)
//
// algorithm (rk3):
//   u* = u^n + dt * L(u^n)
//   u** = ¾u^n + ¼u* + ¼dt * L(u*)
//   u^{n+1} = ⅓u^n + ⅔u** + ⅔dt * L(u**)
//
// usage:
//   euler_step(&mut conserved, &fluxes, dt, dx, device, domain)?;
//   rk2_step(&mut conserved, &fluxes, &mut workspace, dt, dx, device, domain)?;
// =============================================================================

use compute::{Domain, Field};
use xpu_core::Device;

// =============================================================================
// forward euler time integration
// =============================================================================

/// advances conserved fields by one timestep using forward euler method.
///
/// computes: u^{n+1} = u^n - (dt/dx) * ∇·F
///
/// # arguments
/// * `den` - conserved density field (updated in place)
/// * `mom` - conserved momentum fields (updated in place)
/// * `nrg` - conserved energy field (updated in place)
/// * `flux_den` - density fluxes at faces (per axis)
/// * `flux_mom` - momentum fluxes at faces (per axis)
/// * `flux_nrg` - energy fluxes at faces (per axis)
/// * `dt` - timestep size
/// * `dx` - cell spacing per dimension
/// * `device` - device for computation
/// * `domain` - cell-centered domain to update
pub fn euler_step<'d, D, const RANK: usize>(
    den: &mut Field<'d, f64, D, RANK>,
    mom: &mut [Field<'d, f64, D, RANK>; RANK],
    nrg: &mut Field<'d, f64, D, RANK>,
    flux_den: &[Field<'d, f64, D, RANK>; RANK],
    flux_mom: &[[Field<'d, f64, D, RANK>; RANK]; RANK],
    flux_nrg: &[Field<'d, f64, D, RANK>; RANK],
    dt: f64,
    dx: [f64; RANK],
    device: &'d D,
    domain: Domain<RANK>,
) -> Result<(), D::Error>
where
    D: Device,
{
    // update density: D^{n+1} = D^n - (dt/dx) * (F_{i+½} - F_{i-½})
    update_field(den, flux_den, dt, dx, device, domain)?;

    // update momentum components
    for axis in 0..RANK {
        update_field(&mut mom[axis], &flux_mom[axis], dt, dx, device, domain)?;
    }

    // update energy
    update_field(nrg, flux_nrg, dt, dx, device, domain)?;

    Ok(())
}

/// updates single field using flux divergence
fn update_field<'d, D, const RANK: usize>(
    field: &mut Field<'d, f64, D, RANK>,
    fluxes: &[Field<'d, f64, D, RANK>; RANK],
    dt: f64,
    dx: [f64; RANK],
    device: &'d D,
    domain: Domain<RANK>,
) -> Result<(), D::Error>
where
    D: Device,
{
    let mut field_data = field.to_host()?;
    let flux_data: [Vec<f64>; RANK] = std::array::from_fn(|d| fluxes[d].to_host().unwrap());

    let strides = compute_strides(domain);

    // for each cell in domain
    for idx in 0..domain.size() {
        let coord = index_to_coord(idx, domain, &strides);

        // compute flux divergence: ∑_d (F_{i+½} - F_{i-½}) / dx_d
        let mut div_flux = 0.0;

        for axis in 0..RANK {
            // face indices
            let mut face_plus = coord;
            face_plus[axis] += 1; // i+½ face

            let face_minus = coord; // i-½ face

            // get flux values (need to map to flux array indices)
            let flux_plus_idx = coord_to_flux_index(face_plus, domain, &strides, axis);
            let flux_minus_idx = coord_to_flux_index(face_minus, domain, &strides, axis);

            let f_plus = if flux_plus_idx < flux_data[axis].len() {
                flux_data[axis][flux_plus_idx]
            } else {
                0.0
            };

            let f_minus = if flux_minus_idx < flux_data[axis].len() {
                flux_data[axis][flux_minus_idx]
            } else {
                0.0
            };

            div_flux += (f_plus - f_minus) / dx[axis];
        }

        // euler update: u^{n+1} = u^n - dt * div_flux
        field_data[idx] -= dt * div_flux;
    }

    field.from_host(&field_data)?;

    Ok(())
}

// =============================================================================
// runge-kutta 2 (heun's method)
// =============================================================================

/// second-order runge-kutta time integration (heun's method).
///
/// algorithm:
///   u* = u^n + dt * L(u^n)
///   u^{n+1} = ½u^n + ½u* + ½dt * L(u*)
///
/// requires workspace for intermediate state u*.
pub fn rk2_step<'d, D, const RANK: usize>(
    den: &mut Field<'d, f64, D, RANK>,
    mom: &mut [Field<'d, f64, D, RANK>; RANK],
    nrg: &mut Field<'d, f64, D, RANK>,
    flux_den: &[Field<'d, f64, D, RANK>; RANK],
    flux_mom: &[[Field<'d, f64, D, RANK>; RANK]; RANK],
    flux_nrg: &[Field<'d, f64, D, RANK>; RANK],
    den_star: &mut Field<'d, f64, D, RANK>,
    mom_star: &mut [Field<'d, f64, D, RANK>; RANK],
    nrg_star: &mut Field<'d, f64, D, RANK>,
    dt: f64,
    dx: [f64; RANK],
    device: &'d D,
    domain: Domain<RANK>,
) -> Result<(), D::Error>
where
    D: Device,
{
    // save initial state: u^n
    let den_n = den.to_host()?;
    let mom_n: [Vec<f64>; RANK] = std::array::from_fn(|d| mom[d].to_host().unwrap());
    let nrg_n = nrg.to_host()?;

    // stage 1: u* = u^n + dt * L(u^n)
    // (L = -∇·F, so this is euler step)
    euler_step(
        den, mom, nrg, flux_den, flux_mom, flux_nrg, dt, dx, device, domain,
    )?;

    // copy u* to workspace
    let den_star_data = den.to_host()?;
    let mom_star_data: [Vec<f64>; RANK] = std::array::from_fn(|d| mom[d].to_host().unwrap());
    let nrg_star_data = nrg.to_host()?;

    den_star.from_host(&den_star_data)?;
    for d in 0..RANK {
        mom_star[d].from_host(&mom_star_data[d])?;
    }
    nrg_star.from_host(&nrg_star_data)?;

    // note: now need to recompute fluxes from u* (not done here - caller must do)
    // for now, assume fluxes have been recomputed

    // stage 2: u^{n+1} = ½u^n + ½u* + ½dt * L(u*)
    euler_step(
        den_star,
        mom_star,
        nrg_star,
        flux_den,
        flux_mom,
        flux_nrg,
        0.5 * dt,
        dx,
        device,
        domain,
    )?;

    // combine: u = ½u^n + ½(u* + ½dt*L(u*))
    let mut den_final = den_star.to_host()?;
    let mut mom_final: [Vec<f64>; RANK] = std::array::from_fn(|d| mom_star[d].to_host().unwrap());
    let mut nrg_final = nrg_star.to_host()?;

    for i in 0..domain.size() {
        den_final[i] = 0.5 * den_n[i] + 0.5 * den_final[i];
        nrg_final[i] = 0.5 * nrg_n[i] + 0.5 * nrg_final[i];
        for d in 0..RANK {
            mom_final[d][i] = 0.5 * mom_n[d][i] + 0.5 * mom_final[d][i];
        }
    }

    den.from_host(&den_final)?;
    nrg.from_host(&nrg_final)?;
    for d in 0..RANK {
        mom[d].from_host(&mom_final[d])?;
    }

    Ok(())
}

// =============================================================================
// runge-kutta 3 (shu-osher tvd)
// =============================================================================

/// third-order runge-kutta time integration (shu-osher tvd).
///
/// algorithm:
///   u* = u^n + dt * L(u^n)
///   u** = ¾u^n + ¼u* + ¼dt * L(u*)
///   u^{n+1} = ⅓u^n + ⅔u** + ⅔dt * L(u**)
pub fn rk3_step<'d, D, const RANK: usize>(
    den: &mut Field<'d, f64, D, RANK>,
    mom: &mut [Field<'d, f64, D, RANK>; RANK],
    nrg: &mut Field<'d, f64, D, RANK>,
    flux_den: &[Field<'d, f64, D, RANK>; RANK],
    flux_mom: &[[Field<'d, f64, D, RANK>; RANK]; RANK],
    flux_nrg: &[Field<'d, f64, D, RANK>; RANK],
    den_star: &mut Field<'d, f64, D, RANK>,
    mom_star: &mut [Field<'d, f64, D, RANK>; RANK],
    nrg_star: &mut Field<'d, f64, D, RANK>,
    dt: f64,
    dx: [f64; RANK],
    device: &'d D,
    domain: Domain<RANK>,
) -> Result<(), D::Error>
where
    D: Device,
{
    // save initial state: u^n
    let den_n = den.to_host()?;
    let mom_n: [Vec<f64>; RANK] = std::array::from_fn(|d| mom[d].to_host().unwrap());
    let nrg_n = nrg.to_host()?;

    // stage 1: u* = u^n + dt * L(u^n)
    euler_step(
        den, mom, nrg, flux_den, flux_mom, flux_nrg, dt, dx, device, domain,
    )?;

    let den_1 = den.to_host()?;
    let mom_1: [Vec<f64>; RANK] = std::array::from_fn(|d| mom[d].to_host().unwrap());
    let nrg_1 = nrg.to_host()?;

    // note: fluxes must be recomputed from u* here (caller's responsibility)

    // stage 2: u** = ¾u^n + ¼u* + ¼dt * L(u*)
    euler_step(
        den,
        mom,
        nrg,
        flux_den,
        flux_mom,
        flux_nrg,
        0.25 * dt,
        dx,
        device,
        domain,
    )?;

    let mut den_2 = den.to_host()?;
    let mut mom_2: [Vec<f64>; RANK] = std::array::from_fn(|d| mom[d].to_host().unwrap());
    let mut nrg_2 = nrg.to_host()?;

    // combine: u** = ¾u^n + ¼(u* + ¼dt*L(u*))
    for i in 0..domain.size() {
        den_2[i] = 0.75 * den_n[i] + 0.25 * den_2[i];
        nrg_2[i] = 0.75 * nrg_n[i] + 0.25 * nrg_2[i];
        for d in 0..RANK {
            mom_2[d][i] = 0.75 * mom_n[d][i] + 0.25 * mom_2[d][i];
        }
    }

    den.from_host(&den_2)?;
    nrg.from_host(&nrg_2)?;
    for d in 0..RANK {
        mom[d].from_host(&mom_2[d])?;
    }

    // note: fluxes must be recomputed from u** here

    // stage 3: u^{n+1} = ⅓u^n + ⅔u** + ⅔dt * L(u**)
    euler_step(
        den,
        mom,
        nrg,
        flux_den,
        flux_mom,
        flux_nrg,
        (2.0 / 3.0) * dt,
        dx,
        device,
        domain,
    )?;

    let mut den_3 = den.to_host()?;
    let mut mom_3: [Vec<f64>; RANK] = std::array::from_fn(|d| mom[d].to_host().unwrap());
    let mut nrg_3 = nrg.to_host()?;

    // combine: u^{n+1} = ⅓u^n + ⅔(u** + ⅔dt*L(u**))
    for i in 0..domain.size() {
        den_3[i] = (1.0 / 3.0) * den_n[i] + (2.0 / 3.0) * den_3[i];
        nrg_3[i] = (1.0 / 3.0) * nrg_n[i] + (2.0 / 3.0) * nrg_3[i];
        for d in 0..RANK {
            mom_3[d][i] = (1.0 / 3.0) * mom_n[d][i] + (2.0 / 3.0) * mom_3[d][i];
        }
    }

    den.from_host(&den_3)?;
    nrg.from_host(&nrg_3)?;
    for d in 0..RANK {
        mom[d].from_host(&mom_3[d])?;
    }

    Ok(())
}

// =============================================================================
// cfl timestep computation
// =============================================================================

/// computes timestep from cfl condition.
///
/// dt = cfl * min_d(dx_d / max_wave_speed_d)
///
/// # arguments
/// * `vel` - velocity fields
/// * `pre` - pressure field
/// * `rho` - density field
/// * `gamma` - adiabatic index
/// * `dx` - cell spacing per dimension
/// * `cfl` - cfl number (typically 0.4-0.8)
/// * `domain` - domain to compute over
pub fn compute_dt<'d, D, const RANK: usize>(
    vel: &[Field<'d, f64, D, RANK>; RANK],
    pre: &Field<'d, f64, D, RANK>,
    rho: &Field<'d, f64, D, RANK>,
    gamma: f64,
    dx: [f64; RANK],
    cfl: f64,
    domain: Domain<RANK>,
) -> Result<f64, D::Error>
where
    D: Device,
{
    // get data on host
    let vel_data: [Vec<f64>; RANK] = std::array::from_fn(|d| vel[d].to_host().unwrap());
    let pre_data = pre.to_host()?;
    let rho_data = rho.to_host()?;

    let mut max_wave_speed: f64 = 0.0;

    // find maximum wave speed over entire domain
    for i in 0..domain.size() {
        let rho_i = rho_data[i];
        let pre_i = pre_data[i];

        // sound speed
        let cs = (gamma * pre_i / rho_i).sqrt();

        // max wave speed per direction
        for d in 0..RANK {
            let vel_d = vel_data[d][i];
            let lambda = vel_d.abs() + cs;
            max_wave_speed = max_wave_speed.max(lambda);
        }
    }

    // dt = cfl * dx / max_wave_speed
    let mut dt = f64::MAX;
    for d in 0..RANK {
        dt = dt.min(cfl * dx[d] / max_wave_speed);
    }

    Ok(dt)
}

// =============================================================================
// index utilities
// =============================================================================

fn compute_strides<const RANK: usize>(domain: Domain<RANK>) -> [usize; RANK] {
    let shape = domain.shape();
    let mut strides = [1; RANK];

    for d in (0..RANK - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as usize;
    }

    strides
}

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

fn coord_to_flux_index<const RANK: usize>(
    coord: [i64; RANK],
    domain: Domain<RANK>,
    strides: &[usize; RANK],
    _axis: usize,
) -> usize {
    let mut idx = 0;
    for d in 0..RANK {
        idx += ((coord[d] - domain.start[d]) as usize) * strides[d];
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute::Domain;
    use xpu_host::CpuDevice;

    const GAMMA: f64 = 1.4;

    #[test]
    fn test_euler_step_1d() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let face_domain = Domain::from_shape([11]);

        // initial state: uniform
        let mut den = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let mut mom = [Field::<f64, _, 1>::zeros(&device, domain).unwrap()];
        let mut nrg = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        den.from_host(&vec![1.0; 10]).unwrap();
        mom[0].from_host(&vec![0.0; 10]).unwrap();
        nrg.from_host(&vec![2.5; 10]).unwrap();

        // zero fluxes everywhere
        let flux_den = [Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()];
        let flux_mom = [[Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()]];
        let flux_nrg = [Field::<f64, _, 1>::zeros(&device, face_domain).unwrap()];

        let result = euler_step(
            &mut den,
            &mut mom,
            &mut nrg,
            &flux_den,
            &flux_mom,
            &flux_nrg,
            0.01,
            [0.1],
            &device,
            domain,
        );

        assert!(result.is_ok());

        // with zero fluxes, state should be unchanged
        let den_out = den.to_host().unwrap();
        assert!((den_out[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_dt() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);

        let mut vel = [Field::<f64, _, 1>::zeros(&device, domain).unwrap()];
        let mut pre = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let mut rho = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // rho=1, v=0, p=1 -> cs = sqrt(1.4) ≈ 1.183
        rho.from_host(&vec![1.0; 10]).unwrap();
        vel[0].from_host(&vec![0.0; 10]).unwrap();
        pre.from_host(&vec![1.0; 10]).unwrap();

        let dt = compute_dt(&vel, &pre, &rho, GAMMA, [0.1], 0.5, domain).unwrap();

        let cs = (GAMMA * 1.0 / 1.0).sqrt();
        let expected = 0.5 * 0.1 / cs;

        assert!((dt - expected).abs() < 1e-8);
    }

    #[test]
    fn test_compute_dt_with_velocity() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);

        let mut vel = [Field::<f64, _, 1>::zeros(&device, domain).unwrap()];
        let mut pre = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let mut rho = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // rho=1, v=0.5, p=1 -> max_wave_speed = |v| + cs
        rho.from_host(&vec![1.0; 10]).unwrap();
        vel[0].from_host(&vec![0.5; 10]).unwrap();
        pre.from_host(&vec![1.0; 10]).unwrap();

        let dt = compute_dt(&vel, &pre, &rho, GAMMA, [0.1], 0.5, domain).unwrap();

        let cs = (GAMMA * 1.0 / 1.0).sqrt();
        let lambda = 0.5 + cs;
        let expected = 0.5 * 0.1 / lambda;

        assert!((dt - expected).abs() < 1e-8);
    }

    #[test]
    fn test_index_conversion() {
        let domain = Domain::from_shape([5, 5]);
        let strides = compute_strides(domain);

        let coord = [2, 3];
        let idx = coord_to_flux_index(coord, domain, &strides, 0);

        let recovered = index_to_coord(idx, domain, &strides);
        assert_eq!(recovered, coord);
    }
}
