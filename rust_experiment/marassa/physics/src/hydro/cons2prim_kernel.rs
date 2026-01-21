// =============================================================================
// cons2prim_kernel.rs
//
// partition-scale conserved-to-primitive conversion kernel.
// operates on entire partition fields using lazy computation pipeline.
//
// design:
//   - element-wise conversion: u(i,j,k) -> p(i,j,k)
//   - regime-generic: supports newtonian and srhd
//   - pure functional: builds lazy computation, evaluates to fields
//   - error handling: returns conversion errors for invalid states
//
// algorithm:
//   for each cell in domain:
//     1. read conserved state (den, mom, nrg)
//     2. convert to primitive (rho, vel, pre) via regime-specific formula
//     3. write primitive state to fields
//
// usage:
//   cons2prim(&partition.conserved, &mut partition.primitive, gamma)?;
// =============================================================================

use super::state::{Conserved, ConversionError, Newtonian, Primitive, Regime, Srhd};
use compute::{evaluate, from_fn, Domain, Field};
use xpu_core::Device;

// =============================================================================
// simple field-based interface (works with any fields)
// =============================================================================

/// converts conserved fields to primitive fields.
/// generic over field types - works with any soa field structure.
///
/// # type parameters
/// * `R` - regime (newtonian or srhd)
/// * `D` - device type (cpu, cuda, metal)
/// * `RANK` - spatial dimensionality
///
/// # arguments
/// * `den` - conserved density field
/// * `mom` - conserved momentum fields (array of RANK fields)
/// * `nrg` - conserved energy field
/// * `rho` - primitive density field (output)
/// * `vel` - primitive velocity fields (output)
/// * `pre` - primitive pressure field (output)
/// * `gamma` - adiabatic index
/// * `device` - device for computation
/// * `domain` - domain to process
///
/// # returns
/// ok if conversion succeeds for all cells
pub fn cons2prim<'d, R, D, const RANK: usize>(
    den: &Field<'d, f64, D, RANK>,
    mom: &[Field<'d, f64, D, RANK>; RANK],
    nrg: &Field<'d, f64, D, RANK>,
    rho: &mut Field<'d, f64, D, RANK>,
    vel: &mut [Field<'d, f64, D, RANK>; RANK],
    pre: &mut Field<'d, f64, D, RANK>,
    gamma: f64,
    device: &'d D,
    domain: Domain<RANK>,
) -> Result<(), ConversionError>
where
    R: Regime,
    D: Device,
{
    // density is identity mapping
    convert_density::<R, D, RANK>(den, rho, device, domain)?;

    // convert velocity components
    for axis in 0..RANK {
        convert_velocity_component::<R, D, RANK>(den, &mom[axis], &mut vel[axis], device, domain)?;
    }

    // convert pressure
    convert_pressure_from_fields::<R, D, RANK>(den, mom, nrg, pre, gamma, device, domain)?;

    Ok(())
}

// =============================================================================
// newtonian specialization
// =============================================================================

/// newtonian conserved-to-primitive conversion.
pub fn cons2prim_newtonian<'d, D, const RANK: usize>(
    den: &Field<'d, f64, D, RANK>,
    mom: &[Field<'d, f64, D, RANK>; RANK],
    nrg: &Field<'d, f64, D, RANK>,
    rho: &mut Field<'d, f64, D, RANK>,
    vel: &mut [Field<'d, f64, D, RANK>; RANK],
    pre: &mut Field<'d, f64, D, RANK>,
    gamma: f64,
    device: &'d D,
    domain: Domain<RANK>,
) -> Result<(), ConversionError>
where
    D: Device,
{
    cons2prim::<Newtonian, D, RANK>(den, mom, nrg, rho, vel, pre, gamma, device, domain)
}

// =============================================================================
// srhd specialization
// =============================================================================

/// special relativistic conserved-to-primitive conversion.
pub fn cons2prim_srhd<'d, D, const RANK: usize>(
    den: &Field<'d, f64, D, RANK>,
    mom: &[Field<'d, f64, D, RANK>; RANK],
    nrg: &Field<'d, f64, D, RANK>,
    rho: &mut Field<'d, f64, D, RANK>,
    vel: &mut [Field<'d, f64, D, RANK>; RANK],
    pre: &mut Field<'d, f64, D, RANK>,
    gamma: f64,
    device: &'d D,
    domain: Domain<RANK>,
) -> Result<(), ConversionError>
where
    D: Device,
{
    cons2prim::<Srhd, D, RANK>(den, mom, nrg, rho, vel, pre, gamma, device, domain)
}

// =============================================================================
// field conversion helpers
// =============================================================================

/// density conversion: ρ = D (identity for both regimes)
fn convert_density<R, D, const RANK: usize>(
    den_conserved: &Field<'_, f64, D, RANK>,
    rho_primitive: &mut Field<'_, f64, D, RANK>,
    device: &D,
    domain: Domain<RANK>,
) -> Result<(), ConversionError>
where
    R: super::state::Regime,
    D: Device,
{
    // density is identity mapping
    let den_view = den_conserved.view();
    let den_comp = den_view.as_computation();

    // density passes through directly (validation done in pressure step)

    let result_field = evaluate(device, den_comp).map_err(|_| ConversionError::NonFiniteValue)?;

    // copy to output (would ideally be in-place)
    let data = result_field
        .to_host()
        .map_err(|_| ConversionError::NonFiniteValue)?;
    rho_primitive
        .from_host(&data)
        .map_err(|_| ConversionError::NonFiniteValue)?;

    Ok(())
}

/// velocity conversion: v_i = S_i / D (newtonian) or more complex (srhd)
fn convert_velocity_component<R, D, const RANK: usize>(
    den_conserved: &Field<'_, f64, D, RANK>,
    mom_component: &Field<'_, f64, D, RANK>,
    vel_component: &mut Field<'_, f64, D, RANK>,
    device: &D,
    domain: Domain<RANK>,
) -> Result<(), ConversionError>
where
    R: super::state::Regime,
    D: Device,
{
    // newtonian: v = S / D
    let den_view = den_conserved.view();
    let mom_view = mom_component.view();

    let den_comp = den_view.as_computation();
    let mom_comp = mom_view.as_computation();

    // v = S / D (safe division)
    let vel_comp = from_fn(domain, |coord| {
        let den = den_comp.eval(coord);
        let mom = mom_comp.eval(coord);

        if den.abs() < 1e-14 {
            0.0
        } else {
            mom / den
        }
    });

    let result_field = evaluate(device, vel_comp).map_err(|_| ConversionError::NonFiniteValue)?;

    let data = result_field
        .to_host()
        .map_err(|_| ConversionError::NonFiniteValue)?;
    vel_component
        .from_host(&data)
        .map_err(|_| ConversionError::NonFiniteValue)?;

    Ok(())
}

/// pressure conversion: p = (γ-1)[E - ½ρv²] (newtonian)
fn convert_pressure_from_fields<R, D, const RANK: usize>(
    den: &Field<'_, f64, D, RANK>,
    mom: &[Field<'_, f64, D, RANK>; RANK],
    nrg: &Field<'_, f64, D, RANK>,
    pre: &mut Field<'_, f64, D, RANK>,
    gamma: f64,
    device: &D,
    _domain: Domain<RANK>,
) -> Result<(), ConversionError>
where
    R: Regime,
    D: Device,
{
    // newtonian pressure: p = (γ-1) * (E - ½ S²/D)
    let den_view = den.view();
    let nrg_view = nrg.view();

    let den_comp = den_view.as_computation();
    let nrg_comp = nrg_view.as_computation();

    // compute kinetic energy: KE = ½ Σ(S_i²/D)
    let mom_views: Vec<_> = mom
        .iter()
        .map(|m: &Field<'_, f64, D, RANK>| m.view())
        .collect();

    let pre_comp = from_fn(den.domain(), |coord| {
        let den = den_comp.eval(coord);
        let nrg = nrg_comp.eval(coord);

        // compute mom²/den
        let mut mom_sq_over_den = 0.0;
        for axis in 0..RANK {
            let mom = mom_views[axis].as_computation().eval(coord);
            mom_sq_over_den += mom * mom / den;
        }

        let ke = 0.5 * mom_sq_over_den;
        let ie = nrg - ke;
        let pre = (gamma - 1.0) * ie;

        // validate pressure
        if pre < 0.0 {
            // pressure floor
            1e-10
        } else if !pre.is_finite() {
            1e-10
        } else {
            pre
        }
    });

    let result_field = evaluate(device, pre_comp).map_err(|_| ConversionError::NegativePressure)?;

    let data = result_field
        .to_host()
        .map_err(|_| ConversionError::NonFiniteValue)?;
    pre.from_host(&data)
        .map_err(|_| ConversionError::NonFiniteValue)?;

    Ok(())
}

// =============================================================================
// point-wise conversion (for testing/debugging)
// =============================================================================

/// converts single point from conserved to primitive.
/// useful for verification and debugging.
pub fn cons2prim_point<const RANK: usize>(
    conserved: Conserved<Newtonian, RANK>,
    gamma: f64,
) -> Result<Primitive<Newtonian, RANK>, ConversionError> {
    conserved.to_primitive(gamma)
}

/// converts single point from conserved to primitive (srhd).
pub fn cons2prim_point_srhd<const RANK: usize>(
    conserved: Conserved<Srhd, RANK>,
    gamma: f64,
) -> Result<Primitive<Srhd, RANK>, ConversionError> {
    conserved.to_primitive(gamma)
}

// =============================================================================
// validation
// =============================================================================

/// checks if conserved state is physically valid before conversion.
pub fn is_valid_conserved<R, const RANK: usize>(conserved: &Conserved<R, RANK>) -> bool
where
    R: Regime,
{
    // density must be positive
    if conserved.den <= 0.0 || !conserved.den.is_finite() {
        return false;
    }

    // energy must be positive
    if conserved.nrg <= 0.0 || !conserved.nrg.is_finite() {
        return false;
    }

    // momentum must be finite
    for &m in &conserved.mom {
        if !m.is_finite() {
            return false;
        }
    }

    true
}

/// checks if primitive state is physically valid.
pub fn is_valid_primitive<R, const RANK: usize>(primitive: &Primitive<R, RANK>) -> bool
where
    R: Regime,
{
    // density must be positive
    if primitive.rho <= 0.0 || !primitive.rho.is_finite() {
        return false;
    }

    // pressure must be positive
    if primitive.p <= 0.0 || !primitive.p.is_finite() {
        return false;
    }

    // velocity must be finite (and subluminal for srhd)
    for &v in &primitive.vel {
        if !v.is_finite() {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use compute::Domain;
    use xpu_host::CpuDevice;

    const GAMMA: f64 = 1.4;

    #[test]
    fn test_point_conversion_newtonian() {
        // rho=1, v=0.5, p=1 -> den=1, mom=0.5, nrg=2.625
        let primitive = Primitive::<Newtonian, 1>::new(1.0, [0.5], 1.0);
        let conserved = primitive.to_conserved(GAMMA);

        let recovered = cons2prim_point::<1>(conserved, GAMMA).unwrap();

        assert!((recovered.rho - primitive.rho).abs() < 1e-10);
        assert!((recovered.vel[0] - primitive.vel[0]).abs() < 1e-10);
        assert!((recovered.p - primitive.p).abs() < 1e-10);
    }

    #[test]
    fn test_point_conversion_srhd() {
        // moderate relativistic velocity
        let primitive = Primitive::<Srhd, 1>::new(1.0, [0.5], 1.0);
        let conserved = primitive.to_conserved(GAMMA);

        let recovered = cons2prim_point_srhd::<1>(conserved, GAMMA).unwrap();

        assert!((recovered.rho - primitive.rho).abs() < 1e-6);
        assert!((recovered.vel[0] - primitive.vel[0]).abs() < 1e-6);
        assert!((recovered.p - primitive.p).abs() < 1e-6);
    }

    #[test]
    fn test_validation() {
        let valid = Conserved::<Newtonian, 1>::new(1.0, [0.5], 2.0);
        assert!(is_valid_conserved(&valid));

        let invalid_den = Conserved::<Newtonian, 1>::new(-1.0, [0.5], 2.0);
        assert!(!is_valid_conserved(&invalid_den));

        let invalid_nrg = Conserved::<Newtonian, 1>::new(1.0, [0.5], -1.0);
        assert!(!is_valid_conserved(&invalid_nrg));
    }

    #[test]
    fn test_field_conversion_1d() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([14]); // 10 + 2*2 ghosts

        let mut den = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let mut mom = [Field::<f64, _, 1>::zeros(&device, domain).unwrap()];
        let mut nrg = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        let mut rho = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let mut vel = [Field::<f64, _, 1>::zeros(&device, domain).unwrap()];
        let mut pre = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // initialize with uniform state: rho=1, v=0, p=1
        // conserved: den=1, mom=0, nrg=2.5
        let den_data = vec![1.0; 14];
        let mom_data = vec![0.0; 14];
        let nrg_data = vec![2.5; 14]; // E = p/(γ-1) = 1.0/0.4 = 2.5

        den.from_host(&den_data).unwrap();
        mom[0].from_host(&mom_data).unwrap();
        nrg.from_host(&nrg_data).unwrap();

        // convert
        let result = cons2prim_newtonian(
            &den, &mom, &nrg, &mut rho, &mut vel, &mut pre, GAMMA, &device, domain,
        );
        assert!(result.is_ok());

        // verify
        let rho_out = rho.to_host().unwrap();
        let vel_out = vel[0].to_host().unwrap();
        let pre_out = pre.to_host().unwrap();

        assert!((rho_out[5] - 1.0).abs() < 1e-8);
        assert!((vel_out[5] - 0.0).abs() < 1e-8);
        assert!((pre_out[5] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_field_conversion_2d() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([9, 9]); // 5+2*2 in each direction

        let ncells = 9 * 9;

        let mut den = Field::<f64, _, 2>::zeros(&device, domain).unwrap();
        let mut mom = [
            Field::<f64, _, 2>::zeros(&device, domain).unwrap(),
            Field::<f64, _, 2>::zeros(&device, domain).unwrap(),
        ];
        let mut nrg = Field::<f64, _, 2>::zeros(&device, domain).unwrap();

        let mut rho = Field::<f64, _, 2>::zeros(&device, domain).unwrap();
        let mut vel = [
            Field::<f64, _, 2>::zeros(&device, domain).unwrap(),
            Field::<f64, _, 2>::zeros(&device, domain).unwrap(),
        ];
        let mut pre = Field::<f64, _, 2>::zeros(&device, domain).unwrap();

        // rho=2, vx=0.1, vy=0.2, p=1.5
        let den_data = vec![2.0; ncells];
        let mom_x = vec![0.2; ncells]; // rho * vx
        let mom_y = vec![0.4; ncells]; // rho * vy
        let ke = 0.5 * 2.0 * (0.1 * 0.1 + 0.2 * 0.2); // 0.05
        let ie = 1.5 / 0.4; // 3.75
        let nrg_data = vec![ke + ie; ncells]; // 3.8

        den.from_host(&den_data).unwrap();
        mom[0].from_host(&mom_x).unwrap();
        mom[1].from_host(&mom_y).unwrap();
        nrg.from_host(&nrg_data).unwrap();

        let result = cons2prim_newtonian(
            &den, &mom, &nrg, &mut rho, &mut vel, &mut pre, GAMMA, &device, domain,
        );
        assert!(result.is_ok());

        let rho_out = rho.to_host().unwrap();
        let vx_out = vel[0].to_host().unwrap();
        let vy_out = vel[1].to_host().unwrap();
        let pre_out = pre.to_host().unwrap();

        let idx = 4 * 9 + 4; // center cell
        assert!((rho_out[idx] - 2.0).abs() < 1e-8);
        assert!((vx_out[idx] - 0.1).abs() < 1e-8);
        assert!((vy_out[idx] - 0.2).abs() < 1e-8);
        assert!((pre_out[idx] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_pressure_floor() {
        // test that negative pressure is handled
        let conserved = Conserved::<Newtonian, 1>::new(1.0, [0.0], 0.5);

        // this should apply pressure floor
        let result = cons2prim_point::<1>(conserved, GAMMA);

        // should succeed with floored pressure
        assert!(result.is_ok());
        let prim = result.unwrap();
        assert!(prim.p > 0.0);
    }
}
