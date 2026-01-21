// =============================================================================
// reconstruction.rs
//
// spatial reconstruction schemes for finite volume methods.
// implements piecewise constant (pcm) and piecewise linear (plm) reconstruction
// with various slope limiters for shock-capturing.
//
// design:
//   - pure functions: no side effects
//   - generic over numeric types
//   - slope limiters as compile-time strategy (zero overhead)
//   - works seamlessly with stencil views
//
// usage:
//   let samples = stencil.sample([i]);
//   let ul = plm_left(samples, Limiter::MinMod);
//   let ur = plm_right(samples, Limiter::MinMod);
//   let flux = riemann_solver(ul, ur);
//
// theory:
//   pcm: u_interface = u_cell (first-order accurate)
//   plm: u_interface = u_cell + slope * dx/2 (second-order accurate)
//        slope is limited to prevent oscillations at shocks
// =============================================================================

use core::ops::{Add, Div, Mul, Sub};

/// slope limiter functions for plm reconstruction.
/// limiters prevent spurious oscillations near discontinuities while
/// maintaining second-order accuracy in smooth regions.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Limiter {
    /// no limiting (unstable near shocks, use only for smooth flows)
    None,

    /// minmod limiter (most dissipative, tvd)
    /// φ(r) = max(0, min(1, r))
    MinMod,

    /// monotonized central (mc) limiter (less dissipative than minmod, tvd)
    /// φ(r) = max(0, min(2r, 0.5(1+r), 2))
    MC,

    /// van leer limiter (smooth, tvd)
    /// φ(r) = (r + |r|) / (1 + |r|)
    VanLeer,

    /// superbee limiter (least dissipative, tvd)
    /// φ(r) = max(0, min(2r, 1), min(r, 2))
    Superbee,
}

/// trait for types that support reconstruction operations.
pub trait Reconstructible:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
{
    /// returns zero value
    fn zero() -> Self;

    /// returns one value
    fn one() -> Self;

    /// returns two value
    fn two() -> Self;

    /// absolute value
    fn abs(self) -> Self;

    /// signum function (-1, 0, or 1)
    fn signum(self) -> Self;

    /// minimum of two values
    fn min(self, other: Self) -> Self;

    /// maximum of two values
    fn max(self, other: Self) -> Self;
}

// implement for f32 and f64
impl Reconstructible for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn two() -> Self {
        2.0
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline]
    fn signum(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
}

impl Reconstructible for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn two() -> Self {
        2.0
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn signum(self) -> Self {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
}

// =============================================================================
// pcm reconstruction (first-order)
// =============================================================================

/// pcm (piecewise constant) reconstruction for left state.
/// returns the cell value directly (no reconstruction needed).
#[inline]
pub fn pcm_left<T: Copy>(samples: [&T; 1]) -> T {
    *samples[0]
}

/// pcm (piecewise constant) reconstruction for right state.
/// returns the cell value directly (no reconstruction needed).
#[inline]
pub fn pcm_right<T: Copy>(samples: [&T; 1]) -> T {
    *samples[0]
}

// =============================================================================
// plm reconstruction (second-order)
// =============================================================================

/// plm (piecewise linear) reconstruction for left state at interface i.
///
/// samples: [u_{i-2}, u_{i-1}, u_i] (left-biased stencil)
///
/// reconstructs the value at the right edge of cell i-1, which is
/// the left state at interface i.
///
/// algorithm:
///   1. compute forward difference: df = u_i - u_{i-1}
///   2. compute backward difference: db = u_{i-1} - u_{i-2}
///   3. apply slope limiter: slope = limiter(df, db)
///   4. extrapolate: u_L = u_{i-1} + 0.5 * slope
#[inline]
pub fn plm_left<T: Reconstructible>(samples: [&T; 3], limiter: Limiter) -> T {
    let um2 = *samples[0]; // u_{i-2}
    let um1 = *samples[1]; // u_{i-1}
    let u0 = *samples[2]; // u_i

    let df = u0 - um1; // forward difference
    let db = um1 - um2; // backward difference

    let slope = apply_limiter(df, db, limiter);

    // extrapolate to right edge of cell i-1
    um1 + slope * T::one() / T::two()
}

/// plm (piecewise linear) reconstruction for right state at interface i.
///
/// samples: [u_{i-1}, u_i, u_{i+1}] (right-biased stencil)
///
/// reconstructs the value at the left edge of cell i, which is
/// the right state at interface i.
///
/// algorithm:
///   1. compute forward difference: df = u_{i+1} - u_i
///   2. compute backward difference: db = u_i - u_{i-1}
///   3. apply slope limiter: slope = limiter(df, db)
///   4. extrapolate: u_R = u_i - 0.5 * slope
#[inline]
pub fn plm_right<T: Reconstructible>(samples: [&T; 3], limiter: Limiter) -> T {
    let um1 = *samples[0]; // u_{i-1}
    let u0 = *samples[1]; // u_i
    let up1 = *samples[2]; // u_{i+1}

    let df = up1 - u0; // forward difference
    let db = u0 - um1; // backward difference

    let slope = apply_limiter(df, db, limiter);

    // extrapolate to left edge of cell i
    u0 - slope * T::one() / T::two()
}

// =============================================================================
// slope limiters
// =============================================================================

/// applies slope limiter to forward and backward differences.
///
/// all limiters satisfy the tvd condition for stability.
/// the limiter reduces to centered difference in smooth regions
/// and reduces to upwind/downwind near extrema.
#[inline]
fn apply_limiter<T: Reconstructible>(df: T, db: T, limiter: Limiter) -> T {
    match limiter {
        Limiter::None => centered_slope(df, db),
        Limiter::MinMod => minmod_limiter(df, db),
        Limiter::MC => mc_limiter(df, db),
        Limiter::VanLeer => van_leer_limiter(df, db),
        Limiter::Superbee => superbee_limiter(df, db),
    }
}

/// centered slope (no limiting, unstable near shocks).
/// slope = (df + db) / 2
#[inline]
fn centered_slope<T: Reconstructible>(df: T, db: T) -> T {
    (df + db) / T::two()
}

/// minmod limiter (most dissipative).
/// returns the difference with smallest magnitude if same sign, zero otherwise.
#[inline]
fn minmod_limiter<T: Reconstructible>(df: T, db: T) -> T {
    if df * db <= T::zero() {
        T::zero()
    } else if df.abs() < db.abs() {
        df
    } else {
        db
    }
}

/// monotonized central (mc) limiter.
/// compromise between minmod and centered difference.
#[inline]
fn mc_limiter<T: Reconstructible>(df: T, db: T) -> T {
    if df * db <= T::zero() {
        return T::zero();
    }

    let dc = centered_slope(df, db);
    let two = T::two();

    // mc = minmod(2*df, dc, 2*db)
    let a = (two * df).abs();
    let b = dc.abs();
    let c = (two * db).abs();

    let min_abc = a.min(b).min(c);

    dc.signum() * min_abc
}

/// van leer limiter (smooth).
/// uses harmonic mean of differences.
#[inline]
fn van_leer_limiter<T: Reconstructible>(df: T, db: T) -> T {
    if df * db <= T::zero() {
        T::zero()
    } else {
        let two = T::two();
        (two * df * db) / (df + db)
    }
}

/// superbee limiter (least dissipative, sharpest shocks).
/// most aggressive limiter while maintaining tvd property.
#[inline]
fn superbee_limiter<T: Reconstructible>(df: T, db: T) -> T {
    if df * db <= T::zero() {
        return T::zero();
    }

    let two = T::two();
    let sigma1 = (two * df).abs().min(db.abs());
    let sigma2 = df.abs().min((two * db).abs());

    df.signum() * sigma1.max(sigma2)
}

// =============================================================================
// vector reconstruction (for multi-component fields)
// =============================================================================

/// reconstructs an array of values component-wise.
/// useful for primitive variables [rho, vx, vy, vz, p].
#[inline]
pub fn plm_left_vector<T: Reconstructible, const NCOMP: usize>(
    samples: [[&T; 3]; NCOMP],
    limiter: Limiter,
) -> [T; NCOMP] {
    let mut result = [T::zero(); NCOMP];
    let mut ii = 0;
    while ii < NCOMP {
        result[ii] = plm_left(samples[ii], limiter);
        ii += 1;
    }
    result
}

/// reconstructs an array of values component-wise.
#[inline]
pub fn plm_right_vector<T: Reconstructible, const NCOMP: usize>(
    samples: [[&T; 3]; NCOMP],
    limiter: Limiter,
) -> [T; NCOMP] {
    let mut result = [T::zero(); NCOMP];
    let mut ii = 0;
    while ii < NCOMP {
        result[ii] = plm_right(samples[ii], limiter);
        ii += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_reconstruction() {
        let value = 5.0;
        let samples = [&value];

        assert_eq!(pcm_left(samples), 5.0);
        assert_eq!(pcm_right(samples), 5.0);
    }

    #[test]
    fn test_plm_smooth_flow() {
        // linear profile: u = x
        let samples = [&1.0, &2.0, &3.0];

        // in smooth linear region, all limiters should give same result
        let ul = plm_left(samples, Limiter::MinMod);
        let ur = plm_right(samples, Limiter::MinMod);

        // left state at interface: 2.0 + 0.5 * 1.0 = 2.5
        assert!((ul - 2.5).abs() < 1e-10);

        // right state at interface: 2.0 - 0.5 * 1.0 = 1.5
        assert!((ur - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_plm_at_extremum() {
        // local maximum: should reduce to pcm
        let samples = [&1.0, &3.0, &2.0];

        let ul = plm_left(samples, Limiter::MinMod);

        // at extremum, slope should be limited to zero
        // so ul = u_{i-1} = 3.0
        assert!((ul - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_plm_at_shock() {
        // discontinuity
        let samples_left = [&1.0, &1.0, &10.0];

        let ul = plm_left(samples_left, Limiter::MinMod);

        // minmod should see zero backward diff, forward diff = 9
        // slope = 0, so ul = 1.0
        assert!((ul - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmod_limiter() {
        // same sign, df smaller
        let slope = minmod_limiter(1.0, 2.0);
        assert_eq!(slope, 1.0);

        // same sign, db smaller
        let slope = minmod_limiter(2.0, 1.0);
        assert_eq!(slope, 1.0);

        // opposite signs
        let slope = minmod_limiter(1.0, -1.0);
        assert_eq!(slope, 0.0);

        // one is zero
        let slope = minmod_limiter(0.0, 1.0);
        assert_eq!(slope, 0.0);
    }

    #[test]
    fn test_mc_limiter() {
        // smooth region: should be close to centered
        let df = 1.0;
        let db = 1.0;
        let slope = mc_limiter(df, db);
        assert_eq!(slope, 1.0); // mc = min(2, 1, 2) = 1

        // at extremum
        let slope = mc_limiter(1.0, -1.0);
        assert_eq!(slope, 0.0);
    }

    #[test]
    fn test_van_leer_limiter() {
        // smooth region
        let df = 1.0;
        let db = 1.0;
        let slope = van_leer_limiter(df, db);
        assert_eq!(slope, 1.0); // harmonic mean of equal values

        // at extremum
        let slope = van_leer_limiter(1.0, -1.0);
        assert_eq!(slope, 0.0);
    }

    #[test]
    fn test_superbee_limiter() {
        // smooth region
        let df = 1.0;
        let db = 1.0;
        let slope = superbee_limiter(df, db);
        assert_eq!(slope, 1.0);

        // at extremum
        let slope = superbee_limiter(1.0, -1.0);
        assert_eq!(slope, 0.0);
    }

    #[test]
    fn test_limiter_ordering() {
        // in smooth regions: minmod is most dissipative (smallest slope)
        // superbee is least dissipative (largest slope, closest to unlimited)
        let df = 2.0;
        let db = 1.0;

        let s_minmod = minmod_limiter(df, db);
        let s_vl = van_leer_limiter(df, db);
        let s_mc = mc_limiter(df, db);
        let s_superbee = superbee_limiter(df, db);

        // verify minmod gives smallest slope (most dissipative)
        assert_eq!(s_minmod, 1.0); // min(2, 1) = 1

        // all limiters should be >= minmod and <= 2*minmod in this case
        assert!(s_vl >= s_minmod);
        assert!(s_mc >= s_minmod);
        assert!(s_superbee >= s_minmod);
    }

    #[test]
    fn test_plm_different_limiters() {
        let samples = [&1.0, &2.0, &3.5];

        let ul_minmod = plm_left(samples, Limiter::MinMod);
        let ul_mc = plm_left(samples, Limiter::MC);
        let ul_vl = plm_left(samples, Limiter::VanLeer);

        // all should be between cell value and full extrapolation
        assert!(ul_minmod >= 2.0 && ul_minmod <= 3.0);
        assert!(ul_mc >= 2.0 && ul_mc <= 3.0);
        assert!(ul_vl >= 2.0 && ul_vl <= 3.0);

        // ordering: less dissipative limiters give larger slopes
        assert!(ul_mc >= ul_minmod);
    }

    #[test]
    fn test_vector_reconstruction() {
        // reconstruct primitive vector: [rho, vx, p]
        let rho_samples = [&1.0, &2.0, &3.0];
        let vx_samples = [&0.0, &1.0, &2.0];
        let p_samples = [&10.0, &11.0, &12.0];

        let samples = [rho_samples, vx_samples, p_samples];

        let prim_left = plm_left_vector::<f64, 3>(samples, Limiter::MinMod);

        // all components smooth, should extrapolate
        assert!((prim_left[0] - 2.5).abs() < 1e-10); // rho
        assert!((prim_left[1] - 1.5).abs() < 1e-10); // vx
        assert!((prim_left[2] - 11.5).abs() < 1e-10); // p
    }

    #[test]
    fn test_f32_reconstruction() {
        // verify it works with f32 too
        let samples: [&f32; 3] = [&1.0f32, &2.0f32, &3.0f32];
        let ul = plm_left(samples, Limiter::MinMod);
        assert!((ul - 2.5f32).abs() < 1e-6);
    }

    #[test]
    fn test_stencil_reconstruction_integration() {
        // demonstrates full workflow: field -> stencil -> reconstruction
        use crate::domain::Domain;
        use crate::field::Field;
        use crate::stencil::{Reconstruction, StencilView};
        use xpu_core::Device;
        use xpu_host::CpuDevice;

        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut rho = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // smooth density profile
        let data: Vec<f64> = (0..10).map(|i| 1.0 + 0.1 * (i as f64)).collect();
        rho.from_host(&data).unwrap();

        // create stencil views
        let left_stencil = StencilView::<_, 1, 3>::left(rho.view(), Reconstruction::PLM, 0);
        let right_stencil = StencilView::<_, 1, 3>::right(rho.view(), Reconstruction::PLM, 0);

        // reconstruct at interface i=5
        let ul_samples = left_stencil.sample([5]);
        let ur_samples = right_stencil.sample([5]);

        // apply plm reconstruction with minmod limiter
        let ul = plm_left(ul_samples, Limiter::MinMod);
        let ur = plm_right(ur_samples, Limiter::MinMod);

        // left state: reconstructs at right edge of cell i-1 (i=4)
        // u[4] = 1.4, with slope ~0.1, so ul = 1.4 + 0.5*0.1 = 1.45
        // right state: reconstructs at left edge of cell i (i=5)
        // u[5] = 1.5, with slope ~0.1, so ur = 1.5 - 0.5*0.1 = 1.45

        // both should be around 1.45 at the interface
        assert!((ul - 1.45).abs() < 0.01);
        assert!((ur - 1.45).abs() < 0.01);

        // verify they're close to each other (continuous reconstruction)
        assert!((ul - ur).abs() < 0.01);
    }

    #[test]
    fn test_shock_capturing_with_stencils() {
        // demonstrates limiters prevent oscillations at shocks
        use crate::domain::Domain;
        use crate::field::Field;
        use crate::stencil::{Reconstruction, StencilView};
        use xpu_core::Device;
        use xpu_host::CpuDevice;

        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);
        let mut rho = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // shock profile: low density, then sudden jump
        let data: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { 10.0 }).collect();
        rho.from_host(&data).unwrap();

        // stencils at shock interface (i=5)
        let left_stencil = StencilView::<_, 1, 3>::left(rho.view(), Reconstruction::PLM, 0);
        let right_stencil = StencilView::<_, 1, 3>::right(rho.view(), Reconstruction::PLM, 0);

        let ul_samples = left_stencil.sample([5]);
        let ur_samples = right_stencil.sample([5]);

        // with minmod limiter, should not overshoot
        let ul = plm_left(ul_samples, Limiter::MinMod);
        let ur = plm_right(ur_samples, Limiter::MinMod);

        // left state should stay close to 1.0 (no overshoot)
        assert!(ul >= 1.0 && ul <= 1.1);

        // right state should stay close to 10.0 (no overshoot)
        assert!(ur >= 9.9 && ur <= 10.0);
    }

    #[test]
    fn test_multicomponent_reconstruction_workflow() {
        // demonstrates vector reconstruction for primitive variables
        use crate::domain::Domain;
        use crate::field::Field;
        use crate::stencil::{Reconstruction, StencilView};
        use xpu_core::Device;
        use xpu_host::CpuDevice;

        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10]);

        // create fields for rho, vx, p
        let mut rho = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let mut vx = Field::<f64, _, 1>::zeros(&device, domain).unwrap();
        let mut p = Field::<f64, _, 1>::zeros(&device, domain).unwrap();

        // smooth profiles
        let rho_data: Vec<f64> = (0..10).map(|i| 1.0 + 0.1 * (i as f64)).collect();
        let vx_data: Vec<f64> = (0..10).map(|i| 0.1 * (i as f64)).collect();
        let p_data: Vec<f64> = (0..10).map(|i| 10.0 + 0.5 * (i as f64)).collect();

        rho.from_host(&rho_data).unwrap();
        vx.from_host(&vx_data).unwrap();
        p.from_host(&p_data).unwrap();

        // create stencils for each component
        let rho_stencil = StencilView::<_, 1, 3>::left(rho.view(), Reconstruction::PLM, 0);
        let vx_stencil = StencilView::<_, 1, 3>::left(vx.view(), Reconstruction::PLM, 0);
        let p_stencil = StencilView::<_, 1, 3>::left(p.view(), Reconstruction::PLM, 0);

        // sample at interface
        let rho_samples = rho_stencil.sample([5]);
        let vx_samples = vx_stencil.sample([5]);
        let p_samples = p_stencil.sample([5]);

        // reconstruct each component
        let rho_l = plm_left(rho_samples, Limiter::MinMod);
        let vx_l = plm_left(vx_samples, Limiter::MinMod);
        let p_l = plm_left(p_samples, Limiter::MinMod);

        // verify all components reconstructed sensibly
        assert!(rho_l > 1.3 && rho_l < 1.6);
        assert!(vx_l > 0.4 && vx_l < 0.6);
        assert!(p_l > 12.0 && p_l < 13.0);
    }
}
