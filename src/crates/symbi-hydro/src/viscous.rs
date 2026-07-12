// =============================================================================
// viscous.rs
//
// the constant-nu Navier-Stokes shear operator (docs/design/54), carrier-generic
// (f64 oracle / Gv trace / Dual). the shear stress
//   tau_ij = rho nu ( d_i v_j + d_j v_i - (2/3) delta_ij d_k v_k )
// is evaluated at the FOUR cell faces from a halo-1 velocity stencil and
// DIFFERENCED into the momentum, so the update is a conservative flux divergence:
// momentum is conserved to roundoff and angular momentum is TRANSPORTED (the
// r-phi shear), never created. bulk viscosity zeta = 0 (the disk convention).
//
// isothermal only here: no energy channel (the viscous heating is not booked; a
// locally-isothermal disk radiates it instantly by assumption). the r-phi shear
// is what drives disk accretion, so the momentum operator is the load-bearing
// piece; the energy twin (v_i tau_ij flux + heating) is design-54 build step 3.
//
// stencil convention: v[jj][ii] and rho[jj][ii] with jj, ii in {0, 1, 2} for the
// grid offsets {-1, 0, +1} along (y, x); the center cell is [1][1]. the face
// stress is a single-valued function of the two straddling cells (plus their
// transverse neighbors), so adjacent cells agree on the shared face and the
// differenced update telescopes.
//
// usage:
//   let dmom = viscous_mom_update_2d(&v, &rho, nu, dx, dy, dt);
//   cons.mom = cons.mom + dmom;   // additive, post-godunov
// =============================================================================

use symbi_algebra::Tensor;
use symbi_ir::algebra::Scalar;

/// the viscous momentum increment `dt * div(tau)` for the center cell of a 3x3
/// velocity + density + viscosity stencil (2D, cartesian). additive onto
/// `cons.mom`. `nu` is per-cell: constant-nu passes a uniform stencil (the face
/// average `0.5(nu + nu) = nu` is bit-identical to a scalar); alpha passes
/// `nu(x) = alpha c_s^2 / Omega_k(r)`. the FACE viscosity is the average of the
/// two straddling cells' `nu` — single-valued at the face, so the update stays
/// conservative under a spatially varying viscosity.
pub fn viscous_mom_update_2d<S: Scalar>(
    v: &[[Tensor<S, 2>; 3]; 3],
    rho: &[[S; 3]; 3],
    nu: &[[S; 3]; 3],
    dx: S,
    dy: S,
    dt: S,
) -> Tensor<S, 2> {
    let two = S::from_f64(2.0);
    let four = S::from_f64(4.0);
    let two_thirds = S::from_f64(2.0 / 3.0);
    let half = S::from_f64(0.5);

    // vx[jj][ii] / vy[jj][ii] accessors keep the face formulas readable.
    let vx = |jj: usize, ii: usize| v[jj][ii][0];
    let vy = |jj: usize, ii: usize| v[jj][ii][1];

    // --- x-face i+1/2 (between center [1][1] and [1][2]) ---
    // normal gradient: central across the face. transverse gradient: averaged
    // over the two straddling cells (a 4-point central difference in y).
    // the face DYNAMIC viscosity mu = rho_face * nu_face.
    let mu_xp = half * (rho[1][1] + rho[1][2]) * (half * (nu[1][1] + nu[1][2]));
    let dvxdx = (vx(1, 2) - vx(1, 1)) / dx;
    let dvydx = (vy(1, 2) - vy(1, 1)) / dx;
    let dvxdy = ((vx(2, 1) - vx(0, 1)) + (vx(2, 2) - vx(0, 2))) / (four * dy);
    let dvydy = ((vy(2, 1) - vy(0, 1)) + (vy(2, 2) - vy(0, 2))) / (four * dy);
    let div = dvxdx + dvydy;
    let txx_xp = mu_xp * (two * dvxdx - two_thirds * div);
    let txy_xp = mu_xp * (dvxdy + dvydx);

    // --- x-face i-1/2 (between [1][0] and center [1][1]) ---
    let mu_xm = half * (rho[1][0] + rho[1][1]) * (half * (nu[1][0] + nu[1][1]));
    let dvxdx = (vx(1, 1) - vx(1, 0)) / dx;
    let dvydx = (vy(1, 1) - vy(1, 0)) / dx;
    let dvxdy = ((vx(2, 0) - vx(0, 0)) + (vx(2, 1) - vx(0, 1))) / (four * dy);
    let dvydy = ((vy(2, 0) - vy(0, 0)) + (vy(2, 1) - vy(0, 1))) / (four * dy);
    let div = dvxdx + dvydy;
    let txx_xm = mu_xm * (two * dvxdx - two_thirds * div);
    let txy_xm = mu_xm * (dvxdy + dvydx);

    // --- y-face j+1/2 (between center [1][1] and [2][1]) ---
    let mu_yp = half * (rho[1][1] + rho[2][1]) * (half * (nu[1][1] + nu[2][1]));
    let dvxdy = (vx(2, 1) - vx(1, 1)) / dy;
    let dvydy = (vy(2, 1) - vy(1, 1)) / dy;
    let dvxdx = ((vx(1, 2) - vx(1, 0)) + (vx(2, 2) - vx(2, 0))) / (four * dx);
    let dvydx = ((vy(1, 2) - vy(1, 0)) + (vy(2, 2) - vy(2, 0))) / (four * dx);
    let div = dvxdx + dvydy;
    let tyx_yp = mu_yp * (dvxdy + dvydx);
    let tyy_yp = mu_yp * (two * dvydy - two_thirds * div);

    // --- y-face j-1/2 (between [0][1] and center [1][1]) ---
    let mu_ym = half * (rho[0][1] + rho[1][1]) * (half * (nu[0][1] + nu[1][1]));
    let dvxdy = (vx(1, 1) - vx(0, 1)) / dy;
    let dvydy = (vy(1, 1) - vy(0, 1)) / dy;
    let dvxdx = ((vx(1, 2) - vx(1, 0)) + (vx(0, 2) - vx(0, 0))) / (four * dx);
    let dvydx = ((vy(1, 2) - vy(1, 0)) + (vy(0, 2) - vy(0, 0))) / (four * dx);
    let div = dvxdx + dvydy;
    let tyx_ym = mu_ym * (dvxdy + dvydx);
    let tyy_ym = mu_ym * (two * dvydy - two_thirds * div);

    // conservative flux divergence: d_x tau_x. + d_y tau_y.
    let dmom_x = dt * ((txx_xp - txx_xm) / dx + (tyx_yp - tyx_ym) / dy);
    let dmom_y = dt * ((txy_xp - txy_xm) / dx + (tyy_yp - tyy_ym) / dy);
    Tensor::new([dmom_x, dmom_y])
}

#[cfg(test)]
mod tests {
    use super::*;

    // build a 3x3 stencil by sampling v(x, y) and rho(x, y) at the cell centers
    // around (x0, y0). ii/jj in {0,1,2} -> offsets {-1,0,+1} * (dx, dy).
    fn stencil<FV, FR>(
        x0: f64,
        y0: f64,
        dx: f64,
        dy: f64,
        vfun: FV,
        rfun: FR,
    ) -> ([[Tensor<f64, 2>; 3]; 3], [[f64; 3]; 3])
    where
        FV: Fn(f64, f64) -> [f64; 2],
        FR: Fn(f64, f64) -> f64,
    {
        let mut v = [[Tensor::zeros(); 3]; 3];
        let mut r = [[0.0; 3]; 3];
        for jj in 0..3 {
            for ii in 0..3 {
                let x = x0 + (ii as f64 - 1.0) * dx;
                let y = y0 + (jj as f64 - 1.0) * dy;
                v[jj][ii] = Tensor::new(vfun(x, y));
                r[jj][ii] = rfun(x, y);
            }
        }
        (v, r)
    }

    // a uniform (constant-nu) viscosity stencil.
    fn uni(nu: f64) -> [[f64; 3]; 3] {
        [[nu; 3]; 3]
    }

    // null 1: a uniform velocity has zero strain -> zero viscous force, exactly.
    #[test]
    fn uniform_flow_books_zero_force() {
        let (v, r) = stencil(0.3, -0.2, 0.05, 0.05, |_, _| [0.7, -0.4], |_, _| 1.3);
        let d = viscous_mom_update_2d(&v, &r, &uni(0.01), 0.05, 0.05, 0.001);
        assert!(d[0].abs() < 1e-15 && d[1].abs() < 1e-15, "{d:?}");
    }

    // null 2: rigid rotation v = (-omega y, omega x) is strain-free (the
    // symmetric velocity gradient vanishes) -> zero viscous force. THE null that
    // catches a sign error or a missing trace subtraction.
    #[test]
    fn rigid_rotation_books_zero_force() {
        let omega = 1.7;
        let (v, r) = stencil(
            0.4,
            0.25,
            0.05,
            0.05,
            |x, y| [-omega * y, omega * x],
            |_, _| 2.0,
        );
        let d = viscous_mom_update_2d(&v, &r, &uni(0.03), 0.05, 0.05, 0.01);
        assert!(d[0].abs() < 1e-14 && d[1].abs() < 1e-14, "{d:?}");
    }

    // null 3: a linear shear vx = S y has a CONSTANT stress -> zero divergence ->
    // zero force (constant rho). the force appears only when the stress varies.
    #[test]
    fn linear_shear_books_zero_force() {
        let s = 0.9;
        let (v, r) = stencil(0.1, 0.3, 0.05, 0.05, |_, y| [s * y, 0.0], |_, _| 1.0);
        let d = viscous_mom_update_2d(&v, &r, &uni(0.02), 0.05, 0.05, 0.01);
        assert!(d[0].abs() < 1e-14 && d[1].abs() < 1e-14, "{d:?}");
    }

    // positive check: vx = a y^2 gives a varying stress tau_yx = rho nu (2 a y),
    // whose divergence is the exact, constant force d_y tau_yx = 2 a rho nu in x,
    // and zero in y. verifies magnitude, direction, and sign.
    #[test]
    fn quadratic_shear_books_the_analytic_force() {
        let (a, nu, rho, dt) = (1.5, 0.02, 1.0, 0.01);
        let (dx, dy) = (0.05, 0.05);
        let (v, r) =
            stencil(0.2, 0.13, dx, dy, |_, y| [a * y * y, 0.0], |_, _| rho);
        let d = viscous_mom_update_2d(&v, &r, &uni(nu), dx, dy, dt);
        let expect_x = dt * 2.0 * a * rho * nu;
        assert!((d[0] - expect_x).abs() < 1e-14, "fx {} vs {expect_x}", d[0]);
        assert!(d[1].abs() < 1e-14, "fy {}", d[1]);
    }
}
