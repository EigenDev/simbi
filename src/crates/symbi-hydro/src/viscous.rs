// =============================================================================
// viscous.rs
//
// the constant-nu Navier-Stokes shear operator, carrier-generic
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
// piece; the energy twin (the v_i tau_ij flux + viscous heating) is not built here.
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

    // the face DYNAMIC viscosity is mu = rho_face * nu_face. the density uses the
    // HARMONIC (series) mean of the two straddling cells — the physically correct
    // average of a diffusion coefficient across a face, and the one that VANISHES
    // as either cell empties (rho -> 0). a mass sink that retains momentum (the
    // torque-free surface) leaves a mask cell with tiny rho and O(1) momentum, so
    // v = mom/rho is enormous; the arithmetic mean would keep mu ~ rho_healthy and
    // the stress mu*grad(v) would explode, whereas the harmonic mean gives mu ~
    // rho_vacuum, so mu*grad(v) ~ nu*(momentum) stays bounded. the tiny denominator
    // floor guards an empty-empty face against 0/0.
    let tiny = S::from_f64(1e-300);
    let harm = |a: S, b: S| -> S { two * a * b / (a + b + tiny) };

    // vx[jj][ii] / vy[jj][ii] accessors keep the face formulas readable.
    let vx = |jj: usize, ii: usize| v[jj][ii][0];
    let vy = |jj: usize, ii: usize| v[jj][ii][1];

    // --- x-face i+1/2 (between center [1][1] and [1][2]) ---
    // normal gradient: central across the face. transverse gradient: averaged
    // over the two straddling cells (a 4-point central difference in y).
    let mu_xp = harm(rho[1][1], rho[1][2]) * (half * (nu[1][1] + nu[1][2]));
    let dvxdx = (vx(1, 2) - vx(1, 1)) / dx;
    let dvydx = (vy(1, 2) - vy(1, 1)) / dx;
    let dvxdy = ((vx(2, 1) - vx(0, 1)) + (vx(2, 2) - vx(0, 2))) / (four * dy);
    let dvydy = ((vy(2, 1) - vy(0, 1)) + (vy(2, 2) - vy(0, 2))) / (four * dy);
    let div = dvxdx + dvydy;
    let txx_xp = mu_xp * (two * dvxdx - two_thirds * div);
    let txy_xp = mu_xp * (dvxdy + dvydx);

    // --- x-face i-1/2 (between [1][0] and center [1][1]) ---
    let mu_xm = harm(rho[1][0], rho[1][1]) * (half * (nu[1][0] + nu[1][1]));
    let dvxdx = (vx(1, 1) - vx(1, 0)) / dx;
    let dvydx = (vy(1, 1) - vy(1, 0)) / dx;
    let dvxdy = ((vx(2, 0) - vx(0, 0)) + (vx(2, 1) - vx(0, 1))) / (four * dy);
    let dvydy = ((vy(2, 0) - vy(0, 0)) + (vy(2, 1) - vy(0, 1))) / (four * dy);
    let div = dvxdx + dvydy;
    let txx_xm = mu_xm * (two * dvxdx - two_thirds * div);
    let txy_xm = mu_xm * (dvxdy + dvydx);

    // --- y-face j+1/2 (between center [1][1] and [2][1]) ---
    let mu_yp = harm(rho[1][1], rho[2][1]) * (half * (nu[1][1] + nu[2][1]));
    let dvxdy = (vx(2, 1) - vx(1, 1)) / dy;
    let dvydy = (vy(2, 1) - vy(1, 1)) / dy;
    let dvxdx = ((vx(1, 2) - vx(1, 0)) + (vx(2, 2) - vx(2, 0))) / (four * dx);
    let dvydx = ((vy(1, 2) - vy(1, 0)) + (vy(2, 2) - vy(2, 0))) / (four * dx);
    let div = dvxdx + dvydy;
    let tyx_yp = mu_yp * (dvxdy + dvydx);
    let tyy_yp = mu_yp * (two * dvydy - two_thirds * div);

    // --- y-face j-1/2 (between [0][1] and center [1][1]) ---
    let mu_ym = harm(rho[0][1], rho[1][1]) * (half * (nu[0][1] + nu[1][1]));
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

/// the cylindrical `(R, phi)` PHYSICAL-frame deviatoric stress `(tau_RR, tau_pp,
/// tau_Rphi)` from the physical velocity gradients at a point of radius `r`.
/// `u = v_R`, `w = v_phi`. the metric enters as the `u/R` (azimuthal stretch) and
/// `-w/R` (rigid-rotation cancellation) terms — a rigid rotation `w = Omega R`
/// gives `e_Rphi = (Omega - Omega)/2 = 0`, the stress-free null.
#[allow(clippy::too_many_arguments)]
fn cyl_stress<S: Scalar>(du_dr: S, dw_dr: S, du_dp: S, dw_dp: S, u: S, w: S, r: S, mu: S) -> (S, S, S) {
    let two = S::from_f64(2.0);
    let two_thirds = S::from_f64(2.0 / 3.0);
    let half = S::from_f64(0.5);
    let inv_r = S::from_f64(1.0) / r;
    let e_rr = du_dr;
    let e_pp = dw_dp * inv_r + u * inv_r;
    let e_rp = half * (du_dp * inv_r + dw_dr - w * inv_r);
    let theta = e_rr + e_pp;
    let t_rr = mu * (two * e_rr - two_thirds * theta);
    let t_pp = mu * (two * e_pp - two_thirds * theta);
    let t_rp = mu * two * e_rp;
    (t_rr, t_pp, t_rp)
}

/// the viscous momentum increment `dt * div(tau)` for the center cell of a 3x3
/// stencil on a cylindrical `(R, phi)` grid, PHYSICAL orthonormal frame,
/// conservative. `v[j][i] = (v_R, v_phi)`. the radial force uses the area-weighted
/// `(1/R) d_R(R tau_RR)` divergence plus the `-tau_pp/R` hoop stress; the azimuthal
/// force uses the ANGULAR-MOMENTUM-conserving `(1/R^2) d_R(R^2 tau_Rphi)` flux, so
/// `R * mom_phi` (the angular momentum) is transported, never created — the r-phi
/// shear of a differentially rotating disk drives `v_R = -3 nu / (2 R)`. `r_c` is
/// the cell R centroid; neighbours sit at `r_c +- dr`, faces at `r_c +- dr/2`.
pub fn viscous_mom_update_cyl_2d<S: Scalar>(
    v: &[[Tensor<S, 2>; 3]; 3],
    rho: &[[S; 3]; 3],
    nu: &[[S; 3]; 3],
    r_c: S,
    dr: S,
    dphi: S,
    dt: S,
) -> Tensor<S, 2> {
    let half = S::from_f64(0.5);
    let two = S::from_f64(2.0);
    let four = S::from_f64(4.0);
    let tiny = S::from_f64(1e-300);
    let harm = |a: S, b: S| two * a * b / (a + b + tiny);
    let u = |j: usize, i: usize| v[j][i][0];
    let w = |j: usize, i: usize| v[j][i][1];
    let mu_of = |ja: usize, ia: usize, jb: usize, ib: usize| {
        harm(rho[ja][ia], rho[jb][ib]) * (half * (nu[ja][ia] + nu[jb][ib]))
    };
    let r_m = r_c - half * dr;
    let r_p = r_c + half * dr;
    let inv_rc = S::from_f64(1.0) / r_c;

    // outer R-face (i+1/2), radius r_p: normal d/dR compact, transverse d/dphi 4-pt.
    let du_dr = (u(1, 2) - u(1, 1)) / dr;
    let dw_dr = (w(1, 2) - w(1, 1)) / dr;
    let du_dp = ((u(2, 1) - u(0, 1)) + (u(2, 2) - u(0, 2))) / (four * dphi);
    let dw_dp = ((w(2, 1) - w(0, 1)) + (w(2, 2) - w(0, 2))) / (four * dphi);
    let (uf, wf) = (half * (u(1, 1) + u(1, 2)), half * (w(1, 1) + w(1, 2)));
    let (trr_rp, _, trp_rp) =
        cyl_stress(du_dr, dw_dr, du_dp, dw_dp, uf, wf, r_p, mu_of(1, 1, 1, 2));

    // inner R-face (i-1/2), radius r_m.
    let du_dr = (u(1, 1) - u(1, 0)) / dr;
    let dw_dr = (w(1, 1) - w(1, 0)) / dr;
    let du_dp = ((u(2, 0) - u(0, 0)) + (u(2, 1) - u(0, 1))) / (four * dphi);
    let dw_dp = ((w(2, 0) - w(0, 0)) + (w(2, 1) - w(0, 1))) / (four * dphi);
    let (uf, wf) = (half * (u(1, 0) + u(1, 1)), half * (w(1, 0) + w(1, 1)));
    let (trr_rm, _, trp_rm) =
        cyl_stress(du_dr, dw_dr, du_dp, dw_dp, uf, wf, r_m, mu_of(1, 0, 1, 1));

    // outer phi-face (j+1/2), radius r_c: normal d/dphi compact, transverse d/dR 4-pt.
    let dw_dp = (w(2, 1) - w(1, 1)) / dphi;
    let du_dp = (u(2, 1) - u(1, 1)) / dphi;
    let du_dr = ((u(1, 2) - u(1, 0)) + (u(2, 2) - u(2, 0))) / (four * dr);
    let dw_dr = ((w(1, 2) - w(1, 0)) + (w(2, 2) - w(2, 0))) / (four * dr);
    let (uf, wf) = (half * (u(1, 1) + u(2, 1)), half * (w(1, 1) + w(2, 1)));
    let (_, tpp_pp, trp_pp) =
        cyl_stress(du_dr, dw_dr, du_dp, dw_dp, uf, wf, r_c, mu_of(1, 1, 2, 1));

    // inner phi-face (j-1/2), radius r_c.
    let dw_dp = (w(1, 1) - w(0, 1)) / dphi;
    let du_dp = (u(1, 1) - u(0, 1)) / dphi;
    let du_dr = ((u(1, 2) - u(1, 0)) + (u(0, 2) - u(0, 0))) / (four * dr);
    let dw_dr = ((w(1, 2) - w(1, 0)) + (w(0, 2) - w(0, 0))) / (four * dr);
    let (uf, wf) = (half * (u(0, 1) + u(1, 1)), half * (w(0, 1) + w(1, 1)));
    let (_, tpp_pm, trp_pm) =
        cyl_stress(du_dr, dw_dr, du_dp, dw_dp, uf, wf, r_c, mu_of(0, 1, 1, 1));

    // center: the hoop stress tau_pp for the -tau_pp/R source in F_R.
    let du_dr_c = (u(1, 2) - u(1, 0)) / (two * dr);
    let dw_dp_c = (w(2, 1) - w(0, 1)) / (two * dphi);
    let (_, tpp_c, _) =
        cyl_stress(du_dr_c, S::ZERO, S::ZERO, dw_dp_c, u(1, 1), w(1, 1), r_c, rho[1][1] * nu[1][1]);

    let f_r = (r_p * trr_rp - r_m * trr_rm) * (inv_rc / dr)
        + (trp_pp - trp_pm) * (inv_rc / dphi)
        - tpp_c * inv_rc;
    let f_phi = (r_p * r_p * trp_rp - r_m * r_m * trp_rm) * (inv_rc * inv_rc / dr)
        + (tpp_pp - tpp_pm) * (inv_rc / dphi);

    Tensor::new([dt * f_r, dt * f_phi])
}

/// the (tau_0a, tau_1a, tau_2a) stress column at the face normal to axis `a`
/// of the center cell (`hi` = the +a face i+1/2, else the -a face i-1/2), from
/// the 3x3x3 velocity/density/viscosity stencil. the normal gradient is compact
/// across the face; each transverse gradient is averaged over the two straddling
/// cells. the face viscosity is the average of the two, so the column is
/// single-valued and the differenced flux conserves.
fn face_stress_3d<S: Scalar>(
    v: &[[[Tensor<S, 3>; 3]; 3]; 3],
    rho: &[[[S; 3]; 3]; 3],
    nu: &[[[S; 3]; 3]; 3],
    a: usize,
    hi: bool,
    dx: [S; 3],
) -> [S; 3] {
    let half = S::from_f64(0.5);
    let two_thirds = S::from_f64(2.0 / 3.0);
    let (la, ra) = if hi { (1usize, 2usize) } else { (0usize, 1usize) };
    let mut lo = [1usize; 3];
    lo[a] = la;
    let mut ro = [1usize; 3];
    ro[a] = ra;
    let g = |o: [usize; 3], c: usize| v[o[2]][o[1]][o[0]][c];
    let rr = |o: [usize; 3]| rho[o[2]][o[1]][o[0]];
    let nn = |o: [usize; 3]| nu[o[2]][o[1]][o[0]];
    // harmonic (series) density mean: vanishes as either cell empties so the
    // stress stays bounded next to a momentum-retaining mass sink (see the 2D
    // core). arithmetic nu mean (nu never blows up). tiny denom floor guards 0/0.
    let tiny = S::from_f64(1e-300);
    let (rl, rh_) = (rr(lo), rr(ro));
    let mu = (S::from_f64(2.0) * rl * rh_ / (rl + rh_ + tiny)) * (half * (nn(lo) + nn(ro)));

    // grad[j][k] = d v_j / d x_k at the face.
    let mut grad = [[S::ZERO; 3]; 3];
    for j in 0..3 {
        grad[j][a] = (g(ro, j) - g(lo, j)) / dx[a];
        for b in 0..3 {
            if b == a {
                continue;
            }
            let (mut lp, mut lm, mut rp, mut rm) = (lo, lo, ro, ro);
            lp[b] = 2;
            lm[b] = 0;
            rp[b] = 2;
            rm[b] = 0;
            let two = S::from_f64(2.0);
            let cl = (g(lp, j) - g(lm, j)) / (two * dx[b]);
            let cr = (g(rp, j) - g(rm, j)) / (two * dx[b]);
            grad[j][b] = half * (cl + cr);
        }
    }
    let div = grad[0][0] + grad[1][1] + grad[2][2];
    let mut tau = [S::ZERO; 3];
    for i in 0..3 {
        let mut t = grad[i][a] + grad[a][i];
        if i == a {
            t = t - two_thirds * div;
        }
        tau[i] = mu * t;
    }
    tau
}

/// the viscous momentum increment `dt * div(tau)` for the center cell of a 3x3x3
/// velocity + density + viscosity stencil (3D, cartesian). `dmom_i = dt sum_a
/// d_a tau_ia`, each face column single-valued so the update conserves. reduces
/// exactly to `viscous_mom_update_2d` for a z-invariant flow with `v_z = 0`.
pub fn viscous_mom_update_3d<S: Scalar>(
    v: &[[[Tensor<S, 3>; 3]; 3]; 3],
    rho: &[[[S; 3]; 3]; 3],
    nu: &[[[S; 3]; 3]; 3],
    dx: [S; 3],
    dt: S,
) -> Tensor<S, 3> {
    let mut dmom = [S::ZERO; 3];
    for a in 0..3 {
        let sp = face_stress_3d(v, rho, nu, a, true, dx);
        let sm = face_stress_3d(v, rho, nu, a, false, dx);
        for i in 0..3 {
            dmom[i] = dmom[i] + dt * (sp[i] - sm[i]) / dx[a];
        }
    }
    Tensor::new(dmom)
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

    // -- cylindrical (R, phi) operator ---------------------------------------
    // build a 3x3 (v_R, v_phi) + rho stencil about a cell at radius r_c; the
    // neighbours sit at r_c + (i-1) dr, and phi is measured relative to the cell.
    fn cyl_stencil(
        r_c: f64,
        dr: f64,
        dphi: f64,
        vf: impl Fn(f64, f64) -> [f64; 2],
        rf: impl Fn(f64, f64) -> f64,
    ) -> ([[Tensor<f64, 2>; 3]; 3], [[f64; 3]; 3]) {
        let mut v = [[Tensor::zeros(); 3]; 3];
        let mut r = [[0.0; 3]; 3];
        for jj in 0..3 {
            for ii in 0..3 {
                let rr = r_c + (ii as f64 - 1.0) * dr;
                let pp = (jj as f64 - 1.0) * dphi;
                v[jj][ii] = Tensor::new(vf(rr, pp));
                r[jj][ii] = rf(rr, pp);
            }
        }
        (v, r)
    }

    // the load-bearing null: a rigid rotation v_phi = Omega R is strain-free
    // (e_Rphi = (Omega - Omega)/2 = 0), so the viscous force vanishes. catches a
    // wrong metric term or sign in the cylindrical strain rate.
    #[test]
    fn cyl_rigid_rotation_books_zero_force() {
        let (om, nu, rho, dt) = (0.7, 0.05, 1.3, 0.01);
        let (r_c, dr, dphi) = (1.5, 0.05, 0.1);
        let (v, r) = cyl_stencil(r_c, dr, dphi, |rr, _| [0.0, om * rr], |_, _| rho);
        let d = viscous_mom_update_cyl_2d(&v, &r, &uni(nu), r_c, dr, dphi, dt);
        assert!(d[0].abs() < 1e-12, "f_R not zero: {}", d[0]);
        assert!(d[1].abs() < 1e-12, "f_phi not zero: {}", d[1]);
    }

    // a Keplerian profile v_phi = sqrt(GM/R) shears (Omega ~ R^-3/2), so the
    // r-phi stress is active and the azimuthal force is NEGATIVE — viscosity
    // removes angular momentum from the inner gas, driving inflow. the analytic
    // axisymmetric value is F_phi = -(3/4) mu sqrt(GM) R^-5/2.
    #[test]
    fn cyl_keplerian_shear_removes_angular_momentum() {
        let (gm, nu, rho, dt) = (1.0, 0.05, 1.0, 0.01);
        let (r_c, dr, dphi) = (1.0, 0.02, 0.1);
        let (v, r) = cyl_stencil(r_c, dr, dphi, |rr, _| [0.0, (gm / rr).sqrt()], |_, _| rho);
        let d = viscous_mom_update_cyl_2d(&v, &r, &uni(nu), r_c, dr, dphi, dt);
        assert!(d[1].abs() > 1e-8, "keplerian shear produced no torque: {}", d[1]);
        assert!(d[1] < 0.0, "expected angular-momentum loss, got f_phi = {}", d[1]);
        // rough magnitude: -(3/4) mu sqrt(GM) R^-5/2 * dt at R=1.
        let expect = -0.75 * rho * nu * gm.sqrt() * dt;
        assert!((d[1] / expect - 1.0).abs() < 0.1, "f_phi {} vs ~{expect}", d[1]);
    }

    // a mass sink that retains momentum (the torque-free surface) leaves a mask
    // cell with tiny rho and O(1) momentum, so v = mom/rho is enormous. an
    // ARITHMETIC-mean face viscosity keeps mu ~ rho_healthy and the stress
    // mu*grad(v) explodes (a >1e4x kick that FOFC cannot recover -> the freeze).
    // the HARMONIC mean gives mu ~ rho_vacuum, so the stress ~ nu*(momentum): the
    // momentum kick stays BOUNDED BY THE CELL'S OWN MOMENTUM at the viscous-CFL
    // step -- no overshoot, no sign flip, the pathology diffuses away instead.
    #[test]
    fn vacuum_adjacent_stress_stays_bounded_by_momentum() {
        let (nu, dt): (f64, f64) = (0.1, 1e-4);
        let (dx, dy) = (0.01, 0.01);
        // the viscous-CFL step for this nu, dx (the cap the driver would take).
        let dt_visc = 0.1 * dx * dx / nu;
        let dt = dt.min(dt_visc);

        // a healthy disk stencil: rho ~ 1, a smooth shear.
        let (vh, rh) = stencil(0.5, 0.5, dx, dy, |x, y| [x + 0.5 * y, -0.3 * x], |_, _| 1.0);

        // the left neighbour is an evacuated mask cell: rho = 1e-4, momentum
        // retained ~ (1, 0.5) -> v = mom/rho ~ (1e4, 5e3).
        let (mut v, mut r) = (vh, rh);
        let mom_mask = [1.0f64, 0.5];
        r[1][0] = 1e-4;
        v[1][0] = Tensor::new([mom_mask[0] / r[1][0], mom_mask[1] / r[1][0]]);
        let f = viscous_mom_update_2d(&v, &r, &uni(nu), dx, dy, dt);

        // the kick on the mask cell's momentum must not exceed the momentum itself
        // (arithmetic-mean would give ~5x this bound -> overshoot -> breakdown).
        let bound = mom_mask[0].abs().max(mom_mask[1].abs());
        assert!(
            f[0].abs() < bound && f[1].abs() < bound,
            "vacuum-adjacent kick exceeded the retained momentum: f = ({}, {}), bound {bound}",
            f[0],
            f[1],
        );
    }

    // -- stability + conservation battle tests --------------------------------
    // integrate the operator on a periodic grid so EVERY Fourier mode is present;
    // a diffusion that is stable and dissipative must decay every mode. a mode
    // that grows (the checkerboard from a wide cross-stencil, or a CFL over the
    // stability limit) shows up as a rising total kinetic energy.

    // one explicit viscous step on an N x N periodic grid, constant density.
    // v_new = v + dt div(tau) / rho, all reads from the old field (no hazard).
    fn viscous_step_periodic(
        v: &[Vec<[f64; 2]>],
        rho: f64,
        nu: f64,
        dx: f64,
        dt: f64,
    ) -> Vec<Vec<[f64; 2]>> {
        let n = v.len();
        let nust = uni(nu);
        let rst = uni(rho);
        let mut out = v.to_vec();
        for j in 0..n {
            for i in 0..n {
                let mut st = [[Tensor::<f64, 2>::zeros(); 3]; 3];
                for dj in 0..3 {
                    for di in 0..3 {
                        let ii = (i + di + n - 1) % n;
                        let jj = (j + dj + n - 1) % n;
                        st[dj][di] = Tensor::new(v[jj][ii]);
                    }
                }
                let d = viscous_mom_update_2d(&st, &rst, &nust, dx, dx, dt);
                out[j][i] = [v[j][i][0] + d[0] / rho, v[j][i][1] + d[1] / rho];
            }
        }
        out
    }

    // total kinetic energy of the fluctuation about the (conserved) mean.
    fn fluct_energy(v: &[Vec<[f64; 2]>]) -> f64 {
        let n = v.len();
        let (mut mx, mut my) = (0.0, 0.0);
        for row in v {
            for c in row {
                mx += c[0];
                my += c[1];
            }
        }
        let inv = 1.0 / (n * n) as f64;
        let (mx, my) = (mx * inv, my * inv);
        let mut e = 0.0;
        for row in v {
            for c in row {
                e += (c[0] - mx).powi(2) + (c[1] - my).powi(2);
            }
        }
        e
    }

    fn seed_all_modes(n: usize) -> Vec<Vec<[f64; 2]>> {
        use std::f64::consts::PI;
        let mut v = vec![vec![[0.0; 2]; n]; n];
        for j in 0..n {
            for i in 0..n {
                let (fi, fj) = (i as f64 / n as f64, j as f64 / n as f64);
                let cb = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                // a low mode, a mid mode, and the checkerboard (nyquist) in BOTH
                // components — the cross-stencil terms are exercised by having
                // vy vary in x and vx vary in y.
                v[j][i] = [
                    (2.0 * PI * fi).sin() + 0.5 * (6.0 * PI * fj).cos() + 0.4 * cb,
                    (2.0 * PI * fj).cos() + 0.5 * (6.0 * PI * fi).sin() + 0.4 * cb,
                ];
            }
        }
        v
    }

    // conservation: on a torus the flux divergence telescopes, so the summed
    // momentum increment is zero to roundoff.
    #[test]
    fn periodic_box_conserves_total_momentum() {
        let (n, dx, nu, rho, dt) = (12, 0.05, 0.1, 1.3, 1e-4);
        let v = seed_all_modes(n);
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        let nust = uni(nu);
        let rst = uni(rho);
        for j in 0..n {
            for i in 0..n {
                let mut st = [[Tensor::<f64, 2>::zeros(); 3]; 3];
                for dj in 0..3 {
                    for di in 0..3 {
                        let ii = (i + di + n - 1) % n;
                        let jj = (j + dj + n - 1) % n;
                        st[dj][di] = Tensor::new(v[jj][ii]);
                    }
                }
                let d = viscous_mom_update_2d(&st, &rst, &nust, dx, dx, dt);
                sx += d[0];
                sy += d[1];
            }
        }
        assert!(sx.abs() < 1e-12 && sy.abs() < 1e-12, "momentum leak: {sx}, {sy}");
    }

    // stability: at the C_VISC = 0.1 cap the fluctuation energy decays MONOTONE
    // over many steps (no mode grows — including the checkerboard). this is the
    // guard the CFL constant fix is anchored on.
    #[test]
    fn viscous_diffusion_is_monotone_stable_at_the_cfl_cap() {
        let (n, dx, nu, rho) = (16, 0.1, 0.1, 1.0);
        let dt = 0.1 * dx * dx / nu; // C_VISC = 0.1
        let mut v = seed_all_modes(n);
        let mut prev = fluct_energy(&v);
        let e0 = prev;
        for step in 0..400 {
            v = viscous_step_periodic(&v, rho, nu, dx, dt);
            let cur = fluct_energy(&v);
            assert!(
                cur <= prev * (1.0 + 1e-10),
                "energy grew at step {step}: {prev} -> {cur}"
            );
            prev = cur;
        }
        assert!(prev < 1e-3 * e0, "did not decay: {e0} -> {prev}");
    }

    // the limit is real: above ~0.21 dx^2/nu the highest mode amplifies. at
    // 0.3 the fluctuation energy BLOWS UP — confirming C_VISC = 0.1 is the safe
    // side and 0.25 (the plain-Laplacian value) was not.
    #[test]
    fn viscous_diffusion_blows_up_above_the_stability_limit() {
        let (n, dx, nu, rho) = (16, 0.1, 0.1, 1.0);
        let dt = 0.3 * dx * dx / nu; // above the ~0.21 limit
        let mut v = seed_all_modes(n);
        let e0 = fluct_energy(&v);
        for _ in 0..80 {
            v = viscous_step_periodic(&v, rho, nu, dx, dt);
        }
        assert!(
            fluct_energy(&v) > 10.0 * e0,
            "expected blow-up above the CFL limit, got {}",
            fluct_energy(&v)
        );
    }

    // -- spatially varying nu (the alpha case) --------------------------------
    // a smooth nu(x) field and a smooth rho(x) field, periodic. the FACE nu is
    // the average of the two straddling cells (single-valued), so conservation
    // and stability must hold exactly as for constant nu.

    fn seed_nu_field(n: usize, nu_min: f64, nu_max: f64) -> Vec<Vec<f64>> {
        use std::f64::consts::PI;
        let mut f = vec![vec![0.0; n]; n];
        for j in 0..n {
            for i in 0..n {
                let (fi, fj) = (i as f64 / n as f64, j as f64 / n as f64);
                let s = 0.5 * (1.0 + (2.0 * PI * fi).sin() * (2.0 * PI * fj).cos());
                f[j][i] = nu_min + (nu_max - nu_min) * s;
            }
        }
        f
    }

    fn viscous_step_varnu(
        v: &[Vec<[f64; 2]>],
        rho_field: &[Vec<f64>],
        nu_field: &[Vec<f64>],
        dx: f64,
        dt: f64,
    ) -> Vec<Vec<[f64; 2]>> {
        let n = v.len();
        let mut out = v.to_vec();
        for j in 0..n {
            for i in 0..n {
                let mut vst = [[Tensor::<f64, 2>::zeros(); 3]; 3];
                let mut rst = [[0.0; 3]; 3];
                let mut nst = [[0.0; 3]; 3];
                for dj in 0..3 {
                    for di in 0..3 {
                        let ii = (i + di + n - 1) % n;
                        let jj = (j + dj + n - 1) % n;
                        vst[dj][di] = Tensor::new(v[jj][ii]);
                        rst[dj][di] = rho_field[jj][ii];
                        nst[dj][di] = nu_field[jj][ii];
                    }
                }
                let d = viscous_mom_update_2d(&vst, &rst, &nst, dx, dx, dt);
                let rho = rho_field[j][i];
                out[j][i] = [v[j][i][0] + d[0] / rho, v[j][i][1] + d[1] / rho];
            }
        }
        out
    }

    // conservation with a varying nu (and varying rho): the differenced face
    // flux still telescopes, so the total momentum increment is zero.
    #[test]
    fn varying_nu_conserves_total_momentum() {
        let (n, dx, dt) = (12, 0.05, 1e-4);
        let v = seed_all_modes(n);
        let nu_field = seed_nu_field(n, 0.02, 0.2);
        let rho_field = seed_nu_field(n, 0.7, 1.4); // reuse: a smooth positive field
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for j in 0..n {
            for i in 0..n {
                let mut vst = [[Tensor::<f64, 2>::zeros(); 3]; 3];
                let mut rst = [[0.0; 3]; 3];
                let mut nst = [[0.0; 3]; 3];
                for dj in 0..3 {
                    for di in 0..3 {
                        let ii = (i + di + n - 1) % n;
                        let jj = (j + dj + n - 1) % n;
                        vst[dj][di] = Tensor::new(v[jj][ii]);
                        rst[dj][di] = rho_field[jj][ii];
                        nst[dj][di] = nu_field[jj][ii];
                    }
                }
                let d = viscous_mom_update_2d(&vst, &rst, &nst, dx, dx, dt);
                sx += d[0];
                sy += d[1];
            }
        }
        assert!(sx.abs() < 1e-12 && sy.abs() < 1e-12, "leak: {sx}, {sy}");
    }

    // stability with a varying nu: the global cap dt = C_VISC dx^2 / nu_max keeps
    // every cell below its local limit, so the fluctuation energy decays monotone.
    #[test]
    fn varying_nu_diffusion_is_monotone_stable() {
        let (n, dx, rho) = (16, 0.1, 1.0);
        let (nu_min, nu_max) = (0.01, 0.2);
        let dt = 0.1 * dx * dx / nu_max; // cap on the MAX nu
        let nu_field = seed_nu_field(n, nu_min, nu_max);
        let rho_field = vec![vec![rho; n]; n];
        let mut v = seed_all_modes(n);
        let (mut prev, e0) = (fluct_energy(&v), fluct_energy(&v));
        for step in 0..400 {
            v = viscous_step_varnu(&v, &rho_field, &nu_field, dx, dt);
            let cur = fluct_energy(&v);
            assert!(cur <= prev * (1.0 + 1e-10), "grew at step {step}: {prev} -> {cur}");
            prev = cur;
        }
        assert!(prev < 0.1 * e0, "did not decay: {e0} -> {prev}");
    }

    // -- 3D operator ----------------------------------------------------------
    fn uni3(nu: f64) -> [[[f64; 3]; 3]; 3] {
        [[[nu; 3]; 3]; 3]
    }

    fn stencil3(
        p0: [f64; 3],
        dx: [f64; 3],
        vf: impl Fn(f64, f64, f64) -> [f64; 3],
        rf: impl Fn(f64, f64, f64) -> f64,
    ) -> ([[[Tensor<f64, 3>; 3]; 3]; 3], [[[f64; 3]; 3]; 3]) {
        let mut v = [[[Tensor::<f64, 3>::zeros(); 3]; 3]; 3];
        let mut r = [[[0.0; 3]; 3]; 3];
        for kk in 0..3 {
            for jj in 0..3 {
                for ii in 0..3 {
                    let x = p0[0] + (ii as f64 - 1.0) * dx[0];
                    let y = p0[1] + (jj as f64 - 1.0) * dx[1];
                    let z = p0[2] + (kk as f64 - 1.0) * dx[2];
                    v[kk][jj][ii] = Tensor::new(vf(x, y, z));
                    r[kk][jj][ii] = rf(x, y, z);
                }
            }
        }
        (v, r)
    }

    // a uniform translation has zero strain rate -> zero viscous force.
    #[test]
    fn uniform_flow_3d_books_zero_force() {
        let dx = [0.05, 0.07, 0.06];
        let (v, r) = stencil3([0.2, 0.1, 0.3], dx, |_, _, _| [1.3, -0.7, 0.4], |_, _, _| 1.1);
        let d = viscous_mom_update_3d(&v, &r, &uni3(0.05), dx, 0.01);
        assert!(d[0].abs() < 1e-14 && d[1].abs() < 1e-14 && d[2].abs() < 1e-14, "{d:?}");
    }

    // a rigid rotation v = omega x r about an arbitrary axis is strain-free, so
    // the symmetric stress and its divergence vanish. exercises every cross term.
    #[test]
    fn rigid_rotation_3d_books_zero_force() {
        let (wx, wy, wz) = (0.3, 0.5, 0.7);
        let dx = [0.05, 0.05, 0.05];
        let (v, r) = stencil3(
            [0.11, -0.07, 0.13],
            dx,
            |x, y, z| [wy * z - wz * y, wz * x - wx * z, wx * y - wy * x],
            |_, _, _| 1.0,
        );
        let d = viscous_mom_update_3d(&v, &r, &uni3(0.02), dx, 0.01);
        assert!(d[0].abs() < 1e-14 && d[1].abs() < 1e-14 && d[2].abs() < 1e-14, "{d:?}");
    }

    // the load-bearing cross-check: a z-invariant flow with v_z = 0 must give
    // exactly the (already battle-tested) 2D force in x, y and zero in z. ties
    // the 3D operator to the validated 2D one bit-for-bit.
    #[test]
    fn reduces_to_the_2d_operator_for_planar_flow() {
        let (a, b, c, nu, rho, dt) = (1.5, 0.8, -1.1, 0.02, 1.3, 0.01);
        let (dx, dy) = (0.05, 0.06);
        let vf2 = |x: f64, y: f64| [a * y * y + b * x * y, c * x * x];
        let (v2, r2) = stencil(0.2, 0.13, dx, dy, vf2, |_, _| rho);
        let d2 = viscous_mom_update_2d(&v2, &r2, &uni(nu), dx, dy, dt);

        let dx3 = [dx, dy, 0.04];
        let (v3, r3) = stencil3(
            [0.2, 0.13, 0.5],
            dx3,
            |x, y, _| {
                let p = vf2(x, y);
                [p[0], p[1], 0.0]
            },
            |_, _, _| rho,
        );
        let d3 = viscous_mom_update_3d(&v3, &r3, &uni3(nu), dx3, dt);
        assert!((d3[0] - d2[0]).abs() < 1e-15, "x: {} vs {}", d3[0], d2[0]);
        assert!((d3[1] - d2[1]).abs() < 1e-15, "y: {} vs {}", d3[1], d2[1]);
        assert!(d3[2].abs() < 1e-15, "z leak: {}", d3[2]);
    }

    fn seed_all_modes_3d(n: usize) -> Vec<[f64; 3]> {
        use std::f64::consts::PI;
        let mut v = vec![[0.0; 3]; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let (fi, fj, fk) = (i as f64 / n as f64, j as f64 / n as f64, k as f64 / n as f64);
                    let cb = if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 };
                    v[(k * n + j) * n + i] = [
                        (2.0 * PI * fi).sin() + 0.4 * cb,
                        (2.0 * PI * fj).cos() + 0.4 * cb,
                        (2.0 * PI * fk).sin() + 0.4 * cb,
                    ];
                }
            }
        }
        v
    }

    fn viscous_step_periodic_3d(v: &[[f64; 3]], n: usize, rho: f64, nu: f64, dx: f64, dt: f64) -> Vec<[f64; 3]> {
        let at = |i: usize, j: usize, k: usize| v[(k * n + j) * n + i];
        let rst = uni3(rho);
        let nst = uni3(nu);
        let mut out = v.to_vec();
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let mut st = [[[Tensor::<f64, 3>::zeros(); 3]; 3]; 3];
                    for dk in 0..3 {
                        for dj in 0..3 {
                            for di in 0..3 {
                                let ii = (i + di + n - 1) % n;
                                let jj = (j + dj + n - 1) % n;
                                let kk = (k + dk + n - 1) % n;
                                st[dk][dj][di] = Tensor::new(at(ii, jj, kk));
                            }
                        }
                    }
                    let d = viscous_mom_update_3d(&st, &rst, &nst, [dx, dx, dx], dt);
                    let c = at(i, j, k);
                    out[(k * n + j) * n + i] = [c[0] + d[0] / rho, c[1] + d[1] / rho, c[2] + d[2] / rho];
                }
            }
        }
        out
    }

    fn fluct_energy_3d(v: &[[f64; 3]]) -> f64 {
        let inv = 1.0 / v.len() as f64;
        let mut m = [0.0; 3];
        for c in v {
            for d in 0..3 {
                m[d] += c[d] * inv;
            }
        }
        let mut e = 0.0;
        for c in v {
            for d in 0..3 {
                e += (c[d] - m[d]).powi(2);
            }
        }
        e
    }

    #[test]
    fn periodic_box_3d_conserves_total_momentum() {
        let (n, dx, nu, rho, dt) = (6, 0.05, 0.1, 1.2, 1e-4);
        let v = seed_all_modes_3d(n);
        let out = viscous_step_periodic_3d(&v, n, rho, nu, dx, dt);
        let mut s = [0.0; 3];
        for idx in 0..v.len() {
            for d in 0..3 {
                s[d] += (out[idx][d] - v[idx][d]) * rho;
            }
        }
        assert!(s.iter().all(|x| x.abs() < 1e-12), "momentum leak: {s:?}");
    }

    // 3D stability: fluctuation energy decays monotone at the C_VISC = 0.1 cap,
    // the checkerboard mode included.
    #[test]
    fn viscous_3d_is_monotone_stable_at_the_cfl_cap() {
        let (n, dx, nu, rho) = (8, 0.1, 0.1, 1.0);
        let dt = 0.1 * dx * dx / nu;
        let mut v = seed_all_modes_3d(n);
        let (mut prev, e0) = (fluct_energy_3d(&v), fluct_energy_3d(&v));
        for step in 0..300 {
            v = viscous_step_periodic_3d(&v, n, rho, nu, dx, dt);
            let cur = fluct_energy_3d(&v);
            assert!(cur <= prev * (1.0 + 1e-10), "grew at step {step}: {prev} -> {cur}");
            prev = cur;
        }
        assert!(prev < 1e-2 * e0, "did not decay: {e0} -> {prev}");
    }
}
