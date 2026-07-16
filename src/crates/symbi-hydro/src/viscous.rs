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
    viscous_update_2d(v, rho, nu, dx, dy, dt).0
}

/// the viscous increment for the ADIABATIC regime: the momentum `dt * div(tau)` (bit-identical to
/// `viscous_mom_update_2d`) PLUS the total-energy increment `dt * div(tau . v)` — the divergence of
/// the viscous energy flux `F_a = tau_ab v_b`, evaluated on the SAME four faces with the
/// face-interpolated velocity. the conservative flux form conserves total energy exactly, and the
/// irreversible heating `Phi = tau : grad(v) >= 0` emerges in the internal energy `e = E - rho v^2/2`.
pub fn viscous_update_2d<S: Scalar>(
    v: &[[Tensor<S, 2>; 3]; 3],
    rho: &[[S; 3]; 3],
    nu: &[[S; 3]; 3],
    dx: S,
    dy: S,
    dt: S,
) -> (Tensor<S, 2>, S) {
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

    // the viscous ENERGY flux F_a = tau_ab v_b at each face, with v face-interpolated (arithmetic
    // mean of the straddling cells — the velocity is primitive). div F onto the total energy.
    let fx_xp = txx_xp * (half * (vx(1, 1) + vx(1, 2))) + txy_xp * (half * (vy(1, 1) + vy(1, 2)));
    let fx_xm = txx_xm * (half * (vx(1, 0) + vx(1, 1))) + txy_xm * (half * (vy(1, 0) + vy(1, 1)));
    let fy_yp = tyx_yp * (half * (vx(1, 1) + vx(2, 1))) + tyy_yp * (half * (vy(1, 1) + vy(2, 1)));
    let fy_ym = tyx_ym * (half * (vx(0, 1) + vx(1, 1))) + tyy_ym * (half * (vy(0, 1) + vy(1, 1)));
    let dnrg = dt * ((fx_xp - fx_xm) / dx + (fy_yp - fy_ym) / dy);

    (Tensor::new([dmom_x, dmom_y]), dnrg)
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

/// the general ORTHOGONAL-coordinate physical deviatoric stress `(tau_11, tau_22,
/// tau_12)` from the physical velocity gradients + the scale factors `h1, h2` and
/// their gradients at a point. reduces to Cartesian at `h = 1` and to the
/// cylindrical `cyl_stress` at `h = (1, R)` (`d1h2 = 1`). `u1 = v along axis 1`,
/// `u2 = v along axis 2`; `dNhM = d h_M / d x_N`.
#[allow(clippy::too_many_arguments)]
fn ortho_stress<S: Scalar>(
    u1: S, u2: S,
    d1u1: S, d2u1: S, d1u2: S, d2u2: S,
    h1: S, h2: S,
    d2h1: S, d1h2: S,
    mu: S,
) -> (S, S, S) {
    let two = S::from_f64(2.0);
    let half = S::from_f64(0.5);
    let two_thirds = S::from_f64(2.0 / 3.0);
    let inv_h1 = S::from_f64(1.0) / h1;
    let inv_h2 = S::from_f64(1.0) / h2;
    let inv_h1h2 = inv_h1 * inv_h2;
    let e11 = d1u1 * inv_h1 + u2 * d2h1 * inv_h1h2;
    let e22 = d2u2 * inv_h2 + u1 * d1h2 * inv_h1h2;
    let e12 = half * (d2u1 * inv_h2 - u1 * d2h1 * inv_h1h2 + d1u2 * inv_h1 - u2 * d1h2 * inv_h1h2);
    let theta = e11 + e22;
    let t11 = mu * (two * e11 - two_thirds * theta);
    let t22 = mu * (two * e22 - two_thirds * theta);
    let t12 = mu * two * e12;
    (t11, t22, t12)
}

/// the viscous momentum increment `dt * div(tau)` for the center cell of a 3x3
/// stencil on a GENERAL 2D ORTHOGONAL grid, physical frame, given the scale-factor
/// stencils `h1, h2` (their gradients are differenced from the stencil). ONE
/// operator for every diagonal chart: `h = (1, 1)` is Cartesian, `(1, R)` is
/// cylindrical, `(1, r)` is the spherical meridian. axis 2 is the ANGULAR / Killing
/// axis (`h` independent of `x2`): its momentum uses the conservative
/// `(1/(h1 h2^2)) d1(h2^2 tau_12)` flux, so the generalized angular momentum
/// `h2 * mom_2` is transported, never created. axis 1 carries the geometric hoop
/// source `-tau_22 d1(h2)/(h1 h2)`.
#[allow(clippy::too_many_arguments)]
pub fn viscous_mom_update_orthogonal_2d<S: Scalar>(
    v: &[[Tensor<S, 2>; 3]; 3],
    rho: &[[S; 3]; 3],
    nu: &[[S; 3]; 3],
    h1: &[[S; 3]; 3],
    h2: &[[S; 3]; 3],
    dx1: S,
    dx2: S,
    dt: S,
) -> Tensor<S, 2> {
    let half = S::from_f64(0.5);
    let two = S::from_f64(2.0);
    let four = S::from_f64(4.0);
    let tiny = S::from_f64(1e-300);
    let harm = |a: S, b: S| two * a * b / (a + b + tiny);
    let u1a: [[S; 3]; 3] = std::array::from_fn(|j| std::array::from_fn(|i| v[j][i][0]));
    let u2a: [[S; 3]; 3] = std::array::from_fn(|j| std::array::from_fn(|i| v[j][i][1]));

    // stress at the x1-face between columns (ia, ib) of row band: normal d1 compact,
    // transverse d2 four-point. returns (tau_11, tau_12, h2_face).
    let x1_face = |ia: usize, ib: usize| -> (S, S, S) {
        let g1 = |x: &[[S; 3]; 3]| (x[1][ib] - x[1][ia]) / dx1;
        let g2 = |x: &[[S; 3]; 3]| {
            ((x[2][ia] - x[0][ia]) + (x[2][ib] - x[0][ib])) / (four * dx2)
        };
        let fv = |x: &[[S; 3]; 3]| half * (x[1][ia] + x[1][ib]);
        let mu = harm(rho[1][ia], rho[1][ib]) * (half * (nu[1][ia] + nu[1][ib]));
        let (t11, _, t12) = ortho_stress(
            fv(&u1a), fv(&u2a), g1(&u1a), g2(&u1a), g1(&u2a), g2(&u2a),
            fv(h1), fv(h2), g2(h1), g1(h2), mu,
        );
        (t11, t12, fv(h2))
    };
    // stress at the x2-face between rows (ja, jb): normal d2 compact, transverse d1
    // four-point. returns (tau_22, tau_12, h1_face).
    let x2_face = |ja: usize, jb: usize| -> (S, S, S) {
        let g2 = |x: &[[S; 3]; 3]| (x[jb][1] - x[ja][1]) / dx2;
        let g1 = |x: &[[S; 3]; 3]| {
            ((x[ja][2] - x[ja][0]) + (x[jb][2] - x[jb][0])) / (four * dx1)
        };
        let fv = |x: &[[S; 3]; 3]| half * (x[ja][1] + x[jb][1]);
        let mu = harm(rho[ja][1], rho[jb][1]) * (half * (nu[ja][1] + nu[jb][1]));
        let (_, t22, t12) = ortho_stress(
            fv(&u1a), fv(&u2a), g1(&u1a), g2(&u1a), g1(&u2a), g2(&u2a),
            fv(h1), fv(h2), g2(h1), g1(h2), mu,
        );
        (t22, t12, fv(h1))
    };

    let (t11_1p, t12_1p, h2_1p) = x1_face(1, 2);
    let (t11_1m, t12_1m, h2_1m) = x1_face(0, 1);
    let (t22_2p, t12_2p, h1_2p) = x2_face(1, 2);
    let (t22_2m, t12_2m, h1_2m) = x2_face(0, 1);

    // center: the hoop-stress source for the axis-1 force.
    let (h1_c, h2_c) = (h1[1][1], h2[1][1]);
    let inv_h1h2 = S::from_f64(1.0) / (h1_c * h2_c);
    let d1u1_c = (u1a[1][2] - u1a[1][0]) / (two * dx1);
    let d2u1_c = (u1a[2][1] - u1a[0][1]) / (two * dx2);
    let d1u2_c = (u2a[1][2] - u2a[1][0]) / (two * dx1);
    let d2u2_c = (u2a[2][1] - u2a[0][1]) / (two * dx2);
    let d2h1_c = (h1[2][1] - h1[0][1]) / (two * dx2);
    let d1h2_c = (h2[1][2] - h2[1][0]) / (two * dx1);
    let mu_c = rho[1][1] * nu[1][1];
    let (t11_c, t22_c, t12_c) = ortho_stress(
        u1a[1][1], u2a[1][1], d1u1_c, d2u1_c, d1u2_c, d2u2_c,
        h1_c, h2_c, d2h1_c, d1h2_c, mu_c,
    );
    let _ = t11_c;

    // axis-1 (radial) force: area-weighted divergence + the geometric sources.
    let f1 = ((h2_1p * t11_1p - h2_1m * t11_1m) / dx1
        + (h1_2p * t12_2p - h1_2m * t12_2m) / dx2)
        * inv_h1h2
        + (t12_c * d2h1_c - t22_c * d1h2_c) * inv_h1h2;
    // axis-2 (angular) force: the conservative h2^2 flux + the h2 phi-divergence.
    let inv_h1h2sq = S::from_f64(1.0) / (h1_c * h2_c * h2_c);
    let f2 = (h2_1p * h2_1p * t12_1p - h2_1m * h2_1m * t12_1m) * (inv_h1h2sq / dx1)
        + (t22_2p - t22_2m) * (S::from_f64(1.0) / h2_c / dx2);

    Tensor::new([dt * f1, dt * f2])
}

/// the (tau_0a, tau_1a, tau_2a) stress column at the face normal to IN-PLANE axis `a` (0 or 1) for a
/// 2.5D flow: a 3x3 in-plane stencil carrying the FULL 3-vector velocity, with the OUT-OF-PLANE axis
/// frozen (`d_2 = 0`, the 2.5D symmetry). the out-of-plane velocity `v_2` shears in-plane
/// (`tau_2a = mu d_a v_2`), so a rotating disk's toroidal velocity diffuses -- the DOF > ndim case a
/// 2D grid cannot express by dimension alone.
fn face_stress_2p5d<S: Scalar>(
    v: &[[Tensor<S, 3>; 3]; 3],
    rho: &[[S; 3]; 3],
    nu: &[[S; 3]; 3],
    a: usize,
    hi: bool,
    dx: [S; 2],
) -> [S; 3] {
    let half = S::from_f64(0.5);
    let two = S::from_f64(2.0);
    let two_thirds = S::from_f64(2.0 / 3.0);
    let (la, ra) = if hi { (1usize, 2usize) } else { (0usize, 1usize) };
    let b = 1 - a;
    // 2D stencil index [ii, jj] (v[jj][ii]) with axis a at `along`, the transverse axis b at `trans`.
    let mk = |along: usize, trans: usize| -> [usize; 2] {
        let mut o = [1usize; 2];
        o[a] = along;
        o[b] = trans;
        o
    };
    let g = |o: [usize; 2], c: usize| v[o[1]][o[0]][c];
    let rr = |o: [usize; 2]| rho[o[1]][o[0]];
    let nn = |o: [usize; 2]| nu[o[1]][o[0]];
    let (lo, ro) = (mk(la, 1), mk(ra, 1));
    let tiny = S::from_f64(1e-300);
    let (rl, rh_) = (rr(lo), rr(ro));
    let mu = (two * rl * rh_ / (rl + rh_ + tiny)) * (half * (nn(lo) + nn(ro)));
    // grad[j][k] = d v_j / d x_k at the face; k in {a, b} (in-plane), grad[.][2] = 0 (2.5D freeze).
    let mut grad = [[S::ZERO; 3]; 3];
    for j in 0..3 {
        grad[j][a] = (g(ro, j) - g(lo, j)) / dx[a];
        let (lp, lm, rp, rm) = (mk(la, 2), mk(la, 0), mk(ra, 2), mk(ra, 0));
        let cl = (g(lp, j) - g(lm, j)) / (two * dx[b]);
        let cr = (g(rp, j) - g(rm, j)) / (two * dx[b]);
        grad[j][b] = half * (cl + cr);
    }
    let div = grad[0][0] + grad[1][1]; // d_2 v_2 = 0
    let mut tau = [S::ZERO; 3];
    for i in 0..3 {
        // tau_ia = mu (d_a v_i + d_i v_a - 2/3 delta_ia div); d_i v_a = 0 for i = 2 (the frozen axis).
        let mut t = grad[i][a] + grad[a][i];
        if i == a {
            t = t - two_thirds * div;
        }
        tau[i] = mu * t;
    }
    tau
}

/// the ADIABATIC viscous increment for a 2.5D flow (D=2 grid, DOF=3 momentum): `dt div(tau)` over the
/// two in-plane axes onto ALL THREE momentum components (including the out-of-plane one) PLUS
/// `dt div(tau . v)` onto the total energy. the out-of-plane momentum diffuses by the in-plane
/// Laplacian; the energy flux carries its work. for MHD, B is untouched, so the heat warms the gas.
pub fn viscous_update_2p5d<S: Scalar>(
    v: &[[Tensor<S, 3>; 3]; 3],
    rho: &[[S; 3]; 3],
    nu: &[[S; 3]; 3],
    dx: [S; 2],
    dt: S,
) -> (Tensor<S, 3>, S) {
    let half = S::from_f64(0.5);
    let center = [1usize, 1usize];
    let g = |o: [usize; 2], c: usize| v[o[1]][o[0]][c];
    let mut dmom = [S::ZERO; 3];
    let mut dnrg = S::ZERO;
    for a in 0..2 {
        let sp = face_stress_2p5d(v, rho, nu, a, true, dx);
        let sm = face_stress_2p5d(v, rho, nu, a, false, dx);
        for i in 0..3 {
            dmom[i] = dmom[i] + dt * (sp[i] - sm[i]) / dx[a];
        }
        let (mut plus, mut minus) = (center, center);
        plus[a] = 2;
        minus[a] = 0;
        let (mut f_ap, mut f_am) = (S::ZERO, S::ZERO);
        for c in 0..3 {
            f_ap = f_ap + sp[c] * (half * (g(center, c) + g(plus, c)));
            f_am = f_am + sm[c] * (half * (g(minus, c) + g(center, c)));
        }
        dnrg = dnrg + dt * (f_ap - f_am) / dx[a];
    }
    (Tensor::new(dmom), dnrg)
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
    viscous_update_3d(v, rho, nu, dx, dt).0
}

/// the ADIABATIC viscous increment (3D): `dt div(tau)` momentum (bit-identical to
/// `viscous_mom_update_3d`) PLUS `dt div(tau . v)` onto the total energy — the viscous energy flux
/// `F_a = tau_ia v_i` on each of the six faces, with v face-interpolated. conserves total energy;
/// the heating `Phi = tau : grad(v) >= 0` warms the gas. shared by adiabatic hydro AND MHD (viscosity
/// never touches B, so adding dnrg to the MHD total energy heats the gas with 1/2 B^2 untouched).
pub fn viscous_update_3d<S: Scalar>(
    v: &[[[Tensor<S, 3>; 3]; 3]; 3],
    rho: &[[[S; 3]; 3]; 3],
    nu: &[[[S; 3]; 3]; 3],
    dx: [S; 3],
    dt: S,
) -> (Tensor<S, 3>, S) {
    let half = S::from_f64(0.5);
    let center = [1usize, 1, 1];
    let g = |o: [usize; 3], c: usize| v[o[2]][o[1]][o[0]][c];
    let mut dmom = [S::ZERO; 3];
    let mut dnrg = S::ZERO;
    for a in 0..3 {
        let sp = face_stress_3d(v, rho, nu, a, true, dx);
        let sm = face_stress_3d(v, rho, nu, a, false, dx);
        for i in 0..3 {
            dmom[i] = dmom[i] + dt * (sp[i] - sm[i]) / dx[a];
        }
        // the viscous energy flux F_a = tau_ia v_i at the two a-faces, v face-averaged.
        let mut plus = center;
        plus[a] = 2;
        let mut minus = center;
        minus[a] = 0;
        let (mut f_ap, mut f_am) = (S::ZERO, S::ZERO);
        for c in 0..3 {
            f_ap = f_ap + sp[c] * (half * (g(center, c) + g(plus, c)));
            f_am = f_am + sm[c] * (half * (g(minus, c) + g(center, c)));
        }
        dnrg = dnrg + dt * (f_ap - f_am) / dx[a];
    }
    (Tensor::new(dmom), dnrg)
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

    // the CURVED-shear probe the linear tests miss: an axisymmetric keplerian field
    // v_phi = sqrt(GM/r) (constant rho, div v = 0) has a PURELY AZIMUTHAL analytic
    // viscous force F = rho nu laplacian(v), F_phi = -3/4 rho nu sqrt(GM) r^{-5/2},
    // F_r = 0. a consistent operator reproduces it with O(dx^2) error and NO angular
    // (grid-aligned m=4) variation. this prints the operator's actual behavior.
    #[test]
    fn keplerian_disk_viscous_force_probe() {
        let (gm, nu) = (1.0_f64, 1.0_f64);
        let vphi = |r: f64| (gm / r).sqrt();
        let force_at = |r: f64, phi: f64, dx: f64| -> (f64, f64) {
            let (cx, cy) = (r * phi.cos(), r * phi.sin());
            let (v, rho) = stencil(cx, cy, dx, dx, |x, y| {
                let rr = (x * x + y * y).sqrt();
                let vp = vphi(rr);
                [-vp * y / rr, vp * x / rr]
            }, |_, _| 1.0);
            let d = viscous_mom_update_2d(&v, &rho, &uni(nu), dx, dx, 1.0);
            let fr = (d[0] * cx + d[1] * cy) / r;
            let fphi = (-d[0] * cy + d[1] * cx) / r;
            (fr, fphi)
        };
        let analytic = |r: f64| -0.75 * nu * gm.sqrt() * r.powf(-2.5);
        let mut prev_err = f64::INFINITY;
        for &(r, dx) in &[(4.0, 0.05), (4.0, 0.025), (4.0, 0.0125)] {
            let fref = analytic(r);
            let (mut frmax, mut pmin, mut pmax) = (0.0_f64, f64::INFINITY, f64::NEG_INFINITY);
            for k in 0..64 {
                let phi = std::f64::consts::TAU * k as f64 / 64.0;
                let (fr, fp) = force_at(r, phi, dx);
                frmax = frmax.max(fr.abs());
                pmin = pmin.min(fp);
                pmax = pmax.max(fp);
            }
            let spread = (pmax - pmin) / fref.abs();
            let radial = frmax / fref.abs();
            let err = ((0.5 * (pmin + pmax) - fref) / fref).abs();
            // the operator is axisymmetric: NO grid-aligned (m=4) angular variation in the
            // azimuthal force, NO spurious radial force, and the error converges with dx.
            assert!(spread < 0.01, "azimuthal force varies with angle (grid m=4): spread={spread} at dx={dx}");
            assert!(radial < 0.01, "spurious radial viscous force on an axisymmetric field: {radial} at dx={dx}");
            assert!(err < prev_err, "viscous force does not converge: err={err} at dx={dx}");
            prev_err = err;
        }
    }

    // the SAME keplerian field but with the disk's RADIAL density profile (cavity
    // carve-out at r_cav). the analytic viscous force stays axisymmetric (F depends
    // on r only) for ANY rho(r), so any angular (m=4) variation in the discrete force
    // is a grid-anisotropy BUG in the density coupling. probes the cavity edge where
    // rho jumps ~5 decades — exactly where the sim's X originates.
    #[test]
    fn keplerian_disk_with_cavity_density_probe() {
        let (gm, nu, r_cav) = (1.0_f64, 1.0_f64, 2.5_f64);
        let vphi = |r: f64| (gm / r).sqrt();
        let sigma = |r: f64| (1.0 - 1e-5) * (-(r_cav / r).powi(12)).exp() + 1e-5;
        let probe = |r: f64, dx: f64| -> (f64, f64, f64) {
            let (mut frmax, mut pmin, mut pmax) = (0.0_f64, f64::INFINITY, f64::NEG_INFINITY);
            for k in 0..64 {
                let phi = std::f64::consts::TAU * k as f64 / 64.0;
                let (cx, cy) = (r * phi.cos(), r * phi.sin());
                let (v, rho) = stencil(cx, cy, dx, dx, |x, y| {
                    let rr = (x * x + y * y).sqrt();
                    let vp = vphi(rr);
                    [-vp * y / rr, vp * x / rr]
                }, |x, y| sigma((x * x + y * y).sqrt()));
                let d = viscous_mom_update_2d(&v, &rho, &uni(nu), dx, dx, 1.0);
                let fr = (d[0] * cx + d[1] * cy) / r;
                let fphi = (-d[0] * cy + d[1] * cx) / r;
                frmax = frmax.max(fr.abs());
                pmin = pmin.min(fphi);
                pmax = pmax.max(fphi);
            }
            (frmax, pmin, pmax)
        };
        // from the cavity edge outward the force is O(1)-scaled and must be axisymmetric:
        // any angular (grid m=4) variation or spurious radial force is a density-coupling
        // anisotropy bug. measured at dx=0.02: spread <= 0.13%, |F_r| <= 0.06% of |F_phi|.
        for &r in &[2.5, 3.0, 4.0] {
            let (frmax, pmin, pmax) = probe(r, 0.02);
            let scale = pmax.abs().max(pmin.abs()).max(1e-30);
            let spread = (pmax - pmin) / scale;
            assert!(spread < 0.01, "angular force spread {spread} at r={r} (grid anisotropy in the density coupling)");
            assert!(frmax / scale < 0.01, "spurious radial force {frmax} at r={r} ({}% of |F_phi|)", 100.0 * frmax / scale);
        }
        // deep in the cavity (r below the carve-out radius) the density sits on the 1e-5
        // floor and the residual force is dynamically negligible in ABSOLUTE terms — the
        // relative spread there is meaningless (noise over noise).
        let (frmax, pmin, pmax) = probe(2.0, 0.02);
        assert!(pmin.abs().max(pmax.abs()) < 1e-4, "cavity-floor azimuthal force not negligible: [{pmin:.3e},{pmax:.3e}]");
        assert!(frmax < 1e-4, "cavity-floor radial force not negligible: {frmax:.3e}");
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

    // the ADIABATIC energy twin: a linear shear vx = S y has a CONSTANT stress (zero force) but a
    // nonzero viscous HEATING Phi = tau : grad(v) = mu S^2 = rho nu S^2 (>= 0), booked into the total
    // energy via div(tau . v). the momentum stays zero; the energy increment is the exact dissipation.
    #[test]
    fn linear_shear_heats_at_the_dissipation_rate() {
        let (s, nu, rho, dt) = (0.9, 0.02, 1.3, 0.01);
        let (dx, dy) = (0.05, 0.05);
        let (v, r) = stencil(0.1, 0.3, dx, dy, |_, y| [s * y, 0.0], |_, _| rho);
        let (dmom, dnrg) = viscous_update_2d(&v, &r, &uni(nu), dx, dy, dt);
        // constant stress -> no force.
        assert!(dmom[0].abs() < 1e-14 && dmom[1].abs() < 1e-14, "force should vanish: {dmom:?}");
        // heating: dnrg = dt * rho nu S^2, exactly (linear field -> exact central differences).
        let expected = dt * rho * nu * s * s;
        assert!(
            (dnrg - expected).abs() < 1e-14,
            "viscous heating off: dnrg = {dnrg}, expected rho nu S^2 dt = {expected}"
        );
        assert!(dnrg > 0.0, "viscous heating must be positive (irreversible dissipation)");
    }

    // the 2.5D DOF-aware energy: a PURELY out-of-plane shear v_z = S y (v_x = v_y = 0) has a constant
    // stress tau_2y = mu S -> zero force on all 3 momentum components, but heating Phi = mu S^2 booked
    // into the energy. verifies the toroidal-velocity path the plain 2D kernel cannot express.
    #[test]
    fn out_of_plane_shear_heats_2p5d() {
        let (s, nu, rho, dt) = (0.9, 0.02, 1.3, 0.01);
        let (dx, dy) = (0.05, 0.05);
        let mut v = [[Tensor::<f64, 3>::zeros(); 3]; 3];
        let mut r = [[0.0; 3]; 3];
        for jj in 0..3 {
            for ii in 0..3 {
                let y = 0.3 + (jj as f64 - 1.0) * dy;
                v[jj][ii] = Tensor::new([0.0, 0.0, s * y]);
                r[jj][ii] = rho;
            }
        }
        let (dmom, dnrg) = viscous_update_2p5d(&v, &r, &[[nu; 3]; 3], [dx, dy], dt);
        for i in 0..3 {
            assert!(dmom[i].abs() < 1e-14, "constant stress -> zero force, got {dmom:?}");
        }
        let expected = dt * rho * nu * s * s;
        assert!((dnrg - expected).abs() < 1e-14, "2.5D heating off: dnrg {dnrg}, expected {expected}");
        assert!(dnrg > 0.0);
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

    // -- general orthogonal operator -----------------------------------------
    // h = (1, 1) recovers the flat Cartesian operator to roundoff (every metric
    // term is an exact zero).
    #[test]
    fn orthogonal_reduces_to_cartesian() {
        let (nu, rho, dt) = (0.03, 1.1, 0.01);
        let (dx, dy) = (0.05, 0.05);
        let (v, r) = stencil(0.2, 0.13, dx, dy, |x, y| [x + 0.5 * y, -0.3 * x * x], |_, _| rho);
        let ones = [[1.0f64; 3]; 3];
        let d_o = viscous_mom_update_orthogonal_2d(&v, &r, &uni(nu), &ones, &ones, dx, dy, dt);
        let d_c = viscous_mom_update_2d(&v, &r, &uni(nu), dx, dy, dt);
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
        assert!(rel(d_o[0], d_c[0]) < 1e-11, "f1 {} vs {}", d_o[0], d_c[0]);
        assert!(rel(d_o[1], d_c[1]) < 1e-11, "f2 {} vs {}", d_o[1], d_c[1]);
    }

    // h = (1, R) recovers the hand-written cylindrical operator (which is itself
    // bit-gated) to roundoff — the general form and the chart-specific form agree.
    #[test]
    fn orthogonal_reduces_to_cylindrical() {
        let (nu, rho, dt) = (0.04, 1.2, 0.01);
        let (r_c, dr, dphi) = (1.5, 0.05, 0.1);
        let (v, r) =
            cyl_stencil(r_c, dr, dphi, |rr, pp| [0.2 * (rr + pp).cos(), (1.0 / rr).sqrt()], |_, _| rho);
        let ones = [[1.0f64; 3]; 3];
        let h2: [[f64; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|i| r_c + (i as f64 - 1.0) * dr));
        let d_o = viscous_mom_update_orthogonal_2d(&v, &r, &uni(nu), &ones, &h2, dr, dphi, dt);
        let d_c = viscous_mom_update_cyl_2d(&v, &r, &uni(nu), r_c, dr, dphi, dt);
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
        assert!(rel(d_o[0], d_c[0]) < 1e-9, "f1 {} vs {}", d_o[0], d_c[0]);
        assert!(rel(d_o[1], d_c[1]) < 1e-9, "f2 {} vs {}", d_o[1], d_c[1]);
    }

    // rigid rotation is stress-free in the general operator too (h = (1, R)).
    #[test]
    fn orthogonal_rigid_rotation_books_zero_force() {
        let (om, nu, rho, dt) = (0.7, 0.05, 1.3, 0.01);
        let (r_c, dr, dphi) = (1.5, 0.05, 0.1);
        let (v, r) = cyl_stencil(r_c, dr, dphi, |rr, _| [0.0, om * rr], |_, _| rho);
        let ones = [[1.0f64; 3]; 3];
        let h2: [[f64; 3]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|i| r_c + (i as f64 - 1.0) * dr));
        let d = viscous_mom_update_orthogonal_2d(&v, &r, &uni(nu), &ones, &h2, dr, dphi, dt);
        assert!(d[0].abs() < 1e-12 && d[1].abs() < 1e-12, "not stress-free: {d:?}");
    }

    // THE conservation gate: a smooth, doubly-periodic fake orthogonal metric
    // h1 = 1, h2 = 1 + 0.3 sin(2pi i/n), INDEPENDENT of x2 (the angular axis is
    // Killing), so h2 * mom_2 is the conserved generalized angular momentum. every
    // face flux (the single-valued h2^2 tau_12 and h1 h2 tau_22) telescopes on a
    // periodic grid, so the total change is EXACTLY zero — the property a naive
    // non-conservative discretization would violate.
    #[test]
    fn orthogonal_conserves_generalized_angular_momentum() {
        use std::f64::consts::PI;
        let n = 12usize;
        let (dx1, dx2, nu, dt) = (0.1, 0.1, 0.05, 1e-4);
        let h2f = |i: usize| 1.0 + 0.3 * (2.0 * PI * (i as f64) / (n as f64)).sin();
        let vfun = |i: usize, j: usize| {
            let (fi, fj) = (i as f64 / n as f64, j as f64 / n as f64);
            [
                0.4 * (2.0 * PI * fi).cos() + 0.2 * (2.0 * PI * fj).sin(),
                0.3 * (2.0 * PI * fj).cos() + 0.1 * (2.0 * PI * fi).sin(),
            ]
        };
        let rfun = |i: usize, j: usize| {
            1.0 + 0.2 * (2.0 * PI * i as f64 / n as f64).sin() * (2.0 * PI * j as f64 / n as f64).cos()
        };
        let mut total = 0.0f64;
        for jc in 0..n {
            for ic in 0..n {
                let mut v = [[Tensor::<f64, 2>::zeros(); 3]; 3];
                let mut r = [[0.0; 3]; 3];
                let h1s = [[1.0; 3]; 3];
                let mut h2s = [[0.0; 3]; 3];
                for dj in 0..3 {
                    for di in 0..3 {
                        let i = (ic + di + n - 1) % n;
                        let j = (jc + dj + n - 1) % n;
                        v[dj][di] = Tensor::new(vfun(i, j));
                        r[dj][di] = rfun(i, j);
                        h2s[dj][di] = h2f(i); // x2-independent -> Killing
                    }
                }
                let d = viscous_mom_update_orthogonal_2d(&v, &r, &uni(nu), &h1s, &h2s, dx1, dx2, dt);
                // the conserved increment: h1_c h2_c^2 * dmom_2 (common dx dropped).
                let h2c = h2f(ic);
                total += h2c * h2c * d[1];
            }
        }
        assert!(total.abs() < 1e-12, "angular momentum not conserved: {total}");
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
