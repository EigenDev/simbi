// =============================================================================
// deposit.rs
//
// DETERMINISTIC sky-map deposition (Zrake, Xie & MacFadyen 2018): instead of sampling
// photon packets, each fluid patch deposits its lab-frame monochromatic synchrotron
// emissivity directly onto the sky-plane pixel it projects to, gated by the
// equal-arrival-time surface. no shot noise -> a continuous gradient at any resolution.
//
// two paths share one emissivity (`emissivity x spectral_shape`, the same normalization
// the monte-carlo packets carry):
//   - `compute_skymap_deposit_spherical`: a 1d radial profile. the image is a function of
//     projected radius alone, so the azimuth is integrated ANALYTICALLY (exact, fast).
//   - `compute_skymap_deposit`: the general spherical mesh (1/2/3d). each (r, theta) cell
//     sweeps a ring in azimuth; the in-window azimuth ARCS are found analytically, sampled
//     at pixel resolution, and the arrival window selects an EXACT radial sub-interval of
//     the cell along each ray. an optional velocity field captures lateral spreading.
//
// flux bookkeeping (both paths): the image accumulates
// `delta^doppler_power * j'_nu'(nu') * dV_lab * dt_lab` with the ANGLE-INTEGRATED comoving
// emissivity, so the caller reaches a flux density as F_nu = image.sum() / (4 pi d_L^2 dt_obs)
// — the standard optically-thin L_nu / (4 pi d_L^2). doppler_power = 2 is the j_nu / nu^2
// invariant for volume deposition.
//
// usage:
//  let img = compute_skymap_deposit(&cond, &scales, &fields, &mesh, None, nhat,
//                                   t_obs_s, half_win_s, nu_hz, z, 2.0, n_pix, half_cm, dt_s);
// =============================================================================

use crate::constants::{C_LIGHT, PI};
use crate::synchrotron::{beta as beta_of, delta_doppler, emissivity, spectral_shape};
use crate::transfer::{cell_state, radial_shell_edges};
use crate::units::Frequency;
use crate::{HydroFields, Mesh, QuantScales, SimConditions};

/// optional three-velocity components (units of c) on the mesh's coordinate axes, flat arrays
/// indexed like the hydro fields. when absent the flow is taken RADIAL with speed derived from
/// `gamma_beta` — sufficient for a spherical blast, wrong for a laterally-spreading jet.
#[derive(Clone, Copy)]
pub struct VelComponents<'a> {
    pub v1: &'a [f64],
    pub v2: Option<&'a [f64]>,
    pub v3: Option<&'a [f64]>,
}

#[inline]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / m, v[1] / m, v[2] / m]
}

/// an orthonormal basis (e1, e2) spanning the sky plane perpendicular to n (same construction
/// as the packet-catalog reducer, so both imagers share a frame).
#[inline]
fn sky_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let e1 = if n[2].abs() < 0.99 {
        normalize([-n[1], n[0], 0.0])
    } else {
        normalize([0.0, -n[2], n[1]])
    };
    let e2 = [
        n[1] * e1[2] - n[2] * e1[1],
        n[2] * e1[0] - n[0] * e1[2],
        n[0] * e1[1] - n[1] * e1[0],
    ];
    (e1, e2)
}

/// bilinear (cloud-in-cell) deposit of `w` onto the image at sky-plane offset (q1, q2):
/// nearest-pixel deposits alias the smooth brightness field into lattice jaggedness that a
/// post-hoc smoothing kernel smears anisotropically; splitting over the four neighbors keeps
/// the deposited field continuous.
#[inline]
fn cic_deposit(image: &mut [f64], n_pix: usize, half_width: f64, q1: f64, q2: f64, w: f64) {
    let px = 2.0 * half_width / n_pix as f64;
    let fx = (q1 + half_width) / px - 0.5;
    let fy = (q2 + half_width) / px - 0.5;
    let ix0 = fx.floor();
    let iy0 = fy.floor();
    let wx = fx - ix0;
    let wy = fy - iy0;
    for (di, wxi) in [(0i64, 1.0 - wx), (1, wx)] {
        for (dj, wyj) in [(0i64, 1.0 - wy), (1, wy)] {
            let ix = ix0 as i64 + di;
            let iy = iy0 as i64 + dj;
            if ix >= 0 && iy >= 0 && (ix as usize) < n_pix && (iy as usize) < n_pix {
                image[iy as usize * n_pix + ix as usize] += w * wxi * wyj;
            }
        }
    }
}

/// per-cell angular edges from an ascending array of cell CENTERS: arithmetic midpoints
/// between neighbors, boundary cells extended by their adjacent half-gap.
#[inline]
fn angular_edges(centers: &[f64], j: usize) -> (f64, f64) {
    let n = centers.len();
    let lo = if j > 0 {
        0.5 * (centers[j - 1] + centers[j])
    } else if n > 1 {
        centers[0] - 0.5 * (centers[1] - centers[0])
    } else {
        centers[0] - 0.5
    };
    let hi = if j + 1 < n {
        0.5 * (centers[j] + centers[j + 1])
    } else if n > 1 {
        centers[n - 1] + 0.5 * (centers[n - 1] - centers[n - 2])
    } else {
        centers[0] + 0.5
    };
    (lo, hi)
}

/// the interval of ray cosines c = cos(angle to the observer) for which SOME radius in
/// [r_lo, r_hi] projects into the arrival slab [proj_lo, proj_hi] (i.e. r*c lands in it).
/// the admissible set is a single interval because both signed branches touch c = 0 when
/// the slab straddles zero.
#[inline]
fn ray_cos_bounds(r_lo: f64, r_hi: f64, proj_lo: f64, proj_hi: f64) -> Option<(f64, f64)> {
    let c_max = if proj_hi >= 0.0 {
        (proj_hi / r_lo).min(1.0)
    } else {
        proj_hi / r_hi
    };
    let c_min = if proj_lo <= 0.0 {
        (proj_lo / r_lo).max(-1.0)
    } else {
        proj_lo / r_hi
    };
    (c_min <= c_max).then_some((c_min, c_max))
}

/// DETERMINISTIC sky-map deposition for a spherically-symmetric (1d radial) blast: the image
/// is a function of the projected radius ALONE, so the 1d radial flux profile is accumulated
/// (the azimuth integral is the analytic factor 2 pi) and rendered as
/// image[x, y] = profile(sqrt(x^2 + y^2)). the EATS at t_obs selects, per shell radius r, the
/// cone cos(alpha) = r.n / r inside the arrival window; that ring projects to rho = r sin(alpha).
/// cost is O(n_radii * n_alpha) + O(n_pix^2), independent of how thin the shock is.
///
/// `n_mu` / `n_phi` / `theta_max` / `observer_direction` are legacy binding-signature arguments:
/// a sphere looks the same from any angle and the azimuth needs no tessellation.
#[allow(clippy::too_many_arguments)]
pub fn compute_skymap_deposit_spherical(
    cond: &SimConditions,
    scales: &QuantScales,
    fields: &HydroFields,
    x1: &[f64],
    observer_direction: [f64; 3],
    obs_time_s: f64,
    half_window_s: f64,
    frequency_hz: f64,
    redshift: f64,
    doppler_power: f64,
    n_pix: usize,
    half_width: f64,
    emit_dt_s: f64,
    theta_max: f64,
    n_mu: usize,
    n_phi: usize,
) -> Vec<f64> {
    let mut image = vec![0.0; n_pix * n_pix];
    let ni = x1.len();
    if ni == 0 || n_pix == 0 || half_width <= 0.0 {
        return image;
    }
    let _ = (theta_max, n_mu, n_phi, observer_direction);
    let p = cond.p;
    let t_prime_s = (cond.current_time * scales.time).value();
    let one_plus_z = 1.0 + redshift;
    let c = C_LIGHT.value();

    let proj_lo = c * (t_prime_s - (obs_time_s + half_window_s) / one_plus_z);
    let proj_hi = c * (t_prime_s - (obs_time_s - half_window_s) / one_plus_z);

    let n_rho = n_pix;
    let drho = half_width / n_rho as f64;
    let mut radial_flux = vec![0.0; n_rho];

    for ii in 0..ni {
        let r = (x1[ii] * scales.length).value();
        if r <= 0.0 {
            continue;
        }
        // cos(alpha) = r.n / r in the EATS window [proj_lo, proj_hi] / r, clamped to a real cone.
        let cos_lo = (proj_lo / r).max(-1.0);
        let cos_hi = (proj_hi / r).min(1.0);
        if cos_lo >= cos_hi {
            continue; // this shell never crosses the arrival window
        }

        let (x1l, x1r) = radial_shell_edges(x1, ii);
        let cell = cell_state(cond, scales, fields, ii, p, t_prime_s);
        // the ANGLE-INTEGRATED comoving emissivity [erg/(s Hz cm^3)]: the caller's flux
        // bookkeeping is F_nu = sum / (4 pi d_L^2 dt_obs), the standard optically-thin
        // L_nu / (4 pi d_L^2) with L_nu = int emissivity dV — so the emissivity must NOT be
        // reduced to per-steradian here (that double-counts the 4 pi and dims the flux by it).
        // lab transform j_nu = delta^2 j'_nu' (the j_nu / nu^2 invariant), applied below as
        // delta^doppler_power. this normalization is packet-exact: a delta = 1 emitter's
        // deposit equals the monte-carlo catalog's total banded energy in the same window.
        let j_peak = emissivity(cell.bfield, cell.n_e, p).value(); // [erg/(s Hz cm^3)]
        let nu_c = Frequency::new(cell.nu_c);
        let nu_m = Frequency::new(cell.nu_m);
        let shell_vol =
            (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) * scales.length.cubed().value();

        // sample the band uniformly in the polar angle alpha, weighting each sample by its
        // dcos = sin(alpha) dalpha. uniform-in-cos samples space as drho = r (cos/sin) dcos on
        // the sky plane — arbitrarily SPARSE near the image center (sin -> 0), leaving interior
        // radial bins unfilled. uniform alpha bounds the rho spacing by r dalpha; the step is
        // sized to ~half a radial bin so every bin the band crosses receives flux.
        let a_lo = cos_hi.clamp(-1.0, 1.0).acos();
        let a_hi = cos_lo.clamp(-1.0, 1.0).acos();
        let n_a = (((a_hi - a_lo) * r / (0.5 * drho)).ceil() as usize).clamp(2, 16 * n_rho);
        let da = (a_hi - a_lo) / n_a as f64;

        for kc in 0..n_a {
            let alpha = a_lo + (kc as f64 + 0.5) * da;
            let (sin_a, cos_a) = alpha.sin_cos();
            let dcos = sin_a * da;
            let delta = 1.0 / (cell.w * (1.0 - cell.beta * cos_a));
            let nu_prime = Frequency::new(frequency_hz * one_plus_z / delta);
            let j_nu = j_peak * spectral_shape(p, nu_prime, nu_c, nu_m);
            // the FULL ring at rho = r sin_a: its azimuth integral is the factor 2 pi
            // (dV = shell_vol dcos dpsi, integrated over psi -> shell_vol dcos 2 pi).
            let ring_flux =
                delta.powf(doppler_power) * j_nu * shell_vol * dcos * 2.0 * PI * emit_dt_s;
            let rho = r * sin_a;
            // cloud-in-cell deposit between the two adjacent radial bins (bin centers at
            // (k + 1/2) drho): rings land at discrete projected radii, and a nearest-bin
            // deposit leaves interleaving bins EMPTY — holes orders of magnitude deep that a
            // post-hoc smoothing kernel smears anisotropically on the square pixel lattice.
            let s = (rho / half_width * n_rho as f64 - 0.5).max(0.0);
            let k = s as usize;
            let f = s - k as f64;
            if k + 1 < n_rho {
                radial_flux[k] += (1.0 - f) * ring_flux;
                radial_flux[k + 1] += f * ring_flux;
            } else if k < n_rho {
                radial_flux[k] += ring_flux;
            }
        }
    }

    // render the radial profile as a CONTINUOUS surface-brightness density: each bin's flux
    // over its exact annulus area pi (rho_hi^2 - rho_lo^2), linearly interpolated at the
    // pixel's radius. the pixel value depends on its radius alone, so the image is exactly
    // azimuthally symmetric and radially smooth. the image is then rescaled so image.sum()
    // equals the deposited flux — the caller's F_nu bookkeeping divides by pixel area and the
    // observer window.
    let px = 2.0 * half_width / n_pix as f64;
    let density: Vec<f64> = radial_flux
        .iter()
        .enumerate()
        .map(|(k, f)| f / (PI * (2 * k + 1) as f64 * drho * drho))
        .collect();
    for iy in 0..n_pix {
        let y = -half_width + (iy as f64 + 0.5) * px;
        for ix in 0..n_pix {
            let x = -half_width + (ix as f64 + 0.5) * px;
            let s = ((x * x + y * y).sqrt() / drho - 0.5).max(0.0);
            let k = s as usize;
            let f = s - k as f64;
            let d = if k + 1 < n_rho {
                (1.0 - f) * density[k] + f * density[k + 1]
            } else if k < n_rho {
                density[k]
            } else {
                0.0
            };
            image[iy * n_pix + ix] = d * px * px;
        }
    }
    let total_flux: f64 = radial_flux.iter().sum();
    let total_px: f64 = image.iter().sum();
    if total_px > 0.0 {
        let scale = total_flux / total_px;
        for v in image.iter_mut() {
            *v *= scale;
        }
    }
    image
}

/// DETERMINISTIC sky-map deposition for a general SPHERICAL mesh (1/2/3d). a 1d radial
/// profile routes to the exact analytic-azimuth path. for 2d (r, theta) axisymmetric and 3d
/// (r, theta, phi) data, each cell's ring/sector is swept in azimuth:
///   - the in-window azimuth ARCS are found analytically per (cell, theta) — the arrival
///     projection along a ring is A cos(phi - phi0) + B, so the EATS gate is a cosine band,
///   - each azimuth sample is stepped at ~half-pixel resolution on the sky,
///   - the arrival window selects an EXACT radial sub-interval of the cell along each ray
///     (no radial sub-sampling needed),
///   - the contribution delta^doppler_power * j'(nu') * dV_lab * dt_lab is CIC-deposited at
///     the projected position.
///
/// `vels` supplies the three-velocity components for lateral spreading; absent, the flow is
/// radial with speed from `gamma_beta`.
#[allow(clippy::too_many_arguments)]
pub fn compute_skymap_deposit(
    cond: &SimConditions,
    scales: &QuantScales,
    fields: &HydroFields,
    mesh: &Mesh,
    vels: Option<VelComponents>,
    observer_direction: [f64; 3],
    obs_time_s: f64,
    half_window_s: f64,
    frequency_hz: f64,
    redshift: f64,
    doppler_power: f64,
    n_pix: usize,
    half_width: f64,
    emit_dt_s: f64,
) -> Vec<f64> {
    if mesh.data_dim <= 1 {
        return compute_skymap_deposit_spherical(
            cond,
            scales,
            fields,
            mesh.x1,
            observer_direction,
            obs_time_s,
            half_window_s,
            frequency_hz,
            redshift,
            doppler_power,
            n_pix,
            half_width,
            emit_dt_s,
            PI,
            0,
            0,
        );
    }

    let mut image = vec![0.0; n_pix * n_pix];
    let x1 = mesh.x1;
    let x2 = mesh.x2;
    let ni = x1.len();
    let nj = x2.len();
    let resolved_phi = mesh.x3.is_some() && mesh.data_dim > 2;
    let nk = if resolved_phi { mesh.x3.unwrap().len() } else { 1 };
    if ni == 0 || nj == 0 || n_pix == 0 || half_width <= 0.0 {
        return image;
    }

    let p = cond.p;
    let t_prime_s = (cond.current_time * scales.time).value();
    let one_plus_z = 1.0 + redshift;
    let c = C_LIGHT.value();
    let n = normalize(observer_direction);
    let (e1, e2) = sky_basis(n);
    let px = 2.0 * half_width / n_pix as f64;
    let len_scale = scales.length.value();

    // arrival slab: r.n in [proj_lo, proj_hi] selects the EATS window.
    let proj_lo = c * (t_prime_s - (obs_time_s + half_window_s) / one_plus_z);
    let proj_hi = c * (t_prime_s - (obs_time_s - half_window_s) / one_plus_z);

    // ring-projection decomposition of the observer direction: along a ring at polar angle
    // theta, r.n / r = a_perp sin(theta) cos(phi - phi0) + n_z cos(theta).
    let a_perp = (n[0] * n[0] + n[1] * n[1]).sqrt();
    let phi0 = n[1].atan2(n[0]);

    for kk in 0..nk {
        // azimuth domain: the cell's own extent for resolved-phi 3d data; the full circle for
        // axisymmetric 2d data (each (r, theta) cell IS a ring).
        let (cell_phi_lo, cell_phi_hi) = if resolved_phi {
            angular_edges(mesh.x3.unwrap(), kk)
        } else {
            (0.0, 2.0 * PI)
        };
        for jj in 0..nj {
            let (t_lo, t_hi) = angular_edges(x2, jj);
            let (t_lo, t_hi) = (t_lo.max(0.0), t_hi.min(PI));
            if t_hi <= t_lo {
                continue;
            }
            for ii in 0..ni {
                let (x1l, x1r) = radial_shell_edges(x1, ii);
                let (r_lo, r_hi) = (x1l * len_scale, x1r * len_scale);
                if r_hi <= 0.0 {
                    continue;
                }
                let Some((c_min, c_max)) = ray_cos_bounds(r_lo, r_hi, proj_lo, proj_hi) else {
                    continue;
                };

                let idx = kk * ni * nj + jj * ni + ii;
                let cell = cell_state(cond, scales, fields, idx, p, t_prime_s);
                let j_peak = emissivity(cell.bfield, cell.n_e, p).value();
                let nu_c = Frequency::new(cell.nu_c);
                let nu_m = Frequency::new(cell.nu_m);

                // three-velocity components on the coordinate axes (radial default).
                let (v1, v2, v3) = match vels {
                    Some(v) => (
                        v.v1[idx],
                        v.v2.map_or(0.0, |a| a[idx]),
                        v.v3.map_or(0.0, |a| a[idx]),
                    ),
                    None => (beta_of(fields.gamma_beta[idx]), 0.0, 0.0),
                };

                // sub-sample theta so the sky-plane step r dtheta stays under half a pixel.
                let n_t = ((((t_hi - t_lo) * r_hi) / (0.5 * px)).ceil() as usize)
                    .clamp(1, 4 * n_pix);
                let dth = (t_hi - t_lo) / n_t as f64;
                for kt in 0..n_t {
                    let theta = t_lo + (kt as f64 + 0.5) * dth;
                    let (sin_t, cos_t) = theta.sin_cos();
                    let a_ring = a_perp * sin_t;
                    let b_ring = n[2] * cos_t;

                    // in-window azimuth arcs: a_ring cos(phi - phi0) + b_ring in [c_min, c_max].
                    let arcs: [(f64, f64); 2] = if a_ring.abs() < 1.0e-14 {
                        if b_ring >= c_min && b_ring <= c_max {
                            [(cell_phi_lo, cell_phi_hi), (0.0, -1.0)]
                        } else {
                            continue;
                        }
                    } else {
                        let u_lo = ((c_min - b_ring) / a_ring).clamp(-1.0, 1.0);
                        let u_hi = ((c_max - b_ring) / a_ring).clamp(-1.0, 1.0);
                        let (u_lo, u_hi) = (u_lo.min(u_hi), u_lo.max(u_hi));
                        if u_lo >= u_hi
                            && ((c_min - b_ring) / a_ring > 1.0
                                || (c_max - b_ring) / a_ring < -1.0)
                        {
                            continue;
                        }
                        // cos is decreasing on [0, pi]: the +/- alpha branches mirror about phi0.
                        let al_lo = u_hi.acos();
                        let al_hi = u_lo.acos();
                        [(phi0 + al_lo, phi0 + al_hi), (phi0 - al_hi, phi0 - al_lo)]
                    };

                    for &(arc_lo, arc_hi) in &arcs {
                        if arc_hi <= arc_lo {
                            continue;
                        }
                        // step the arc at ~half-pixel sky resolution.
                        let step_scale = (r_hi * sin_t).max(px);
                        let n_f = ((((arc_hi - arc_lo) * step_scale) / (0.5 * px)).ceil()
                            as usize)
                            .clamp(1, 8 * n_pix);
                        let dphi = (arc_hi - arc_lo) / n_f as f64;
                        for kf in 0..n_f {
                            let phi = arc_lo + (kf as f64 + 0.5) * dphi;
                            // resolved-phi data: only the overlap with the cell's own sector
                            // counts (map the sample into [cell_lo, cell_lo + 2 pi) first).
                            if resolved_phi {
                                let span = 2.0 * PI;
                                let wrapped =
                                    (phi - cell_phi_lo).rem_euclid(span) + cell_phi_lo;
                                if wrapped < cell_phi_lo || wrapped > cell_phi_hi {
                                    continue;
                                }
                            }
                            let (sin_p, cos_p) = phi.sin_cos();
                            let xhat = [sin_t * cos_p, sin_t * sin_p, cos_t];
                            let cos_psi = xhat[0] * n[0] + xhat[1] * n[1] + xhat[2] * n[2];

                            // exact radial sub-interval of the cell inside the arrival slab.
                            let (ra, rb) = if cos_psi.abs() < 1.0e-14 {
                                if proj_lo <= 0.0 && proj_hi >= 0.0 {
                                    (r_lo, r_hi)
                                } else {
                                    continue;
                                }
                            } else {
                                let q1 = proj_lo / cos_psi;
                                let q2 = proj_hi / cos_psi;
                                let (q_lo, q_hi) = (q1.min(q2), q1.max(q2));
                                let ra = q_lo.max(r_lo);
                                let rb = q_hi.min(r_hi);
                                if rb <= ra {
                                    continue;
                                }
                                (ra, rb)
                            };

                            let dvol =
                                (1.0 / 3.0) * (rb * rb * rb - ra * ra * ra) * sin_t * dth * dphi;
                            let r_mid =
                                (0.5 * (ra * ra * ra + rb * rb * rb)).cbrt();

                            // velocity vector in the lab frame from the coordinate components.
                            let that = [cos_t * cos_p, cos_t * sin_p, -sin_t];
                            let phat = [-sin_p, cos_p, 0.0];
                            let beta_vec = [
                                v1 * xhat[0] + v2 * that[0] + v3 * phat[0],
                                v1 * xhat[1] + v2 * that[1] + v3 * phat[1],
                                v1 * xhat[2] + v2 * that[2] + v3 * phat[2],
                            ];
                            let bsq = (beta_vec[0] * beta_vec[0]
                                + beta_vec[1] * beta_vec[1]
                                + beta_vec[2] * beta_vec[2])
                                .min(1.0 - 1.0e-15);
                            let w_lor = 1.0 / (1.0 - bsq).sqrt();
                            let delta = delta_doppler(w_lor, beta_vec, n);

                            let nu_prime = Frequency::new(frequency_hz * one_plus_z / delta);
                            let j_nu = j_peak * spectral_shape(p, nu_prime, nu_c, nu_m);
                            let flux =
                                delta.powf(doppler_power) * j_nu * dvol * emit_dt_s;

                            let pos = [r_mid * xhat[0], r_mid * xhat[1], r_mid * xhat[2]];
                            let q1 = pos[0] * e1[0] + pos[1] * e1[1] + pos[2] * e1[2];
                            let q2 = pos[0] * e2[0] + pos[1] * e2[1] + pos[2] * e2[2];
                            cic_deposit(&mut image, n_pix, half_width, q1, q2, flux);
                        }
                    }
                }
            }
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{EnergyDensity, Length, MassDensity, Time, Velocity};

    fn conditions() -> SimConditions {
        SimConditions {
            dt: 1.0,
            theta_obs: 0.0,
            adiabatic_index: 4.0 / 3.0,
            current_time: 1.0e6,
            p: 2.5,
            redshift: 0.0,
            eps_e: 0.1,
            eps_b: 0.01,
            d_l: Length::new(1.0e26),
            nus: vec![],
        }
    }
    fn scales() -> QuantScales {
        QuantScales {
            time: Time::new(1.0),
            pre: EnergyDensity::new(1.0),
            rho: MassDensity::new(1.0),
            velocity: Velocity::new(1.0),
            length: Length::new(1.0),
        }
    }

    fn radial_profile() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let ni = 12;
        let x1: Vec<f64> = (0..ni).map(|i| 1.0e17 * (1.0 + 0.01 * i as f64)).collect();
        (x1, vec![1.0e-24; ni], vec![2.0; ni], vec![1.0e-6; ni])
    }

    // full-sphere window bounds for the profile: covers every arrival from the shell.
    fn full_window(cond: &SimConditions, x1: &[f64]) -> (f64, f64) {
        let c = C_LIGHT.value();
        let r_max = x1[x1.len() - 1];
        let t_mid = cond.current_time;
        (t_mid, 1.05 * r_max / c)
    }

    // the deterministic deposit renders a CONTINUOUS radial brightness profile: rings land at
    // discrete projected radii, so a nearest-bin deposit leaves half-pixel radial bins with no
    // flux — holes four decades deep that gaussian smoothing then smears ANISOTROPICALLY on the
    // square lattice (a dark cross along the image axes). the gate: inside the lit disk, no
    // pixel may sit far below its radial neighbors.
    #[test]
    fn deposit_radial_profile_has_no_holes() {
        let cond = conditions();
        let sc = scales();
        let ni = 16;
        let x1: Vec<f64> = (0..ni).map(|i| 1.0e17 * (1.0 + 0.02 * i as f64)).collect();
        let (rho, gb, pre) = (vec![1.0e-24; ni], vec![2.0; ni], vec![1.0e-6; ni]);
        let fields = HydroFields { rho: &rho, gamma_beta: &gb, pre: &pre };

        // observer window wide enough that the EATS cos band covers the whole sphere, so the
        // projected rings sweep the full disk and the profile should be continuous out to the limb.
        let r_max_s = x1[ni - 1] / C_LIGHT.value();
        let n_pix = 128;
        let img = compute_skymap_deposit_spherical(
            &cond, &sc, &fields, &x1, [0.0, 0.0, 1.0], cond.current_time, 1.2 * r_max_s, 1.0e9,
            0.0, 2.0, n_pix, 1.3e17, 1.0, PI, 64, 128,
        );

        // walk the center row outward; inside the lit region every pixel must hold its own
        // against the brighter of its neighbors (a smooth profile varies by O(10%) per pixel;
        // a missed radial bin sits orders of magnitude low).
        let row: Vec<f64> = (0..n_pix).map(|ix| img[(n_pix / 2) * n_pix + ix]).collect();
        let lit: Vec<usize> = (1..n_pix - 1).filter(|&ix| row[ix] > 0.0).collect();
        assert!(lit.len() > 20, "deposit produced too small an image: {} lit", lit.len());
        let (lo, hi) = (lit[2], lit[lit.len() - 3]);
        for ix in lo..=hi {
            let nbr = row[ix - 1].max(row[ix + 1]);
            assert!(
                row[ix] > 0.2 * nbr,
                "radial hole at pixel {ix}: {} vs neighbor {nbr}",
                row[ix]
            );
        }
    }

    // a 2d (r, theta) mesh with UNIFORM fields describes the same sphere as the 1d radial
    // profile: the general deposit must reproduce the analytic-azimuth 1d image — total flux
    // and radial brightness profile alike. this is the dimensional-equivalence gate for the
    // generalization.
    #[test]
    fn deposit_2d_uniform_sphere_matches_1d() {
        let cond = conditions();
        let sc = scales();
        let (x1, rho1, gb1, pre1) = radial_profile();
        let ni = x1.len();
        let nj = 96;
        let x2: Vec<f64> = (0..nj).map(|j| PI * (j as f64 + 0.5) / nj as f64).collect();

        // broadcast the radial profile over theta (row-major j*ni + i).
        let mut rho = vec![0.0; ni * nj];
        let mut gb = vec![0.0; ni * nj];
        let mut pre = vec![0.0; ni * nj];
        for jj in 0..nj {
            for ii in 0..ni {
                rho[jj * ni + ii] = rho1[ii];
                gb[jj * ni + ii] = gb1[ii];
                pre[jj * ni + ii] = pre1[ii];
            }
        }
        let f1 = HydroFields { rho: &rho1, gamma_beta: &gb1, pre: &pre1 };
        let f2 = HydroFields { rho: &rho, gamma_beta: &gb, pre: &pre };
        let mesh2 = Mesh { x1: &x1, x2: &x2, x3: None, data_dim: 2 };

        let (t_obs, half_win) = full_window(&cond, &x1);
        let n_pix = 96;
        let hw = 1.3e17;
        let nu0 = 1.0e9;

        let img1 = compute_skymap_deposit_spherical(
            &cond, &sc, &f1, &x1, [0.0, 0.0, 1.0], t_obs, half_win, nu0, 0.0, 2.0, n_pix, hw,
            1.0, PI, 0, 0,
        );
        // an off-axis observer: the sphere must look identical from any direction.
        let th = 0.6_f64;
        let nhat = [th.sin(), 0.0, th.cos()];
        let img2 = compute_skymap_deposit(
            &cond, &sc, &f2, &mesh2, None, nhat, t_obs, half_win, nu0, 0.0, 2.0, n_pix, hw, 1.0,
        );

        let (s1, s2) = (img1.iter().sum::<f64>(), img2.iter().sum::<f64>());
        assert!(s1 > 0.0 && s2 > 0.0);
        assert!(
            (s2 / s1 - 1.0).abs() < 0.03,
            "2d total flux vs 1d: ratio {}",
            s2 / s1
        );

        // radial brightness profiles agree ring by ring (equal-area annuli).
        let prof = |img: &[f64]| -> Vec<f64> {
            let n_r = 8;
            let mut sum = vec![0.0; n_r];
            for iy in 0..n_pix {
                for ix in 0..n_pix {
                    let x = -hw + (ix as f64 + 0.5) * 2.0 * hw / n_pix as f64;
                    let y = -hw + (iy as f64 + 0.5) * 2.0 * hw / n_pix as f64;
                    let frac = (x * x + y * y) / (hw * hw);
                    if frac < 1.0 {
                        sum[((frac * n_r as f64) as usize).min(n_r - 1)] +=
                            img[iy * n_pix + ix];
                    }
                }
            }
            sum
        };
        let (p1, p2) = (prof(&img1), prof(&img2));
        for k in 0..p1.len() {
            if p1[k] > 0.02 * s1 {
                assert!(
                    (p2[k] / p1[k] - 1.0).abs() < 0.1,
                    "ring {k}: 2d {} vs 1d {}",
                    p2[k],
                    p1[k]
                );
            }
        }
    }

    // a 3d (r, theta, phi) mesh with uniform fields is the same sphere again: total flux
    // matches the 1d analytic path within quadrature tolerance.
    #[test]
    fn deposit_3d_uniform_sphere_matches_1d() {
        let cond = conditions();
        let sc = scales();
        let (x1, rho1, gb1, pre1) = radial_profile();
        let ni = x1.len();
        let nj = 48;
        let nk = 64;
        let x2: Vec<f64> = (0..nj).map(|j| PI * (j as f64 + 0.5) / nj as f64).collect();
        let x3: Vec<f64> =
            (0..nk).map(|k| 2.0 * PI * (k as f64 + 0.5) / nk as f64).collect();

        let len = ni * nj * nk;
        let mut rho = vec![0.0; len];
        let mut gb = vec![0.0; len];
        let mut pre = vec![0.0; len];
        for kk in 0..nk {
            for jj in 0..nj {
                for ii in 0..ni {
                    let idx = kk * ni * nj + jj * ni + ii;
                    rho[idx] = rho1[ii];
                    gb[idx] = gb1[ii];
                    pre[idx] = pre1[ii];
                }
            }
        }
        let f1 = HydroFields { rho: &rho1, gamma_beta: &gb1, pre: &pre1 };
        let f3 = HydroFields { rho: &rho, gamma_beta: &gb, pre: &pre };
        let mesh3 = Mesh { x1: &x1, x2: &x2, x3: Some(&x3), data_dim: 3 };

        let (t_obs, half_win) = full_window(&cond, &x1);
        let n_pix = 64;
        let hw = 1.3e17;
        let nu0 = 1.0e9;

        let img1 = compute_skymap_deposit_spherical(
            &cond, &sc, &f1, &x1, [0.0, 0.0, 1.0], t_obs, half_win, nu0, 0.0, 2.0, n_pix, hw,
            1.0, PI, 0, 0,
        );
        let th = 0.4_f64;
        let nhat = [th.sin(), 0.0, th.cos()];
        let img3 = compute_skymap_deposit(
            &cond, &sc, &f3, &mesh3, None, nhat, t_obs, half_win, nu0, 0.0, 2.0, n_pix, hw, 1.0,
        );

        let (s1, s3) = (img1.iter().sum::<f64>(), img3.iter().sum::<f64>());
        assert!(s1 > 0.0 && s3 > 0.0);
        assert!(
            (s3 / s1 - 1.0).abs() < 0.05,
            "3d total flux vs 1d: ratio {}",
            s3 / s1
        );
    }

    // a relativistic polar-cap jet is doppler-beamed along its axis: the on-axis observer
    // integrates far more flux than one at 60 degrees, and providing the velocity FIELD with
    // a lateral (theta) component toward the observer brightens the image relative to the
    // same jet spreading away — the signature a radial-only treatment cannot produce.
    #[test]
    fn deposit_2d_jet_beams_along_axis() {
        let cond = conditions();
        let sc = scales();
        let (x1, _, _, _) = radial_profile();
        let ni = x1.len();
        let nj = 64;
        let x2: Vec<f64> = (0..nj).map(|j| PI * (j as f64 + 0.5) / nj as f64).collect();

        // emission concentrated in the polar cap theta < 0.35 with gamma ~ 5 radial flow.
        let mut rho = vec![1.0e-30; ni * nj];
        let mut gb = vec![0.0; ni * nj];
        let mut pre = vec![1.0e-12; ni * nj];
        for jj in 0..nj {
            if x2[jj] < 0.35 {
                for ii in 0..ni {
                    rho[jj * ni + ii] = 1.0e-24;
                    gb[jj * ni + ii] = 5.0;
                    pre[jj * ni + ii] = 1.0e-6;
                }
            }
        }
        let f = HydroFields { rho: &rho, gamma_beta: &gb, pre: &pre };
        let mesh = Mesh { x1: &x1, x2: &x2, x3: None, data_dim: 2 };
        let (t_obs, half_win) = full_window(&cond, &x1);
        let total = |nhat: [f64; 3], vels: Option<VelComponents>| {
            compute_skymap_deposit(
                &cond, &sc, &f, &mesh, vels, nhat, t_obs, half_win, 1.0e9, 0.0, 2.0, 64,
                1.3e17, 1.0,
            )
            .iter()
            .sum::<f64>()
        };

        let on_axis = total([0.0, 0.0, 1.0], None);
        let off_axis = total([0.6_f64.sin(), 0.0, 0.6_f64.cos()], None);
        assert!(on_axis > 3.0 * off_axis, "beaming: on {on_axis} vs off {off_axis}");

        // lateral spreading: a theta-velocity component beams flux toward larger polar angles,
        // brightening the 60-degree observer relative to the purely radial flow.
        let b = 5.0_f64 / (1.0 + 25.0_f64).sqrt(); // |v| for gamma*beta = 5
        let v1: Vec<f64> = gb.iter().map(|&g| if g > 0.0 { 0.8 * b } else { 0.0 }).collect();
        let v2: Vec<f64> = gb.iter().map(|&g| if g > 0.0 { 0.5 * b } else { 0.0 }).collect();
        let spreading = total(
            [0.6_f64.sin(), 0.0, 0.6_f64.cos()],
            Some(VelComponents { v1: &v1, v2: Some(&v2), v3: None }),
        );
        assert!(
            spreading > off_axis,
            "lateral velocity toward the observer must brighten: {spreading} vs {off_axis}"
        );
    }
}
