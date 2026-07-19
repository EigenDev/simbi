// =============================================================================
// observe.rs
//
// observer-side reductions over a photon catalog: collapse a `Vec<PhotonEvent>` into
// the three afterglow
// observables for a chosen line of sight —
//   - `compute_lightcurve_from_events`   : flux density vs observer time & frequency,
//   - `compute_skymap_from_events`       : surface brightness on the plane of the sky,
//   - `compute_polarization_from_events` : stokes / polarization degree & angle vs time.
//
// these operate purely on the raw-f64 events (the serialization boundary), so they
// carry no `units` types — they are binning + normalization over a fixed catalog.
// absorbed packets are skipped; the arrival time is the SAME equal-arrival-time-surface
// form (1+z)(t_em - r.n/c) everywhere (light curve, sky map, polarization). the light
// curve is the sky map integrated over the sky: identical per-packet selection (EATS,
// observer-direction doppler delta^power, observed-frequency band) binned in observer
// TIME (the sky map bins the same packets by sky position), so F_nu(t) and the image are the same physical quantity.
//
// usage:
//  let lc = compute_lightcurve_from_events(&events, [s,0,c], &nus, z, d_l, &tbins, 3.0, 0.1);
// =============================================================================

use crate::constants::{C_LIGHT, H_PLANCK, PI, SECONDS_PER_DAY};
use crate::event::PhotonEvent;

/// doppler weighting exponent for a monochromatic / narrow-band image: the relativistic
/// specific-intensity invariant I_nu/nu^3 gives I_nu(obs) = delta^3 I'_nu(emit) (the spectral
/// slope correction is already carried by the sampled photon frequencies, so it is delta^3, not
/// delta^{3+alpha}). pass to `compute_skymap` as `doppler_power`.
pub const DOPPLER_BAND: f64 = 3.0;

/// doppler weighting exponent for a frequency-integrated (bolometric) image: delta^4.
pub const DOPPLER_BOLOMETRIC: f64 = 4.0;

/// convert a CGS spectral flux density [erg/(s cm^2 Hz)] to milli-janskys:
/// 1 Jy = 1e-23 erg/(s cm^2 Hz), and 1 mJy = 1e-3 Jy, so [erg/(s cm^2 Hz)] * 1e26 = [mJy].
pub const MJY_PER_CGS_FLUX: f64 = 1.0e26;

/// flux density binned by observer time and frequency. `fluxes` is `n_times * n_freqs`,
/// indexed `t_bin * n_freqs + f_bin` in [mJy].
#[derive(Clone, Debug)]
pub struct ObserverLightcurve {
    pub times:       Vec<f64>,
    pub fluxes:      Vec<f64>,
    pub frequencies: Vec<f64>,
}

/// a sky-plane surface-brightness image on a UNIFORM CARTESIAN pixel grid (not polar): the
/// afterglow image is a tiny patch of sky, so equal-area pixels are the natural, singularity-
/// free representation — unlike a polar (theta, phi) grid whose `1/theta` cell area spuriously
/// over-brightens the center and erases the limb-brightened ring. `intensity` is row-major
/// `[iy * n_pix + ix]` in surface-brightness units (weight per cm^2 of sky plane); `half_width`
/// is the image half-width as a projected length [cm] (divide by the luminosity distance for
/// an angular half-width).
#[derive(Clone, Debug)]
pub struct SkyImage {
    pub intensity:  Vec<f64>,
    pub n_pix:      usize,
    pub half_width: f64,
}

impl SkyImage {
    /// surface brightness at pixel (ix, iy).
    #[inline]
    pub fn pixel(&self, ix: usize, iy: usize) -> f64 {
        self.intensity[iy * self.n_pix + ix]
    }

    /// the axisymmetric radial surface-brightness profile in `n_rings` EQUAL-AREA annuli
    /// (binned by `(R / half_width)^2`, so each ring covers the same sky area). ring 0 is the
    /// center; this is the limb-brightening diagnostic — a ring shows up as an off-center peak.
    pub fn radial_profile(&self, n_rings: usize) -> Vec<f64> {
        let mut sum = vec![0.0; n_rings];
        let mut cnt = vec![0.0; n_rings];
        let px = 2.0 * self.half_width / self.n_pix as f64;
        for iy in 0..self.n_pix {
            for ix in 0..self.n_pix {
                let x = -self.half_width + (ix as f64 + 0.5) * px;
                let y = -self.half_width + (iy as f64 + 0.5) * px;
                let frac = (x * x + y * y) / (self.half_width * self.half_width);
                if frac >= 1.0 {
                    continue;
                }
                let ring = ((frac * n_rings as f64) as usize).min(n_rings - 1);
                sum[ring] += self.pixel(ix, iy);
                cnt[ring] += 1.0;
            }
        }
        sum.iter().zip(cnt).map(|(s, c)| if c > 0.0 { s / c } else { 0.0 }).collect()
    }
}

/// polarization evolution: normalized stokes Q/U/V, degree, and angle vs observer time.
#[derive(Clone, Debug)]
pub struct PolarizationCurve {
    pub times:               Vec<f64>,
    pub polarization_degree: Vec<f64>,
    pub polarization_angle:  Vec<f64>,
    pub stokes_q:            Vec<f64>,
    pub stokes_u:            Vec<f64>,
    pub stokes_v:            Vec<f64>,
}

#[inline]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / m, v[1] / m, v[2] / m]
}

/// the index of the bin in `edges` (ascending) containing `x`, or `None` if out of range.
#[inline]
fn bin_index(edges: &[f64], x: f64) -> Option<usize> {
    if edges.len() < 2 {
        return None;
    }
    for i in 0..edges.len() - 1 {
        if x >= edges[i] && x < edges[i + 1] {
            return Some(i);
        }
    }
    None
}

/// light curve for a chosen line of sight = the SKY MAP integrated over the sky. each
/// non-absorbed packet is binned by its EATS arrival time and contributes the SAME beamed
/// flux the sky map would place on the image: `F_nu = energy * delta^doppler_power /
/// (4 pi d_L^2 dt dnu)` in [mJy]. `time_bins` are day edges; `frequencies` are the CENTER
/// observed frequencies [Hz] (one light curve each), each with a monochromatic band of width
/// `dnu = nu0 * frac_bandwidth`. because the catalog's `nu_emit` are importance-sampled from
/// the SPN98 synchrotron spectrum, banding on the OBSERVED frequency recovers the true per-Hz
/// flux density (not the crude all-energy/dnu approximation). `doppler_power` is 3 for the
/// specific-intensity (per-Hz) flux — see `DOPPLER_BAND`. additive across catalogs (streams).
#[allow(clippy::too_many_arguments)]
pub fn compute_lightcurve_from_events(
    events: &[PhotonEvent],
    observer_direction: [f64; 3],
    frequencies: &[f64],
    redshift: f64,
    luminosity_distance: f64,
    time_bins: &[f64],
    doppler_power: f64,
    frac_bandwidth: f64,
) -> ObserverLightcurve {
    let n_times = time_bins.len();
    let n_freqs = frequencies.len();
    let mut fluxes = vec![0.0; n_times * n_freqs];

    let obs_hat = normalize(observer_direction);
    let d_l_sq = luminosity_distance * luminosity_distance;
    let one_plus_z = 1.0 + redshift;
    let time_bins_s: Vec<f64> = time_bins.iter().map(|t| t * SECONDS_PER_DAY.value()).collect();

    for evt in events.iter().filter(|e| !e.absorbed) {
        // equal-arrival-time surface, IDENTICAL to compute_skymap: t_obs = (1+z)(t_em - r.n/c).
        let r_dot_n = evt.x * obs_hat[0] + evt.y * obs_hat[1] + evt.z * obs_hat[2];
        let t_arrival = one_plus_z * (evt.t_emission - r_dot_n / C_LIGHT.value());
        let Some(t_bin) = bin_index(&time_bins_s, t_arrival) else { continue };
        let dt = time_bins_s[t_bin + 1] - time_bins_s[t_bin];

        // observer-direction doppler from the lab-frame fluid VELOCITY (not the random photon
        // direction, which is a sampling artifact the sky map also ignores): delta = 1/(gamma(1-beta.n)).
        let b = evt.beta_vec;
        let beta_dot_n = b[0] * obs_hat[0] + b[1] * obs_hat[1] + b[2] * obs_hat[2];
        let delta = 1.0 / (evt.lorentz_factor() * (1.0 - beta_dot_n));
        let nu_obs = delta * evt.nu_emit / one_plus_z;
        let beamed = evt.energy_weight * evt.stokes_i * delta.powf(doppler_power);

        // accumulate into every requested frequency whose monochromatic band brackets nu_obs.
        for (f_idx, &nu0) in frequencies.iter().enumerate() {
            let dnu = nu0 * frac_bandwidth;
            if (nu_obs - nu0).abs() <= 0.5 * dnu {
                let f_nu = beamed / (4.0 * PI * d_l_sq * dt * dnu) * MJY_PER_CGS_FLUX;
                fluxes[t_bin * n_freqs + f_idx] += f_nu;
            }
        }
    }

    ObserverLightcurve {
        times:       time_bins.to_vec(),
        fluxes,
        frequencies: frequencies.to_vec(),
    }
}

/// sky-plane image at a given observer time / energy band, on a uniform Cartesian pixel grid.
///
/// each in-window, in-band packet is projected onto the plane perpendicular to the line of
/// sight `observer_direction` and binned into `n_pix * n_pix` equal-area pixels, weighted by
/// the OBSERVER-DIRECTION doppler boost `delta^doppler_power` (delta recomputed from the fluid
/// velocity — radial, magnitude from the stored lorentz factor — toward the line of sight, NOT
/// the packet's own random emission direction). this is the relativistic beaming that produces
/// the limb-brightened ring. the EATS arrival time
/// t_obs = (1+z)(t_em - r.n/c) within `+/- time_window/2` selects the surface; the image extent
/// is set by the in-window events (so the visible ring fills the frame).
///
/// `doppler_power` is the one physics knob: 3 for specific intensity (I_nu/nu^3 invariant),
/// 4 bolometric — calibrate against the analytic Granot-Sari image.
///
/// `frequency_hz > 0` selects the OBSERVED frequency band `nu0 +/- frac_bandwidth/2 * nu0`
/// per packet (nu_obs = delta * nu_emit / (1+z), the same selection the light curve uses), so
/// the image is the monochromatic per-Hz flux map a `1/dnu` calibration expects. 0 disables
/// banding (an all-band energy image — divide by a bandwidth at your peril).
#[allow(clippy::too_many_arguments)]
pub fn compute_skymap(
    events: &[PhotonEvent],
    observer_direction: [f64; 3],
    observer_time: f64,
    time_window: f64,
    energy_min: f64,
    energy_max: f64,
    redshift: f64,
    doppler_power: f64,
    n_pix: usize,
    fixed_half_width: f64,
    frequency_hz: f64,
    frac_bandwidth: f64,
) -> SkyImage {
    let n = normalize(observer_direction);
    // an orthonormal basis (e1, e2) spanning the sky plane perpendicular to n.
    let e1 = if n[2].abs() < 0.99 { normalize([-n[1], n[0], 0.0]) } else { normalize([0.0, -n[2], n[1]]) };
    let e2 = [
        n[1] * e1[2] - n[2] * e1[1],
        n[2] * e1[0] - n[0] * e1[2],
        n[0] * e1[1] - n[1] * e1[0],
    ];
    let one_plus_z = 1.0 + redshift;
    let t_obs_s = observer_time * SECONDS_PER_DAY.value();
    let half_window = 0.5 * time_window * SECONDS_PER_DAY.value();

    // collect the in-window, in-band sky-plane offsets and their beamed weights.
    let mut pts: Vec<([f64; 2], f64)> = Vec::new();
    for evt in events.iter().filter(|e| !e.absorbed) {
        // energy band filter on the PHOTON energy h*nu_emit, which sets the spectral band.
        let photon_energy = H_PLANCK.value() * evt.nu_emit;
        if photon_energy < energy_min || photon_energy > energy_max {
            continue;
        }
        let r_dot_n = evt.x * n[0] + evt.y * n[1] + evt.z * n[2];
        let t_arrival = one_plus_z * (evt.t_emission - r_dot_n / C_LIGHT.value());
        if (t_arrival - t_obs_s).abs() > half_window {
            continue;
        }

        // observer-direction doppler from the stored lab-frame fluid VELOCITY VECTOR — valid for
        // any flow direction (radial or laterally spreading):
        // delta = 1 / (gamma (1 - beta . n)).
        let b = evt.beta_vec;
        let beta_dot_n = b[0] * n[0] + b[1] * n[1] + b[2] * n[2];
        let delta = 1.0 / (evt.lorentz_factor() * (1.0 - beta_dot_n));

        // monochromatic band on the OBSERVED frequency (identical to the light curve's
        // selection), so the accumulated energy corresponds to the calibration bandwidth.
        if frequency_hz > 0.0 {
            let nu_obs = delta * evt.nu_emit / one_plus_z;
            if (nu_obs - frequency_hz).abs() > 0.5 * frequency_hz * frac_bandwidth {
                continue;
            }
        }
        let weight = evt.energy_weight * evt.stokes_i * delta.powf(doppler_power);

        let proj1 = evt.x * e1[0] + evt.y * e1[1] + evt.z * e1[2];
        let proj2 = evt.x * e2[0] + evt.y * e2[1] + evt.z * e2[2];
        pts.push(([proj1, proj2], weight));
    }

    // image extent: a CALLER-FIXED half-width (a shared field of view, so per-checkpoint images
    // accumulate onto one grid for streaming) when `fixed_half_width > 0`; else auto-size from the
    // in-window events with a little padding.
    let half_width = if fixed_half_width > 0.0 {
        fixed_half_width
    } else {
        let mut hw = 0.0_f64;
        for (q, _) in &pts {
            hw = hw.max(q[0].abs()).max(q[1].abs());
        }
        hw * 1.1
    };

    let mut intensity = vec![0.0; n_pix * n_pix];
    if half_width <= 0.0 || n_pix == 0 {
        return SkyImage { intensity, n_pix, half_width };
    }
    let px = 2.0 * half_width / n_pix as f64;
    for (q, w) in &pts {
        let ix = (((q[0] + half_width) / (2.0 * half_width)) * n_pix as f64) as isize;
        let iy = (((q[1] + half_width) / (2.0 * half_width)) * n_pix as f64) as isize;
        if ix < 0 || iy < 0 || ix as usize >= n_pix || iy as usize >= n_pix {
            continue;
        }
        intensity[iy as usize * n_pix + ix as usize] += w;
    }
    // normalize to a surface brightness (weight per unit sky-plane area).
    let pixel_area = px * px;
    for v in intensity.iter_mut() {
        *v /= pixel_area;
    }

    SkyImage { intensity, n_pix, half_width }
}

/// polarization evolution for a chosen line of sight: accumulate energy-weighted stokes
/// parameters per observer-time bin, normalize by intensity, and derive the linear
/// polarization degree sqrt(Q^2+U^2) and angle 0.5 atan2(U, Q).
pub fn compute_polarization_from_events(
    events: &[PhotonEvent],
    observer_direction: [f64; 3],
    time_bins: &[f64],
    energy_min: f64,
    energy_max: f64,
    redshift: f64,
) -> PolarizationCurve {
    let n_times = time_bins.len();
    let mut stokes_q = vec![0.0; n_times];
    let mut stokes_u = vec![0.0; n_times];
    let mut stokes_v = vec![0.0; n_times];
    let mut stokes_i_total = vec![0.0; n_times];

    let obs_hat = normalize(observer_direction);
    let time_bins_s: Vec<f64> = time_bins.iter().map(|t| t * SECONDS_PER_DAY.value()).collect();

    for evt in events.iter().filter(|e| !e.absorbed) {
        // energy band filter on the PHOTON energy h*nu_emit, which sets the spectral band.
        let photon_energy = H_PLANCK.value() * evt.nu_emit;
        if photon_energy < energy_min || photon_energy > energy_max {
            continue;
        }
        let cos_angle = evt.px * obs_hat[0] + evt.py * obs_hat[1] + evt.pz * obs_hat[2];
        if cos_angle < 0.5 {
            continue;
        }
        // same equal-arrival-time surface as the light curve / skymap: t_obs = (1+z)(t_em - r.n/c).
        let r_dot_n = evt.x * obs_hat[0] + evt.y * obs_hat[1] + evt.z * obs_hat[2];
        let t_arrival = (1.0 + redshift) * (evt.t_emission - r_dot_n / C_LIGHT.value());
        let Some(t_bin) = bin_index(&time_bins_s, t_arrival) else { continue };

        stokes_i_total[t_bin] += evt.energy_weight * evt.stokes_i;
        stokes_q[t_bin] += evt.energy_weight * evt.stokes_q;
        stokes_u[t_bin] += evt.energy_weight * evt.stokes_u;
        stokes_v[t_bin] += evt.energy_weight * evt.stokes_v;
    }

    let mut polarization_degree = vec![0.0; n_times];
    let mut polarization_angle = vec![0.0; n_times];
    for i in 0..n_times {
        if stokes_i_total[i] > 0.0 {
            stokes_q[i] /= stokes_i_total[i];
            stokes_u[i] /= stokes_i_total[i];
            stokes_v[i] /= stokes_i_total[i];
            let (q, u) = (stokes_q[i], stokes_u[i]);
            polarization_degree[i] = (q * q + u * u).sqrt();
            polarization_angle[i] = 0.5 * u.atan2(q);
        }
    }

    PolarizationCurve {
        times: time_bins.to_vec(),
        polarization_degree,
        polarization_angle,
        stokes_q,
        stokes_u,
        stokes_v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a single visible packet heading at the observer along +x. `nu_emit` sets the observed
    // frequency (doppler=1 here); `weight` is the flux/intensity contribution.
    fn packet_toward_x(nu_emit: f64, weight: f64, radius: f64) -> PhotonEvent {
        PhotonEvent {
            t_emission: 0.0, x: radius, y: 0.0, z: 0.0,
            nu_emit, energy_weight: weight, px: 1.0, py: 0.0, pz: 0.0,
            stokes_i: 1.0, stokes_q: 0.0, stokes_u: 0.0, stokes_v: 0.0,
            doppler_factor: 1.0, beta_vec: [0.0, 0.0, 0.0], optical_depth: 0.0,
            cell_id: 0, absorbed: false, n_scatter: 0,
        }
    }

    // a visible packet lands flux in the right (time, freq) bin; an absorbed one is skipped.
    #[test]
    fn lightcurve_bins_visible_flux() {
        let r = 1.0e16;
        // nu_obs = doppler * nu_emit / (1+z) = 1e15, inside [1e14, 1e16).
        let visible = packet_toward_x(1.0e15, 1.0, r);
        let mut absorbed = visible;
        absorbed.absorbed = true;

        // a single CENTER frequency at the packet's observed nu (delta=1 here -> nu_obs=1e15).
        let nus = vec![1.0e15];
        // EATS t_obs = (1+z)(t_em - r.n/c) = -(1e16/c) ~ -3.86 day: a near-side emitter (at +x,
        // toward the +x observer) arrives EARLY relative to the origin -> inside [-10, 0] day.
        let tbins = vec![-10.0, 0.0];
        let obs = [1.0, 0.0, 0.0];

        let lc = compute_lightcurve_from_events(&[visible], obs, &nus, 0.0, 1.0e26, &tbins, 3.0, 0.1);
        assert!(lc.fluxes.iter().any(|&f| f > 0.0), "visible packet should land flux");

        let lc0 = compute_lightcurve_from_events(&[absorbed], obs, &nus, 0.0, 1.0e26, &tbins, 3.0, 0.1);
        assert!(lc0.fluxes.iter().all(|&f| f == 0.0), "absorbed packet contributes nothing");
    }

    // the monochromatic band selects only packets whose OBSERVED frequency brackets the target:
    // a packet at nu_obs=1e15 lands in a band centered there but is absent from a far band.
    #[test]
    fn lightcurve_bands_by_observed_frequency() {
        let evt = packet_toward_x(1.0e15, 1.0, 1.0e16);
        let obs = [1.0, 0.0, 0.0];
        let tbins = vec![-10.0, 0.0];
        let on = compute_lightcurve_from_events(&[evt], obs, &[1.0e15], 0.0, 1.0e26, &tbins, 3.0, 0.1);
        let off = compute_lightcurve_from_events(&[evt], obs, &[1.0e12], 0.0, 1.0e26, &tbins, 3.0, 0.1);
        assert!(on.fluxes[0] > 0.0, "packet lands in its own frequency band");
        assert!(off.fluxes.iter().all(|&f| f == 0.0), "packet absent from a far band");
    }

    // unpolarized packets (Q=U=V=0) give zero polarization degree but finite intensity.
    #[test]
    fn polarization_of_unpolarized_is_zero() {
        let evts: Vec<PhotonEvent> = (0..16).map(|_| packet_toward_x(1.0e15, 1.0, 1.0e16)).collect();
        let pc = compute_polarization_from_events(&evts, [1.0, 0.0, 0.0], &[-10.0, 0.0], 0.0, 1.0e30, 0.0);
        assert!(pc.polarization_degree[0].abs() < 1e-12, "unpolarized -> zero degree");
    }

    // a partially-polarized population recovers its linear polarization degree and angle.
    #[test]
    fn polarization_recovers_q() {
        let mut e = packet_toward_x(1.0e15, 1.0, 1.0e16);
        e.stokes_q = 0.4; // 40% along Q
        let pc = compute_polarization_from_events(&[e], [1.0, 0.0, 0.0], &[-10.0, 0.0], 0.0, 1.0e30, 0.0);
        assert!((pc.polarization_degree[0] - 0.4).abs() < 1e-9);
        assert!(pc.polarization_angle[0].abs() < 1e-9, "pure +Q -> angle 0");
    }

    // a thin emitting sphere (a snapshot of an expanding shell), uniform lorentz factor.
    fn shell_events(radius: f64, gamma: f64, n_theta: usize, n_phi: usize) -> Vec<PhotonEvent> {
        let mut v = Vec::new();
        for it in 0..n_theta {
            let theta = PI * (it as f64 + 0.5) / n_theta as f64;
            let (st, ct) = (theta.sin(), theta.cos());
            let beta = (1.0 - 1.0 / (gamma * gamma)).sqrt();
            for ip in 0..n_phi {
                let phi = 2.0 * PI * (ip as f64 + 0.5) / n_phi as f64;
                let rhat = [st * phi.cos(), st * phi.sin(), ct];
                v.push(PhotonEvent {
                    t_emission: 0.0,
                    x: radius * rhat[0], y: radius * rhat[1], z: radius * rhat[2],
                    nu_emit: 1.0e15, energy_weight: 1.0,
                    px: rhat[0], py: rhat[1], pz: rhat[2],
                    stokes_i: 1.0, stokes_q: 0.0, stokes_u: 0.0, stokes_v: 0.0,
                    doppler_factor: 1.0,
                    beta_vec: [beta * rhat[0], beta * rhat[1], beta * rhat[2]],
                    optical_depth: 0.0,
                    cell_id: 0, absorbed: false, n_scatter: 0,
                });
            }
        }
        v
    }

    fn argmax(v: &[f64]) -> usize {
        let mut best = 0;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best
    }

    // the cartesian grid has NO central singularity: a uniform disk of (delta=1) emitters maps
    // to a roughly uniform image — no spurious bright center. (a polar 1/theta
    // normalization would spike the center; this pins that the cartesian binning does not.)
    #[test]
    fn skymap_has_no_central_singularity() {
        let r = 1.0e16;
        let ng = 80;
        let mut evts = Vec::new();
        for i in 0..ng {
            for j in 0..ng {
                let x = -r + 2.0 * r * (i as f64 + 0.5) / ng as f64;
                let y = -r + 2.0 * r * (j as f64 + 0.5) / ng as f64;
                if x * x + y * y < r * r {
                    // emitters in the sky plane (z=0), at rest (gamma=1 -> delta=1).
                    evts.push(PhotonEvent {
                        t_emission: 0.0, x, y, z: 0.0,
                        nu_emit: 1.0e15, energy_weight: 1.0,
                        px: 0.0, py: 0.0, pz: 1.0,
                        stokes_i: 1.0, stokes_q: 0.0, stokes_u: 0.0, stokes_v: 0.0,
                        doppler_factor: 1.0, beta_vec: [0.0, 0.0, 0.0], optical_depth: 0.0,
                        cell_id: 0, absorbed: false, n_scatter: 0,
                    });
                }
            }
        }
        let img = compute_skymap(&evts, [0.0, 0.0, 1.0], 0.0, 1.0e9, 0.0, 1.0e30, 0.0, 3.0, 16, 0.0, 0.0, 0.1);
        let nonzero: Vec<f64> = img.intensity.iter().copied().filter(|&v| v > 0.0).collect();
        let mean = nonzero.iter().sum::<f64>() / nonzero.len() as f64;
        let maxv = img.intensity.iter().copied().fold(0.0, f64::max);
        assert!(maxv < 5.0 * mean, "central singularity: max {maxv} vs mean {mean}");
    }

    // the EATS time window slices the shell into a RING: the radial brightness profile peaks
    // off-center and the image center is dark — the canonical limb-brightened appearance.
    #[test]
    fn eats_selects_off_center_ring() {
        // shell radius R, R/c ~ 38.6 day; select theta ~ 60 deg (z = 0.5 R -> arrival -19.3 day).
        let r = 1.0e17;
        let evts = shell_events(r, 10.0, 240, 240);
        let img = compute_skymap(&evts, [0.0, 0.0, 1.0], -19.3, 4.0, 0.0, 1.0e30, 0.0, 3.0, 64, 0.0, 0.0, 0.1);
        let prof = img.radial_profile(10);
        let peak = argmax(&prof);
        assert!(peak >= 5, "ring should peak off-center (got ring {peak}): {prof:?}");
        assert!(prof[0] < 0.5 * prof[peak], "center should be dark vs the ring: {prof:?}");
    }

    // the observer-direction doppler weighting boosts a faster (more beamed) emitter: a gamma=10
    // shell element pointed near the line of sight outshines a gamma=2 one at the same geometry.
    #[test]
    fn skymap_doppler_weighting_boosts_faster_fluid() {
        // single emitter near the axis (theta=10 deg, strongly beamed for high gamma).
        let r = 1.0e16;
        let theta = 10.0_f64.to_radians();
        let (st, ct) = (theta.sin(), theta.cos());
        let arrival_day = -(r * ct / C_LIGHT.value()) / SECONDS_PER_DAY.value();
        let one = |gamma: f64| {
            let beta = (1.0 - 1.0 / (gamma * gamma)).sqrt();
            // radial velocity along the position direction (st, 0, ct).
            vec![PhotonEvent {
                t_emission: 0.0, x: r * st, y: 0.0, z: r * ct,
                nu_emit: 1.0e15, energy_weight: 1.0,
                px: st, py: 0.0, pz: ct,
                stokes_i: 1.0, stokes_q: 0.0, stokes_u: 0.0, stokes_v: 0.0,
                doppler_factor: 1.0, beta_vec: [beta * st, 0.0, beta * ct], optical_depth: 0.0,
                cell_id: 0, absorbed: false, n_scatter: 0,
            }]
        };
        let total = |g: f64| {
            let img = compute_skymap(&one(g), [0.0, 0.0, 1.0], arrival_day, 4.0, 0.0, 1.0e30, 0.0, 3.0, 16, 0.0, 0.0, 0.1);
            img.intensity.iter().sum::<f64>()
        };
        assert!(total(10.0) > total(2.0), "delta^3 should favor the faster fluid");
    }
}
