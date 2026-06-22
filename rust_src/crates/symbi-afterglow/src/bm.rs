// =============================================================================
// bm.rs
//
// the Blandford-McKee (1976) self-similar ultrarelativistic blast wave for a
// homogeneous external medium (k = 0) — the synthetic test bed for afterglow
// imaging. `bm_profile` returns the radial fluid profile at one lab time; it feeds
// the spherical photon generator to make synthetic GRB afterglow observations.
//
// self-similar solution (k = 0), with chi = [1 + 2(m+1) gamma_sh^2](1 - r/ct), m = 3:
//   gamma^2(chi) = gamma_sh^2 / (2 chi)            (fluid lorentz factor)
//   p(chi)       = (2/3) gamma_sh^2 rho_1 c^2 chi^{-17/12}
//   n'(chi)      = 4 gamma_sh n_0 chi^{-7/4}       (proper number density)
// chi = 1 at the shock (r = R), increasing inward. the energy-conserving shock
// lorentz factor is gamma_sh^2 = 17 E / (8 pi rho_1 c^5 t^3), rho_1 = n_0 m_p.
//
// `synthesize_afterglow_events` builds an image-ready photon catalog by integrating
// the EATS across the blast's TIME EVOLUTION: a single snapshot cannot reproduce the
// canonical limb-brightened ring (the line-of-sight image center samples the LATER,
// decelerated, dimmer blast while the limb samples the bright blast at the current
// time), so it generates spherical events at a range of lab times and accumulates.
//
// usage:
//  let prof = bm_profile(1e53, 1.0, 2.5e7, 50.0, 64);
//  let events = synthesize_afterglow_events(1e53, 1.0, 2.5, .., t_obs_day, ..);
//  let image = compute_skymap(&events, [0.,0.,1.], t_obs_day, .., 3.0, n_pix);
// =============================================================================

use crate::constants::{C_LIGHT, M_P, PI, SECONDS_PER_DAY};
use crate::event::PhotonEvent;
use crate::transfer::generate_photon_events_spherical;
use crate::units::{Energy, EnergyDensity, Length, MassDensity, NumberDensity, Time, Velocity};
use crate::{HydroFields, QuantScales, SimConditions};

/// a Blandford-McKee radial profile at one lab time. all CGS; arrays are radial cell values
/// (ascending radius, `x1[ni-1]` at the shock). `rho` is the proper mass density, `gamma_beta`
/// the fluid four-velocity magnitude, `pre` the pressure.
#[derive(Clone, Debug)]
pub struct BmProfile {
    pub x1:          Vec<f64>,
    pub rho:         Vec<f64>,
    pub gamma_beta:  Vec<f64>,
    pub pre:         Vec<f64>,
    pub r_shock:     f64,
    pub gamma_shock: f64,
}

/// the shock lorentz factor squared at lab time `t` [s] for an adiabatic k=0 blast (energy
/// conserving): gamma_sh^2 = 17 E / (8 pi rho_1 c^5 t^3). dimensionally checked via `units`.
pub fn shock_lorentz_factor_sq(e_iso: f64, n0: f64, t: f64) -> f64 {
    let rho1: MassDensity = M_P * NumberDensity::new(n0);
    let c5 = C_LIGHT.squared() * C_LIGHT.squared() * C_LIGHT; // velocity^5
    let denom = 8.0 * PI * rho1 * c5 * Time::new(t).cubed(); // = energy
    ((17.0 * Energy::new(e_iso)) / denom).value()
}

/// the Blandford-McKee radial profile at lab time `t` [s], sampled over `n_cells` log-spaced
/// values of the self-similar variable `chi` in [1, `chi_max`] (chi = 1 at the shock).
pub fn bm_profile(e_iso: f64, n0: f64, t: f64, chi_max: f64, n_cells: usize) -> BmProfile {
    let gamma_sh_sq = shock_lorentz_factor_sq(e_iso, n0, t);
    let gamma_sh = gamma_sh_sq.sqrt();
    let a = 1.0 + 8.0 * gamma_sh_sq; // 1 + 2(m+1) gamma_sh^2, m = 3
    let ct = C_LIGHT.value() * t;
    let r_shock = ct * (1.0 - 1.0 / a);

    // post-shock normalizations (standard ultrarelativistic k=0 jump conditions).
    let p2: EnergyDensity = (2.0 / 3.0) * gamma_sh_sq * (M_P * NumberDensity::new(n0)) * C_LIGHT.squared();
    let p2 = p2.value();
    let np2 = 4.0 * gamma_sh * n0; // proper number density just behind the shock [cm^-3]
    let mp = M_P.value();

    let mut x1 = Vec::with_capacity(n_cells);
    let mut rho = Vec::with_capacity(n_cells);
    let mut gamma_beta = Vec::with_capacity(n_cells);
    let mut pre = Vec::with_capacity(n_cells);

    for k in 0..n_cells {
        let frac = if n_cells > 1 { k as f64 / (n_cells - 1) as f64 } else { 0.0 };
        // chi from chi_max (k=0, inner) down to 1 (k=n-1, shock) -> radius ascending.
        let chi = chi_max.powf(1.0 - frac);
        let gamma_sq = gamma_sh_sq / (2.0 * chi);
        x1.push(ct * (1.0 - chi / a));
        gamma_beta.push((gamma_sq - 1.0).max(0.0).sqrt());
        pre.push(p2 * chi.powf(-17.0 / 12.0));
        rho.push(np2 * chi.powf(-7.0 / 4.0) * mp);
    }

    BmProfile { x1, rho, gamma_beta, pre, r_shock, gamma_shock: gamma_sh }
}

/// build an image-ready photon catalog for a Blandford-McKee afterglow observed on-axis at
/// observer time `t_obs_day` [day], by integrating the EATS over the blast's time evolution.
///
/// lab times are sampled log-spaced over `[lo_factor, hi_factor] * t_obs_s/(1+z)` (the limb of
/// the image comes from `t ~ t_obs`, the center from much later/decelerated stages, so the span
/// must reach beyond `t_obs`); each lab-time snapshot is a BM profile generated over a synthesized
/// equal-solid-angle cone (sector) of half-angle `theta_sector` about the +z axis (use PI for a
/// full sphere, a smaller angle for a collimated jet OR any angularly-bounded relativistic outflow
/// — a ring/partial shell, not necessarily a canonical jet), weighted by its lab-time bin width.
/// the returned events
/// are reduced into an image by `compute_skymap` for a chosen observer direction (which applies the
/// observer-direction delta^k beaming) — pass an off-axis direction for off-axis jet imaging.
/// `n_mu` x `n_phi` sets angular resolution; `n_radial` the shell resolution.
#[allow(clippy::too_many_arguments)]
pub fn synthesize_afterglow_events(
    e_iso: f64,
    n0: f64,
    p: f64,
    eps_e: f64,
    eps_b: f64,
    redshift: f64,
    d_l: f64,
    t_obs_day: f64,
    theta_sector: f64,
    lo_factor: f64,
    hi_factor: f64,
    n_snapshots: usize,
    chi_max: f64,
    n_radial: usize,
    n_mu: u64,
    n_phi: u64,
    photons_per_dir: u64,
    seed: u64,
    max_events: u64,
) -> Vec<PhotonEvent> {
    let mut events = Vec::new();
    if n_snapshots == 0 {
        return events;
    }
    // the observer time maps to lab times near t_peak = (8 C t_obs)^{1/4}, where C = gamma_sh^2
    // at t = 1 s (front arrival t_obs ~ t/A ~ t^4 / 8C). the bright, strongly-beamed limb of the
    // image comes from t ~ t_peak; sample from a fraction of the (much earlier) limb-arrival time
    // tn up to a multiple of t_peak so the EATS is spanned.
    let tn = t_obs_day * SECONDS_PER_DAY.value() / (1.0 + redshift);
    let c_const = shock_lorentz_factor_sq(e_iso, n0, 1.0);
    let t_peak = (8.0 * c_const * tn).powf(0.25);
    let t_lo = lo_factor * tn;
    let t_hi = hi_factor * t_peak;

    let scales = QuantScales {
        time:     Time::new(1.0),
        pre:      EnergyDensity::new(1.0),
        rho:      MassDensity::new(1.0),
        velocity: Velocity::new(1.0),
        length:   Length::new(1.0),
    };

    for k in 0..n_snapshots {
        if events.len() as u64 >= max_events {
            break;
        }
        let frac = if n_snapshots > 1 { k as f64 / (n_snapshots - 1) as f64 } else { 0.0 };
        let t = t_lo * (t_hi / t_lo).powf(frac); // log-spaced lab time
        // log-spaced lab-time bin width -> emission energy weight for this snapshot.
        let dt = if n_snapshots > 1 {
            t * (t_hi / t_lo).ln() / (n_snapshots as f64 - 1.0)
        } else {
            t
        };

        let prof = bm_profile(e_iso, n0, t, chi_max, n_radial);
        let cond = SimConditions {
            dt,
            theta_obs: 0.0,
            adiabatic_index: 4.0 / 3.0,
            current_time: t,
            p,
            redshift,
            eps_e,
            eps_b,
            d_l: Length::new(d_l),
            nus: vec![],
        };
        let fields = HydroFields { rho: &prof.rho, gamma_beta: &prof.gamma_beta, pre: &prof.pre };
        let remaining = max_events - events.len() as u64;
        let mut ev = generate_photon_events_spherical(
            &cond,
            &scales,
            &fields,
            &prof.x1,
            seed.wrapping_add(k as u64),
            theta_sector,
            n_mu,
            n_phi,
            photons_per_dir,
            remaining,
        );
        events.append(&mut ev);
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    // energy conservation: gamma_sh^2 ~ t^-3, so doubling the lab time drops it by 8x.
    #[test]
    fn shock_lorentz_factor_scales_as_t_minus_three() {
        let g1 = shock_lorentz_factor_sq(1.0e53, 1.0, 1.0e7);
        let g2 = shock_lorentz_factor_sq(1.0e53, 1.0, 2.0e7);
        assert!((g1 / g2 - 8.0).abs() / 8.0 < 1e-12, "gamma_sh^2 should scale as t^-3");
    }

    // post-shock jump: the outermost (shock) cell has gamma = gamma_sh / sqrt(2), and the shock
    // radius is just below ct (the ultrarelativistic blast trails the light front by ~1/gamma^2).
    #[test]
    fn post_shock_jump_conditions() {
        let prof = bm_profile(1.0e53, 1.0, 2.5e7, 100.0, 256);
        let last = prof.x1.len() - 1;
        let gamma_fluid = (prof.gamma_beta[last].powi(2) + 1.0).sqrt();
        let expected = prof.gamma_shock / 2.0_f64.sqrt();
        assert!((gamma_fluid / expected - 1.0).abs() < 1e-3, "post-shock gamma = gamma_sh/sqrt(2)");
        let ct = C_LIGHT.value() * 2.5e7;
        assert!(prof.x1[last] <= ct && prof.x1[last] > 0.99 * ct, "shock radius just below ct");
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

    // the synthetic BM afterglow image is LIMB-BRIGHTENED: integrating the EATS across the
    // blast's time evolution and applying delta^3 beaming, the radial surface-brightness profile
    // peaks off-center with a dimmer center (the canonical Granot-Sari ring). this is the payoff
    // test for the whole imaging path; the profile is printed on failure for calibration.
    #[test]
    fn synthetic_bm_image_is_limb_brightened() {
        let t_obs_day = 1.0;
        let events = synthesize_afterglow_events(
            1.0e53, 1.0, 2.5, 0.1, 0.01, 0.0, 1.0e26, t_obs_day, PI, 0.7, 2.0, 30, 20.0, 8, 32, 64,
            1, 7, 8_000_000,
        );
        assert!(!events.is_empty(), "synthesis produced no events");
        let img = crate::observe::compute_skymap(
            &events, [0.0, 0.0, 1.0], t_obs_day, t_obs_day, 0.0, 1.0e30, 0.0, 3.0, 48,
        );
        let prof = img.radial_profile(10);
        let peak = argmax(&prof);
        assert!(prof.iter().all(|v| v.is_finite()), "non-finite profile: {prof:?}");
        assert!(peak >= 3, "image should be limb-brightened (peak ring {peak}): {prof:?}");
        assert!(prof[0] < prof[peak], "center should be dimmer than the ring: {prof:?}");
    }

    // the self-similar exponents: between two cells, the profile ratios match chi^exponent
    // (p ~ chi^-17/12, rho ~ chi^-7/4) with chi reconstructed from the lorentz factor.
    #[test]
    fn self_similar_exponents() {
        let prof = bm_profile(1.0e53, 1.0, 2.5e7, 100.0, 256);
        let g_sh_sq = prof.gamma_shock.powi(2);
        let chi = |i: usize| g_sh_sq / (2.0 * (prof.gamma_beta[i].powi(2) + 1.0));
        let (i, j) = (prof.x1.len() - 1, prof.x1.len() / 2);
        let ratio = chi(i) / chi(j);
        assert!((prof.pre[i] / prof.pre[j] / ratio.powf(-17.0 / 12.0) - 1.0).abs() < 1e-6);
        assert!((prof.rho[i] / prof.rho[j] / ratio.powf(-7.0 / 4.0) - 1.0).abs() < 1e-6);
        // density / field / lorentz factor all peak at the shock and decline inward.
        assert!(prof.rho[i] > prof.rho[0] && prof.pre[i] > prof.pre[0]);
        assert!(prof.gamma_beta[i] > prof.gamma_beta[0]);
    }
}
