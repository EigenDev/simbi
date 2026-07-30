// =============================================================================
// lightcurve.rs
//
// the equal-arrival-time-surface (EATS) synchrotron light curve — the workhorse
// afterglow observable. each cell of a spherical
// hydro snapshot emits a doppler-boosted broken-power-law synchrotron spectrum; the
// emission arrives at the observer at t_obs = t' - r (rhat . obs_hat) / c, so a single
// snapshot smears across a range of observer times. the flux of every (frequency, cell)
// is binned into the observer-time bins and accumulated across snapshots.
//
// dimensionful intermediates are typed `Quantity` values (src/units.rs): the code-unit
// hydro/mesh numbers are bare f64 and become CGS quantities only when multiplied by their
// `QuantScales`, so the unit conversions themselves are dimension-checked. CGS throughout;
// output is the spectral flux density [erg/(s cm^2 Hz)] (x1e26 for mJy), laid out
// `flux[fidx * (n_tbins - 1) + tidx]`.
//
// usage:
//  let flux = light_curve(&cond, &scales, &fields, &mesh, &tbin_edges, checkpoint_index);
// =============================================================================

use crate::constants::{C_LIGHT, M_P, PI, SECONDS_PER_DAY};
use crate::synchrotron::{
    beta, critical_lorentz, delta_doppler, emissivity, gyration_frequency, lorentz_factor,
    minimum_lorentz, nu, powerlaw_flux, shock_bfield,
};
use crate::units::{EnergyDensity, Frequency, NumberDensity, Time, Volume};
use crate::{HydroFields, Mesh, QuantScales, SimConditions};

/// accumulate one snapshot's EATS synchrotron flux into the observer-time / frequency bins.
///
/// `tbin_edges` are observer-time bin edges [day] (`n_tbins` edges -> `n_tbins - 1` bins); the
/// returned array is `cond.nus.len() * (n_tbins - 1)` long, indexed `fidx * (n_tbins - 1) + tidx`.
/// `checkpoint_index > 0` weights each cell by its effective lifetime `dt_obs_of_cell / bin_width`
/// (so a sequence of snapshots integrates to the true light curve); the first snapshot uses 1.0.
pub fn light_curve(
    cond: &SimConditions,
    scales: &QuantScales,
    fields: &HydroFields,
    mesh: &Mesh,
    tbin_edges: &[f64],
    checkpoint_index: i64,
) -> Vec<f64> {
    // observer direction in the (sin theta_obs, 0, cos theta_obs) plane.
    let obs_hat = [cond.theta_obs.sin(), 0.0, cond.theta_obs.cos()];

    let nt = tbin_edges.len();
    let nf = cond.nus.len();
    let mut flux = vec![0.0f64; nf * nt.saturating_sub(1)];
    if nt < 2 || nf == 0 {
        return flux;
    }

    let x1 = mesh.x1;
    let x2 = mesh.x2;
    let ni = x1.len();
    let nj = x2.len();
    let (x1min, x1max) = (x1[0], x1[ni - 1]);
    let dlogx1 = (x1max / x1min).log10() / (ni as f64 - 1.0);
    let (x2min, x2max) = (x2[0], x2[nj - 1]);
    let dx2 = (x2max - x2min) / (nj as f64 - 1.0).max(1.0);

    // on-axis observer (|cos theta_obs| == 1) is axisymmetric: one phi cell spanning 2 pi.
    let at_pole = cond.theta_obs.cos().abs() == 1.0;
    let (nk, x3) = if at_pole {
        (1usize, None)
    } else {
        (mesh.x3.map_or(1, |a| a.len()), mesh.x3)
    };
    let (x3min, x3max) = match x3 {
        Some(a) if a.len() > 1 => (a[0], a[a.len() - 1]),
        _ => (0.0, 0.0),
    };

    let p = cond.p;
    let d = cond.d_l; // luminosity distance [cm]
    let t_prime: Time = cond.current_time * scales.time; // snapshot time [s]
    let dt: Time = cond.dt * scales.time; // timestep [s]
    let flux_denom = 1.0 / (4.0 * PI * d.squared()); // 1 / (4 pi d^2), inverse area
    let length3: Volume = scales.length.cubed();

    for kk in 0..nk {
        let (sin_phi, cos_phi, dx3);
        if at_pole {
            sin_phi = 0.0;
            cos_phi = 1.0;
            dx3 = 2.0 * PI;
        } else {
            let a3 = x3.expect("off-axis run requires an x3 (phi) mesh");
            let dphi = (x3max - x3min) / (a3.len() as f64 - 1.0).max(1.0);
            // half-cell at the phi boundaries, full cell interior.
            let x3l = if kk > 0 {
                x3min + (kk as f64 - 0.5) * dphi
            } else {
                x3min
            };
            let x3r = if kk < nk - 1 {
                x3l + dphi * if kk == 0 { 0.5 } else { 1.0 }
            } else {
                x3max
            };
            sin_phi = a3[kk].sin();
            cos_phi = a3[kk].cos();
            dx3 = x3r - x3l;
        }

        // a 3d field has a real k index; lower-dim data broadcasts (kreal stays 0).
        let kreal = if mesh.data_dim > 2 { kk } else { 0 };

        for jj in 0..nj {
            let x2l = if jj > 0 {
                x2min + (jj as f64 - 0.5) * dx2
            } else {
                x2min
            };
            let x2r = if jj < nj - 1 {
                x2l + dx2 * if jj == 0 { 0.5 } else { 1.0 }
            } else {
                x2max
            };
            let dcos = x2l.cos() - x2r.cos();

            // radial unit vector for this (theta, phi).
            let rhat = [x2[jj].sin() * cos_phi, x2[jj].sin() * sin_phi, x2[jj].cos()];
            let jreal = if mesh.data_dim > 1 { jj } else { 0 };

            for ii in 0..ni {
                let idx = kreal * ni * nj + jreal * ni + ii;
                let gb = fields.gamma_beta[idx];
                let bb = beta(gb);
                let w = lorentz_factor(gb);
                let t_emitter: Time = t_prime / w;

                // code-unit pressure/density -> CGS energy density / number density (dimension
                // checked: a code number times its scale yields the typed CGS quantity).
                let rho_e: EnergyDensity =
                    fields.pre[idx] * scales.pre / (cond.adiabatic_index - 1.0);
                let bfield = shock_bfield(rho_e, cond.eps_b);
                let n_e: NumberDensity = fields.rho[idx] * scales.rho / M_P;
                let nu_g = gyration_frequency(bfield);
                let gamma_min = minimum_lorentz(cond.eps_e, rho_e, n_e, p);
                let gamma_crit = critical_lorentz(bfield, t_emitter);

                // log-spaced radial cell edges -> shell volume (cm^3): the code-length edges are
                // bare f64; multiplying by `length3` (cm^3 per code-length^3) yields a Volume.
                let x1l = if ii > 0 {
                    x1min * 10.0_f64.powf((ii as f64 - 0.5) * dlogx1)
                } else {
                    x1min
                };
                let x1r = if ii < ni - 1 {
                    x1l * 10.0_f64.powf(dlogx1 * if ii == 0 { 0.5 } else { 1.0 })
                } else {
                    x1max
                };
                let dvolume: Volume =
                    (dx3 * dcos * (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l)) * length3;

                // EATS: arrival time at the observer for emission at this radius. x1 (code length)
                // times the length scale is a cm radius; divided by c it is a light-travel Time.
                let r_proj = rhat[0] * obs_hat[0] + rhat[1] * obs_hat[1] + rhat[2] * obs_hat[2];
                let t_obs: Time = t_prime - (x1[ii] * scales.length) * r_proj / C_LIGHT;
                let t_obs_day = (t_obs / SECONDS_PER_DAY).value();

                let beta_vec = [bb * rhat[0], bb * rhat[1], bb * rhat[2]];
                let nu_c: Frequency = nu(gamma_crit, nu_g);
                let nu_m: Frequency = nu(gamma_min, nu_g);
                let delta = delta_doppler(w, beta_vec, obs_hat);
                let eps_m = emissivity(bfield, n_e, p);

                // peak emitted power per unit frequency from this cell (doppler-boosted, delta^2).
                let power_prime = dvolume * eps_m * (delta * delta);

                for (fidx, &nu_obs) in cond.nus.iter().enumerate() {
                    // the observed frequency maps to a higher emitter-frame frequency by 1/delta.
                    let nu_source: Frequency = nu_obs / delta;
                    let power_cool = powerlaw_flux(power_prime, p, nu_source, nu_c, nu_m);
                    let f_nu = (power_cool * flux_denom).value();

                    for tidx in 0..nt - 1 {
                        let (t1, t2) = (tbin_edges[tidx], tbin_edges[tidx + 1]);
                        if t1 < t_obs_day && t_obs_day < t2 {
                            // weight by the cell's effective observed lifetime / bin width once a
                            // sequence of snapshots is being integrated (checkpoint_index > 0).
                            let dt_day = (dt / SECONDS_PER_DAY).value();
                            let trat = if checkpoint_index > 0 {
                                dt_day / (t2 - t1)
                            } else {
                                1.0
                            };
                            flux[fidx * (nt - 1) + tidx] += trat * f_nu;
                            break;
                        }
                    }
                }
            }
        }
    }

    flux
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{EnergyDensity, Frequency, Length, MassDensity, Time, Velocity};

    // a minimal on-axis uniform radial shell fixture: log-spaced radii, gamma*beta=10, ~1 p/cc.
    struct Case {
        cond: SimConditions,
        scales: QuantScales,
        x1: Vec<f64>,
        x2: Vec<f64>,
        rho: Vec<f64>,
        gb: Vec<f64>,
        pre: Vec<f64>,
    }

    fn uniform_shell() -> Case {
        let ni = 8;
        // log-spaced radii ~1e16..1e17 cm (length scale 1).
        let x1: Vec<f64> = (0..ni)
            .map(|i| 1.0e16 * 10.0_f64.powf(i as f64 / (ni as f64 - 1.0)))
            .collect();
        Case {
            cond: SimConditions {
                dt: 1.0,
                theta_obs: 0.0,
                adiabatic_index: 4.0 / 3.0,
                current_time: 1.0e6,
                p: 2.5,
                redshift: 0.0,
                eps_e: 0.1,
                eps_b: 0.01,
                d_l: Length::new(1.0e26),
                nus: vec![Frequency::new(1.0e9), Frequency::new(1.0e15)], // radio + X-ray-ish
            },
            scales: QuantScales {
                time: Time::new(1.0),
                pre: EnergyDensity::new(1.0),
                rho: MassDensity::new(1.0),
                velocity: Velocity::new(1.0),
                length: Length::new(1.0),
            },
            x1,
            // several theta cells (1D radial data broadcasts over them); a single cell would have
            // zero solid angle (dcos = cos(theta) - cos(theta) = 0) and emit nothing.
            x2: vec![0.2, 0.4, 0.6, 0.8],
            rho: vec![1.0e-24; ni], // ~1 proton/cc in g/cm^3
            gb: vec![10.0; ni],     // gamma*beta = 10
            pre: vec![1.0e-6; ni],
        }
    }

    // the integrator runs, lands flux in observer-time bins, and the flux is finite + positive.
    // exercises the full per-cell pipeline (equipartition B, electron breaks, EATS arrival time,
    // broken-power-law, binning).
    #[test]
    fn light_curve_produces_finite_positive_flux() {
        let c = uniform_shell();
        let fields = HydroFields {
            rho: &c.rho,
            gamma_beta: &c.gb,
            pre: &c.pre,
        };
        let mesh = Mesh {
            x1: &c.x1,
            x2: &c.x2,
            x3: None,
            data_dim: 1,
        };
        let tbins: Vec<f64> = (0..=20).map(|i| i as f64 * 2.0).collect();

        let flux = light_curve(&c.cond, &c.scales, &fields, &mesh, &tbins, 0);
        assert_eq!(flux.len(), c.cond.nus.len() * (tbins.len() - 1));
        assert!(flux.iter().all(|f| f.is_finite()), "all fluxes finite");
        assert!(
            flux.iter().any(|&f| f > 0.0),
            "some bin received positive flux"
        );
    }

    // for this slow-cooling setup the X-ray band is on a steeper segment than radio, so its
    // integrated flux should not exceed the radio band's — a sanity check on spectral ordering.
    #[test]
    fn higher_frequency_is_not_brighter() {
        let c = uniform_shell();
        let fields = HydroFields {
            rho: &c.rho,
            gamma_beta: &c.gb,
            pre: &c.pre,
        };
        let mesh = Mesh {
            x1: &c.x1,
            x2: &c.x2,
            x3: None,
            data_dim: 1,
        };
        let tbins: Vec<f64> = (0..=20).map(|i| i as f64 * 2.0).collect();
        let flux = light_curve(&c.cond, &c.scales, &fields, &mesh, &tbins, 0);
        let nb = tbins.len() - 1;
        let radio: f64 = (0..nb).map(|t| flux[t]).sum();
        let xray: f64 = (0..nb).map(|t| flux[nb + t]).sum();
        assert!(
            radio >= xray,
            "radio {radio} should be >= x-ray {xray} (slow cooling)"
        );
    }
}
