// =============================================================================
// lib.rs
//
// the radiation post-processing python module — `rad_hydro`, the rust
// replacement for the legacy C++ module of the same name. unlike the C++ API
// (which made python marshal fields/mesh dicts in), the rust afterglow reads the
// checkpoint itself (symbi_afterglow_io::read_cells), so the whole pipeline is
// self-contained: read_cells -> generate_events_from_cells -> compute_skymap /
// compute_lightcurve. python passes only the checkpoint path(s) + the code->cgs
// unit scales + the synchrotron microphysics.
//
// usage (from python, via simbi.libs.rad_hydro):
//  intensity, n_pix = rad_hydro.skymap(checkpoint, *scales, *micro, theta_obs,
//                                      t_obs, window, n_pix)
//  times, fluxes, freqs = rad_hydro.lightcurve(checkpoints, *scales, *micro,
//                                      theta_obs, freqs, z, d_l, time_bins)
// =============================================================================

use std::path::Path;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use symbi_afterglow::observe::{
    compute_lightcurve_from_events, compute_skymap, DOPPLER_BAND, DOPPLER_BOLOMETRIC,
};
use symbi_afterglow::{generate_events_from_cells, Microphysics};
use symbi_afterglow_io::{read_cells, read_sequence, CgsScales, Synth};

/// a sky image (surface-brightness map) from one checkpoint. returns the flat
/// `n_pix * n_pix` intensity buffer + `n_pix` (reshape to `(n_pix, n_pix)` in
/// python). `theta_obs`/`t_obs`/`window` are the observer angle [rad] / time +
/// window [days]; `bolometric` selects the doppler beaming power.
#[pyfunction]
#[pyo3(signature = (checkpoint, length_scale, density_scale, pressure_scale, time_scale,
                    p, eps_e, eps_b, gamma, dt, theta_obs, t_obs, window, n_pix,
                    photons_per_cell = 4, seed = 1, max_events = 60_000_000, bolometric = false))]
#[allow(clippy::too_many_arguments)]
fn skymap(
    py: Python<'_>,
    checkpoint: String,
    length_scale: f64,
    density_scale: f64,
    pressure_scale: f64,
    time_scale: f64,
    p: f64,
    eps_e: f64,
    eps_b: f64,
    gamma: f64,
    dt: f64,
    theta_obs: f64,
    t_obs: f64,
    window: f64,
    n_pix: usize,
    photons_per_cell: u64,
    seed: u64,
    max_events: u64,
    bolometric: bool,
) -> PyResult<(Vec<f64>, usize)> {
    py.allow_threads(|| -> Result<(Vec<f64>, usize), String> {
        let scales = CgsScales {
            length: length_scale,
            density: density_scale,
            pressure: pressure_scale,
            time: time_scale,
        };
        let cells = read_cells(Path::new(&checkpoint), &scales, &Synth::default())
            .map_err(|e| format!("read_cells({checkpoint}): {e:?}"))?;
        let micro = Microphysics { p, eps_e, eps_b, adiabatic_index: gamma, dt };
        let events = generate_events_from_cells(&cells, &micro, seed, photons_per_cell, max_events);
        let obs = [theta_obs.sin(), 0.0, theta_obs.cos()];
        let doppler = if bolometric { DOPPLER_BOLOMETRIC } else { DOPPLER_BAND };
        let img = compute_skymap(&events, obs, t_obs, window, 0.0, 1.0e30, 0.0, doppler, n_pix);
        Ok((img.intensity, img.n_pix))
    })
    .map_err(PyRuntimeError::new_err)
}

/// the observer light curve F_nu(t) from a TIME SEQUENCE of checkpoints. returns
/// `(times, fluxes, frequencies)` (flat fluxes laid out by `compute_lightcurve`).
#[pyfunction]
#[pyo3(signature = (checkpoints, length_scale, density_scale, pressure_scale, time_scale,
                    p, eps_e, eps_b, gamma, dt, theta_obs, frequencies, redshift,
                    luminosity_distance, time_bins, photons_per_cell = 4, seed = 1,
                    max_events = 60_000_000))]
#[allow(clippy::too_many_arguments)]
fn lightcurve(
    py: Python<'_>,
    checkpoints: Vec<String>,
    length_scale: f64,
    density_scale: f64,
    pressure_scale: f64,
    time_scale: f64,
    p: f64,
    eps_e: f64,
    eps_b: f64,
    gamma: f64,
    dt: f64,
    theta_obs: f64,
    frequencies: Vec<f64>,
    redshift: f64,
    luminosity_distance: f64,
    time_bins: Vec<f64>,
    photons_per_cell: u64,
    seed: u64,
    max_events: u64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    py.allow_threads(|| -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let scales = CgsScales {
            length: length_scale,
            density: density_scale,
            pressure: pressure_scale,
            time: time_scale,
        };
        let paths: Vec<&Path> = checkpoints.iter().map(|s| Path::new(s.as_str())).collect();
        let (cells, _t_max) = read_sequence(&paths, &scales, &Synth::default())
            .map_err(|e| format!("read_sequence: {e:?}"))?;
        let micro = Microphysics { p, eps_e, eps_b, adiabatic_index: gamma, dt };
        let events = generate_events_from_cells(&cells, &micro, seed, photons_per_cell, max_events);
        let obs = [theta_obs.sin(), 0.0, theta_obs.cos()];
        let lc = compute_lightcurve_from_events(
            &events, obs, &frequencies, redshift, luminosity_distance, &time_bins,
        );
        Ok((lc.times, lc.fluxes, lc.frequencies))
    })
    .map_err(PyRuntimeError::new_err)
}

#[pymodule]
fn rad_hydro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skymap, m)?)?;
    m.add_function(wrap_pyfunction!(lightcurve, m)?)?;
    Ok(())
}
