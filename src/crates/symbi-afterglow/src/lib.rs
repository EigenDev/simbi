// =============================================================================
// symbi-afterglow/src/lib.rs
//
// synchrotron afterglow post-processing for relativistic-blast-wave simulations
// (GRB-like): turn a hydro snapshot into observed light curves / spectra via the
// Sari, Piran & Narayan model + equal-arrival-time (EATS) integration.
//
// the physics is a pure CGS core operating on neutral hydro arrays (`HydroFields`)
// over a spherical `Mesh`, with NO symbi dependency, in two complementary paths:
//   - the deterministic EATS integrator: `synchrotron` (per-cell primitives + the
//     broken-power-law spectrum) feeding `lightcurve` (the equal-arrival-time flux),
//   - the Monte-Carlo photon-transfer path: `transfer` (`generate_photon_events` +
//     `monte_carlo_radiative_transfer` with self-absorption / scattering / pair
//     production) producing a catalog of `event::PhotonEvent`s, reduced by `observe`
//     into light curves, sky maps, and polarization curves for any line of sight.
// dimensional correctness is enforced by the `units` system (compile-time M/L/T).
//
// PENDING: a thin symbi-checkpoint adapter (symbi-io HDF5 -> these inputs) and a
// python frontend; both are sibling adapters above this pure core, never baked in.
//
// all physics is CGS (gauss, Hz, erg, cm, s); `QuantScales` converts the sim's
// code units to CGS, mirroring the legacy `quant_scales_t`.
//
// usage:
//  let flux = symbi_afterglow::light_curve(&cond, &scales, &fields, &mesh, &tbins, ckpt_index);
//  // flux[fidx * (tbins.len()-1) + tidx] is the CGS spectral flux [erg/(s cm^2 Hz)]
//  // (multiply by 1e26 for mJy) summed over the snapshot into observer-time bin `tidx`.
// =============================================================================

pub mod bm;
pub mod constants;
pub mod coords;
pub mod event;
pub mod ingest;
pub mod lightcurve;
pub mod observe;
pub mod rng;
pub mod synchrotron;
pub mod transfer;
pub mod units;

pub use bm::{BmProfile, bm_profile, synthesize_afterglow_events};
pub use coords::Coords;
pub use event::PhotonEvent;
pub use ingest::{Cell, Microphysics, generate_events_from_cells};
pub use lightcurve::light_curve;
pub use observe::{
    DOPPLER_BAND, DOPPLER_BOLOMETRIC, ObserverLightcurve, PolarizationCurve, SkyImage,
    compute_lightcurve_from_events, compute_polarization_from_events, compute_skymap,
};
pub use transfer::{
    generate_photon_events, generate_photon_events_spherical, monte_carlo_radiative_transfer,
};

use units::{EnergyDensity, Frequency, Length, MassDensity, Time, Velocity};

/// the observation + microphysics conditions for one snapshot's flux contribution. angles in
/// radians, distances in cm, frequencies in Hz; `current_time`/`dt` are in CODE units (scaled to
/// seconds via `QuantScales::time`). `redshift` is carried for the phase-3 transfer path; the
/// phase-1 EATS light curve uses the luminosity distance directly (matching the legacy `calc_fnu`).
#[derive(Clone, Debug)]
pub struct SimConditions {
    /// snapshot timestep [code units] — the emitting cell's effective lifetime.
    pub dt: f64,
    /// observer polar angle from the jet axis [rad]. 0 = on-axis (axisymmetric).
    pub theta_obs: f64,
    /// adiabatic index gamma_ad (sets internal energy = pre / (gamma_ad - 1)).
    pub adiabatic_index: f64,
    /// snapshot time [code units].
    pub current_time: f64,
    /// electron power-law index p (N(gamma) ~ gamma^-p).
    pub p: f64,
    /// cosmological redshift (phase-3 path; unused by the phase-1 light curve).
    pub redshift: f64,
    /// fraction of shock energy in the electrons.
    pub eps_e: f64,
    /// fraction of shock energy in the magnetic field.
    pub eps_b: f64,
    /// luminosity distance to the source [cm].
    pub d_l: Length,
    /// observer frequencies to evaluate [Hz].
    pub nus: Vec<Frequency>,
}

/// code-unit -> CGS conversion scales (mirrors the legacy `quant_scales_t`). a quantity in code
/// units is multiplied by its scale to reach CGS: `rho_cgs = rho_code * rho`, etc.
#[derive(Clone, Copy, Debug)]
pub struct QuantScales {
    /// time scale [s per code-time].
    pub time: Time,
    /// pressure / energy-density scale [erg/cm^3 per code-pressure].
    pub pre: EnergyDensity,
    /// mass-density scale [g/cm^3 per code-density].
    pub rho: MassDensity,
    /// velocity scale [cm/s per code-velocity] (carried for completeness; the light curve uses
    /// the dimensionless `gamma_beta` directly).
    pub velocity: Velocity,
    /// length scale [cm per code-length].
    pub length: Length,
}

/// the hydro snapshot fields, flat row-major arrays indexed `k*ni*nj + j*ni + i` (lower-dimensional
/// data broadcasts over the missing axes via `Mesh::data_dim`). `gamma_beta` is the four-velocity
/// magnitude |gamma * beta| (dimensionless); `rho`/`pre` are in CODE units (scaled by `QuantScales`).
#[derive(Clone, Copy, Debug)]
pub struct HydroFields<'a> {
    /// mass density [code units].
    pub rho: &'a [f64],
    /// four-velocity magnitude gamma*beta [dimensionless].
    pub gamma_beta: &'a [f64],
    /// pressure [code units].
    pub pre: &'a [f64],
}

/// the spherical mesh the snapshot lives on: `x1` = radius (log-spaced, code-length), `x2` = polar
/// angle theta [rad], `x3` = azimuth phi [rad] (present only for genuine 3D / off-axis). `data_dim`
/// is the intrinsic dimensionality of the field arrays (1/2/3) — a 1D run broadcasts over theta/phi.
#[derive(Clone, Copy, Debug)]
pub struct Mesh<'a> {
    /// radial cell centers [code-length], log-spaced (ascending).
    pub x1: &'a [f64],
    /// polar-angle cell centers [rad] (ascending).
    pub x2: &'a [f64],
    /// azimuthal cell centers [rad]; `None` for axisymmetric / on-axis runs.
    pub x3: Option<&'a [f64]>,
    /// intrinsic field dimensionality (1, 2, or 3).
    pub data_dim: i64,
}
