// =============================================================================
// event.rs
//
// the lab-frame photon packet — the data product of the monte-carlo transfer path
// (ported from the legacy `photon_event_t`). a packet carries TWO distinct physical
// quantities that the legacy conflated into one `energy` field:
//   - `nu_emit`       : the comoving (emitter-frame) photon frequency [Hz], sampled
//                       from the cell's synchrotron spectrum — this is what sets the
//                       observed frequency (nu_obs = delta * nu_emit / (1+z)) and the
//                       photon energy (h * nu_emit) for self-absorption / pair tests,
//   - `energy_weight` : the comoving energy [erg] the packet represents (equal per
//                       cell) — this is what accumulates into flux / intensity.
// keeping them separate is what lets the monte-carlo spectrum reproduce the analytic
// broken power law; the old single `energy` field (used BOTH as a weight AND, via
// energy/h, as a frequency) could not.
//
// the fields are raw f64 (CGS): a `Vec<PhotonEvent>` IS the serialization boundary
// — written to disk, reduced into observables, and (eventually) handed to numpy. the
// dimensional `units` system guards the physics that PRODUCES these numbers
// (src/transfer.rs); the events themselves are where the type system is exited.
//
// usage:
//  let events = generate_photon_events(&cond, &scales, &fields, &mesh, seed, ..);
//  let lc = compute_lightcurve_from_events(&events, obs_hat, &nus, z, d_l, &tbins);
// =============================================================================

/// one lab-frame photon packet. polarization is zero for SRHD (no field geometry);
/// SRMHD runs populate the stokes parameters from the magnetic-field geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhotonEvent {
    /// emission time in the lab frame [s].
    pub t_emission: f64,
    /// emission position [cm].
    pub x: f64,
    pub y: f64,
    pub z: f64,

    /// comoving (emitter-frame) photon frequency [Hz], sampled from the cell spectrum.
    pub nu_emit: f64,
    /// comoving energy the packet represents [erg] (the flux/intensity weight).
    pub energy_weight: f64,
    /// propagation direction (unit vector).
    pub px: f64,
    pub py: f64,
    pub pz: f64,

    /// stokes intensity (>= 0).
    pub stokes_i: f64,
    /// stokes linear polarization (0 / 90 deg).
    pub stokes_q: f64,
    /// stokes linear polarization (+/- 45 deg).
    pub stokes_u: f64,
    /// stokes circular polarization.
    pub stokes_v: f64,

    /// doppler boost factor toward the packet's own emission direction (set at generation).
    pub doppler_factor: f64,
    /// the emitting fluid element's lab-frame three-velocity (units of c), as a Cartesian VECTOR
    /// — not assumed radial. this is what lets the image beam toward ANY observer and capture a
    /// laterally-spreading jet/ring: the observer-direction doppler is recomputed from this.
    pub beta_vec: [f64; 3],
    /// optical depth integrated along the path (filled by the transfer step).
    pub optical_depth: f64,

    /// source cell index (row-major `k*ni*nj + j*ni + i`, matching the light curve).
    pub cell_id: u32,
    /// true if absorbed/destroyed during transfer (skipped by the observer reductions).
    pub absorbed: bool,
    /// number of scattering events the packet underwent.
    pub n_scatter: u32,
}

impl PhotonEvent {
    /// the emission radius |r| [cm].
    #[inline]
    pub fn radius(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// the emitting fluid's lorentz factor W = 1/sqrt(1 - beta.beta).
    #[inline]
    pub fn lorentz_factor(&self) -> f64 {
        let b = &self.beta_vec;
        let bsq = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).min(1.0 - 1e-15);
        1.0 / (1.0 - bsq).sqrt()
    }
}
