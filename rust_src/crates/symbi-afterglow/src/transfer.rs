// =============================================================================
// transfer.rs
//
// the monte-carlo photon-transfer path (ported from the legacy `rad.cpp`):
//   - `generate_photon_events` samples relativistically-beamed synchrotron photon
//     packets from a hydro snapshot. each packet's COMOVING FREQUENCY is drawn from
//     the cell's broken-power-law synchrotron spectrum, and it carries an equal share
//     of the cell's emitted energy. (this is the proper energy/frequency model: the
//     legacy stored a single `energy` and misused energy/h as a frequency, so the
//     monte-carlo spectrum could not reproduce the analytic one — now it does.)
//   - `monte_carlo_radiative_transfer` propagates packets through the medium with
//     synchrotron self-absorption, thomson scattering, and optional pair production.
//
// cell physics (equipartition field, electron density, breaks, emitted energy) is
// computed with typed `Quantity` values; results are stored into raw-f64 `PhotonEvent`s.
// the one exception is the empirical SSA coefficient, whose calibrated prefactor carries
// implicit units and is therefore computed in raw f64 (documented at its site).
//
// deviations from the legacy (all deliberate; "C++ is reference, not gospel"):
//   - the proper energy/frequency model above (separate nu_emit and energy_weight),
//   - seeded deterministic RNG (src/rng.rs) instead of std::random_device,
//   - correct relativistic-aberration beaming (a rotation, not a magnitude scale),
//   - per-photon absorption path length (0.1 * emission radius), not 0.1 * x1[0],
//   - SSA / pair-production keyed on the photon energy h*nu_emit, not the packet weight.
//
// usage:
//  let mut ev = generate_photon_events(&cond, &scales, &fields, &mesh, seed, 1_000_000, 0);
//  monte_carlo_radiative_transfer(&mut ev, &cond, &scales, &fields, &mesh, seed, true, false);
// =============================================================================

use crate::constants::{C_LIGHT, H_PLANCK, M_P, PI, SIGMA_THOMSON};
use crate::event::PhotonEvent;
use crate::rng::Rng;
use crate::synchrotron::{
    beta, critical_lorentz, delta_doppler, gyration_frequency, lorentz_factor, minimum_lorentz, nu,
    shock_bfield,
};
use crate::units::{Energy, EnergyDensity, Length, MagneticField, NumberDensity};
use crate::{HydroFields, Mesh, QuantScales, SimConditions};

// the comoving emission spectrum is sampled over this many decades on each side of the
// outer/inner spectral break, broad enough to cover the observable bands after boosting.
const SPECTRUM_DECADES: f64 = 4.0;

// small fixed-size vector helpers (3-component lab-frame directions / positions).
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// the lower/upper radial edges of cell `i` as geometric midpoints between neighbors (boundary
/// cells extrapolate the local ratio). works for ANY radial spacing — log-spaced sim meshes or
/// the quasi-linear Blandford-McKee shell alike — so the shell volume is correct either way.
#[inline]
fn radial_shell_edges(x1: &[f64], i: usize) -> (f64, f64) {
    let ni = x1.len();
    let lo = if i > 0 {
        (x1[i - 1] * x1[i]).sqrt()
    } else if ni > 1 {
        x1[0] * (x1[0] / x1[1]).sqrt()
    } else {
        x1[0]
    };
    let hi = if i + 1 < ni {
        (x1[i] * x1[i + 1]).sqrt()
    } else if ni > 1 {
        x1[ni - 1] * (x1[ni - 1] / x1[ni - 2]).sqrt()
    } else {
        x1[0]
    };
    (lo, hi)
}

/// the integral of `nu^a` over [lo, hi] (the a = -1 case is the logarithm).
#[inline]
fn power_integral(a: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        0.0
    } else if (a + 1.0).abs() < 1.0e-9 {
        (hi / lo).ln()
    } else {
        (hi.powf(a + 1.0) - lo.powf(a + 1.0)) / (a + 1.0)
    }
}

/// draw a frequency from `nu^a` on [lo, hi] by inverse-CDF of a uniform deviate `v`.
#[inline]
fn sample_power_segment(a: f64, lo: f64, hi: f64, v: f64) -> f64 {
    if (a + 1.0).abs() < 1.0e-9 {
        lo * (hi / lo).powf(v)
    } else {
        let g = a + 1.0;
        (lo.powf(g) + v * (hi.powf(g) - lo.powf(g))).powf(1.0 / g)
    }
}

/// sample a comoving photon frequency from the broken-power-law synchrotron spectrum
/// F_nu (Sari, Piran & Narayan 1998) on [nu_lo, nu_hi]. the spectrum is three power-law
/// segments split at the breaks nu_m, nu_c; the frequency is drawn proportional to F_nu
/// (energy spectrum) by an EXACT piecewise inverse-CDF: choose a segment with probability
/// equal to its integrated energy, then invert the power-law CDF within it. equal-energy
/// packets drawn this way reproduce F_nu when histogrammed as `sum(weight) / d_nu`.
fn sample_emission_frequency(
    p: f64,
    nu_lo: f64,
    nu_hi: f64,
    nu_c: f64,
    nu_m: f64,
    rng: &mut Rng,
) -> f64 {
    // F_nu ~ nu^a exponents, low -> high frequency (same shape `powerlaw_flux` evaluates).
    let slow_cool = nu_c > nu_m;
    let a_mid = if slow_cool { -0.5 * (p - 1.0) } else { -0.5 };
    let exps = [1.0 / 3.0, a_mid, -0.5 * p];

    // segment bounds: the breaks split [nu_lo, nu_hi] at b1 = lower break, b2 = upper break.
    let b1 = nu_m.min(nu_c).clamp(nu_lo, nu_hi);
    let b2 = nu_m.max(nu_c).clamp(nu_lo, nu_hi);
    let bounds = [nu_lo, b1, b2, nu_hi];

    // amplitudes by continuity across the breaks (A_0 = 1; absolute scale is irrelevant).
    let mut amp = [1.0, 0.0, 0.0];
    amp[1] = amp[0] * bounds[1].powf(exps[0] - exps[1]);
    amp[2] = amp[1] * bounds[2].powf(exps[1] - exps[2]);

    // integrated energy per segment -> segment selection probabilities.
    let w = [
        amp[0] * power_integral(exps[0], bounds[0], bounds[1]),
        amp[1] * power_integral(exps[1], bounds[1], bounds[2]),
        amp[2] * power_integral(exps[2], bounds[2], bounds[3]),
    ];
    let total = w[0] + w[1] + w[2];
    if !(total > 0.0) {
        return (nu_lo * nu_hi).sqrt();
    }

    let mut u = rng.uniform() * total;
    let mut k = 0;
    while k < 2 && u > w[k] {
        u -= w[k];
        k += 1;
    }
    sample_power_segment(exps[k], bounds[k], bounds[k + 1], rng.uniform())
}

/// the lab-frame propagation direction of a photon emitted isotropically along `nprime`
/// (unit vector, fluid frame) by a fluid element moving along `rhat` (unit vector) with
/// speed `beta` (units of c). relativistic aberration changes the angle to `rhat` from
/// acos(mu') to acos((mu'+beta)/(1+beta mu')); the result is that rotation applied in the
/// (rhat, nprime) plane — a proper rotation yielding a UNIT vector. (the legacy scaled
/// `nprime` by cos(rotation), which de-normalizes the direction; this is the fix.)
fn beam_direction(rhat: [f64; 3], nprime: [f64; 3], beta: f64) -> [f64; 3] {
    let mu = dot(rhat, nprime);
    let mu_beam = (mu + beta) / (1.0 + beta * mu);
    // the component of nprime perpendicular to rhat, normalized (the in-plane tangent).
    let perp = [nprime[0] - mu * rhat[0], nprime[1] - mu * rhat[1], nprime[2] - mu * rhat[2]];
    let sin_a = norm(perp);
    if sin_a < 1.0e-12 {
        // emission along the boost axis: aberration leaves it along rhat.
        return rhat;
    }
    let that = [perp[0] / sin_a, perp[1] / sin_a, perp[2] / sin_a];
    let sin_beam = (1.0 - mu_beam * mu_beam).max(0.0).sqrt();
    [
        mu_beam * rhat[0] + sin_beam * that[0],
        mu_beam * rhat[1] + sin_beam * that[1],
        mu_beam * rhat[2] + sin_beam * that[2],
    ]
}

/// per-cell physical state needed to weight, beam, and spectrally sample a photon packet.
pub(crate) struct CellState {
    pub(crate) bfield:    MagneticField,
    pub(crate) n_e:       NumberDensity,
    pub(crate) beta:      f64,
    pub(crate) w:         f64,
    pub(crate) nu_c:      f64,
    pub(crate) nu_m:      f64,
    pub(crate) gamma_min: f64,
}

impl CellState {
    /// build from PHYSICAL cgs quantities: proper mass density `rho` [g/cm^3], pressure `pre`
    /// [erg/cm^3], speed `beta` (units of c), adiabatic index, microphysics, and the emitter-frame
    /// time `t_emitter_s` [s] (sets the cooling break). this is the geometry-agnostic entry shared
    /// by the radial mesh path and the coordinate-agnostic cell path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_physical(
        rho: f64,
        pre: f64,
        beta: f64,
        adiabatic_index: f64,
        eps_e: f64,
        eps_b: f64,
        p: f64,
        t_emitter_s: f64,
    ) -> CellState {
        let w = 1.0 / (1.0 - (beta * beta).min(1.0 - 1e-15)).sqrt();
        let rho_e = EnergyDensity::new(pre / (adiabatic_index - 1.0));
        let bfield = shock_bfield(rho_e, eps_b);
        let n_e = NumberDensity::new(rho / M_P.value());
        let nu_g = gyration_frequency(bfield);
        let gamma_min = minimum_lorentz(eps_e, rho_e, n_e, p);
        let gamma_crit = critical_lorentz(bfield, crate::units::Time::new(t_emitter_s));
        CellState {
            bfield,
            n_e,
            beta,
            w,
            nu_c: nu(gamma_crit, nu_g).value(),
            nu_m: nu(gamma_min, nu_g).value(),
            gamma_min,
        }
    }
}

/// compute the cell state for the radial mesh path: `gamma_beta` is the four-velocity magnitude;
/// `t_prime_s` is the lab snapshot time (the emitter-frame time is t_prime_s / W).
fn cell_state(
    cond: &SimConditions,
    scales: &QuantScales,
    fields: &HydroFields,
    idx: usize,
    p: f64,
    t_prime_s: f64,
) -> CellState {
    let gb = fields.gamma_beta[idx];
    let w = lorentz_factor(gb);
    let rho = (fields.rho[idx] * scales.rho).value();
    let pre = (fields.pre[idx] * scales.pre).value();
    CellState::from_physical(
        rho, pre, beta(gb), cond.adiabatic_index, cond.eps_e, cond.eps_b, p, t_prime_s / w,
    )
}

/// emit `n_packets` equal-weight packets from one cell at lab-frame `position` [cm] whose fluid
/// moves along the unit direction `vhat` with speed `cell.beta`: each isotropic in the fluid
/// frame, aberrated about the VELOCITY direction (not the radius — this is what captures lateral
/// spreading), with a comoving frequency drawn from the cell's broken-power-law spectrum. shared
/// by the radial mesh / spherical generators (vhat = rhat) and the cell path (vhat = velocity dir).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_packets(
    out: &mut Vec<PhotonEvent>,
    rng: &mut Rng,
    cell: &CellState,
    p: f64,
    t_emission: f64,
    position: [f64; 3],
    vhat: [f64; 3],
    packet_weight: f64,
    n_packets: u64,
    max_events: u64,
    cell_id: u32,
) {
    // sample comoving frequencies over a broad band bracketing the spectral breaks.
    let nu_lo = cell.nu_c.min(cell.nu_m) * 10.0_f64.powf(-SPECTRUM_DECADES);
    let nu_hi = cell.nu_c.max(cell.nu_m) * 10.0_f64.powf(SPECTRUM_DECADES);
    let beta_vec = [cell.beta * vhat[0], cell.beta * vhat[1], cell.beta * vhat[2]];

    for _ in 0..n_packets {
        if out.len() as u64 >= max_events {
            break;
        }
        // isotropic emission direction in the fluid frame, aberrated about the velocity to the lab.
        let phi_prime = 2.0 * PI * rng.uniform();
        let mu_prime = 2.0 * rng.uniform() - 1.0;
        let sin_t = (1.0 - mu_prime * mu_prime).max(0.0).sqrt();
        let nprime = [sin_t * phi_prime.cos(), sin_t * phi_prime.sin(), mu_prime];
        let dir = beam_direction(vhat, nprime, cell.beta);
        let nu_emit = sample_emission_frequency(p, nu_lo, nu_hi, cell.nu_c, cell.nu_m, rng);

        out.push(PhotonEvent {
            t_emission,
            x:              position[0],
            y:              position[1],
            z:              position[2],
            nu_emit,
            energy_weight:  packet_weight,
            px:             dir[0],
            py:             dir[1],
            pz:             dir[2],
            stokes_i:       1.0,
            stokes_q:       0.0,
            stokes_u:       0.0,
            stokes_v:       0.0,
            doppler_factor: delta_doppler(cell.w, beta_vec, dir),
            beta_vec,
            optical_depth:  0.0,
            cell_id,
            absorbed:       false,
            n_scatter:      0,
        });
    }
}

/// generate relativistically-beamed synchrotron photon packets from a hydro snapshot.
///
/// each cell radiates a total synchrotron energy over the snapshot timestep; that energy is
/// split into `photons_per_cell` equal-weight packets (or `max(10, max_events / n_cells)` if
/// 0). each packet is emitted isotropically in the fluid frame, aberrated into the lab frame,
/// and assigned a comoving frequency drawn from the cell's broken-power-law spectrum. `seed`
/// makes the catalog reproducible; `max_events` caps total packets. SRHD emission is
/// unpolarized (stokes = [1,0,0,0]); SRMHD polarization is a later increment.
pub fn generate_photon_events(
    cond: &SimConditions,
    scales: &QuantScales,
    fields: &HydroFields,
    mesh: &Mesh,
    seed: u64,
    max_events: u64,
    photons_per_cell: u64,
) -> Vec<PhotonEvent> {
    let mut rng = Rng::seed(seed);
    let mut events = Vec::new();

    let x1 = mesh.x1;
    let x2 = mesh.x2;
    let x3 = mesh.x3;
    let ni = x1.len();
    let nj = x2.len();
    let nk = x3.map_or(1, |a| a.len());
    if ni == 0 || nj == 0 {
        return events;
    }

    let p = cond.p;
    let t_prime = cond.current_time * scales.time;
    let t_prime_s = t_prime.value();
    let dt = cond.dt * scales.time;

    // a 2d (r, theta) sim is AXISYMMETRIC — the physical 3d blast is the slice swept around the
    // jet axis. when phi is NOT a resolved data axis, REVOLVE each (r, theta) cell over this many
    // azimuths to fill the ring (else everything collapses onto the phi=0 plane and the sky image
    // is a flat cross-section, not the 3d blast).
    const AXISYM_N_PHI: usize = 64;
    let resolved_phi = x3.is_some() && mesh.data_dim > 2;
    let n_azimuth = if resolved_phi { nk } else { AXISYM_N_PHI };

    let total_cells = (ni * nj * n_azimuth) as u64;
    let photons_target =
        if photons_per_cell > 0 { photons_per_cell } else { (max_events / total_cells).max(4) };

    // <gamma^2> for a power law N(gamma)~gamma^-p above gamma_min: (p-1)/(p-3) gamma_min^2 for
    // p>3, else gamma_min^2 (the high-gamma integral does not converge, use the lower bound).
    let power_law_factor = if p > 3.0 { (p - 1.0) / (p - 3.0) } else { 1.0 };

    let energy_prefactor = (4.0 / 3.0) * SIGMA_THOMSON * C_LIGHT;

    for ii in 0..ni {
        if events.len() as u64 >= max_events {
            break;
        }
        let r_center = (x1[ii] * scales.length).value();
        let (x1l, x1r) = radial_shell_edges(x1, ii);

        for jj in 0..nj {
            let dx2 = if nj > 1 { x2[1] - x2[0] } else { 2.0 * PI };
            let dcos = if nj > 1 { (x2[jj].cos() - (x2[jj] + dx2).cos()).abs() } else { 2.0 };
            let jreal = if mesh.data_dim > 1 { jj } else { 0 };

            for kk in 0..n_azimuth {
                if events.len() as u64 >= max_events {
                    break;
                }
                // resolved phi cells in 3d, else REVOLVE the axisymmetric data: phi at the
                // azimuth-cell center, dphi = 2pi / n_azimuth (summing over kk recovers 2pi).
                let (sin_phi, cos_phi, dx3) = if resolved_phi {
                    let a = x3.unwrap();
                    let d = if a.len() > 1 { a[1] - a[0] } else { 2.0 * PI };
                    (a[kk].sin(), a[kk].cos(), d)
                } else {
                    let phi = (kk as f64 + 0.5) * 2.0 * PI / n_azimuth as f64;
                    (phi.sin(), phi.cos(), 2.0 * PI / n_azimuth as f64)
                };
                let kreal = if resolved_phi { kk } else { 0 };
                let rhat = [x2[jj].sin() * cos_phi, x2[jj].sin() * sin_phi, x2[jj].cos()];

                let idx = kreal * ni * nj + jreal * ni + ii;
                let cell = cell_state(cond, scales, fields, idx, p, t_prime_s);

                let dvolume = (dx3 * dcos * (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l))
                    * scales.length.cubed();
                let u_b: EnergyDensity = cell.bfield.squared() / (8.0 * PI);
                let gamma_e_sq_avg = cell.gamma_min * cell.gamma_min * power_law_factor;

                // total radiated energy from the cell over dt [erg]; split into packet weights.
                let total_energy: Energy = energy_prefactor
                    * (cell.beta * cell.beta)
                    * u_b
                    * cell.n_e
                    * dvolume
                    * dt
                    * gamma_e_sq_avg;
                let packet_weight = (total_energy / photons_target as f64).value();

                // radial flow: cell at r_center along rhat, velocity also along rhat.
                let position = [r_center * rhat[0], r_center * rhat[1], r_center * rhat[2]];
                emit_packets(
                    &mut events, &mut rng, &cell, p, t_prime_s, position, rhat, packet_weight,
                    photons_target, max_events, idx as u32,
                );
            }
        }
    }

    events
}

/// generate photon packets from a 1D RADIAL profile over a SYNTHESIZED equal-solid-angle
/// sphere — the right tool for imaging a spherically-symmetric (e.g. Blandford-McKee) blast,
/// where the emission sphere must be tessellated independently of the (degenerate) hydro
/// angular mesh. directions are sampled as `mu = cos(theta)` uniform on `[cos(theta_max), 1]`
/// (so every direction cell has equal solid angle) crossed with uniform `phi`; the radial
/// profile (`fields` indexed by the radial cell, length `x1.len()`) is read at each direction.
///
/// `theta_max` is the half-opening angle [rad] (use PI for a full sphere); `n_mu` x `n_phi` is
/// the angular resolution; `photons_per_dir` packets are emitted per (radius, direction) patch.
#[allow(clippy::too_many_arguments)]
pub fn generate_photon_events_spherical(
    cond: &SimConditions,
    scales: &QuantScales,
    fields: &HydroFields,
    x1: &[f64],
    seed: u64,
    theta_max: f64,
    n_mu: u64,
    n_phi: u64,
    photons_per_dir: u64,
    max_events: u64,
) -> Vec<PhotonEvent> {
    let mut rng = Rng::seed(seed);
    let mut events = Vec::new();
    let ni = x1.len();
    if ni == 0 || n_mu == 0 || n_phi == 0 {
        return events;
    }

    let p = cond.p;
    let t_prime = cond.current_time * scales.time;
    let t_prime_s = t_prime.value();
    let dt = cond.dt * scales.time;
    let power_law_factor = if p > 3.0 { (p - 1.0) / (p - 3.0) } else { 1.0 };
    let energy_prefactor = (4.0 / 3.0) * SIGMA_THOMSON * C_LIGHT;

    // equal-solid-angle direction grid: mu = cos(theta) uniform on [cos(theta_max), 1].
    let mu_lo = theta_max.cos();
    let dmu = (1.0 - mu_lo) / n_mu as f64;
    let dphi = 2.0 * PI / n_phi as f64;

    for ii in 0..ni {
        if events.len() as u64 >= max_events {
            break;
        }
        let r_center = (x1[ii] * scales.length).value();
        let (x1l, x1r) = radial_shell_edges(x1, ii);

        let cell = cell_state(cond, scales, fields, ii, p, t_prime_s);
        let u_b: EnergyDensity = cell.bfield.squared() / (8.0 * PI);
        let gamma_e_sq_avg = cell.gamma_min * cell.gamma_min * power_law_factor;

        // patch volume = (radial shell) x (solid angle dmu*dphi); same energy in every direction
        // because the profile is radial, so the packet weight is computed once per radius.
        let dvolume = ((1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) * dmu * dphi)
            * scales.length.cubed();
        let total_energy: Energy = energy_prefactor
            * (cell.beta * cell.beta)
            * u_b
            * cell.n_e
            * dvolume
            * dt
            * gamma_e_sq_avg;
        let packet_weight = (total_energy / photons_per_dir as f64).value();

        for km in 0..n_mu {
            if events.len() as u64 >= max_events {
                break;
            }
            // mu at the cell center (never exactly 1, which would put the strongly-beamed forward
            // material onto the image center as a spike); the azimuth phi is jittered within its
            // cell to break the rigid-grid "spoke" artifacts without disturbing the radial profile.
            let mu = mu_lo + (km as f64 + 0.5) * dmu;
            let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
            for kp in 0..n_phi {
                if events.len() as u64 >= max_events {
                    break;
                }
                let phi = (kp as f64 + rng.uniform()) * dphi;
                let rhat = [sin_theta * phi.cos(), sin_theta * phi.sin(), mu];
                let position = [r_center * rhat[0], r_center * rhat[1], r_center * rhat[2]];
                emit_packets(
                    &mut events, &mut rng, &cell, p, t_prime_s, position, rhat, packet_weight,
                    photons_per_dir, max_events, ii as u32,
                );
            }
        }
    }

    events
}

/// synchrotron self-absorption optical depth over `path_length` (dimensionless). the SSA
/// coefficient alpha ~ n_e B (nu_g/nu)^{(p+4)/2} below the synchrotron peak; the `3.3e-10`
/// prefactor is an empirical GRB-afterglow calibration that absorbs implicit units, so this
/// one product is computed in raw f64 — it sits OUTSIDE the dimensional system by design.
fn ssa_optical_depth(
    photon_energy: Energy,
    n_e: NumberDensity,
    bfield: MagneticField,
    path_length: Length,
    p: f64,
) -> f64 {
    let nu_photon = photon_energy / H_PLANCK; // Frequency
    // cyclotron frequency nu_g = e B / (2 pi m_e c); reuse the synchrotron gyration scale,
    // which is (3/4 pi)(e/m_e c) B, and rescale to the (1/2 pi) cyclotron convention.
    let nu_gyro = gyration_frequency(bfield) * (2.0 / 3.0);
    let nu_ratio = (nu_gyro / nu_photon).value();
    if nu_ratio < 1.0 {
        return 0.0; // above the synchrotron peak, SSA is negligible
    }
    let alpha_ssa = 3.3e-10 * n_e.value() * bfield.value() * nu_ratio.powf((p + 4.0) / 2.0);
    alpha_ssa * path_length.value()
}

/// thomson scattering optical depth over `path_length` (dimensionless): tau = n_e sigma_T L.
/// fully typed — number density x area x length is dimensionless.
fn thomson_optical_depth(n_e: NumberDensity, path_length: Length) -> f64 {
    (n_e * SIGMA_THOMSON * path_length).value()
}

/// isotropically redirect a scattered photon and partially depolarize it.
fn scatter_photon(photon: &mut PhotonEvent, rng: &mut Rng) {
    let phi = 2.0 * PI * rng.uniform();
    let mu = 2.0 * rng.uniform() - 1.0;
    let sin_t = (1.0 - mu * mu).max(0.0).sqrt();
    photon.px = sin_t * phi.cos();
    photon.py = sin_t * phi.sin();
    photon.pz = mu;
    // each scatter reduces linear/circular polarization by a factor 1/e.
    let depol = (-1.0_f64).exp();
    photon.stokes_q *= depol;
    photon.stokes_u *= depol;
    photon.stokes_v *= depol;
    photon.n_scatter += 1;
}

/// propagate `events` through the medium, filling `optical_depth` and updating `absorbed` /
/// `n_scatter` in place. processes: synchrotron self-absorption + thomson scattering (an
/// absorbed photon may instead scatter and survive), and optional gamma-gamma pair production
/// above the ~0.5 MeV threshold. self-absorption and pair production are keyed on the PHOTON
/// energy h*nu_emit (not the packet weight). `seed` makes the transfer reproducible.
pub fn monte_carlo_radiative_transfer(
    events: &mut [PhotonEvent],
    cond: &SimConditions,
    scales: &QuantScales,
    fields: &HydroFields,
    _mesh: &Mesh,
    seed: u64,
    include_scattering: bool,
    include_pair_production: bool,
) {
    let mut rng = Rng::seed(seed);
    let p = cond.p;
    // gamma-gamma -> e+e- threshold ~ 0.5 MeV ~ 8e-7 erg.
    const PAIR_THRESHOLD_ERG: f64 = 8.0e-7;

    for photon in events.iter_mut() {
        let idx = photon.cell_id as usize;
        let rho_e: EnergyDensity = fields.pre[idx] * scales.pre / (cond.adiabatic_index - 1.0);
        let bfield = shock_bfield(rho_e, cond.eps_b);
        let n_e: NumberDensity = fields.rho[idx] * scales.rho / M_P;
        let photon_energy = Energy::new(H_PLANCK.value() * photon.nu_emit);

        // path length ~ 10% of the photon's emission radius (per-photon, not a global scale).
        let path_length = Length::new(0.1 * photon.radius());

        let tau_ssa = ssa_optical_depth(photon_energy, n_e, bfield, path_length, p);
        let tau_thomson = thomson_optical_depth(n_e, path_length);
        let tau_total = tau_ssa + tau_thomson;
        photon.optical_depth = tau_total;

        // monte-carlo absorption test against the survival probability exp(-tau).
        if rng.uniform() > (-tau_total).exp() {
            photon.absorbed = true;
            // an "absorbed" photon may instead have scattered (and thus survive).
            if include_scattering && tau_thomson > 0.0 {
                let scatter_prob = tau_thomson / tau_total;
                if rng.uniform() < scatter_prob {
                    scatter_photon(photon, &mut rng);
                    photon.absorbed = false;
                }
            }
        }

        if include_pair_production && photon_energy.value() > PAIR_THRESHOLD_ERG {
            // simplified: destroy photons above the pair-production threshold. a full treatment
            // would integrate the gamma-gamma opacity against the ambient photon field.
            photon.absorbed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synchrotron::powerlaw_flux;
    use crate::units::{EnergyDensity, Frequency, Length, MassDensity, SpectralPower, Time, Velocity};

    fn conditions() -> SimConditions {
        SimConditions {
            dt:              1.0,
            theta_obs:       0.0,
            adiabatic_index: 4.0 / 3.0,
            current_time:    1.0e6,
            p:               2.5,
            redshift:        0.0,
            eps_e:           0.1,
            eps_b:           0.01,
            d_l:             Length::new(1.0e26),
            nus:             vec![],
        }
    }
    fn scales() -> QuantScales {
        QuantScales {
            time:     Time::new(1.0),
            pre:      EnergyDensity::new(1.0),
            rho:      MassDensity::new(1.0),
            velocity: Velocity::new(1.0),
            length:   Length::new(1.0),
        }
    }

    struct Snap {
        x1:  Vec<f64>,
        x2:  Vec<f64>,
        rho: Vec<f64>,
        gb:  Vec<f64>,
        pre: Vec<f64>,
    }
    fn snapshot() -> Snap {
        let ni = 6;
        let nj = 4;
        let x1: Vec<f64> =
            (0..ni).map(|i| 1.0e16 * 10.0_f64.powf(i as f64 / (ni as f64 - 1.0))).collect();
        let x2: Vec<f64> = (0..nj).map(|j| 0.2 + 0.4 * j as f64).collect();
        Snap { x1, x2, rho: vec![1.0e-24; ni], gb: vec![10.0; ni], pre: vec![1.0e-6; ni] }
    }

    // the frequency sampler reproduces the analytic broken power law: histogramming many
    // equal-weight draws as count/d_nu recovers the canonical Sari slopes in every segment.
    // THIS is the "is the monte-carlo working" test bed — the MC spectrum == the analytic one.
    #[test]
    fn sampled_spectrum_matches_broken_power_law() {
        let p = 2.5;
        let (nu_m, nu_c) = (1.0e10_f64, 1.0e14_f64); // slow cooling
        let nu_lo = 1.0e6_f64;
        let nu_hi = 1.0e18_f64;
        let n = 4_000_000;

        // log-spaced histogram bins; count/d_nu (linear bin width) estimates F_nu.
        let nbins = 48;
        let lg_lo = nu_lo.log10();
        let lg_hi = nu_hi.log10();
        let edges: Vec<f64> = (0..=nbins).map(|i| 10.0_f64.powf(lg_lo + (lg_hi - lg_lo) * i as f64 / nbins as f64)).collect();
        let mut counts = vec![0.0f64; nbins];

        let mut rng = Rng::seed(2024);
        for _ in 0..n {
            let nu = sample_emission_frequency(p, nu_lo, nu_hi, nu_c, nu_m, &mut rng);
            let lg = nu.log10();
            let b = (((lg - lg_lo) / (lg_hi - lg_lo)) * nbins as f64) as usize;
            if b < nbins {
                counts[b] += 1.0;
            }
        }

        // recovered F_nu at each bin center, and the analytic shape there.
        let center = |b: usize| (edges[b] * edges[b + 1]).sqrt();
        let f_mc = |b: usize| counts[b] / (edges[b + 1] - edges[b]);
        let f_an = |nu: f64| powerlaw_flux(SpectralPower::new(1.0), p, Frequency::new(nu), Frequency::new(nu_c), Frequency::new(nu_m)).value();

        // compare recovered vs analytic slope across each spectral segment (two interior bins
        // a decade-ish apart). the slope is regime-independent of the unknown overall scale.
        let slope = |fa: f64, fb: f64, na: f64, nb: f64| (fb / fa).ln() / (nb / na).ln();
        let probe = |nu_a: f64, nu_b: f64, expected: f64| {
            let (ba, bb) = (bin_of(&edges, nu_a), bin_of(&edges, nu_b));
            let s_mc = slope(f_mc(ba), f_mc(bb), center(ba), center(bb));
            let s_an = slope(f_an(center(ba)), f_an(center(bb)), center(ba), center(bb));
            assert!((s_mc - expected).abs() < 0.08, "mc slope {s_mc} vs {expected} (analytic {s_an})");
        };
        probe(1.0e7, 1.0e9, 1.0 / 3.0); // below nu_m
        probe(1.0e11, 1.0e13, -0.5 * (p - 1.0)); // nu_m..nu_c
        probe(1.0e15, 1.0e17, -0.5 * p); // above nu_c
    }

    fn bin_of(edges: &[f64], nu: f64) -> usize {
        for i in 0..edges.len() - 1 {
            if nu >= edges[i] && nu < edges[i + 1] {
                return i;
            }
        }
        edges.len() - 2
    }

    // beaming returns a unit vector at the aberrated angle; beta=0 leaves the direction alone.
    #[test]
    fn beaming_is_a_unit_rotation() {
        let rhat = [0.0, 0.0, 1.0];
        let nprime = {
            let v = [0.3, 0.4, 0.6];
            let n = norm(v);
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let same = beam_direction(rhat, nprime, 0.0);
        for k in 0..3 {
            assert!((same[k] - nprime[k]).abs() < 1e-12);
        }
        let beta = 0.8;
        let dir = beam_direction(rhat, nprime, beta);
        assert!((norm(dir) - 1.0).abs() < 1e-12, "beamed direction must be a unit vector");
        let mu = dot(rhat, nprime);
        let mu_beam = (mu + beta) / (1.0 + beta * mu);
        assert!((dot(rhat, dir) - mu_beam).abs() < 1e-12, "aberrated angle mismatch");
    }

    // generation is reproducible, respects the event cap, emits finite positive weights, and
    // assigns each packet a positive comoving frequency.
    #[test]
    fn generation_is_reproducible_and_bounded() {
        let cond = conditions();
        let sc = scales();
        let s = snapshot();
        let fields = HydroFields { rho: &s.rho, gamma_beta: &s.gb, pre: &s.pre };
        let mesh = Mesh { x1: &s.x1, x2: &s.x2, x3: None, data_dim: 1 };

        let a = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 10_000, 5);
        let b = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 10_000, 5);
        assert_eq!(a, b, "same seed -> identical catalog");
        assert!(!a.is_empty());
        assert!(a.iter().all(|e| e.energy_weight.is_finite() && e.energy_weight > 0.0));
        assert!(a.iter().all(|e| e.nu_emit.is_finite() && e.nu_emit > 0.0), "positive frequencies");
        assert!(a.iter().all(|e| (norm([e.px, e.py, e.pz]) - 1.0).abs() < 1e-9), "unit directions");

        let capped = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 7, 5);
        assert!(capped.len() as u64 <= 7, "max_events respected");
    }

    // transfer is reproducible and only flips photons to absorbed (or scatters them).
    #[test]
    fn transfer_is_reproducible() {
        let cond = conditions();
        let sc = scales();
        let s = snapshot();
        let fields = HydroFields { rho: &s.rho, gamma_beta: &s.gb, pre: &s.pre };
        let mesh = Mesh { x1: &s.x1, x2: &s.x2, x3: None, data_dim: 1 };

        let base = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 10_000, 5);
        let mut e1 = base.clone();
        let mut e2 = base.clone();
        monte_carlo_radiative_transfer(&mut e1, &cond, &sc, &fields, &mesh, 9, true, false);
        monte_carlo_radiative_transfer(&mut e2, &cond, &sc, &fields, &mesh, 9, true, false);
        assert_eq!(e1, e2, "same seed -> identical transfer outcome");
        assert!(e1.iter().all(|e| e.optical_depth >= 0.0), "non-negative optical depth");
    }

    // the spherical generator synthesizes a full sphere from a 1d radial profile with
    // equal-solid-angle sampling: <mu> ~ 0 and both hemispheres are populated (the legacy tied
    // emission directions to the hydro mesh, so a 1d run had no sphere at all).
    #[test]
    fn spherical_generation_fills_the_sphere_uniformly() {
        let cond = conditions();
        let sc = scales();
        let ni = 4;
        let x1: Vec<f64> =
            (0..ni).map(|i| 1.0e16 * 10.0_f64.powf(i as f64 / (ni as f64 - 1.0))).collect();
        let (rho, gb, pre) = (vec![1.0e-24; ni], vec![10.0; ni], vec![1.0e-6; ni]);
        let fields = HydroFields { rho: &rho, gamma_beta: &gb, pre: &pre };

        let a = generate_photon_events_spherical(&cond, &sc, &fields, &x1, 5, PI, 20, 40, 2, 1_000_000);
        let b = generate_photon_events_spherical(&cond, &sc, &fields, &x1, 5, PI, 20, 40, 2, 1_000_000);
        assert_eq!(a, b, "same seed -> identical catalog");
        assert!(!a.is_empty());

        let mus: Vec<f64> = a.iter().map(|e| e.z / e.radius()).collect();
        let mean = mus.iter().sum::<f64>() / mus.len() as f64;
        assert!(mean.abs() < 0.05, "equal-solid-angle sampling -> <mu> ~ 0, got {mean}");
        assert!(mus.iter().any(|&m| m > 0.5) && mus.iter().any(|&m| m < -0.5), "both hemispheres");
    }

    // pair production destroys photons whose ENERGY h*nu_emit is above threshold.
    #[test]
    fn pair_production_absorbs_high_energy() {
        let cond = conditions();
        let sc = scales();
        let s = snapshot();
        let fields = HydroFields { rho: &s.rho, gamma_beta: &s.gb, pre: &s.pre };
        let mesh = Mesh { x1: &s.x1, x2: &s.x2, x3: None, data_dim: 1 };
        // nu_emit chosen so h*nu_emit >> 8e-7 erg threshold.
        let mut ev = vec![PhotonEvent {
            t_emission: 0.0, x: 1.0e16, y: 0.0, z: 0.0,
            nu_emit: 1.0e21, energy_weight: 1.0,
            px: 1.0, py: 0.0, pz: 0.0,
            stokes_i: 1.0, stokes_q: 0.0, stokes_u: 0.0, stokes_v: 0.0,
            doppler_factor: 1.0, beta_vec: [0.0, 0.0, 0.0], optical_depth: 0.0,
            cell_id: 0, absorbed: false, n_scatter: 0,
        }];
        monte_carlo_radiative_transfer(&mut ev, &cond, &sc, &fields, &mesh, 3, false, true);
        assert!(ev[0].absorbed, "super-threshold photon should pair-produce away");
    }
}
