// =============================================================================
// transfer.rs
//
// the monte-carlo photon-transfer path:
//   - `generate_photon_events` samples relativistically-beamed synchrotron photon
//     packets from a hydro snapshot. each packet's COMOVING FREQUENCY is drawn from
//     the cell's broken-power-law synchrotron spectrum, and it carries an equal share
//     of the cell's emitted energy. (this is the proper energy/frequency model:
//     storing a single `energy` and misusing energy/h as a frequency would prevent the
//     monte-carlo spectrum from reproducing the analytic one; separate fields do.)
//   - `monte_carlo_radiative_transfer` propagates packets through the medium with
//     synchrotron self-absorption, thomson scattering, and optional pair production.
//
// cell physics (equipartition field, electron density, breaks, emitted energy) is
// computed with typed `Quantity` values; results are stored into raw-f64 `PhotonEvent`s.
// the one exception is the empirical SSA coefficient, whose calibrated prefactor carries
// implicit units and is therefore computed in raw f64 (documented at its site).
//
// design properties:
//   - the proper energy/frequency model above (separate nu_emit and energy_weight),
//   - seeded deterministic RNG (src/rng.rs) for reproducibility,
//   - correct relativistic-aberration beaming (a direction rotation),
//   - per-photon absorption path length (0.1 * emission radius),
//   - SSA / pair-production keyed on the photon energy h*nu_emit, which sets the spectral band.
//
// usage:
//  let mut ev = generate_photon_events(&cond, &scales, &fields, &mesh, seed, 1_000_000, 0);
//  monte_carlo_radiative_transfer(&mut ev, &cond, &scales, &fields, &mesh, seed, true, false);
// =============================================================================

use crate::constants::{H_PLANCK, M_P, PI, SIGMA_THOMSON};
use crate::event::PhotonEvent;
use crate::rng::Rng;
use crate::synchrotron::{
    band_integrated_shape, beta, critical_lorentz, delta_doppler, emissivity, gyration_frequency,
    lorentz_factor, minimum_lorentz, nu, power_integral, shock_bfield, spectral_segments,
};
use crate::units::{Energy, EnergyDensity, Length, MagneticField, NumberDensity, PowerDensity};
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
pub(crate) fn radial_shell_edges(x1: &[f64], i: usize) -> (f64, f64) {
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
    // the segment structure is the SAME decomposition the band-energy integral uses
    // (synchrotron::spectral_segments), so the sampled packet frequencies and the packet
    // energy normalization describe one spectrum by construction.
    let seg = spectral_segments(p, nu_lo, nu_hi, nu_c, nu_m);

    // integrated energy per segment -> segment selection probabilities.
    let w: [f64; 3] = core::array::from_fn(|k| {
        seg.amps[k] * power_integral(seg.exps[k], seg.bounds[k], seg.bounds[k + 1])
    });
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
    sample_power_segment(seg.exps[k], seg.bounds[k], seg.bounds[k + 1], rng.uniform())
}

/// the lab-frame propagation direction of a photon emitted isotropically along `nprime`
/// (unit vector, fluid frame) by a fluid element moving along `rhat` (unit vector) with
/// speed `beta` (units of c). relativistic aberration changes the angle to `rhat` from
/// acos(mu') to acos((mu'+beta)/(1+beta mu')); the result is that rotation applied in the
/// (rhat, nprime) plane — a proper rotation yielding a UNIT vector. (scaling
/// `nprime` by cos(rotation) would de-normalize the direction; the rotation does not.)
fn beam_direction(rhat: [f64; 3], nprime: [f64; 3], beta: f64) -> [f64; 3] {
    let mu = dot(rhat, nprime);
    let mu_beam = (mu + beta) / (1.0 + beta * mu);
    // the component of nprime perpendicular to rhat, normalized (the in-plane tangent).
    let perp = [
        nprime[0] - mu * rhat[0],
        nprime[1] - mu * rhat[1],
        nprime[2] - mu * rhat[2],
    ];
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
    pub(crate) bfield: MagneticField,
    pub(crate) n_e: NumberDensity,
    pub(crate) beta: f64,
    pub(crate) w: f64,
    pub(crate) nu_c: f64,
    pub(crate) nu_m: f64,
}

impl CellState {
    /// the sampled comoving band [nu_lo, nu_hi] [Hz]: SPECTRUM_DECADES decades beyond the
    /// outer/inner spectral break, broad enough to cover the observable bands after boosting.
    /// the frequency sampler and the packet energy normalization both use THIS band, so the
    /// equal-weight packets sum to exactly the band-integrated emitted energy.
    pub(crate) fn band(&self) -> (f64, f64) {
        let lo_break = self.nu_c.min(self.nu_m);
        // an effectively-uncooled cell (emitter time -> 0 gives gamma_c -> inf) has no finite
        // cooling break; the ~nu^{-(p-1)/2} segment then diverges with the band top, so cap
        // the upper break a fixed 2x SPECTRUM_DECADES above the lower one.
        let hi_break = self
            .nu_c
            .max(self.nu_m)
            .min(lo_break * 10.0_f64.powf(2.0 * SPECTRUM_DECADES));
        (
            lo_break * 10.0_f64.powf(-SPECTRUM_DECADES),
            hi_break * 10.0_f64.powf(SPECTRUM_DECADES),
        )
    }

    /// the comoving synchrotron power density radiated in the sampled band [erg/(s cm^3)]:
    /// `emissivity x int spectral_shape dnu` — the SAME per-Hz emissivity the deterministic
    /// deposit integrates, so the monte-carlo catalog and the deposit share one normalization.
    /// (a bolometric (4/3) sigma_T c gamma^2 u_B budget is NOT equivalent: for p < 3 the
    /// gamma^2 average is dominated by the tail up to gamma_c and a gamma_min^2 truncation
    /// underestimates the radiated power by (p-1)/(3-p) (gamma_c/gamma_min)^{3-p}.)
    pub(crate) fn band_power_density(&self, p: f64) -> PowerDensity {
        let (nu_lo, nu_hi) = self.band();
        emissivity(self.bfield, self.n_e, p)
            * band_integrated_shape(p, nu_lo, nu_hi, self.nu_c, self.nu_m)
    }

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
        }
    }
}

/// compute the cell state for the radial mesh path: `gamma_beta` is the four-velocity magnitude;
/// `t_prime_s` is the lab snapshot time (the emitter-frame time is t_prime_s / W).
pub(crate) fn cell_state(
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
        rho,
        pre,
        beta(gb),
        cond.adiabatic_index,
        cond.eps_e,
        cond.eps_b,
        p,
        t_prime_s / w,
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
    // sample comoving frequencies over the SAME band the packet energy was integrated over,
    // so the equal-weight packets tile exactly the band-integrated emitted energy.
    let (nu_lo, nu_hi) = cell.band();
    let beta_vec = [
        cell.beta * vhat[0],
        cell.beta * vhat[1],
        cell.beta * vhat[2],
    ];

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
            x: position[0],
            y: position[1],
            z: position[2],
            nu_emit,
            energy_weight: packet_weight,
            px: dir[0],
            py: dir[1],
            pz: dir[2],
            stokes_i: 1.0,
            stokes_q: 0.0,
            stokes_u: 0.0,
            stokes_v: 0.0,
            doppler_factor: delta_doppler(cell.w, beta_vec, dir),
            beta_vec,
            optical_depth: 0.0,
            cell_id,
            absorbed: false,
            n_scatter: 0,
        });
    }
}

/// generate relativistically-beamed synchrotron photon packets from a hydro snapshot.
///
/// each cell radiates a total synchrotron energy over the snapshot timestep; that energy is
/// split into `photons_per_cell` equal-weight packets (or `max(10, max_events / n_cells)` if
/// 0). each packet is emitted isotropically in the fluid frame, aberrated into the lab frame,
/// and assigned a comoving frequency drawn from the cell's broken-power-law spectrum. `seed`
/// makes the catalog reproducible; `max_events` caps total packets. RHD emission is
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
    // is a flat cross-section).
    const AXISYM_N_PHI: usize = 64;
    let resolved_phi = x3.is_some() && mesh.data_dim > 2;
    let n_azimuth = if resolved_phi { nk } else { AXISYM_N_PHI };

    let total_cells = (ni * nj * n_azimuth) as u64;
    let photons_target = if photons_per_cell > 0 {
        photons_per_cell
    } else {
        (max_events / total_cells).max(4)
    };

    for ii in 0..ni {
        if events.len() as u64 >= max_events {
            break;
        }
        let (x1l, x1r) = radial_shell_edges(x1, ii);

        for jj in 0..nj {
            let dx2 = if nj > 1 { x2[1] - x2[0] } else { 2.0 * PI };
            // mu bounds of the polar cell (mu = cos theta decreases with theta).
            let (mu_hi, mu_lo) = if nj > 1 {
                (x2[jj].cos(), (x2[jj] + dx2).cos())
            } else {
                (1.0, -1.0)
            };
            let dcos = (mu_hi - mu_lo).abs();
            let jreal = if mesh.data_dim > 1 { jj } else { 0 };

            for kk in 0..n_azimuth {
                if events.len() as u64 >= max_events {
                    break;
                }
                // resolved phi cells in 3d, else REVOLVE the axisymmetric data: dphi =
                // 2pi / n_azimuth (summing over kk recovers 2pi).
                let (phi_lo, dx3) = if resolved_phi {
                    let a = x3.unwrap();
                    let d = if a.len() > 1 { a[1] - a[0] } else { 2.0 * PI };
                    (a[kk] - 0.5 * d, d)
                } else {
                    let d = 2.0 * PI / n_azimuth as f64;
                    (kk as f64 * d, d)
                };
                let kreal = if resolved_phi { kk } else { 0 };

                let idx = kreal * ni * nj + jreal * ni + ii;
                let cell = cell_state(cond, scales, fields, idx, p, t_prime_s);

                let dvolume = (dx3 * dcos * (1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l))
                    * scales.length.cubed();

                // total comoving energy radiated in the sampled band over the represented lab
                // interval: (band-integrated SPN98 emissivity) x dV_lab x dt_lab. the proper
                // volume is W dV_lab and the comoving interval is dt_lab / W, so the lorentz
                // factors cancel and the lab pair is exact. split into equal packet weights.
                let total_energy: Energy = cell.band_power_density(p) * dvolume * dt;
                let packet_weight = (total_energy / photons_target as f64).value();

                // each packet's position is sampled CONTINUOUSLY within the cell (volume-weighted
                // radius, uniform mu and phi): cell-centered positions would quantize the EATS
                // arrival time t_em - r.n/c onto the angular grid, biasing any observer window
                // narrower than the lattice spacing. radial flow: velocity along the sampled rhat.
                for _ in 0..photons_target {
                    if events.len() as u64 >= max_events {
                        break;
                    }
                    let mu = mu_lo + rng.uniform() * (mu_hi - mu_lo);
                    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
                    let phi = phi_lo + rng.uniform() * dx3;
                    let u = rng.uniform();
                    let r3 = x1l * x1l * x1l + u * (x1r * x1r * x1r - x1l * x1l * x1l);
                    let r = r3.cbrt() * scales.length.value();
                    let rhat = [sin_theta * phi.cos(), sin_theta * phi.sin(), mu];
                    let position = [r * rhat[0], r * rhat[1], r * rhat[2]];
                    emit_packets(
                        &mut events,
                        &mut rng,
                        &cell,
                        p,
                        t_prime_s,
                        position,
                        rhat,
                        packet_weight,
                        1,
                        max_events,
                        idx as u32,
                    );
                }
            }
        }
    }

    events
}

/// the (n_mu, n_phi) angular tessellation that fits `max_events` over `ni` radial cells at
/// `photons_per_dir` packets per direction — sized so EVERY radial cell emits. a fixed
/// tessellation larger than the budget would make the `max_events` cap truncate the radial
/// loop, silently dropping the outer shells (where the shock lives). packet positions are
/// jittered uniformly within each (mu, phi) cell, so a coarse tessellation still covers the
/// full sphere without bias.
pub fn spherical_tessellation_for_budget(
    ni: usize,
    photons_per_dir: u64,
    max_events: u64,
) -> (u64, u64) {
    let ni = ni.max(1) as u64;
    let ppd = photons_per_dir.max(1);
    let per_cell = (max_events / (ni * ppd)).max(8);
    let n_mu = ((per_cell as f64 / 2.0).sqrt().floor() as u64).max(2);
    let n_phi = (per_cell / n_mu).max(2);
    (n_mu, n_phi)
}

/// generate photon packets from a 1D RADIAL profile over a SYNTHESIZED equal-solid-angle
/// sphere — the right tool for imaging a spherically-symmetric (e.g., Blandford-McKee) blast,
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

    // equal-solid-angle direction grid: mu = cos(theta) uniform on [cos(theta_max), 1].
    let mu_lo = theta_max.cos();
    let dmu = (1.0 - mu_lo) / n_mu as f64;
    let dphi = 2.0 * PI / n_phi as f64;

    for ii in 0..ni {
        if events.len() as u64 >= max_events {
            break;
        }
        let (x1l, x1r) = radial_shell_edges(x1, ii);

        let cell = cell_state(cond, scales, fields, ii, p, t_prime_s);

        // patch volume = (radial shell) x (solid angle dmu*dphi); same energy in every direction
        // because the profile is radial, so the packet weight is computed once per radius. the
        // energy is the band-integrated SPN98 emissivity x dV_lab x dt_lab (proper volume W dV
        // and comoving interval dt/W cancel), the SAME normalization the deposit integrates.
        let dvolume = ((1.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) * dmu * dphi)
            * scales.length.cubed();
        let total_energy: Energy = cell.band_power_density(p) * dvolume * dt;
        let packet_weight = (total_energy / photons_per_dir as f64).value();

        for km in 0..n_mu {
            if events.len() as u64 >= max_events {
                break;
            }
            for kp in 0..n_phi {
                if events.len() as u64 >= max_events {
                    break;
                }
                // every packet's position is sampled CONTINUOUSLY within its (r, mu, phi) cell —
                // volume-weighted in radius, uniform in mu (solid angle) and phi. cell-centered
                // positions would quantize the EATS arrival time t_em - r mu / c onto an n_mu
                // lattice, so an observer window narrower than the lattice spacing catches either
                // a whole mu-ring or nothing: a large flux bias and a hollow image center (the
                // mu -> 1 forward material has no lattice point). continuous sampling makes every
                // arrival window catch a packet share proportional to its width, leaving only
                // shot noise.
                for _ in 0..photons_per_dir {
                    if events.len() as u64 >= max_events {
                        break;
                    }
                    let mu = mu_lo + (km as f64 + rng.uniform()) * dmu;
                    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
                    let phi = (kp as f64 + rng.uniform()) * dphi;
                    let u = rng.uniform();
                    let r3 = x1l * x1l * x1l + u * (x1r * x1r * x1r - x1l * x1l * x1l);
                    let r = r3.cbrt() * scales.length.value();
                    let rhat = [sin_theta * phi.cos(), sin_theta * phi.sin(), mu];
                    let position = [r * rhat[0], r * rhat[1], r * rhat[2]];
                    emit_packets(
                        &mut events,
                        &mut rng,
                        &cell,
                        p,
                        t_prime_s,
                        position,
                        rhat,
                        packet_weight,
                        1,
                        max_events,
                        ii as u32,
                    );
                }
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

        // path length ~ 10% of the photon's emission radius (per-photon).
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
    use crate::constants::C_LIGHT;
    use crate::synchrotron::powerlaw_flux;
    use crate::units::{
        EnergyDensity, Frequency, Length, MassDensity, SpectralPower, Time, Velocity,
    };

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

    struct Snap {
        x1: Vec<f64>,
        x2: Vec<f64>,
        rho: Vec<f64>,
        gb: Vec<f64>,
        pre: Vec<f64>,
    }
    fn snapshot() -> Snap {
        let ni = 6;
        let nj = 4;
        let x1: Vec<f64> = (0..ni)
            .map(|i| 1.0e16 * 10.0_f64.powf(i as f64 / (ni as f64 - 1.0)))
            .collect();
        let x2: Vec<f64> = (0..nj).map(|j| 0.2 + 0.4 * j as f64).collect();
        Snap {
            x1,
            x2,
            rho: vec![1.0e-24; ni],
            gb: vec![10.0; ni],
            pre: vec![1.0e-6; ni],
        }
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
        let edges: Vec<f64> = (0..=nbins)
            .map(|i| 10.0_f64.powf(lg_lo + (lg_hi - lg_lo) * i as f64 / nbins as f64))
            .collect();
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
        let f_an = |nu: f64| {
            powerlaw_flux(
                SpectralPower::new(1.0),
                p,
                Frequency::new(nu),
                Frequency::new(nu_c),
                Frequency::new(nu_m),
            )
            .value()
        };

        // compare recovered vs analytic slope across each spectral segment (two interior bins
        // a decade-ish apart). the slope is regime-independent of the unknown overall scale.
        let slope = |fa: f64, fb: f64, na: f64, nb: f64| (fb / fa).ln() / (nb / na).ln();
        let probe = |nu_a: f64, nu_b: f64, expected: f64| {
            let (ba, bb) = (bin_of(&edges, nu_a), bin_of(&edges, nu_b));
            let s_mc = slope(f_mc(ba), f_mc(bb), center(ba), center(bb));
            let s_an = slope(f_an(center(ba)), f_an(center(bb)), center(ba), center(bb));
            assert!(
                (s_mc - expected).abs() < 0.08,
                "mc slope {s_mc} vs {expected} (analytic {s_an})"
            );
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
        assert!(
            (norm(dir) - 1.0).abs() < 1e-12,
            "beamed direction must be a unit vector"
        );
        let mu = dot(rhat, nprime);
        let mu_beam = (mu + beta) / (1.0 + beta * mu);
        assert!(
            (dot(rhat, dir) - mu_beam).abs() < 1e-12,
            "aberrated angle mismatch"
        );
    }

    // generation is reproducible, respects the event cap, emits finite positive weights, and
    // assigns each packet a positive comoving frequency.
    #[test]
    fn generation_is_reproducible_and_bounded() {
        let cond = conditions();
        let sc = scales();
        let s = snapshot();
        let fields = HydroFields {
            rho: &s.rho,
            gamma_beta: &s.gb,
            pre: &s.pre,
        };
        let mesh = Mesh {
            x1: &s.x1,
            x2: &s.x2,
            x3: None,
            data_dim: 1,
        };

        let a = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 10_000, 5);
        let b = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 10_000, 5);
        assert_eq!(a, b, "same seed -> identical catalog");
        assert!(!a.is_empty());
        assert!(
            a.iter()
                .all(|e| e.energy_weight.is_finite() && e.energy_weight > 0.0)
        );
        assert!(
            a.iter().all(|e| e.nu_emit.is_finite() && e.nu_emit > 0.0),
            "positive frequencies"
        );
        assert!(
            a.iter()
                .all(|e| (norm([e.px, e.py, e.pz]) - 1.0).abs() < 1e-9),
            "unit directions"
        );

        let capped = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 7, 5);
        assert!(capped.len() as u64 <= 7, "max_events respected");
    }

    // transfer is reproducible and only flips photons to absorbed (or scatters them).
    #[test]
    fn transfer_is_reproducible() {
        let cond = conditions();
        let sc = scales();
        let s = snapshot();
        let fields = HydroFields {
            rho: &s.rho,
            gamma_beta: &s.gb,
            pre: &s.pre,
        };
        let mesh = Mesh {
            x1: &s.x1,
            x2: &s.x2,
            x3: None,
            data_dim: 1,
        };

        let base = generate_photon_events(&cond, &sc, &fields, &mesh, 1, 10_000, 5);
        let mut e1 = base.clone();
        let mut e2 = base.clone();
        monte_carlo_radiative_transfer(&mut e1, &cond, &sc, &fields, &mesh, 9, true, false);
        monte_carlo_radiative_transfer(&mut e2, &cond, &sc, &fields, &mesh, 9, true, false);
        assert_eq!(e1, e2, "same seed -> identical transfer outcome");
        assert!(
            e1.iter().all(|e| e.optical_depth >= 0.0),
            "non-negative optical depth"
        );
    }

    // the spherical generator synthesizes a full sphere from a 1d radial profile with
    // equal-solid-angle sampling: <mu> ~ 0 and both hemispheres are populated (tying
    // emission directions to the hydro mesh would leave a 1d run with no sphere at all).
    #[test]
    fn spherical_generation_fills_the_sphere_uniformly() {
        let cond = conditions();
        let sc = scales();
        let ni = 4;
        let x1: Vec<f64> = (0..ni)
            .map(|i| 1.0e16 * 10.0_f64.powf(i as f64 / (ni as f64 - 1.0)))
            .collect();
        let (rho, gb, pre) = (vec![1.0e-24; ni], vec![10.0; ni], vec![1.0e-6; ni]);
        let fields = HydroFields {
            rho: &rho,
            gamma_beta: &gb,
            pre: &pre,
        };

        let a =
            generate_photon_events_spherical(&cond, &sc, &fields, &x1, 5, PI, 20, 40, 2, 1_000_000);
        let b =
            generate_photon_events_spherical(&cond, &sc, &fields, &x1, 5, PI, 20, 40, 2, 1_000_000);
        assert_eq!(a, b, "same seed -> identical catalog");
        assert!(!a.is_empty());

        let mus: Vec<f64> = a.iter().map(|e| e.z / e.radius()).collect();
        let mean = mus.iter().sum::<f64>() / mus.len() as f64;
        assert!(
            mean.abs() < 0.05,
            "equal-solid-angle sampling -> <mu> ~ 0, got {mean}"
        );
        assert!(
            mus.iter().any(|&m| m > 0.5) && mus.iter().any(|&m| m < -0.5),
            "both hemispheres"
        );
    }

    // a budget-sized tessellation reaches the OUTERMOST radial cell: with a tessellation
    // larger than the budget, the max_events cap truncates the ascending radius loop and the
    // outer shells (the shock, in a blast-wave profile) silently never emit — a large flux
    // bias toward the blast interior.
    #[test]
    fn budget_tessellation_covers_all_radii() {
        let cond = conditions();
        let sc = scales();
        let ni = 500;
        let x1: Vec<f64> = (0..ni)
            .map(|i| 1.0e17 * (1.0 + 1.0e-4 * i as f64))
            .collect();
        let (rho, gb, pre) = (vec![1.0e-24; ni], vec![2.0; ni], vec![1.0e-6; ni]);
        let fields = HydroFields {
            rho: &rho,
            gamma_beta: &gb,
            pre: &pre,
        };

        let budget = 50_000_u64;
        let (n_mu, n_phi) = spherical_tessellation_for_budget(ni, 1, budget);
        let ev = generate_photon_events_spherical(
            &cond, &sc, &fields, &x1, 3, PI, n_mu, n_phi, 1, budget,
        );
        assert!(!ev.is_empty());
        let r_max_seen = ev.iter().map(|e| e.radius()).fold(0.0_f64, f64::max);
        let (x1l_last, _) = radial_shell_edges(&x1, ni - 1);
        assert!(
            r_max_seen >= x1l_last,
            "outermost shell never emitted: max packet radius {r_max_seen} < shell edge {x1l_last}"
        );
    }

    // packet positions are sampled CONTINUOUSLY within their (r, mu, phi) cells, so EATS
    // arrival times t_obs = t_em - r.n/c fill their span continuously.
    // cell-centered positions quantize the arrivals into n_mu discrete rings ~span/n_mu
    // apart; an observer window narrower than that spacing then catches either one full ring or
    // nothing — a 200x flux bias and a hollow image. the gate: windows much narrower than the
    // old lattice spacing must each catch a packet share proportional to their width.
    #[test]
    fn spherical_arrival_times_are_continuous() {
        let cond = conditions();
        let sc = scales();
        let ni = 8;
        let x1: Vec<f64> = (0..ni).map(|i| 1.0e17 * (1.0 + 0.001 * i as f64)).collect();
        let (rho, gb, pre) = (vec![1.0e-24; ni], vec![10.0; ni], vec![1.0e-6; ni]);
        let fields = HydroFields {
            rho: &rho,
            gamma_beta: &gb,
            pre: &pre,
        };

        let n_mu = 16;
        let ev = generate_photon_events_spherical(
            &cond, &sc, &fields, &x1, 11, PI, n_mu, 32, 8, 1_000_000,
        );
        assert!(!ev.is_empty());

        // observer along +z: arrival = t_em - z/c; the span is set by the sphere diameter.
        let c = C_LIGHT.value();
        let arrivals: Vec<f64> = ev.iter().map(|e| e.t_emission - e.z / c).collect();
        let lo = arrivals.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = arrivals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = hi - lo;
        assert!(span > 0.0);

        // 8 windows, each 1/8 of the OLD lattice spacing (span/n_mu), spread across the span:
        // continuous sampling puts packets in EVERY one; lattice sampling leaves most empty.
        let w = span / (n_mu as f64 * 8.0);
        for k in 0..8 {
            let center = lo + span * (k as f64 + 0.5) / 8.0;
            let n_in = arrivals
                .iter()
                .filter(|&&t| (t - center).abs() <= 0.5 * w)
                .count();
            assert!(
                n_in > 0,
                "window {k} at {center} (width {w}) caught no packets: arrivals are latticed"
            );
        }

        // the packet radii fill the shells continuously.
        let radii: Vec<f64> = ev.iter().map(|e| e.radius()).collect();
        let distinct = {
            let mut r = radii.clone();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            r.dedup_by(|a, b| (*a - *b).abs() < 1.0e-3);
            r.len()
        };
        assert!(
            distinct > 4 * ni,
            "only {distinct} distinct radii for {ni} shells: radii are latticed"
        );
    }

    // THE UNIFICATION GATE: the monte-carlo packet catalog and the deterministic deposit
    // measure the SAME flux density, because both are normalized by the SPN98 per-Hz
    // emissivity (packets carry the band-integrated energy, importance-sampled in frequency;
    // the deposit integrates emissivity x shape directly). one wide observer-time bin over the
    // full arrival span, one frequency, z = 0: F_mc / F_deposit = 1 up to shot noise. (the two
    // paths previously used unrelated normalizations — a bolometric gamma_min^2 budget vs the
    // spectral emissivity — and disagreed by ~50x.)
    #[test]
    fn monte_carlo_flux_matches_deposit() {
        let cond = conditions();
        let sc = scales();
        let ni = 16;
        let x1: Vec<f64> = (0..ni).map(|i| 1.0e17 * (1.0 + 0.01 * i as f64)).collect();
        let (rho, gb, pre) = (vec![1.0e-24; ni], vec![2.0; ni], vec![1.0e-6; ni]);
        let fields = HydroFields {
            rho: &rho,
            gamma_beta: &gb,
            pre: &pre,
        };

        let nu0 = 1.0e9;
        let d_l = 1.0e26;
        let c = C_LIGHT.value();
        let day = 86_400.0;

        // one observer-time bin covering the whole shell's arrival span [t_em - r/c, t_em + r/c].
        let r_max = x1[ni - 1];
        let t_lo_s = cond.current_time - 1.05 * r_max / c;
        let t_hi_s = cond.current_time + 1.05 * r_max / c;

        // monte-carlo flux [mJy] from the packet catalog.
        let ev =
            generate_photon_events_spherical(&cond, &sc, &fields, &x1, 42, PI, 64, 64, 4, u64::MAX);
        let lc = crate::observe::compute_lightcurve_from_events(
            &ev,
            [0.0, 0.0, 1.0],
            &[nu0],
            0.0,
            d_l,
            &[t_lo_s / day, t_hi_s / day],
            3.0,
            0.2,
        );
        let f_mc = lc.fluxes[0];

        // deposit flux [mJy]: image sum / (4 pi d_L^2 dt_obs), same window and frequency.
        let n_pix = 96;
        let img = crate::deposit::compute_skymap_deposit_spherical(
            &cond,
            &sc,
            &fields,
            &x1,
            [0.0, 0.0, 1.0],
            0.5 * (t_lo_s + t_hi_s),
            0.5 * (t_hi_s - t_lo_s),
            nu0,
            0.0,
            2.0,
            n_pix,
            1.3e17,
            cond.dt,
            PI,
            64,
            128,
        );
        let f_dep = img.iter().sum::<f64>() / (4.0 * PI * d_l * d_l * (t_hi_s - t_lo_s)) * 1.0e26;

        assert!(
            f_mc > 0.0 && f_dep > 0.0,
            "both paths must see flux: mc {f_mc} dep {f_dep}"
        );
        let ratio = f_mc / f_dep;
        assert!(
            (0.8..1.25).contains(&ratio),
            "unified emissivity: F_mc / F_deposit = {ratio} (mc {f_mc} mJy, deposit {f_dep} mJy)"
        );
    }

    // pair production destroys photons whose ENERGY h*nu_emit is above threshold.
    #[test]
    fn pair_production_absorbs_high_energy() {
        let cond = conditions();
        let sc = scales();
        let s = snapshot();
        let fields = HydroFields {
            rho: &s.rho,
            gamma_beta: &s.gb,
            pre: &s.pre,
        };
        let mesh = Mesh {
            x1: &s.x1,
            x2: &s.x2,
            x3: None,
            data_dim: 1,
        };
        // nu_emit chosen so h*nu_emit >> 8e-7 erg threshold.
        let mut ev = vec![PhotonEvent {
            t_emission: 0.0,
            x: 1.0e16,
            y: 0.0,
            z: 0.0,
            nu_emit: 1.0e21,
            energy_weight: 1.0,
            px: 1.0,
            py: 0.0,
            pz: 0.0,
            stokes_i: 1.0,
            stokes_q: 0.0,
            stokes_u: 0.0,
            stokes_v: 0.0,
            doppler_factor: 1.0,
            beta_vec: [0.0, 0.0, 0.0],
            optical_depth: 0.0,
            cell_id: 0,
            absorbed: false,
            n_scatter: 0,
        }];
        monte_carlo_radiative_transfer(&mut ev, &cond, &sc, &fields, &mesh, 3, false, true);
        assert!(
            ev[0].absorbed,
            "super-threshold photon should pair-produce away"
        );
    }
}
