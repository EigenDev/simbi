# =============================================================================
# postprocess.py
#
# post-processing functions for photon events.
# reads HDF5 files, computes observer-dependent quantities.
# pure python/numpy, no C++ dependencies.
#
# functions:
#   - read_photon_events: load events from HDF5
#   - compute_lightcurve: time-binned flux for observer
#   - compute_skymap: spatial flux map at specific time
#   - compute_polarization: polarization evolution
#   - compute_spectrum: spectral flux at specific time
# =============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# =============================================================================
# data structures
# =============================================================================

@dataclass
class photon_events_t:
    """photon event data from HDF5"""
    # spacetime
    t_emission: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # momentum
    energy: np.ndarray
    px: np.ndarray
    py: np.ndarray
    pz: np.ndarray

    # polarization
    stokes_I: np.ndarray
    stokes_Q: np.ndarray
    stokes_U: np.ndarray
    stokes_V: np.ndarray

    # physics
    doppler_factor: np.ndarray
    lorentz_factor: np.ndarray
    optical_depth: np.ndarray

    # metadata
    cell_id: np.ndarray
    absorbed: np.ndarray
    n_scatter: np.ndarray

    @property
    def n_events(self) -> int:
        return len(self.energy)

    def filter(self, mask: np.ndarray) -> 'photon_events_t':
        """return filtered copy"""
        return photon_events_t(
            t_emission=self.t_emission[mask],
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            energy=self.energy[mask],
            px=self.px[mask],
            py=self.py[mask],
            pz=self.pz[mask],
            stokes_I=self.stokes_I[mask],
            stokes_Q=self.stokes_Q[mask],
            stokes_U=self.stokes_U[mask],
            stokes_V=self.stokes_V[mask],
            doppler_factor=self.doppler_factor[mask],
            lorentz_factor=self.lorentz_factor[mask],
            optical_depth=self.optical_depth[mask],
            cell_id=self.cell_id[mask],
            absorbed=self.absorbed[mask],
            n_scatter=self.n_scatter[mask],
        )


@dataclass
class metadata_t:
    """simulation metadata from HDF5"""
    dt: float
    theta_obs: float
    adiabatic_index: float
    current_time: float
    p: float
    z: float
    eps_e: float
    eps_b: float
    d_L: float
    time_scale: float
    pre_scale: float
    rho_scale: float
    v_scale: float
    length_scale: float
    n_events: int
    data_dim: int
    hydro_type: int
    frequencies: np.ndarray


@dataclass
class lightcurve_t:
    """observer lightcurve result"""
    times: np.ndarray          # observer times [day]
    fluxes: Dict[float, np.ndarray]  # flux densities [mJy] per frequency
    frequencies: np.ndarray    # frequencies [Hz]


@dataclass
class skymap_t:
    """sky intensity map"""
    theta: np.ndarray          # polar angles [rad]
    phi: np.ndarray            # azimuthal angles [rad]
    intensity: np.ndarray      # [n_theta, n_phi]
    time: float                # observer time [day]


@dataclass
class polarization_t:
    """polarization evolution"""
    times: np.ndarray                # observer times [day]
    polarization_degree: np.ndarray  # 0 to 1
    polarization_angle: np.ndarray   # radians
    stokes_Q: np.ndarray             # normalized
    stokes_U: np.ndarray             # normalized
    stokes_V: np.ndarray             # normalized


@dataclass
class spectrum_t:
    """spectral flux at specific time"""
    frequencies: np.ndarray    # [Hz]
    fluxes: np.ndarray         # [mJy]
    time: float                # observer time [day]


# =============================================================================
# i/o functions
# =============================================================================

def read_photon_events(filename: str) -> Tuple[photon_events_t, metadata_t]:
    """
    load photon events from HDF5 file.

    returns:
        (events, metadata)
    """
    with h5py.File(filename, 'r') as f:
        events = photon_events_t(
            t_emission=f['t_emission'][:],
            x=f['x'][:],
            y=f['y'][:],
            z=f['z'][:],
            energy=f['energy'][:],
            px=f['px'][:],
            py=f['py'][:],
            pz=f['pz'][:],
            stokes_I=f['stokes_I'][:],
            stokes_Q=f['stokes_Q'][:],
            stokes_U=f['stokes_U'][:],
            stokes_V=f['stokes_V'][:],
            doppler_factor=f['doppler_factor'][:],
            lorentz_factor=f['lorentz_factor'][:],
            optical_depth=f['optical_depth'][:],
            cell_id=f['cell_id'][:],
            absorbed=f['absorbed'][:].astype(bool),
            n_scatter=f['n_scatter'][:],
        )

        meta = metadata_t(
            dt=f.attrs['dt'],
            theta_obs=f.attrs['theta_obs'],
            adiabatic_index=f.attrs['adiabatic_index'],
            current_time=f.attrs['current_time'],
            p=f.attrs['p'],
            z=f.attrs['z'],
            eps_e=f.attrs['eps_e'],
            eps_b=f.attrs['eps_b'],
            d_L=f.attrs['d_L'],
            time_scale=f.attrs['time_scale'],
            pre_scale=f.attrs['pre_scale'],
            rho_scale=f.attrs['rho_scale'],
            v_scale=f.attrs['v_scale'],
            length_scale=f.attrs['length_scale'],
            n_events=f.attrs['n_events'],
            data_dim=0,  # not stored yet
            hydro_type=f.attrs['hydro_type'],
            frequencies=f['frequencies'][:] if 'frequencies' in f else np.array([]),
        )

    return events, meta


# =============================================================================
# observer transformations
# =============================================================================

def compute_observer_time(
    t_emission: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    observer_direction: np.ndarray,
    redshift: float = 0.0
) -> np.ndarray:
    """
    compute observer arrival time from emission coordinates.

    t_obs = (1 + z) * (t_em + r·n_obs / c)

    args:
        t_emission: emission time [s]
        x, y, z: emission position [cm]
        observer_direction: unit vector [nx, ny, nz]
        redshift: cosmological redshift

    returns:
        observer time [day]
    """
    c = 2.99792458e10  # cm/s

    # distance along line of sight
    r_dot_n = x * observer_direction[0] + y * observer_direction[1] + z * observer_direction[2]

    # arrival time
    t_obs = t_emission + r_dot_n / c

    # redshift correction
    t_obs *= (1.0 + redshift)

    # convert to days
    return t_obs / 86400.0


def compute_observed_energy(
    energy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    observer_direction: np.ndarray,
    redshift: float = 0.0
) -> np.ndarray:
    """
    compute observed photon energy including beaming and redshift.

    E_obs = E_em * delta / (1 + z)

    delta correction already in doppler_factor, but we recompute
    from propagation direction for flexibility.

    args:
        energy: emitted energy [erg]
        px, py, pz: photon direction (unit vector)
        observer_direction: unit vector [nx, ny, nz]
        redshift: cosmological redshift

    returns:
        observed energy [erg]
    """
    # beaming factor (should match doppler_factor from generation)
    # for photons propagating at c, this is geometric projection
    cos_theta = px * observer_direction[0] + py * observer_direction[1] + pz * observer_direction[2]

    # observed energy (beaming already encoded in emission, just apply redshift)
    E_obs = energy / (1.0 + redshift)

    return E_obs


# =============================================================================
# lightcurve computation
# =============================================================================

def compute_lightcurve(
    events: photon_events_t,
    meta: metadata_t,
    observer_angle: float,
    frequencies: List[float],
    time_bins: Optional[np.ndarray] = None,
    n_bins: int = 50,
    energy_cut: float = 0.0
) -> lightcurve_t:
    """
    compute observer lightcurve from photon events.

    args:
        events: photon event data
        meta: simulation metadata
        observer_angle: viewing angle [radians]
        frequencies: observed frequencies [Hz]
        time_bins: bin edges [day] (auto if None)
        n_bins: number of time bins (if time_bins=None)
        energy_cut: minimum energy threshold [erg]

    returns:
        lightcurve_t with times and fluxes
    """
    # filter: unabsorbed photons above threshold
    mask = ~events.absorbed & (events.energy > energy_cut)
    filtered = events.filter(mask)

    if filtered.n_events == 0:
        raise ValueError("no events passed filter")

    # observer direction
    observer_dir = np.array([
        np.sin(observer_angle),
        0.0,
        np.cos(observer_angle)
    ])

    # compute observer times
    t_obs = compute_observer_time(
        filtered.t_emission,
        filtered.x,
        filtered.y,
        filtered.z,
        observer_dir,
        meta.z
    )

    # auto time bins
    if time_bins is None:
        t_min, t_max = t_obs.min(), t_obs.max()
        time_bins = np.geomspace(t_min * 0.9, t_max * 1.1, n_bins + 1)

    bin_centers = np.sqrt(time_bins[1:] * time_bins[:-1])

    # compute flux density for each frequency
    # simple energy binning (monochromatic approximation)
    # proper version would convolve with synchrotron spectrum

    h = 6.62607015e-27  # planck constant [erg·s]
    c = 2.99792458e10   # speed of light [cm/s]
    Jy = 1e-23          # jansky [erg/cm^2/s/Hz]

    fluxes = {}

    for nu in frequencies:
        # photon energy corresponding to frequency
        E_nu = h * nu

        # simple binning: count photons near this energy
        # (real implementation should integrate spectrum)
        E_obs = compute_observed_energy(
            filtered.energy,
            filtered.px,
            filtered.py,
            filtered.pz,
            observer_dir,
            meta.z
        )

        # bin photons in time and energy
        flux_density = np.zeros(len(bin_centers))

        for ii in range(len(bin_centers)):
            t_mask = (t_obs >= time_bins[ii]) & (t_obs < time_bins[ii + 1])
            dt = (time_bins[ii + 1] - time_bins[ii]) * 86400.0  # [s]

            if t_mask.sum() > 0:
                # total energy in bin
                E_total = E_obs[t_mask].sum()

                # luminosity distance
                d_L = meta.d_L

                # flux: L / (4π d_L^2 Δt Δν)
                # simplified: assume all energy at frequency nu
                dnu = nu * 0.1  # bandwidth (10% of frequency)
                flux = E_total / (4.0 * np.pi * d_L**2 * dt * dnu)

                # convert to mJy
                flux_density[ii] = flux / Jy * 1e3

        fluxes[nu] = flux_density

    return lightcurve_t(
        times=bin_centers,
        fluxes=fluxes,
        frequencies=np.array(frequencies)
    )


# =============================================================================
# skymap computation
# =============================================================================

def compute_skymap(
    events: photon_events_t,
    meta: metadata_t,
    time: float,
    energy_min: float = 0.0,
    energy_max: float = np.inf,
    n_theta: int = 128,
    n_phi: int = 256,
    time_window: float = 0.1
) -> skymap_t:
    """
    compute sky intensity map at specific observer time.

    args:
        events: photon event data
        meta: simulation metadata
        time: observer time [day]
        energy_min, energy_max: energy range [erg]
        n_theta: polar resolution
        n_phi: azimuthal resolution
        time_window: integration window [day]

    returns:
        skymap_t with intensity map
    """
    # filter: unabsorbed, energy range
    mask = (~events.absorbed &
            (events.energy > energy_min) &
            (events.energy < energy_max))
    filtered = events.filter(mask)

    if filtered.n_events == 0:
        raise ValueError("no events in energy range")

    # compute observer times for all directions
    # this is expensive - loop over sky directions
    theta_grid = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi)
    intensity = np.zeros((n_theta, n_phi))

    for ii, theta in enumerate(theta_grid):
        for jj, phi in enumerate(phi_grid):
            observer_dir = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])

            t_obs = compute_observer_time(
                filtered.t_emission,
                filtered.x,
                filtered.y,
                filtered.z,
                observer_dir,
                meta.z
            )

            # photons arriving in time window
            t_mask = np.abs(t_obs - time) < time_window / 2.0

            if t_mask.sum() > 0:
                # sum intensity (stokes I)
                intensity[ii, jj] = filtered.stokes_I[t_mask].sum()

    return skymap_t(
        theta=theta_grid,
        phi=phi_grid,
        intensity=intensity,
        time=time
    )


# =============================================================================
# polarization computation
# =============================================================================

def compute_polarization(
    events: photon_events_t,
    meta: metadata_t,
    observer_angle: float,
    time_bins: Optional[np.ndarray] = None,
    n_bins: int = 50,
    energy_min: float = 0.0,
    energy_max: float = np.inf
) -> polarization_t:
    """
    compute polarization evolution for observer.

    args:
        events: photon event data
        meta: simulation metadata
        observer_angle: viewing angle [radians]
        time_bins: bin edges [day] (auto if None)
        n_bins: number of time bins
        energy_min, energy_max: energy range [erg]

    returns:
        polarization_t with time series
    """
    # filter: unabsorbed, energy range
    mask = (~events.absorbed &
            (events.energy > energy_min) &
            (events.energy < energy_max))
    filtered = events.filter(mask)

    if filtered.n_events == 0:
        raise ValueError("no events in energy range")

    # observer direction
    observer_dir = np.array([
        np.sin(observer_angle),
        0.0,
        np.cos(observer_angle)
    ])

    # compute observer times
    t_obs = compute_observer_time(
        filtered.t_emission,
        filtered.x,
        filtered.y,
        filtered.z,
        observer_dir,
        meta.z
    )

    # auto time bins
    if time_bins is None:
        t_min, t_max = t_obs.min(), t_obs.max()
        time_bins = np.geomspace(t_min * 0.9, t_max * 1.1, n_bins + 1)

    bin_centers = np.sqrt(time_bins[1:] * time_bins[:-1])

    # compute polarization in each bin
    pol_degree = np.zeros(len(bin_centers))
    pol_angle = np.zeros(len(bin_centers))
    Q_norm = np.zeros(len(bin_centers))
    U_norm = np.zeros(len(bin_centers))
    V_norm = np.zeros(len(bin_centers))

    for ii in range(len(bin_centers)):
        t_mask = (t_obs >= time_bins[ii]) & (t_obs < time_bins[ii + 1])

        if t_mask.sum() > 0:
            # sum stokes parameters
            I_total = filtered.stokes_I[t_mask].sum()
            Q_total = filtered.stokes_Q[t_mask].sum()
            U_total = filtered.stokes_U[t_mask].sum()
            V_total = filtered.stokes_V[t_mask].sum()

            if I_total > 0:
                # normalized stokes parameters
                Q_norm[ii] = Q_total / I_total
                U_norm[ii] = U_total / I_total
                V_norm[ii] = V_total / I_total

                # polarization degree: sqrt(Q^2 + U^2 + V^2) / I
                pol_degree[ii] = np.sqrt(Q_total**2 + U_total**2 + V_total**2) / I_total

                # polarization angle: 0.5 * arctan2(U, Q)
                pol_angle[ii] = 0.5 * np.arctan2(U_total, Q_total)

    return polarization_t(
        times=bin_centers,
        polarization_degree=pol_degree,
        polarization_angle=pol_angle,
        stokes_Q=Q_norm,
        stokes_U=U_norm,
        stokes_V=V_norm
    )


# =============================================================================
# spectrum computation
# =============================================================================

def compute_spectrum(
    events: photon_events_t,
    meta: metadata_t,
    observer_angle: float,
    time: float,
    frequencies: np.ndarray,
    time_window: float = 0.1
) -> spectrum_t:
    """
    compute spectral flux at specific observer time.

    args:
        events: photon event data
        meta: simulation metadata
        observer_angle: viewing angle [radians]
        time: observer time [day]
        frequencies: frequency grid [Hz]
        time_window: integration window [day]

    returns:
        spectrum_t with spectral flux
    """
    # filter: unabsorbed
    mask = ~events.absorbed
    filtered = events.filter(mask)

    if filtered.n_events == 0:
        raise ValueError("no unabsorbed events")

    # observer direction
    observer_dir = np.array([
        np.sin(observer_angle),
        0.0,
        np.cos(observer_angle)
    ])

    # compute observer times
    t_obs = compute_observer_time(
        filtered.t_emission,
        filtered.x,
        filtered.y,
        filtered.z,
        observer_dir,
        meta.z
    )

    # photons in time window
    t_mask = np.abs(t_obs - time) < time_window / 2.0

    if t_mask.sum() == 0:
        raise ValueError(f"no photons at time {time} day")

    # compute observed energies
    E_obs = compute_observed_energy(
        filtered.energy[t_mask],
        filtered.px[t_mask],
        filtered.py[t_mask],
        filtered.pz[t_mask],
        observer_dir,
        meta.z
    )

    # bin photons by energy → frequency
    h = 6.62607015e-27  # erg·s
    nu_obs = E_obs / h

    # histogram into frequency bins
    flux_density = np.zeros(len(frequencies) - 1)

    for ii in range(len(frequencies) - 1):
        nu_mask = (nu_obs >= frequencies[ii]) & (nu_obs < frequencies[ii + 1])

        if nu_mask.sum() > 0:
            E_total = E_obs[nu_mask].sum()
            dnu = frequencies[ii + 1] - frequencies[ii]
            dt = time_window * 86400.0  # s
            d_L = meta.d_L

            # flux: E / (4π d_L^2 Δt Δν)
            flux = E_total / (4.0 * np.pi * d_L**2 * dt * dnu)

            # convert to mJy
            Jy = 1e-23
            flux_density[ii] = flux / Jy * 1e3

    # bin centers
    freq_centers = np.sqrt(frequencies[1:] * frequencies[:-1])

    return spectrum_t(
        frequencies=freq_centers,
        fluxes=flux_density,
        time=time
    )
