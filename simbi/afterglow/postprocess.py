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

from .units import C_CGS, DAY_CGS

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

    def filter(self, mask: np.ndarray) -> "photon_events_t":
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
    hydro_type: int
    frequencies: np.ndarray


@dataclass
class lightcurve_t:
    """observer lightcurve result"""

    times: np.ndarray  # observer times [day]
    fluxes: Dict[float, np.ndarray]  # flux densities [mJy] per frequency
    frequencies: np.ndarray  # frequencies [Hz]


@dataclass
class skymap_t:
    """sky intensity map"""

    theta: np.ndarray  # polar angles [rad]
    phi: np.ndarray  # azimuthal angles [rad]
    intensity: np.ndarray  # [n_theta, n_phi]
    time: float  # observer time [day]
    d_L: float = 1e28  # luminosity distance [cm]


@dataclass
class polarization_t:
    """polarization evolution"""

    times: np.ndarray  # observer times [day]
    polarization_degree: np.ndarray  # 0 to 1
    polarization_angle: np.ndarray  # radians
    stokes_Q: np.ndarray  # normalized
    stokes_U: np.ndarray  # normalized
    stokes_V: np.ndarray  # normalized


@dataclass
class spectrum_t:
    """spectral flux at specific time"""

    frequencies: np.ndarray  # [Hz]
    fluxes: np.ndarray  # [mJy]
    time: float  # observer time [day]


# =============================================================================
# i/o functions
# =============================================================================


def read_photon_events(filename: str) -> Tuple[photon_events_t, metadata_t]:
    """
    load photon events from HDF5 file.

    returns:
        (events, metadata)
    """
    with h5py.File(filename, "r") as f:
        events = photon_events_t(
            t_emission=f["t_emission"][:],
            x=f["x"][:],
            y=f["y"][:],
            z=f["z"][:],
            energy=f["energy"][:],
            px=f["px"][:],
            py=f["py"][:],
            pz=f["pz"][:],
            stokes_I=f["stokes_I"][:],
            stokes_Q=f["stokes_Q"][:],
            stokes_U=f["stokes_U"][:],
            stokes_V=f["stokes_V"][:],
            doppler_factor=f["doppler_factor"][:],
            lorentz_factor=f["lorentz_factor"][:],
            optical_depth=f["optical_depth"][:],
            cell_id=f["cell_id"][:],
            absorbed=f["absorbed"][:].astype(bool),
            n_scatter=f["n_scatter"][:],
        )

        meta = metadata_t(
            dt=f.attrs["dt"],
            theta_obs=f.attrs["theta_obs"],
            adiabatic_index=f.attrs["adiabatic_index"],
            current_time=f.attrs["current_time"],
            p=f.attrs["p"],
            z=f.attrs["z"],
            eps_e=f.attrs["eps_e"],
            eps_b=f.attrs["eps_b"],
            d_L=f.attrs["d_L"],
            time_scale=f.attrs["time_scale"],
            pre_scale=f.attrs["pre_scale"],
            rho_scale=f.attrs["rho_scale"],
            v_scale=f.attrs["v_scale"],
            length_scale=f.attrs["length_scale"],
            n_events=f.attrs["n_events"],
            hydro_type=f.attrs["hydro_type"],
            frequencies=f["frequencies"][:]
            if "frequencies" in f
            else np.array([]),
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
    redshift: float = 0.0,
) -> np.ndarray:
    """
    compute observer arrival time from emission coordinates.

    t_obs = (1 + z) * (t_em + r dot n_obs / c)

    args:
        t_emission: emission time [s]
        x, y, z: emission position [cm]
        observer_direction: unit vector [nx, ny, nz]
        redshift: cosmological redshift

    returns:
        observer time [day]
    """
    # distance along line of sight
    r_dot_n = (
        x * observer_direction[0]
        + y * observer_direction[1]
        + z * observer_direction[2]
    )

    # arrival time (photons from far side arrive later)
    t_obs = t_emission - r_dot_n / C_CGS

    # redshift correction
    t_obs *= 1.0 + redshift

    # convert to days
    return t_obs / DAY_CGS


def compute_observed_energy(
    energy: np.ndarray,
    redshift: float = 0.0,
) -> np.ndarray:
    """
    compute observed photon energy with redshift correction.

    E_obs = E_em / (1 + z)

    args:
        energy: emitted energy [erg]
        redshift: cosmological redshift

    returns:
        observed energy [erg]
    """
    return energy / (1.0 + redshift)


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
    energy_cut: float = 0.0,
) -> lightcurve_t:
    """
    compute observer lightcurve from photon events.
    uses fast C++ implementation with zero-copy numpy arrays.

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
    from ..libs import rad_hydro

    # auto time bins from emission times
    if time_bins is None:
        t_min, t_max = events.t_emission.min(), events.t_emission.max()
        time_bins = np.geomspace(t_min * 0.9, t_max * 1.1, n_bins + 1)

    # observer direction
    observer_dir = np.array(
        [np.sin(observer_angle), 0.0, np.cos(observer_angle)], dtype=np.float64
    )

    # ensure contiguous arrays for zero-copy
    freq_arr = np.ascontiguousarray(frequencies, dtype=np.float64)
    time_bins = np.ascontiguousarray(time_bins, dtype=np.float64)
    absorbed = np.ascontiguousarray(events.absorbed, dtype=np.uint8)

    # call fast C++ implementation with numpy arrays directly
    cpp_lc = rad_hydro.compute_lightcurve_from_arrays(
        t_emission=np.ascontiguousarray(events.t_emission, dtype=np.float64),
        x=np.ascontiguousarray(events.x, dtype=np.float64),
        y=np.ascontiguousarray(events.y, dtype=np.float64),
        z=np.ascontiguousarray(events.z, dtype=np.float64),
        energy=np.ascontiguousarray(events.energy, dtype=np.float64),
        px=np.ascontiguousarray(events.px, dtype=np.float64),
        py=np.ascontiguousarray(events.py, dtype=np.float64),
        pz=np.ascontiguousarray(events.pz, dtype=np.float64),
        stokes_I=np.ascontiguousarray(events.stokes_I, dtype=np.float64),
        absorbed=absorbed,
        observer_direction=observer_dir,
        frequencies=freq_arr,
        redshift=float(meta.z),
        luminosity_distance=float(meta.d_L),
        time_bins=time_bins,
    )

    # convert fluxes from flat array to dict by frequency
    n_times = len(time_bins) - 1
    n_freqs = len(frequencies)
    fluxes_flat = np.array(cpp_lc.fluxes)
    fluxes = {}
    for jj, nu in enumerate(frequencies):
        fluxes[nu] = fluxes_flat[jj::n_freqs][:n_times]

    bin_centers = np.sqrt(time_bins[1:] * time_bins[:-1])

    return lightcurve_t(
        times=bin_centers, fluxes=fluxes, frequencies=np.array(frequencies)
    )


# =============================================================================
# skymap computation
# =============================================================================


def compute_skymap(
    events: photon_events_t,
    meta: metadata_t,
    observer_angle: float,
    time: float,
    energy_min: float = 0.0,
    energy_max: float = np.inf,
    n_theta: int = 128,
    n_phi: int = 256,
    time_window: float = 0.1,
    distance_override: Optional[float] = None,
) -> skymap_t:
    """
    compute sky intensity map at specific observer time.
    uses fast C++ implementation with zero-copy numpy arrays.

    for a spherically symmetric blast wave viewed on-axis (observer_angle=0),
    the skymap shows a ring structure due to the equal arrival time surface.

    args:
        events: photon event data (from read_photon_events)
        meta: simulation metadata
        observer_angle: viewing angle [radians] (0 = on-axis)
        time: observer time [day]
        energy_min, energy_max: energy range [erg]
        n_theta: polar resolution (radial bins in sky image)
        n_phi: azimuthal resolution (angular bins around ring)
        time_window: integration window [day]
        distance_override: override luminosity distance [cm] for angular scaling

    returns:
        skymap_t with intensity map
    """
    from ..libs import rad_hydro

    # use override distance if provided
    d_L = distance_override if distance_override is not None else meta.d_L

    # observer direction (unit vector toward observer)
    observer_dir = np.array(
        [np.sin(observer_angle), 0.0, np.cos(observer_angle)], dtype=np.float64
    )

    # ensure contiguous arrays for zero-copy
    absorbed = np.ascontiguousarray(events.absorbed, dtype=np.uint8)

    # call fast C++ implementation with numpy arrays directly
    cpp_skymap = rad_hydro.compute_skymap_from_arrays(
        t_emission=np.ascontiguousarray(events.t_emission, dtype=np.float64),
        x=np.ascontiguousarray(events.x, dtype=np.float64),
        y=np.ascontiguousarray(events.y, dtype=np.float64),
        z=np.ascontiguousarray(events.z, dtype=np.float64),
        energy=np.ascontiguousarray(events.energy, dtype=np.float64),
        stokes_I=np.ascontiguousarray(events.stokes_I, dtype=np.float64),
        absorbed=absorbed,
        observer_direction=observer_dir,
        observer_time=float(time),
        energy_min=float(energy_min),
        energy_max=float(energy_max),
        redshift=float(meta.z),
        luminosity_distance=float(d_L),
        time_window=float(time_window),
        n_theta=int(n_theta),
        n_phi=int(n_phi),
    )

    return skymap_t(
        theta=np.array(cpp_skymap.theta),
        phi=np.array(cpp_skymap.phi),
        intensity=np.array(cpp_skymap.intensity),
        time=time,
        d_L=d_L,
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
    energy_max: float = np.inf,
) -> polarization_t:
    """
    compute polarization evolution for observer.
    uses fast C++ implementation with zero-copy numpy arrays.

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
    from ..libs import rad_hydro

    # auto time bins from emission times
    if time_bins is None:
        t_min, t_max = events.t_emission.min(), events.t_emission.max()
        time_bins = np.geomspace(t_min * 0.9, t_max * 1.1, n_bins + 1)

    # observer direction
    observer_dir = np.array(
        [np.sin(observer_angle), 0.0, np.cos(observer_angle)], dtype=np.float64
    )

    # ensure contiguous arrays for zero-copy
    time_bins = np.ascontiguousarray(time_bins, dtype=np.float64)
    absorbed = np.ascontiguousarray(events.absorbed, dtype=np.uint8)

    # call fast C++ implementation with numpy arrays directly
    cpp_pol = rad_hydro.compute_polarization_from_arrays(
        t_emission=np.ascontiguousarray(events.t_emission, dtype=np.float64),
        x=np.ascontiguousarray(events.x, dtype=np.float64),
        y=np.ascontiguousarray(events.y, dtype=np.float64),
        z=np.ascontiguousarray(events.z, dtype=np.float64),
        energy=np.ascontiguousarray(events.energy, dtype=np.float64),
        px=np.ascontiguousarray(events.px, dtype=np.float64),
        py=np.ascontiguousarray(events.py, dtype=np.float64),
        pz=np.ascontiguousarray(events.pz, dtype=np.float64),
        stokes_I=np.ascontiguousarray(events.stokes_I, dtype=np.float64),
        stokes_Q=np.ascontiguousarray(events.stokes_Q, dtype=np.float64),
        stokes_U=np.ascontiguousarray(events.stokes_U, dtype=np.float64),
        stokes_V=np.ascontiguousarray(events.stokes_V, dtype=np.float64),
        absorbed=absorbed,
        observer_direction=observer_dir,
        time_bins=time_bins,
        energy_min=float(energy_min),
        energy_max=float(energy_max),
    )

    bin_centers = np.sqrt(time_bins[1:] * time_bins[:-1])

    return polarization_t(
        times=bin_centers,
        polarization_degree=np.array(cpp_pol.polarization_degree),
        polarization_angle=np.array(cpp_pol.polarization_angle),
        stokes_Q=np.array(cpp_pol.stokes_Q),
        stokes_U=np.array(cpp_pol.stokes_U),
        stokes_V=np.array(cpp_pol.stokes_V),
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
    time_window: float = 0.1,
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
    observer_dir = np.array(
        [np.sin(observer_angle), 0.0, np.cos(observer_angle)]
    )

    # compute observer times
    t_obs = compute_observer_time(
        filtered.t_emission,
        filtered.x,
        filtered.y,
        filtered.z,
        observer_dir,
        meta.z,
    )

    # photons in time window
    t_mask = np.abs(t_obs - time) < time_window / 2.0

    if t_mask.sum() == 0:
        raise ValueError(f"no photons at time {time} day")

    # compute observed energies
    E_obs = compute_observed_energy(
        filtered.energy[t_mask],
        meta.z,
    )

    # bin photons by energy -> frequency
    h = 6.62607015e-27  # erg*s
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

            # flux: E / (4pi d_L^2 Delta t Deltanu)
            flux = E_total / (4.0 * np.pi * d_L**2 * dt * dnu)

            # convert to mJy
            Jy = 1e-23
            flux_density[ii] = flux / Jy * 1e3

    # bin centers
    freq_centers = np.sqrt(frequencies[1:] * frequencies[:-1])

    return spectrum_t(frequencies=freq_centers, fluxes=flux_density, time=time)
