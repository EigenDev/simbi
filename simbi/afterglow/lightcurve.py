# =============================================================================
# lightcurve.py
#
# observer light curve F_nu(t), STREAMED over checkpoints via the cpu_ext catalog
# path. each checkpoint -> in-process synchrotron catalog -> EATS reduce into the
# fixed time bins -> accumulate -> DISCARD the events. memory is O(time_bins x freqs),
# so it scales to many checkpoints (fine time binning) without
# holding every photon. this is the SINGLE afterglow path (replaces symbi-rad-py's
# self-contained read_cells/rad_hydro.lightcurve), so it inherits the lab-radius,
# 2d-revolve, and units fixes.
# usage:
#  times, fluxes_flat, freqs = stream_lightcurve(files, qscales, micro, theta_obs,
#                                                freqs, z, d_l, time_edges)
# =============================================================================

from typing import Any, Sequence

import numpy as np

_C_CGS = 2.997924580e10


def _is_event_catalog(path: str) -> bool:
    """true if `path` is a photon-event catalog (vs a hydro checkpoint)."""
    import h5py

    with h5py.File(path, "r") as f:
        return "n_events" in f.attrs or "t_emission" in f


def stream_lightcurve(
    checkpoints: Sequence[str],
    qscales: dict[str, float],
    micro: dict[str, float],
    theta_obs: float,
    frequencies: Sequence[float],
    redshift: float,
    luminosity_distance: float,
    time_edges: Sequence[float],
    max_events: int = 2_000_000,
    chunk_size: int = 4_000_000,
):
    """accumulate the EATS light curve over `checkpoints`, one at a time. the EATS reduction
    is ADDITIVE over event subsets, so a saved catalog is read in row-CHUNKS (O(chunk_size)
    memory, not O(file)) — a huge generate-once events file reduces without being held whole.
    returns (times [day], fluxes flat [n_time*n_freq] mJy, frequencies [Hz])."""
    from simbi.libs import cpu_ext
    from simbi.reader import read_simulation

    from .inputs import build_fields, build_mesh

    from .generate import _read_snapshot_time, _snapshot_emission_durations

    nhat = [float(np.sin(theta_obs)), 0.0, float(np.cos(theta_obs))]
    edges = [float(t) for t in time_edges]
    freqs = [float(f) for f in frequencies]

    # each hydro checkpoint emits over the lab-time interval it REPRESENTS (its trapezoidal
    # share of the snapshot-time axis) — weighting by the CFL dt would
    # undercount the emitted energy by (checkpoint cadence) / (CFL dt), typically ~1e5.
    hydro_paths = sorted(
        (p for p in checkpoints if not _is_event_catalog(p)), key=_read_snapshot_time
    )
    durations = dict(
        zip(
            hydro_paths,
            _snapshot_emission_durations([_read_snapshot_time(p) for p in hydro_paths]),
        )
    )

    times: Any = None
    total: Any = None

    def _reduce(catalog):
        nonlocal times, total
        t, fl, _ = cpu_ext.lightcurve_from_events(
            catalog, nhat, freqs, redshift, luminosity_distance, edges
        )
        times = np.asarray(t)
        total = np.asarray(fl) if total is None else total + np.asarray(fl)

    for path in checkpoints:
        # accept EITHER an events catalog (read it back) or a hydro checkpoint (generate in
        # place) — the events file is the canonical product, but a checkpoint works directly too.
        if _is_event_catalog(path):
            n = cpu_ext.photon_event_count(path)
            for start in range(0, max(n, 1), chunk_size):
                catalog = cpu_ext.read_photon_events_chunk(path, start, chunk_size)
                if len(catalog) == 0:
                    break
                _reduce(catalog)
                del catalog  # release this chunk before the next
        else:
            data = read_simulation(path)
            emit_dt = durations.get(path, 0.0)
            sim_cond = {
                "dt": emit_dt if emit_dt > 0.0 else data.metadata.dt,
                "theta_obs": theta_obs,
                "adiabatic_index": data.metadata.gamma,
                "current_time": data.metadata.time,
                "p": micro["p"],
                "z": redshift,
                "eps_e": micro["eps_e"],
                "eps_b": micro["eps_b"],
                "d_L": luminosity_distance,
                "nus": freqs,
            }
            catalog = cpu_ext.generate_photon_events(
                sim_cond=sim_cond,
                qscales=qscales,
                fields=build_fields(data),
                mesh=build_mesh(data),
                max_events=max_events,
                photons_per_cell=0,
            )
            _reduce(catalog)
            del catalog  # release this checkpoint's events before the next one

    return times, total, np.asarray(freqs)


def afterglow_lightcurve(
    checkpoints,
    length_scale,
    density_scale,
    pressure_scale,
    time_scale,
    p,
    eps_e,
    eps_b,
    gamma,
    dt,
    theta_obs,
    frequencies,
    redshift,
    luminosity_distance,
    time_bins,
):
    """backwards-compatible shim with the old `rad_hydro.lightcurve` signature, now backed by
    the streaming cpu_ext path. returns (times, fluxes_flat, frequencies) as lists."""
    qscales = {
        "length": float(length_scale),
        "rho": float(density_scale),
        "pre": float(pressure_scale),
        "time": float(time_scale),
        "velocity": _C_CGS,
    }
    micro = {"p": float(p), "eps_e": float(eps_e), "eps_b": float(eps_b)}
    times, fluxes, freqs = stream_lightcurve(
        list(checkpoints), qscales, micro, float(theta_obs), frequencies,
        float(redshift), float(luminosity_distance), time_bins,
    )
    return list(times), list(fluxes), list(freqs)
