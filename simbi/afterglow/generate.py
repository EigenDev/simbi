# =============================================================================
# generate.py
#
# workflow for generating photon events from hydro snapshots.
# - loads each checkpoint via the simbi reader
# - maps it to the rust event-generator contract (inputs.build_afterglow_inputs)
# - calls the symbi-afterglow rust binding (cpu_ext) to generate + transfer photons
# - writes the lab-frame catalog to HDF5 (postprocess.read_photon_events schema)
#
# a grb afterglow observation integrates emission over the EQUAL-ARRIVAL-TIME surface,
# which draws from a RANGE of emission epochs -- so the catalog must span an ARRAY of
# snapshots. each snapshot emits over the lab-time interval it REPRESENTS (its
# trapezoidal share of the snapshot-time axis); weighting by
# the represented interval is what makes the stacked snapshots tile the blast's history.
# a single snapshot has no interval to tile with -- it is one t_em slice.
# usage:
#  generate_from_files(["s0.h5", "s1.h5", ...], "events.h5", scale_model="blandford-mckee")
# =============================================================================

from typing import List, Optional


def _snapshot_emission_durations(times: List[float]) -> List[float]:
    """the lab-time interval each snapshot REPRESENTS, as composite-trapezoid quadrature
    weights over the (sorted) snapshot times: w_0 = (t_1 - t_0)/2, w_{N-1} =
    (t_{N-1} - t_{N-2})/2, interior w_i = (t_{i+1} - t_{i-1})/2. summing power * w_i over
    snapshots approximates the integral of power over the covered epoch [t_0, t_{N-1}].
    a single snapshot has no neighbor -> returns [0.0] (caller substitutes a fallback).
    NO snapshots -> no weights: the consumers partition their inputs into hydro
    checkpoints and pre-built catalogs and call this on the hydro side alone, so an
    all-catalog invocation (the ordinary way to reuse `afterglow generate` output across
    viewing angles) lands here with an empty list. a catalog already carries the emission
    weighting applied when it was generated, so an empty weight table is the correct
    answer rather than an edge case to refuse."""
    n = len(times)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    w = [0.0] * n
    w[0] = 0.5 * (times[1] - times[0])
    w[-1] = 0.5 * (times[-1] - times[-2])
    for ii in range(1, n - 1):
        w[ii] = 0.5 * (times[ii + 1] - times[ii - 1])
    return w


def _read_snapshot_time(path: str) -> float:
    """the lab-frame simulation time [code units] of a checkpoint, read WITHOUT loading
    fields (just the `metadata/time` attribute)."""
    import h5py

    with h5py.File(path, "r") as f:
        return float(f["metadata"].attrs["time"])


def generate_from_files(
    files: List[str],
    output: str,
    max_events: int = 1000000,
    photons_per_cell: int = 0,
    eps_e: float = 0.1,
    eps_b: float = 0.01,
    p: float = 2.5,
    theta_obs: float = 0.0,
    z: float = 0.0,
    d_L: float = 1e28,
    apply_mcrt: bool = False,
    include_scattering: bool = True,
    scale_model: str = "blandford-mckee",
    qscales: Optional[dict] = None,
) -> None:
    """
    generate photon events from an ARRAY of hydro checkpoint files.

    workflow:
        1. read each snapshot's lab time -> the interval it represents (trapezoid weight)
        2. load hydro data from each checkpoint
        3. call the rust event generator (each snapshot emits over its represented interval)
        4. optionally apply monte carlo radiative transfer
        5. merge + write the catalog to HDF5

    args:
        files: checkpoint files spanning the blast evolution (one snapshot is NOT an afterglow)
        output: output HDF5 filename
        max_events: maximum number of events to generate (split across files)
        photons_per_cell: sampling density (0=auto)
        eps_e: electron energy fraction
        eps_b: magnetic field energy fraction
        p: electron distribution power-law index
        theta_obs: observer angle [radians] (vestigial: the catalog is angle-INDEPENDENT;
            the line of sight is chosen at REDUCTION, in skymap/lightcurve)
        z: redshift
        d_L: luminosity distance [cm]
        apply_mcrt: apply monte carlo radiative transfer
        include_scattering: include thomson scattering in MCRT
        scale_model: named code->cgs scale model (used only when `qscales` is not supplied)
        qscales: explicit code->cgs factors (from a SystemManifest); overrides scale_model
    """
    # imported here to avoid a circular import (libs pulls the compiled extension).
    from ..libs import cpu_ext as rad_hydro
    from ..reader import read_simulation
    from .inputs import build_fields, build_mesh, build_qscales

    if not files:
        raise ValueError("no checkpoint files supplied")

    # code->cgs factors: an explicit manifest wins, else build from the named model.
    if qscales is None:
        qscales = build_qscales(scale_model)

    # SORT the snapshots by lab time and weight each by the interval it REPRESENTS, so the
    # stacked emission tiles the blast history (a single emit-duration = the tiny CFL dt would
    # under-count and mis-weight). a lone snapshot has no interval -> warn; the EATS integral
    # over epochs is undefined from one t_em slice.
    files = sorted(files, key=_read_snapshot_time)
    times = [_read_snapshot_time(f) for f in files]
    durations = _snapshot_emission_durations(times)
    if len(files) == 1:
        print(
            "  WARNING: a single snapshot is not a grb afterglow -- the equal-arrival-time\n"
            "  surface integrates emission over a RANGE of epochs. supply an array of\n"
            "  snapshots spanning the blast evolution. falling back to the CFL dt for now."
        )

    # the microphysics + observer block is file-independent; dt / adiabatic_index /
    # current_time are overwritten per snapshot.
    sim_cond = {
        "dt": 0.0,
        "theta_obs": theta_obs,
        "adiabatic_index": 4.0 / 3.0,
        "current_time": 0.0,
        "p": p,
        "z": z,
        "eps_e": eps_e,
        "eps_b": eps_b,
        "d_L": d_L,
        "nus": [1e9],
    }

    # the rust binding returns an opaque PhotonEvents handle; catalogs from multiple
    # files are merged into the first handle via its `extend` method (no python list).
    catalog = None

    for idx, file in enumerate(files):
        print(f"processing {file} ({idx + 1}/{len(files)})...")

        data = read_simulation(file)
        fields = build_fields(data)
        mesh = build_mesh(data)

        # emission duration = the lab-time interval this snapshot represents; a lone snapshot
        # (duration 0) falls back to the CFL dt so it still produces SOMETHING (with the warning).
        emit_dt = durations[idx] if durations[idx] > 0.0 else data.metadata.dt
        sim_cond["dt"] = emit_dt
        sim_cond["adiabatic_index"] = data.metadata.gamma
        sim_cond["current_time"] = data.metadata.time

        events = rad_hydro.generate_photon_events(
            sim_cond=sim_cond,
            qscales=qscales,
            fields=fields,
            mesh=mesh,
            max_events=max_events // len(files),
            photons_per_cell=photons_per_cell,
        )
        print(f"  generated {len(events)} photons")

        if apply_mcrt:
            rad_hydro.monte_carlo_radiative_transfer(
                events,
                sim_cond=sim_cond,
                qscales=qscales,
                fields=fields,
                mesh=mesh,
                include_scattering=include_scattering,
                include_pair_production=False,
            )
            n_absorbed = events.n_absorbed
            print(f"  MCRT: {n_absorbed} absorbed, {events.n_surviving} surviving")

        if catalog is None:
            catalog = events
        else:
            catalog.extend(events)

    if catalog is None:
        raise ValueError("no checkpoint files supplied")

    print(f"\ntotal events: {len(catalog)}")
    print(f"writing events to {output}...")
    rad_hydro.write_photon_events(output, catalog, sim_cond, qscales)
    print("done")
