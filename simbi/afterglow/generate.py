# =============================================================================
# generate.py
#
# workflow for generating photon events from hydro snapshots.
# - loads each checkpoint via the simbi reader
# - maps it to the rust event-generator contract (inputs.build_afterglow_inputs)
# - calls the symbi-afterglow rust binding (cpu_ext) to generate + transfer photons
# - writes the lab-frame catalog to HDF5 (postprocess.read_photon_events schema)
# usage:
#  generate_from_files(["chkpt.h5"], "events.h5", scale_model="blandford-mckee")
# =============================================================================

from typing import List


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
) -> None:
    """
    generate photon events from hydro checkpoint files.

    workflow:
        1. load hydro data from checkpoint(s)
        2. call the rust event generator
        3. optionally apply monte carlo radiative transfer
        4. write the catalog to HDF5

    args:
        files: list of checkpoint files
        output: output HDF5 filename
        max_events: maximum number of events to generate (split across files)
        photons_per_cell: sampling density (0=auto)
        eps_e: electron energy fraction
        eps_b: magnetic field energy fraction
        p: electron distribution power-law index
        theta_obs: observer angle [radians]
        z: redshift
        d_L: luminosity distance [cm]
        apply_mcrt: apply monte carlo radiative transfer
        include_scattering: include thomson scattering in MCRT
        scale_model: named code->cgs scale model (matches the hydro run)
    """
    # imported here to avoid a circular import (libs pulls the compiled extension).
    from ..libs import cpu_ext as rad_hydro
    from ..reader import read_simulation
    from .inputs import build_fields, build_mesh, build_qscales

    # code->cgs factors depend only on the scale model -> build once.
    qscales = build_qscales(scale_model)

    # the microphysics + observer block is file-independent; dt / adiabatic_index /
    # current_time are overwritten from each checkpoint's metadata.
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

        sim_cond["dt"] = data.metadata.dt
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
