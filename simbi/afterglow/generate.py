# =============================================================================
# generate.py
#
# workflow for generating photon events from hydro snapshots.
# calls C++ bindings, handles file I/O, orchestrates MCRT.
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
) -> None:
    """
    generate photon events from hydro checkpoint files.

    workflow:
        1. load hydro data from checkpoint(s)
        2. call C++ event generation
        3. optionally apply MCRT
        4. write events to HDF5

    args:
        files: list of checkpoint files
        output: output HDF5 filename
        max_events: maximum number of events to generate
        photons_per_cell: sampling density (0=auto)
        eps_e: electron energy fraction
        eps_b: magnetic field energy fraction
        p: electron distribution power-law index
        theta_obs: observer angle [radians]
        z: redshift
        d_L: luminosity distance [cm]
        apply_mcrt: apply monte carlo radiative transfer
        include_scattering: include thomson scattering in MCRT
    """
    # import here to avoid circular dependencies
    from . import rad_hydro
    from .scales import get_scale_model

    # get scale model (assume solar for now)
    scales = get_scale_model("solar")

    # prepare simulation conditions
    sim_cond = {
        "dt": 0.0,  # will be filled from checkpoint
        "theta_obs": theta_obs,
        "adiabatic_index": 4.0 / 3.0,  # will be filled from checkpoint
        "current_time": 0.0,  # will be filled from checkpoint
        "p": p,
        "z": z,
        "eps_e": eps_e,
        "eps_b": eps_b,
        "d_L": d_L,
        "nus": [1e9],  # placeholder
    }

    qscales = {
        "time_scale": scales.time_scale.value,
        "length_scale": scales.length_scale.value,
        "rho_scale": scales.rho_scale.value,
        "pre_scale": scales.pre_scale.value,
        "v_scale": 1.0,
    }

    all_events = []

    for idx, file in enumerate(files):
        print(f"processing {file} ({idx+1}/{len(files)})...")

        # read checkpoint using simbi's reader
        from ..tools.reader import read_simulation

        data = read_simulation(file)

        # extract fields
        fields = {
            name: data.get_field(name)
            for name in data.available_fields()
        }

        # build mesh
        mesh = {
            "x1": 0.5 * (data.mesh.x1v[:-1] + data.mesh.x1v[1:]),
        }
        if data.metadata.dimensions >= 2:
            mesh["x2"] = 0.5 * (data.mesh.x2v[:-1] + data.mesh.x2v[1:])
        if data.metadata.dimensions >= 3:
            mesh["x3"] = 0.5 * (data.mesh.x3v[:-1] + data.mesh.x3v[1:])

        # update sim conditions from checkpoint
        sim_cond["dt"] = data.metadata.dt
        sim_cond["adiabatic_index"] = data.metadata.gamma
        sim_cond["current_time"] = data.metadata.time

        # determine dimensionality
        data_dim = data.metadata.dimensions

        # call C++ event generation
        # NOTE: This requires proper pybind11 bindings to be implemented
        # For now, this is a stub showing the intended interface
        try:
            events = rad_hydro.generate_photon_events(
                sim_cond=sim_cond,
                qscales=qscales,
                fields=fields,
                mesh=mesh,
                data_dim=data_dim,
                max_events=max_events // len(files),  # distribute across files
                photons_per_cell=photons_per_cell,
            )

            print(f"  generated {len(events)} photons")

            # apply MCRT if requested
            if apply_mcrt:
                rad_hydro.monte_carlo_radiative_transfer(
                    events=events,
                    sim_cond=sim_cond,
                    qscales=qscales,
                    fields=fields,
                    mesh=mesh,
                    data_dim=data_dim,
                    include_scattering=include_scattering,
                    include_pair_production=False,
                )
                n_absorbed = sum(1 for e in events if e.absorbed)
                print(f"  MCRT: {n_absorbed} absorbed, {len(events)-n_absorbed} surviving")

            all_events.extend(events)

        except AttributeError:
            print("  ERROR: C++ bindings not yet implemented")
            print("  Need to add pybind11 wrappers for:")
            print("    - generate_photon_events")
            print("    - monte_carlo_radiative_transfer")
            raise NotImplementedError("C++ bindings not yet implemented")

    print(f"\ntotal events: {len(all_events)}")

    # write to HDF5
    print(f"writing events to {output}...")

    # use C++ write function (already in photon_event_io.cpp)
    try:
        rad_hydro.write_photon_events(output, all_events, sim_cond, qscales)
    except AttributeError:
        print("  ERROR: C++ write_photon_events not bound to Python")
        print("  Need to add pybind11 wrapper for write_photon_events")
        raise NotImplementedError("write_photon_events binding not implemented")

    print("done")
