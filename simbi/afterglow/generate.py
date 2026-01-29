# =============================================================================
# generate.py
#
# generate photon events from hydro snapshots using compiled C++ backend.
# uses rad_hydro extension module (compiled via meson from bindings/binding.cpp)
# =============================================================================

from typing import List, Optional
from typing import Optional as Opt

import numpy as np

from .mesh_expansion import expand_to_3d, validate_field_dict
from .scale_config import load_scale_config, scale_config_t


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
    d_L: Optional[float] = None,
    apply_mcrt: bool = False,
    include_scattering: bool = True,
    hydro_type: str = "SRHD",
    scale_config: Opt[scale_config_t] = None,
    n_theta: int = 32,
    n_phi: int = 16,
) -> None:
    """
    generate photon events from hydro checkpoint files.

    workflow:
        1. load hydro data from checkpoint(s)
        2. call C++ event generation (via rad_hydro module)
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
        d_L: luminosity distance [cm] (auto from z if None)
        apply_mcrt: apply monte carlo radiative transfer
        include_scattering: include thomson scattering in MCRT
        hydro_type: "SRHD" or "SRMHD"
        scale_config: scale configuration (auto-loads "grb_standard" if None)
        n_theta: number of theta zones for 1D/2D mesh expansion
        n_phi: number of phi zones for 1D/2D mesh expansion
    """
    try:
        from ..libs import rad_hydro
    except ImportError as e:
        raise ImportError(
            "rad_hydro extension not found. ensure meson build completed successfully.\n"
            "run: meson setup build && ninja -C build"
        ) from e

    from ..reader import read_simulation
    from .helpers import get_dL

    # load scale configuration
    if scale_config is None:
        scale_config = load_scale_config("grb_standard")
        print(f"using default scale config: {scale_config.name}")

    scales = scale_config

    # auto luminosity distance from redshift
    if d_L is None:
        d_L = get_dL(z).value

    # validate hydro type
    if hydro_type not in ["SRHD", "SRMHD"]:
        raise ValueError(
            f"hydro_type must be 'SRHD' or 'SRMHD', got '{hydro_type}'"
        )

    # prepare simulation conditions dict
    sim_cond = {
        "dt": 0.0,  # filled from checkpoint
        "theta_obs": theta_obs,
        "adiabatic_index": 4.0 / 3.0,  # filled from checkpoint
        "current_time": 0.0,  # filled from checkpoint
        "p": p,
        "z": z,
        "eps_e": eps_e,
        "eps_b": eps_b,
        "d_L": d_L,
        "nus": [1e9],  # placeholder frequency
        "hydro_type": hydro_type,
    }

    qscales = {
        "time_scale": scales.time_scale,
        "length_scale": scales.length_scale,
        "rho_scale": scales.rho_scale,
        "pre_scale": scales.pre_scale,
        "v_scale": scales.v_scale,
    }

    all_events = []
    n_files = len(files)

    print(f"generating photon events from {n_files} checkpoint(s)...")
    print(f"  max_events: {max_events}")
    print(f"  eps_e: {eps_e}, eps_b: {eps_b}, p: {p}")
    print(f"  observer angle: {np.degrees(theta_obs):.1f} deg")
    print(f"  hydro type: {hydro_type}")
    print(f"  MCRT: {'enabled' if apply_mcrt else 'disabled'}")
    print()

    for idx, file in enumerate(files):
        print(f"[{idx + 1}/{n_files}] processing {file}...")

        # read checkpoint
        data = read_simulation(file)

        # determine dimensionality first
        data_dim = int(data.metadata.dimensions)

        # extract fields needed for radiation: rho, gamma_beta, pressure
        # C++ code expects fields in order: [rho, gamma_beta, pressure]
        fields = {}

        # density
        fields["rho"] = np.ascontiguousarray(
            data.get_field("rho"), dtype=np.float64
        )

        # four-velocity magnitude (gamma*beta)
        try:
            fields["gamma_beta"] = np.ascontiguousarray(
                data.get_field("gamma_beta"), dtype=np.float64
            )
        except KeyError:
            # fallback: compute from velocity components
            # gamma_beta = lorentz * beta = sqrt(v^2 / (1 - v^2))
            v1 = data.get_field("v1")
            try:
                v2 = (
                    data.get_field("v2") if data_dim >= 2 else np.zeros_like(v1)
                )
            except (KeyError, IndexError):
                v2 = np.zeros_like(v1)
            try:
                v3 = (
                    data.get_field("v3") if data_dim >= 3 else np.zeros_like(v1)
                )
            except (KeyError, IndexError):
                v3 = np.zeros_like(v1)

            vsq = v1**2 + v2**2 + v3**2
            gamma = 1.0 / np.sqrt(1.0 - vsq)
            beta = np.sqrt(vsq)
            fields["gamma_beta"] = np.ascontiguousarray(
                gamma * beta, dtype=np.float64
            )

        # pressure
        fields["p"] = np.ascontiguousarray(
            data.get_field("p"), dtype=np.float64
        )

        # validate fields
        validate_field_dict(fields)

        # build initial mesh dict
        mesh = {
            "x1": np.ascontiguousarray(
                0.5 * (data.mesh.x1v[:-1] + data.mesh.x1v[1:]), dtype=np.float64
            ),
        }
        if data_dim >= 2:
            mesh["x2"] = np.ascontiguousarray(
                0.5 * (data.mesh.x2v[:-1] + data.mesh.x2v[1:]), dtype=np.float64
            )
        if data_dim >= 3:
            mesh["x3"] = np.ascontiguousarray(
                0.5 * (data.mesh.x3v[:-1] + data.mesh.x3v[1:]), dtype=np.float64
            )

        # expand to 3D spherical coordinates using geometry-aware mapping
        coord_system = data.metadata.coord_system.lower()
        original_dim = data_dim
        fields, mesh = expand_to_3d(
            fields=fields,
            mesh=mesh,
            coord_system=coord_system,
            dimensions=data_dim,
            n_theta=n_theta,  # polar zones for expansion
            n_phi=n_phi,  # azimuthal zones for expansion
        )

        # after expansion, mesh is always 3D (has x1, x2, x3)
        expanded_dim = 3 if original_dim < 3 else original_dim

        # update sim conditions from checkpoint
        sim_cond["dt"] = float(data.metadata.dt)
        sim_cond["adiabatic_index"] = float(data.metadata.gamma)
        sim_cond["current_time"] = float(data.metadata.time)

        # distribute max_events across files
        max_events_this_file = max_events // n_files
        if idx < (max_events % n_files):
            max_events_this_file += 1

        # call C++ event generation
        # use expanded_dim so C++ uses the full 3D mesh coordinates
        events = rad_hydro.generate_photon_events(
            sim_cond=sim_cond,
            qscales=qscales,
            fields=fields,
            mesh=mesh,
            data_dim=expanded_dim,
            max_events=max_events_this_file,
            photons_per_cell=photons_per_cell,
        )

        n_generated = len(events)
        print(f"  generated {n_generated} photon events")

        # apply MCRT if requested
        if apply_mcrt and n_generated > 0:
            print(f"  applying MCRT (scattering={include_scattering})...")
            rad_hydro.monte_carlo_radiative_transfer(
                events=events,
                sim_cond=sim_cond,
                qscales=qscales,
                fields=fields,
                mesh=mesh,
                data_dim=expanded_dim,
                include_scattering=include_scattering,
                include_pair_production=False,
            )

            # count absorbed photons
            n_absorbed = sum(1 for e in events if e.absorbed)
            n_surviving = n_generated - n_absorbed
            print(
                f"  MCRT complete: {n_absorbed} absorbed, {n_surviving} surviving"
            )

        all_events.extend(events)

    print()
    print(f"total events generated: {len(all_events)}")

    if len(all_events) == 0:
        print("WARNING: no events generated, not writing output file")
        return

    # write to HDF5
    print(f"writing events to {output}...")
    rad_hydro.write_photon_events(output, all_events, sim_cond, qscales)
    print("done!")


def read_events(filename: str):
    """
    read photon events from HDF5 file.

    returns:
        (events_list, metadata_dict)

    example:
        >>> events, meta = read_events("photons.h5")
        >>> print(f"loaded {len(events)} events")
        >>> print(f"observer angle: {np.degrees(meta['theta_obs']):.1f} deg")
    """
    try:
        from ..libs import rad_hydro
    except ImportError as e:
        raise ImportError(
            "rad_hydro extension not found. ensure meson build completed successfully."
        ) from e

    return rad_hydro.read_photon_events(filename)


def read_metadata(filename: str):
    """
    read metadata from HDF5 file without loading event data.

    returns:
        metadata_dict

    example:
        >>> meta = read_metadata("photons.h5")
        >>> print(f"file contains {meta['n_events']} events")
    """
    try:
        from ..libs import rad_hydro
    except ImportError as e:
        raise ImportError(
            "rad_hydro extension not found. ensure meson build completed successfully."
        ) from e

    return rad_hydro.read_photon_event_metadata(filename)
