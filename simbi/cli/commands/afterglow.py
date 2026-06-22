# =============================================================================
# simbi/cli/commands/afterglow.py
#
# afterglow command with subcommands for photon event workflow.
# clean separation: generate events (C++), analyze events (Python)
# =============================================================================
from argparse import Namespace
from typing import Optional

import numpy as np

from ..utils.formatter import HelpFormatter


def setup_parser(subparsers) -> None:
    """setup afterglow command with subcommands"""
    afterglow_parser = subparsers.add_parser(
        "afterglow",
        help="photon event generation and analysis",
        usage="simbi afterglow <subcommand> [options]",
        formatter_class=HelpFormatter,
    )

    afterglow_subparsers = afterglow_parser.add_subparsers(
        dest="afterglow_subcommand",
        title="subcommands",
        metavar="<subcommand>",
        required=True,
    )

    # subcommand: generate
    setup_generate_parser(afterglow_subparsers)

    # subcommand: lightcurve
    setup_lightcurve_parser(afterglow_subparsers)

    # subcommand: skymap
    setup_skymap_parser(afterglow_subparsers)

    # subcommand: polarization
    setup_polarization_parser(afterglow_subparsers)

    # subcommand: spectrum
    setup_spectrum_parser(afterglow_subparsers)


def setup_generate_parser(subparsers) -> None:
    """generate photon events from hydro snapshots"""
    parser = subparsers.add_parser(
        "generate",
        help="generate photon events from simulation data",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="hydro checkpoint file(s)",
    )

    parser.add_argument(
        "--output",
        default="events.h5",
        help="output HDF5 file for photon events",
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=1000000,
        help="maximum number of events to generate",
    )

    parser.add_argument(
        "--photons-per-cell",
        type=int,
        default=0,
        help="photons per cell (0=auto)",
    )

    parser.add_argument(
        "--eps-e",
        type=float,
        default=0.1,
        help="electron energy fraction",
    )

    parser.add_argument(
        "--eps-b",
        type=float,
        default=0.01,
        help="magnetic field energy fraction",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=2.5,
        help="electron distribution power-law index",
    )

    parser.add_argument(
        "--theta-obs",
        type=float,
        default=0.0,
        help="observer angle [degrees]",
    )

    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="redshift",
    )

    parser.add_argument(
        "--d-L",
        type=float,
        default=1e28,
        help="luminosity distance [cm]",
    )

    parser.add_argument(
        "--mcrt",
        action="store_true",
        help="apply monte carlo radiative transfer",
    )

    parser.add_argument(
        "--no-scattering",
        action="store_true",
        help="disable thomson scattering in MCRT",
    )

    parser.set_defaults(func=execute_generate)


def setup_lightcurve_parser(subparsers) -> None:
    """compute lightcurve from events"""
    parser = subparsers.add_parser(
        "lightcurve",
        help="compute observer lightcurve from photon events",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="hydro checkpoint HDF5 file(s) — the rust afterglow reads them "
        "directly and integrates the EATS over all frames",
    )

    parser.add_argument(
        "--scale",
        default="solar",
        help="unit scale model (code -> cgs): length/density/pressure/time",
    )
    parser.add_argument("--eps-e", type=float, default=0.1, help="electron energy fraction")
    parser.add_argument("--eps-b", type=float, default=0.01, help="magnetic energy fraction")
    parser.add_argument("--p", type=float, default=2.5, help="electron power-law index")
    parser.add_argument("--redshift", type=float, default=0.0, help="source redshift z")
    parser.add_argument(
        "--d-l", type=float, default=None,
        help="luminosity distance [cm] (auto from z if omitted)",
    )

    parser.add_argument(
        "--observer-angle",
        type=float,
        default=0.0,
        help="viewing angle [degrees]",
    )

    parser.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=[1e9],
        help="observed frequencies [Hz]",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="number of time bins",
    )

    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        default=None,
        help="time range [day] (auto if omitted)",
    )

    parser.add_argument(
        "--energy-cut",
        type=float,
        default=0.0,
        help="minimum photon energy [erg]",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="save lightcurve data to file",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="show plot",
    )

    parser.add_argument(
        "--save-fig",
        default=None,
        help="save figure to file",
    )

    parser.set_defaults(func=execute_lightcurve)


def setup_skymap_parser(subparsers) -> None:
    """compute skymap at specific time"""
    parser = subparsers.add_parser(
        "skymap",
        help="compute sky intensity map at observer time",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "checkpoint",
        help="hydro checkpoint HDF5 file (the rust afterglow reads it directly)",
    )

    parser.add_argument(
        "--scale",
        default="solar",
        help="unit scale model (code -> cgs)",
    )
    parser.add_argument("--eps-e", type=float, default=0.1, help="electron energy fraction")
    parser.add_argument("--eps-b", type=float, default=0.01, help="magnetic energy fraction")
    parser.add_argument("--p", type=float, default=2.5, help="electron power-law index")
    parser.add_argument("--observer-angle", type=float, default=0.0, help="viewing angle [deg]")

    parser.add_argument(
        "--time",
        type=float,
        required=True,
        help="observer time [day]",
    )

    parser.add_argument(
        "--n-pix",
        type=int,
        default=256,
        help="image resolution (n_pix x n_pix, cartesian sky plane)",
    )

    parser.add_argument(
        "--time-window",
        type=float,
        default=0.1,
        help="integration window [day]",
    )

    parser.add_argument(
        "--bolometric",
        action="store_true",
        help="bolometric beaming (doppler^4) instead of in-band (doppler^3)",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="save skymap data to file",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="show plot",
    )

    parser.add_argument(
        "--save-fig",
        default=None,
        help="save figure to file",
    )

    parser.set_defaults(func=execute_skymap)


def setup_polarization_parser(subparsers) -> None:
    """compute polarization evolution"""
    parser = subparsers.add_parser(
        "polarization",
        help="compute polarization evolution for observer",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "events",
        help="photon events HDF5 file",
    )

    parser.add_argument(
        "--observer-angle",
        type=float,
        default=0.0,
        help="viewing angle [degrees]",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="number of time bins",
    )

    parser.add_argument(
        "--energy-min",
        type=float,
        default=0.0,
        help="minimum energy [erg]",
    )

    parser.add_argument(
        "--energy-max",
        type=float,
        default=1e10,
        help="maximum energy [erg]",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="save polarization data to file",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="show plot",
    )

    parser.add_argument(
        "--save-fig",
        default=None,
        help="save figure to file",
    )

    parser.set_defaults(func=execute_polarization)


def setup_spectrum_parser(subparsers) -> None:
    """compute spectrum at specific time"""
    parser = subparsers.add_parser(
        "spectrum",
        help="compute spectral flux at observer time",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "events",
        help="photon events HDF5 file",
    )

    parser.add_argument(
        "--time",
        type=float,
        required=True,
        help="observer time [day]",
    )

    parser.add_argument(
        "--observer-angle",
        type=float,
        default=0.0,
        help="viewing angle [degrees]",
    )

    parser.add_argument(
        "--freq-min",
        type=float,
        default=1e8,
        help="minimum frequency [Hz]",
    )

    parser.add_argument(
        "--freq-max",
        type=float,
        default=1e12,
        help="maximum frequency [Hz]",
    )

    parser.add_argument(
        "--n-freq",
        type=int,
        default=50,
        help="number of frequency bins",
    )

    parser.add_argument(
        "--time-window",
        type=float,
        default=0.1,
        help="integration window [day]",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="save spectrum data to file",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="show plot",
    )

    parser.add_argument(
        "--save-fig",
        default=None,
        help="save figure to file",
    )

    parser.set_defaults(func=execute_spectrum)


# =============================================================================
# execution functions
# =============================================================================

def execute_generate(args: Namespace, remaining: Optional[list] = None) -> None:
    """execute generate subcommand"""
    from ...afterglow.generate import generate_from_files

    print(f"generating photon events from {len(args.files)} snapshot(s)...")

    generate_from_files(
        files=args.files,
        output=args.output,
        max_events=args.max_events,
        photons_per_cell=args.photons_per_cell,
        eps_e=args.eps_e,
        eps_b=args.eps_b,
        p=args.p,
        theta_obs=np.deg2rad(args.theta_obs),
        z=args.z,
        d_L=args.d_L,
        apply_mcrt=args.mcrt,
        include_scattering=not args.no_scattering,
    )

    print(f"saved photon events to {args.output}")


def execute_lightcurve(args: Namespace, remaining: Optional[list] = None) -> None:
    """compute the observer light curve F_nu(t) directly from hydro checkpoints
    via the rust afterglow (read_cells -> synchrotron catalog -> EATS reduce)."""
    import h5py

    from simbi import afterglow_lightcurve
    from simbi.reader import read_simulation

    from ...afterglow.scales import get_scale_model

    if afterglow_lightcurve is None:
        raise SystemExit("rad_hydro (rust afterglow) not installed in simbi/libs")

    scales = get_scale_model(args.scale)
    meta0 = read_simulation(args.files[0]).metadata
    # luminosity distance: explicit, else from z (0 -> a 10 pc reference).
    if args.d_l is not None:
        d_l = args.d_l
    elif args.redshift > 0.0:
        from ...afterglow.radiation import get_dL
        d_l = float(get_dL(args.redshift).value)
    else:
        d_l = 3.086e19  # 10 pc in cm

    # observer-time bin edges [day].
    if args.time_range:
        time_edges = np.geomspace(args.time_range[0], args.time_range[1], args.n_bins + 1)
    else:
        # default: a couple decades around the checkpoint light-crossing scale.
        time_edges = np.geomspace(1e-3, 1e3, args.n_bins + 1)

    print(f"computing light curve from {len(args.files)} checkpoint(s)...")
    times, fluxes_flat, freqs = afterglow_lightcurve(
        list(args.files),
        scales.length_scale.value,
        scales.rho_scale.value,
        scales.pre_scale.value,
        scales.time_scale.value,
        args.p,
        args.eps_e,
        args.eps_b,
        meta0.gamma,
        meta0.dt,
        np.deg2rad(args.observer_angle),
        [float(f) for f in args.frequencies],
        args.redshift,
        d_l,
        [float(t) for t in time_edges],
    )
    fluxes = np.array(fluxes_flat).reshape(len(times), len(freqs))[: args.n_bins]
    times = np.asarray(times)[: args.n_bins]
    print(f"computed {len(times)} time bins x {len(freqs)} frequencies")

    if args.output:
        with h5py.File(args.output, "w") as f:
            f.create_dataset("times", data=times)
            f.create_dataset("frequencies", data=freqs)
            for i, nu in enumerate(freqs):
                f.create_dataset(f"flux_{nu:.2e}", data=fluxes[:, i])
        print(f"saved lightcurve to {args.output}")

    if args.plot or args.save_fig:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, nu in enumerate(freqs):
            ax.loglog(times, fluxes[:, i], label=f"{nu:.2e} Hz")
        ax.set_xlabel("observer time [day]")
        ax.set_ylabel(r"$F_\nu$ [mJy]")
        ax.legend()
        if args.save_fig:
            fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
            print(f"saved figure to {args.save_fig}")
        if args.plot:
            plt.show()


def execute_skymap(args: Namespace, remaining: Optional[list] = None) -> None:
    """compute the sky intensity map directly from a hydro checkpoint via the
    rust afterglow (read_cells -> synchrotron catalog -> sky-plane reduce)."""
    from simbi import afterglow_skymap
    from simbi.reader import read_simulation

    from ...afterglow.scales import get_scale_model

    if afterglow_skymap is None:
        raise SystemExit("rad_hydro (rust afterglow) not installed in simbi/libs")

    scales = get_scale_model(args.scale)
    meta0 = read_simulation(args.checkpoint).metadata

    print(f"computing skymap at t={args.time} day from {args.checkpoint}...")
    flat, n_pix = afterglow_skymap(
        args.checkpoint,
        scales.length_scale.value,
        scales.rho_scale.value,
        scales.pre_scale.value,
        scales.time_scale.value,
        args.p,
        args.eps_e,
        args.eps_b,
        meta0.gamma,
        meta0.dt,
        np.deg2rad(args.observer_angle),
        args.time,
        args.time_window,
        args.n_pix,
        bolometric=args.bolometric,
    )
    image = np.array(flat).reshape(n_pix, n_pix)
    print(f"computed {n_pix}x{n_pix} skymap (max={image.max():.3e})")

    if args.output:
        import h5py
        with h5py.File(args.output, "w") as f:
            f.create_dataset("intensity", data=image)
            f.attrs["time"] = args.time
            f.attrs["n_pix"] = n_pix
        print(f"saved skymap to {args.output}")

    if args.plot or args.save_fig:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(image, origin="lower", cmap="inferno")
        ax.set_title(f"t = {args.time} day")
        if args.save_fig:
            fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
            print(f"saved figure to {args.save_fig}")
        if args.plot:
            plt.show()


def execute_polarization(args: Namespace, remaining: Optional[list] = None) -> None:
    """execute polarization subcommand"""
    from ...afterglow.plotting import plot_polarization
    from ...afterglow.postprocess import compute_polarization, read_photon_events

    print(f"loading photon events from {args.events}...")
    events, meta = read_photon_events(args.events)

    print(f"computing polarization for observer angle {args.observer_angle}°...")

    pol = compute_polarization(
        events,
        meta,
        observer_angle=np.deg2rad(args.observer_angle),
        n_bins=args.n_bins,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
    )

    print(f"computed {len(pol.times)} time bins")

    if args.output:
        import h5py
        with h5py.File(args.output, 'w') as f:
            f.create_dataset('times', data=pol.times)
            f.create_dataset('polarization_degree', data=pol.polarization_degree)
            f.create_dataset('polarization_angle', data=pol.polarization_angle)
            f.create_dataset('stokes_Q', data=pol.stokes_Q)
            f.create_dataset('stokes_U', data=pol.stokes_U)
            f.create_dataset('stokes_V', data=pol.stokes_V)
        print(f"saved polarization to {args.output}")

    if args.plot or args.save_fig:
        plot_polarization(pol, save=args.save_fig)


def execute_spectrum(args: Namespace, remaining: Optional[list] = None) -> None:
    """execute spectrum subcommand"""
    from ...afterglow.plotting import plot_spectrum
    from ...afterglow.postprocess import compute_spectrum, read_photon_events

    print(f"loading photon events from {args.events}...")
    events, meta = read_photon_events(args.events)

    print(f"computing spectrum at t={args.time} day...")

    frequencies = np.geomspace(args.freq_min, args.freq_max, args.n_freq + 1)

    spec = compute_spectrum(
        events,
        meta,
        observer_angle=np.deg2rad(args.observer_angle),
        time=args.time,
        frequencies=frequencies,
        time_window=args.time_window,
    )

    print(f"computed {len(spec.frequencies)} frequency bins")

    if args.output:
        import h5py
        with h5py.File(args.output, 'w') as f:
            f.create_dataset('frequencies', data=spec.frequencies)
            f.create_dataset('fluxes', data=spec.fluxes)
            f.attrs['time'] = spec.time
        print(f"saved spectrum to {args.output}")

    if args.plot or args.save_fig:
        plot_spectrum(spec, save=args.save_fig)
