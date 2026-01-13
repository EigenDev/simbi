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
        "--n-theta",
        type=int,
        default=128,
        help="polar resolution",
    )

    parser.add_argument(
        "--n-phi",
        type=int,
        default=256,
        help="azimuthal resolution",
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
    """execute lightcurve subcommand"""
    from ...afterglow.plotting import plot_lightcurve
    from ...afterglow.postprocess import compute_lightcurve, read_photon_events

    print(f"loading photon events from {args.events}...")
    events, meta = read_photon_events(args.events)
    print(f"loaded {meta.n_events} events")

    print(f"computing lightcurve for observer angle {args.observer_angle}°...")

    # time bins
    if args.time_range:
        time_bins = np.geomspace(args.time_range[0], args.time_range[1], args.n_bins + 1)
    else:
        time_bins = None

    lc = compute_lightcurve(
        events,
        meta,
        observer_angle=np.deg2rad(args.observer_angle),
        frequencies=args.frequencies,
        time_bins=time_bins,
        n_bins=args.n_bins,
        energy_cut=args.energy_cut,
    )

    print(f"computed {len(lc.times)} time bins, {len(lc.frequencies)} frequencies")

    if args.output:
        import h5py
        with h5py.File(args.output, 'w') as f:
            f.create_dataset('times', data=lc.times)
            f.create_dataset('frequencies', data=lc.frequencies)
            for nu, flux in lc.fluxes.items():
                f.create_dataset(f'flux_{nu:.2e}', data=flux)
        print(f"saved lightcurve to {args.output}")

    if args.plot or args.save_fig:
        plot_lightcurve(lc, save=args.save_fig)


def execute_skymap(args: Namespace, remaining: Optional[list] = None) -> None:
    """execute skymap subcommand"""
    from ...afterglow.plotting import plot_skymap
    from ...afterglow.postprocess import compute_skymap, read_photon_events

    print(f"loading photon events from {args.events}...")
    events, meta = read_photon_events(args.events)

    print(f"computing skymap at t={args.time} day...")

    skymap = compute_skymap(
        events,
        meta,
        time=args.time,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        time_window=args.time_window,
    )

    print(f"computed {args.n_theta}x{args.n_phi} skymap")

    if args.output:
        import h5py
        with h5py.File(args.output, 'w') as f:
            f.create_dataset('theta', data=skymap.theta)
            f.create_dataset('phi', data=skymap.phi)
            f.create_dataset('intensity', data=skymap.intensity)
            f.attrs['time'] = skymap.time
        print(f"saved skymap to {args.output}")

    if args.plot or args.save_fig:
        plot_skymap(skymap, save=args.save_fig)


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
