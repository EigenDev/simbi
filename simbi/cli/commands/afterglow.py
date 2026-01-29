# =============================================================================
# simbi/cli/commands/afterglow.py
#
# afterglow command with subcommands for photon event workflow.
# clean separation: generate events (C++), analyze events (Python)
# =============================================================================
import sys
from argparse import Namespace
from typing import Optional

import numpy as np

from ..utils.file_utils import glob_files
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

    # subcommand: inspect
    setup_inspect_parser(afterglow_subparsers)


def setup_inspect_parser(subparsers) -> None:
    """inspect photon events file"""
    parser = subparsers.add_parser(
        "inspect",
        help="inspect photon events file and print summary statistics",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "events",
        help="photon events HDF5 file",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="show detailed statistics and histograms",
    )

    parser.set_defaults(func=execute_inspect)


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
        help="hydro checkpoint file(s) or directory",
        action=glob_files,
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
        "--n-theta",
        type=int,
        default=32,
        help="number of theta zones for 1D/2D mesh expansion",
    )

    parser.add_argument(
        "--n-phi",
        type=int,
        default=16,
        help="number of phi zones for 1D/2D mesh expansion",
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

    parser.add_argument(
        "--scale-config",
        type=str,
        default="grb_standard",
        help="scale configuration: standard name (grb_standard, kilonova, etc.) or yaml file path",
    )

    parser.add_argument(
        "--list-scales",
        action="store_true",
        help="list available standard scale configurations and exit",
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
        nargs="+",
        default=None,
        help="observer time(s) [day] - single value or multiple for animation. if omitted, shows available times",
    )

    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="minimum energy [erg] (default: use data range)",
    )

    parser.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help="maximum energy [erg] (default: use data range)",
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
        "--observer-angle",
        type=float,
        default=0.0,
        help="observer viewing angle [degrees] (0 = on-axis)",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=None,
        help="telescope beam FWHM [arcsec] for PSF convolution",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="override luminosity distance [Mpc] for angular scaling",
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
        help="save figure to file (.pdf for single, .mp4/.gif for animation)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="frames per second for animation",
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


def execute_inspect(args: Namespace, remaining: Optional[list] = None) -> None:
    """execute inspect subcommand"""
    import h5py
    import numpy as np

    print(f"inspecting photon events file: {args.events}\n")

    with h5py.File(args.events, "r") as f:
        # read metadata
        print("=" * 80)
        print("METADATA")
        print("=" * 80)
        for key in sorted(f.attrs.keys()):
            value = f.attrs[key]
            print(f"  {key:20s} = {value}")

        # read event data
        print("\n" + "=" * 80)
        print("EVENT STATISTICS")
        print("=" * 80)

        n_events = f.attrs["n_events"]
        print(f"  total events: {n_events:,}")

        # energy range
        energy = f["energy"][:]
        print("\n  energy [erg]:")
        print(f"    min:    {energy.min():.4e}")
        print(f"    max:    {energy.max():.4e}")
        print(f"    median: {np.median(energy):.4e}")
        print(f"    mean:   {energy.mean():.4e}")

        # convert to eV for intuition
        erg_to_eV = 6.242e11
        print("\n  energy [eV]:")
        print(f"    min:    {energy.min() * erg_to_eV:.4e}")
        print(f"    max:    {energy.max() * erg_to_eV:.4e}")
        print(f"    median: {np.median(energy) * erg_to_eV:.4e}")

        # emission time range
        t_em = f["t_emission"][:]
        print("\n  emission time [s]:")
        print(f"    min:  {t_em.min():.4e}")
        print(f"    max:  {t_em.max():.4e}")
        print(f"    span: {t_em.max() - t_em.min():.4e}")

        # convert to days
        print("\n  emission time [day]:")
        print(f"    min:  {t_em.min() / 86400:.4f}")
        print(f"    max:  {t_em.max() / 86400:.4f}")
        print(f"    span: {(t_em.max() - t_em.min()) / 86400:.4f}")

        # spatial extent
        x = f["x"][:]
        y = f["y"][:]
        z = f["z"][:]
        r = np.sqrt(x**2 + y**2 + z**2)
        print("\n  spatial extent [cm]:")
        print(f"    min radius: {r.min():.4e}")
        print(f"    max radius: {r.max():.4e}")

        # absorption
        absorbed = f["absorbed"][:]
        n_absorbed = absorbed.sum()
        print("\n  absorption:")
        print(
            f"    absorbed:  {n_absorbed:,} ({100 * n_absorbed / n_events:.1f}%)"
        )
        print(
            f"    surviving: {n_events - n_absorbed:,} ({100 * (1 - n_absorbed / n_events):.1f}%)"
        )

        # scattering
        n_scatter = f["n_scatter"][:]
        print("\n  scattering:")
        print(f"    mean scatters: {n_scatter.mean():.2f}")
        print(f"    max scatters:  {n_scatter.max()}")

        # stokes parameters (polarization)
        stokes_I = f["stokes_I"][:]
        print("\n  stokes I (intensity):")
        print(f"    min:    {stokes_I.min():.4e}")
        print(f"    max:    {stokes_I.max():.4e}")
        print(f"    median: {np.median(stokes_I):.4e}")

        if args.detailed:
            # detailed histograms
            print("\n" + "=" * 80)
            print("DETAILED STATISTICS")
            print("=" * 80)

            # energy histogram
            print("\n  energy histogram (log10 erg):")
            counts, bins = np.histogram(np.log10(energy), bins=10)
            for i in range(len(counts)):
                bar = "#" * int(50 * counts[i] / counts.max())
                print(
                    f"    [{bins[i]:7.2f}, {bins[i + 1]:7.2f}): {counts[i]:8,} {bar}"
                )

            # time histogram
            print("\n  emission time histogram (day):")
            counts, bins = np.histogram(t_em / 86400, bins=10)
            for i in range(len(counts)):
                bar = "#" * int(50 * counts[i] / counts.max())
                print(
                    f"    [{bins[i]:8.2f}, {bins[i + 1]:8.2f}): {counts[i]:8,} {bar}"
                )

            # radial histogram
            print("\n  radial distribution histogram (log10 cm):")
            counts, bins = np.histogram(np.log10(r), bins=10)
            for i in range(len(counts)):
                bar = "#" * int(50 * counts[i] / counts.max())
                print(
                    f"    [{bins[i]:7.2f}, {bins[i + 1]:7.2f}): {counts[i]:8,} {bar}"
                )

        print("\n" + "=" * 80)
        print("RECOMMENDED PARAMETERS")
        print("=" * 80)

        # suggest reasonable ranges for analysis
        E_med = np.median(energy)
        E_min_suggest = E_med * 0.1
        E_max_suggest = E_med * 10.0

        print("\n  for skymap/spectrum commands:")
        print(f"    --energy-min {E_min_suggest:.2e}")
        print(f"    --energy-max {E_max_suggest:.2e}")

        t_med_day = np.median(t_em) / 86400
        print("\n  for time-based analysis:")
        print(f"    --time {t_med_day:.2f}  (median emission time)")
        print("    --time-window 0.1  (captures ~10% of time range in days)")

        print("\n" + "=" * 80)


def execute_generate(args: Namespace, remaining: Optional[list] = None) -> None:
    """execute generate subcommand"""
    from ...afterglow.generate import generate_from_files
    from ...afterglow.scale_config import (
        list_standard_scales,
        load_scale_config,
    )
    from ...reader import read_simulation

    # handle --list-scales
    if args.list_scales:
        list_standard_scales()
        return

    # load scale configuration
    try:
        scales = load_scale_config(args.scale_config)
        print(f"using scale configuration: {scales.name}")
        scales.print_info()
        print()
    except Exception as e:
        raise ValueError(
            f"failed to load scale config '{args.scale_config}': {e}"
        )

    print(f"generating photon events from {len(args.files)} snapshot(s)...")

    # validate regime from first checkpoint
    print(f"checking simulation regime from {args.files[0]}...")
    first_checkpoint = read_simulation(args.files[0])
    regime = first_checkpoint.metadata.regime.lower()

    if regime not in ["srhd", "srmhd"]:
        raise ValueError(
            f"afterglow module only supports SRHD or SRMHD regimes.\n"
            f"detected regime: '{regime}'\n"
            f"your simulation must be special relativistic to compute synchrotron radiation."
        )

    # auto-detect hydro_type from regime
    hydro_type = "SRMHD" if regime == "srmhd" else "SRHD"
    print(f"detected regime: {regime.upper()} → using hydro_type={hydro_type}")

    if hydro_type == "SRHD":
        print("warning: SRHD has no magnetic field → unpolarized emission")

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
        hydro_type=hydro_type,
        scale_config=scales,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
    )

    print(f"saved photon events to {args.output}")


def execute_lightcurve(
    args: Namespace, remaining: Optional[list] = None
) -> None:
    """execute lightcurve subcommand"""
    from ...afterglow.plotting import plot_lightcurve
    from ...afterglow.postprocess import compute_lightcurve, read_photon_events

    print(f"loading photon events from {args.events}...")
    events, meta = read_photon_events(args.events)
    print(f"loaded {meta.n_events} events")

    print(f"computing lightcurve for observer angle {args.observer_angle}°...")

    # time bins
    if args.time_range:
        time_bins = np.geomspace(
            args.time_range[0], args.time_range[1], args.n_bins + 1
        )
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

    print(
        f"computed {len(lc.times)} time bins, {len(lc.frequencies)} frequencies"
    )

    if args.output:
        import h5py

        with h5py.File(args.output, "w") as f:
            f.create_dataset("times", data=lc.times)
            f.create_dataset("frequencies", data=lc.frequencies)
            for nu, flux in lc.fluxes.items():
                f.create_dataset(f"flux_{nu:.2e}", data=flux)
        print(f"saved lightcurve to {args.output}")

    if args.plot or args.save_fig:
        plot_lightcurve(lc, save=args.save_fig)


def execute_skymap(args: Namespace, remaining: Optional[list] = None) -> None:
    """execute skymap subcommand"""
    from ...afterglow.plotting import plot_skymap, plot_skymap_animation
    from ...afterglow.postprocess import compute_skymap, read_photon_events

    print(f"loading photon events from {args.events}...")
    events, meta = read_photon_events(args.events)

    # check if we should launch TUI (interactive mode, no --time specified)
    if args.time is None and sys.stdin.isatty():
        from ..tui.skymap import run_skymap_tui

        params = run_skymap_tui(args.events, events, meta)
        if params is None:
            print("cancelled.")
            return

        # run skymap with TUI-selected parameters
        distance_cm = params.distance * 3.086e24 if params.distance else None
        if distance_cm:
            print(f"using distance override: {params.distance} Mpc")

        print(f"computing skymap at t={params.time} day...")
        skymap = compute_skymap(
            events,
            meta,
            observer_angle=np.radians(params.observer_angle),
            time=params.time,
            energy_min=params.energy_min,
            energy_max=params.energy_max,
            n_theta=params.n_theta,
            n_phi=params.n_phi,
            time_window=params.time_window,
            distance_override=distance_cm,
        )

        intensity = np.array(skymap.intensity)
        print(f"computed {params.n_theta}x{params.n_phi} skymap")
        print(
            f"  intensity: min={intensity.min():.2e}, max={intensity.max():.2e}"
        )

        if params.output:
            import h5py

            with h5py.File(params.output, "w") as f:
                f.create_dataset("theta", data=skymap.theta)
                f.create_dataset("phi", data=skymap.phi)
                f.create_dataset("intensity", data=skymap.intensity)
                f.attrs["time"] = skymap.time
                f.attrs["d_L"] = skymap.d_L
            print(f"saved skymap to {params.output}")

        plot_skymap(skymap, save=params.save_fig, beam_fwhm_arcsec=params.beam)
        return

    # non-interactive mode requires --time
    if args.time is None:
        # compute stats for error message
        c_cgs = 2.998e10
        day_s = 86400.0
        one_plus_z = 1.0 + meta.z
        observer_dir = np.array([0.0, 0.0, 1.0])
        r_dot_n = (
            events.x * observer_dir[0]
            + events.y * observer_dir[1]
            + events.z * observer_dir[2]
        )
        t_arrival = one_plus_z * (events.t_emission - r_dot_n / c_cgs) / day_s
        mask = ~events.absorbed
        t_filtered = t_arrival[mask]
        t_median = np.median(t_filtered)

        print()
        print("error: --time is required in non-interactive mode")
        print(
            f"  example: simbi afterglow skymap {args.events} --time {t_median:.1f}"
        )
        return

    # CLI mode with explicit parameters
    observer_angle = np.radians(args.observer_angle)
    observer_dir = np.array(
        [np.sin(observer_angle), 0.0, np.cos(observer_angle)]
    )
    c_cgs = 2.998e10
    day_s = 86400.0
    one_plus_z = 1.0 + meta.z
    r_dot_n = (
        events.x * observer_dir[0]
        + events.y * observer_dir[1]
        + events.z * observer_dir[2]
    )
    t_arrival = one_plus_z * (events.t_emission - r_dot_n / c_cgs) / day_s

    # set energy range from data if not specified
    energy_min = (
        args.energy_min if args.energy_min is not None else events.energy.min()
    )
    energy_max = (
        args.energy_max if args.energy_max is not None else events.energy.max()
    )

    # filter by absorbed and energy
    mask = ~events.absorbed
    if energy_min > 0:
        mask &= events.energy >= energy_min
    if energy_max < np.inf:
        mask &= events.energy <= energy_max

    t_arrival_filtered = t_arrival[mask]
    n_filtered = mask.sum()

    if n_filtered == 0:
        print("error: no events pass the filter (all absorbed?)")
        print(f"  energy range: [{energy_min:.2e}, {energy_max:.2e}] erg")
        print(f"  non-absorbed events: {(~events.absorbed).sum()}")
        return

    t_min = t_arrival_filtered.min()
    t_max = t_arrival_filtered.max()

    print(f"observer arrival times ({n_filtered} events after filtering):")
    print(f"  min: {t_min:.1f} day, max: {t_max:.1f} day")
    print(f"  energy range: [{energy_min:.2e}, {energy_max:.2e}] erg")

    times = args.time if isinstance(args.time, list) else [args.time]

    # validate times are within range
    for t in times:
        if t < t_min or t > t_max:
            print(
                f"warning: requested time {t:.1f} outside range [{t_min:.1f}, {t_max:.1f}]"
            )

    # convert distance override from Mpc to cm if provided
    distance_cm = args.distance * 3.086e24 if args.distance else None
    if distance_cm:
        print(
            f"  using distance override: {args.distance} Mpc ({distance_cm:.2e} cm)"
        )

    # single skymap
    if len(times) == 1:
        print(f"computing skymap at t={times[0]} day...")

        skymap = compute_skymap(
            events,
            meta,
            observer_angle=np.radians(args.observer_angle),
            time=times[0],
            energy_min=energy_min,
            energy_max=energy_max,
            n_theta=args.n_theta,
            n_phi=args.n_phi,
            time_window=args.time_window,
            distance_override=distance_cm,
        )

        intensity = np.array(skymap.intensity)
        print(f"computed {args.n_theta}x{args.n_phi} skymap")
        print(
            f"  intensity: min={intensity.min():.2e}, max={intensity.max():.2e}, sum={intensity.sum():.2e}"
        )

        if args.output:
            import h5py

            with h5py.File(args.output, "w") as f:
                f.create_dataset("theta", data=skymap.theta)
                f.create_dataset("phi", data=skymap.phi)
                f.create_dataset("intensity", data=skymap.intensity)
                f.attrs["time"] = skymap.time
                f.attrs["d_L"] = skymap.d_L
            print(f"saved skymap to {args.output}")

        if args.plot or args.save_fig:
            plot_skymap(skymap, save=args.save_fig, beam_fwhm_arcsec=args.beam)

    # animation
    else:
        print(f"computing skymap animation at {len(times)} times...")
        skymaps = []
        for t in times:
            print(f"  computing t={t:.2f} day...")
            skymap = compute_skymap(
                events,
                meta,
                observer_angle=np.radians(args.observer_angle),
                time=t,
                energy_min=energy_min,
                energy_max=energy_max,
                n_theta=args.n_theta,
                n_phi=args.n_phi,
                time_window=args.time_window,
                distance_override=distance_cm,
            )
            skymaps.append(skymap)

        print(f"creating animation with {len(skymaps)} frames...")

        if args.save_fig or args.plot:
            plot_skymap_animation(
                skymaps, save=args.save_fig, fps=args.fps, show=args.plot
            )


def execute_polarization(
    args: Namespace, remaining: Optional[list] = None
) -> None:
    """execute polarization subcommand"""
    from ...afterglow.plotting import plot_polarization
    from ...afterglow.postprocess import (
        compute_polarization,
        read_photon_events,
    )

    print(f"loading photon events from {args.events}...")
    events, meta = read_photon_events(args.events)

    print(
        f"computing polarization for observer angle {args.observer_angle}°..."
    )

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

        with h5py.File(args.output, "w") as f:
            f.create_dataset("times", data=pol.times)
            f.create_dataset(
                "polarization_degree", data=pol.polarization_degree
            )
            f.create_dataset("polarization_angle", data=pol.polarization_angle)
            f.create_dataset("stokes_Q", data=pol.stokes_Q)
            f.create_dataset("stokes_U", data=pol.stokes_U)
            f.create_dataset("stokes_V", data=pol.stokes_V)
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

        with h5py.File(args.output, "w") as f:
            f.create_dataset("frequencies", data=spec.frequencies)
            f.create_dataset("fluxes", data=spec.fluxes)
            f.attrs["time"] = spec.time
        print(f"saved spectrum to {args.output}")

    if args.plot or args.save_fig:
        plot_spectrum(spec, save=args.save_fig)
