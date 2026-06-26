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


def _is_event_catalog(path: str) -> bool:
    """true if `path` is a photon-event catalog (from `afterglow generate`) rather than a
    hydro checkpoint. catalogs carry the `n_events` root attr / the `t_emission` dataset;
    a v2.0 checkpoint carries `format_version`."""
    import h5py

    with h5py.File(path, "r") as f:
        return "n_events" in f.attrs or "t_emission" in f


def _report_spec_sources(near, observer_arg):
    """print where the system + observer specs are coming from (discovered next to the
    data / explicit / built-in defaults), so units provenance is visible at a glance."""
    from pathlib import Path

    from ...afterglow.spec import OBSERVER_PARAMS_NAME, SYSTEM_MANIFEST_NAME

    data_dir = Path(near).resolve().parent
    sys_path = data_dir / SYSTEM_MANIFEST_NAME
    obs_path = Path(observer_arg) if observer_arg else data_dir / OBSERVER_PARAMS_NAME
    print(f"system.yaml  : {sys_path if sys_path.is_file() else '(none -> --scale fallback)'}")
    print(
        f"observer.yaml: {obs_path if obs_path.is_file() else '(none -> defaults: 10 pc, p=2.5)'}"
    )


def _load_catalog(path, args, rad_hydro, observer, manifest):
    """load a photon catalog from one input: a saved catalog read back directly, or
    generated in-process from a hydro checkpoint. returns the opaque PhotonEvents handle."""
    if _is_event_catalog(path):
        print(f"reading photon catalog {path}...")
        return rad_hydro.read_photon_events(path)

    from simbi.reader import read_simulation

    from ...afterglow.inputs import build_fields, build_mesh

    data = read_simulation(path)
    sim_cond = {
        "dt": data.metadata.dt,
        "theta_obs": np.deg2rad(args.observer_angle),
        "adiabatic_index": data.metadata.gamma,
        "current_time": data.metadata.time,
        "p": observer.p,
        "z": observer.redshift,
        "eps_e": observer.eps_e,
        "eps_b": observer.eps_b,
        "d_L": observer.luminosity_distance_cm(),
        "nus": list(observer.frequencies),
    }
    print(f"generating synchrotron catalog from {path}...")
    return rad_hydro.generate_photon_events(
        sim_cond=sim_cond,
        qscales=manifest.to_qscales(),
        fields=build_fields(data),
        mesh=build_mesh(data),
        max_events=getattr(args, "max_events", 1_000_000),
        photons_per_cell=getattr(args, "photons_per_cell", 0),
    )


def _combine_catalogs(paths, args, rad_hydro, observer, manifest):
    """load each input (checkpoint or catalog) and merge into ONE handle, so a movie can
    span the epochs the user provides. each checkpoint carries its own scale_factor_a, so
    the lab-frame radii are epoch-consistent."""
    catalog = None
    for path in paths:
        part = _load_catalog(path, args, rad_hydro, observer, manifest)
        if catalog is None:
            catalog = part
        else:
            catalog.extend(part)
    return catalog


def _plot_skymap(
    image,
    half_mas,
    title,
    save_fig=None,
    show=False,
    vmax=None,
    cbar_label="surface brightness [mJy/mas$^2$]",
):
    """render a skymap with milliarcsecond axes relative to the image center."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = ax.imshow(
        image,
        origin="lower",
        cmap="inferno",
        extent=[-half_mas, half_mas, -half_mas, half_mas],
        vmax=vmax,
    )
    ax.set_xlabel("relative R.A. [mas]")
    ax.set_ylabel("relative Dec. [mas]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    if save_fig:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"saved figure to {save_fig}")
    if show:
        plt.show()
    plt.close(fig)


def _catalog_arrival_window(path, nhat, redshift, power=3.0, lo=0.02, hi=0.98):
    """flux-weighted [lo, hi] percentile observer-arrival times [day] for a catalog,
    used to auto-range a movie sweep so frames land where there is emission."""
    from ...afterglow.postprocess import read_photon_events

    ev, _ = read_photon_events(path)
    c = 2.997924580e10
    day = 86400.0
    r_dot_n = ev.x * nhat[0] + ev.y * nhat[1] + ev.z * nhat[2]
    t_arr = (1.0 + redshift) * (ev.t_emission - r_dot_n / c) / day
    weight = ev.stokes_I * ev.doppler_factor**power
    order = np.argsort(t_arr)
    cw = np.cumsum(weight[order])
    if cw[-1] <= 0.0:
        return float(t_arr.min()), float(t_arr.max())
    cw = cw / cw[-1]
    t_lo = t_arr[order[min(np.searchsorted(cw, lo), len(order) - 1)]]
    t_hi = t_arr[order[min(np.searchsorted(cw, hi), len(order) - 1)]]
    return float(t_lo), float(t_hi)


def _handle_arrival_window(catalog, rad_hydro, manifest, observer, nhat, power):
    """flux-weighted arrival window [day] for an in-memory merged handle: dump it to a
    throwaway catalog and reuse `_catalog_arrival_window`, so the auto-range spans every
    merged epoch rather than a single input."""
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    os.remove(tmp)  # keep the unique name; the writer creates the file fresh
    sim_cond = {
        "dt": 1.0,
        "theta_obs": 0.0,
        "adiabatic_index": 4.0 / 3.0,
        "current_time": 0.0,
        "p": observer.p,
        "z": observer.redshift,
        "eps_e": observer.eps_e,
        "eps_b": observer.eps_b,
        "d_L": observer.luminosity_distance_cm(),
        "nus": list(observer.frequencies),
    }
    try:
        rad_hydro.write_photon_events(tmp, catalog, sim_cond, manifest.to_qscales())
        return _catalog_arrival_window(tmp, nhat, observer.redshift, power)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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

    # subcommand: movie
    setup_movie_parser(afterglow_subparsers)

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
        help="hydro checkpoint HDF5 file (read -> in-process synchrotron catalog "
        "-> sky-plane reduce at the line of sight)",
    )

    parser.add_argument(
        "--observer",
        default=None,
        help="observer yaml (redshift, luminosity_distance, microphysics, "
        "frequencies); defaults to 10 pc / p=2.5 if omitted",
    )
    parser.add_argument(
        "--scale",
        default="blandford-mckee",
        help="fallback code->cgs scale model when no system.yaml sits next to the "
        "input (the manifest is preferred)",
    )
    parser.add_argument("--observer-angle", type=float, default=0.0, help="viewing angle [deg]")
    parser.add_argument(
        "--max-events", type=int, default=1000000, help="max photon packets to generate"
    )
    parser.add_argument(
        "--photons-per-cell", type=int, default=0, help="sampling density (0=auto)"
    )

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
        d_L=getattr(args, "d_l", None) or 1e28,
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
    """resolved sky-plane image at an observer time + viewing angle. the input is either a
    hydro checkpoint (read -> in-process synchrotron catalog) OR a saved photon-event catalog
    from `afterglow generate` (read back directly, generate-once-reduce-many); it is reduced
    onto the sky plane perpendicular to the line of sight, the observer-direction doppler boost
    recomputed per angle, so one catalog serves any `--observer-angle`."""
    from simbi.libs import cpu_ext as rad_hydro

    from ...afterglow.spec import ObserverParams, SystemManifest

    _report_spec_sources(args.checkpoint, args.observer)
    observer = ObserverParams.resolve(args.observer, near=args.checkpoint)
    manifest = SystemManifest.resolve(args.checkpoint, scale_fallback=args.scale)
    catalog = _load_catalog(args.checkpoint, args, rad_hydro, observer, manifest)

    theta = np.deg2rad(args.observer_angle)
    # line of sight in the (x, z) plane at polar angle theta from the symmetry axis.
    nhat = [float(np.sin(theta)), 0.0, float(np.cos(theta))]
    doppler_power = 4.0 if args.bolometric else 3.0
    print(
        f"reducing skymap at t={args.time} day, angle={args.observer_angle} deg "
        f"({len(catalog)} packets)..."
    )
    intensity, n_pix, half_width = rad_hydro.skymap_from_events(
        catalog,
        nhat,
        args.time,
        time_window=args.time_window,
        redshift=observer.redshift,
        doppler_power=doppler_power,
        n_pix=args.n_pix,
    )
    image = np.array(intensity).reshape(n_pix, n_pix)
    half_mas = observer.length_to_mas(half_width)

    if image.max() <= 0.0:
        print(
            f"  empty image: no photons arrived within +/-{args.time_window / 2:g} day of "
            f"t={args.time} day. pick a --time inside the catalog's arrival window "
            "(or widen --time-window)."
        )
        return

    from ...afterglow.spec import calibrate_skymap

    nu = observer.frequencies[0]
    sb, flux_mjy = calibrate_skymap(image, half_width, observer, args.time_window, nu)
    print(
        f"computed {n_pix}x{n_pix} skymap: half_width={half_mas:.3f} mas, "
        f"F_nu={flux_mjy:.4g} mJy at {nu:.2e} Hz (peak {sb.max():.3g} mJy/mas^2)"
    )

    if args.output:
        import h5py

        with h5py.File(args.output, "w") as f:
            f.create_dataset("surface_brightness", data=sb)  # mJy/mas^2
            f.create_dataset("intensity", data=image)  # raw beamed energy/cm^2
            f.attrs["time"] = args.time
            f.attrs["n_pix"] = n_pix
            f.attrs["half_width_cm"] = half_width
            f.attrs["half_width_mas"] = half_mas
            f.attrs["frequency_hz"] = nu
            f.attrs["flux_mjy"] = flux_mjy
        print(f"saved skymap to {args.output}")

    if args.plot or args.save_fig:
        _plot_skymap(
            sb,
            half_mas,
            f"t = {args.time} day   ({nu:.1e} Hz, {flux_mjy:.3g} mJy)",
            args.save_fig,
            args.plot,
            cbar_label="surface brightness [mJy/mas$^2$]",
        )


def setup_movie_parser(subparsers) -> None:
    """observer-time-sweep skymap movie from a single catalog"""
    parser = subparsers.add_parser(
        "movie",
        help="observer-time-sweep skymap movie (one catalog, many frames)",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="one or more checkpoints OR catalogs; multiple are MERGED into one catalog "
        "so the movie spans the epochs you provide (the physically-correct EATS movie)",
    )
    parser.add_argument("--observer", default=None, help="observer yaml (z, distance, ...)")
    parser.add_argument(
        "--scale", default="blandford-mckee", help="fallback scale model (system.yaml preferred)"
    )
    parser.add_argument("--observer-angle", type=float, default=0.0, help="viewing angle [deg]")
    parser.add_argument(
        "--max-events", type=int, default=1_000_000, help="max packets per checkpoint generated"
    )
    parser.add_argument(
        "--photons-per-cell", type=int, default=0, help="sampling density (0=auto)"
    )
    parser.add_argument(
        "--t-start", type=float, default=None, help="first observer time [day] (auto if omitted)"
    )
    parser.add_argument(
        "--t-stop", type=float, default=None, help="last observer time [day] (auto if omitted)"
    )
    parser.add_argument("--n-frames", type=int, default=60, help="number of frames")
    parser.add_argument("--log-time", action="store_true", help="log-spaced observer times")
    parser.add_argument(
        "--time-window", type=float, default=None, help="per-frame window [day] (auto: frame spacing)"
    )
    parser.add_argument("--n-pix", type=int, default=128, help="image resolution")
    parser.add_argument(
        "--bolometric", action="store_true", help="bolometric beaming (doppler^4)"
    )
    parser.add_argument("--log-color", action="store_true", help="log color scale across frames")
    parser.add_argument("--fps", type=int, default=12, help="frames per second")
    parser.add_argument(
        "--output", default="skymap_movie.mp4", help="output movie (.mp4 via ffmpeg, or .gif)"
    )
    parser.set_defaults(func=execute_movie)


def execute_movie(args: Namespace, remaining: Optional[list] = None) -> None:
    """assemble an apparent-image movie from the N inputs (checkpoints/catalogs) the user
    provides. all inputs are MERGED into one catalog so each observer-time frame integrates
    the right emission epochs (a single checkpoint gives a degenerate thin ring); the sweep
    runs in a fixed mas field of view so the ring visibly grows and brightens."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.colors import LogNorm, Normalize

    from simbi.libs import cpu_ext as rad_hydro

    from ...afterglow.spec import ObserverParams, SystemManifest

    _report_spec_sources(args.inputs[0], args.observer)
    observer = ObserverParams.resolve(args.observer, near=args.inputs[0])
    manifest = SystemManifest.resolve(args.inputs[0], scale_fallback=args.scale)
    theta = np.deg2rad(args.observer_angle)
    nhat = [float(np.sin(theta)), 0.0, float(np.cos(theta))]
    doppler_power = 4.0 if args.bolometric else 3.0

    print(f"merging {len(args.inputs)} input(s) into one catalog...")
    catalog = _combine_catalogs(args.inputs, args, rad_hydro, observer, manifest)

    # auto-range the sweep to where the flux is, unless the user pinned the endpoints. the
    # window is read from the MERGED handle via a throwaway catalog dump (so it spans all
    # provided epochs, not just one input).
    if args.t_start is None or args.t_stop is None:
        t0, t1 = _handle_arrival_window(
            catalog, rad_hydro, manifest, observer, nhat, doppler_power
        )
    else:
        t0, t1 = args.t_start, args.t_stop
    t_start = args.t_start if args.t_start is not None else t0
    t_stop = args.t_stop if args.t_stop is not None else t1
    if args.log_time:
        times = np.geomspace(max(t_start, 1e-3), max(t_stop, 1e-2), args.n_frames)
    else:
        times = np.linspace(t_start, t_stop, args.n_frames)
    window = (
        args.time_window
        if args.time_window is not None
        else float(np.diff(times).mean() * 2.0)
    )

    print(
        f"sweeping {args.n_frames} frames over t=[{t_start:.1f}, {t_stop:.1f}] day "
        f"(window {window:.1f} day, angle {args.observer_angle:g} deg)..."
    )

    from ...afterglow.spec import calibrate_skymap

    nu = observer.frequencies[0]
    frames = []  # (surface_brightness [mJy/mas^2], half_mas, t_day, flux_mjy)
    for t in times:
        intensity, n_pix, half_width = rad_hydro.skymap_from_events(
            catalog,
            nhat,
            float(t),
            time_window=window,
            redshift=observer.redshift,
            doppler_power=doppler_power,
            n_pix=args.n_pix,
        )
        img = np.array(intensity).reshape(n_pix, n_pix)
        sb, flux_mjy = calibrate_skymap(img, half_width, observer, window, nu)
        frames.append((sb, observer.length_to_mas(half_width), float(t), flux_mjy))

    half_max = max((hm for _, hm, _, _ in frames if hm > 0), default=1.0)
    vmax = max((sb.max() for sb, _, _, _ in frames), default=0.0)
    if vmax <= 0.0:
        raise SystemExit("no emission in any frame; widen --t-start/--t-stop or --time-window.")

    # fixed mas field of view (half_max) so the ring expands within the frame; each image
    # is drawn at its true extent inside that fixed window.
    norm = LogNorm(vmax * 1e-3, vmax) if args.log_color else Normalize(0.0, vmax)
    fig, ax = plt.subplots()
    im = ax.imshow(
        frames[0][0],
        origin="lower",
        cmap="inferno",
        norm=norm,
        extent=[-half_max, half_max, -half_max, half_max],
    )
    ax.set_xlim(-half_max, half_max)
    ax.set_ylim(-half_max, half_max)
    ax.set_xlabel("relative R.A. [mas]")
    ax.set_ylabel("relative Dec. [mas]")
    fig.colorbar(im, ax=ax, label="surface brightness [mJy/mas$^2$]")
    title = ax.set_title("")

    def update(idx):
        sb, hmm, t, flux_mjy = frames[idx]
        extent = hmm if hmm > 0 else half_max
        im.set_data(sb)
        im.set_extent([-extent, extent, -extent, extent])
        title.set_text(f"t = {t:.1f} day   F$_\\nu$ = {flux_mjy:.3g} mJy")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = (
        PillowWriter(fps=args.fps)
        if args.output.endswith(".gif")
        else FFMpegWriter(fps=args.fps, bitrate=2400)
    )
    anim.save(args.output, writer=writer, dpi=120)
    plt.close(fig)
    print(f"saved {len(frames)}-frame movie to {args.output}")


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
