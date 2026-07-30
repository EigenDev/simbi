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


def _generate_catalog(path, args, rad_hydro, observer, manifest, emit_dt=None):
    """generate a photon catalog in-process from a hydro checkpoint. `emit_dt` is the lab-time
    interval [code units] this snapshot REPRESENTS in a multi-checkpoint array (its trapezoidal
    share of the snapshot-time axis); omitted, the CFL step is the lone-snapshot fallback —
    weighting an array member by the CFL dt undercounts its emitted energy by the cadence/CFL
    ratio, typically ~1e5. returns the handle."""
    from simbi.reader import read_simulation

    from ...afterglow.inputs import build_fields, build_mesh

    data = read_simulation(path)
    sim_cond = {
        "dt": emit_dt if emit_dt and emit_dt > 0.0 else data.metadata.dt,
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


def _load_catalog(path, args, rad_hydro, observer, manifest):
    """load a photon catalog from one input: a saved catalog read back WHOLE, or generated
    in-process from a hydro checkpoint. returns the opaque PhotonEvents handle. for
    bounded-memory reduction of a large catalog, prefer `_iter_event_handles` (row-chunks)."""
    if _is_event_catalog(path):
        print(f"reading photon catalog {path}...")
        return rad_hydro.read_photon_events(path)
    return _generate_catalog(path, args, rad_hydro, observer, manifest)


def _iter_event_handles(path, args, rad_hydro, observer, manifest, chunk_size=4_000_000, emit_dt=None):
    """yield PhotonEvents handles for one input: a saved catalog is read in row-CHUNKS of
    `chunk_size` (O(chunk) memory, not O(file)), each yielded then discarded; a hydro
    checkpoint yields a single generated handle. additive reductions (skymap/lightcurve)
    accumulate across the chunks, so a huge generate-once events file reduces without being
    held whole."""
    if _is_event_catalog(path):
        n = rad_hydro.photon_event_count(path)
        print(f"reading photon catalog {path} ({n} events, {chunk_size}/chunk)...")
        for start in range(0, max(n, 1), chunk_size):
            cat = rad_hydro.read_photon_events_chunk(path, start, chunk_size)
            if len(cat) == 0:
                break
            try:
                yield cat
            finally:
                del cat
    else:
        yield _generate_catalog(path, args, rad_hydro, observer, manifest, emit_dt=emit_dt)


def _stream_skymap(paths, args, rad_hydro, observer, manifest, obs_time, nhat, doppler_power):
    """reduce each checkpoint into the image accumulator on a SHARED grid and DISCARD it, so
    memory is O(n_pix^2) regardless of checkpoint count (fine time binning -> smooth images
    without holding every photon). the shared field of view is --fov [mas] (one pass) or a
    two-pass auto (pass 1 finds the max extent, pass 2 accumulates). returns (image_2d, half_cm)."""
    from ...afterglow.generate import _read_snapshot_time, _snapshot_emission_durations

    n_pix = args.n_pix
    # each checkpoint emits over its trapezoidal share of the snapshot-time axis.
    hydro = sorted((p for p in paths if not _is_event_catalog(p)), key=_read_snapshot_time)
    durations = dict(
        zip(hydro, _snapshot_emission_durations([_read_snapshot_time(p) for p in hydro]))
    )

    def _reduce(path, half_width_cm):
        # accumulate over the input's event handles: a saved catalog streams in row-chunks
        # (bounded memory), a checkpoint is one handle. with a FIXED half_width the chunks
        # share the grid so summing intensities is exact; in the sizing pass (half_width=0)
        # the per-chunk grids differ but only the max extent `hw` is used.
        total = None
        hw_max = 0.0
        for cat in _iter_event_handles(
            path, args, rad_hydro, observer, manifest, emit_dt=durations.get(path)
        ):
            intensity, _, hw = rad_hydro.skymap_from_events(
                cat, nhat, obs_time, time_window=args.time_window, redshift=observer.redshift,
                doppler_power=doppler_power, n_pix=n_pix, half_width=half_width_cm,
                frequency=float(observer.frequencies[0]), frac_bandwidth=0.1,
            )
            arr = np.asarray(intensity)
            total = arr if total is None else total + arr
            hw_max = max(hw_max, hw)
        if total is None:
            total = np.zeros(n_pix * n_pix)
        return total, hw_max

    if args.fov is not None:
        half_width_cm = observer.mas_to_length(args.fov)
    else:
        print("  pass 1/2: sizing the shared field of view...")
        half_width_cm = 0.0
        for path in paths:
            _, hw = _reduce(path, 0.0)
            half_width_cm = max(half_width_cm, hw)
        if half_width_cm <= 0.0:
            return np.zeros((n_pix, n_pix)), 0.0

    print(f"  accumulating {len(paths)} checkpoints onto a fixed {n_pix}x{n_pix} grid...")
    image = np.zeros(n_pix * n_pix)
    for path in paths:
        intensity, _ = _reduce(path, half_width_cm)
        image += intensity
    return image.reshape(n_pix, n_pix), half_width_cm


def _combine_catalogs(paths, args, rad_hydro, observer, manifest):
    """load each input (checkpoint or catalog) and merge into ONE handle, so a movie can
    span the epochs the user provides. each checkpoint carries its own scale_factor_a, so
    the lab-frame radii are epoch-consistent; each emits over its trapezoidal share of the
    snapshot-time axis (not the CFL dt)."""
    from ...afterglow.generate import _read_snapshot_time, _snapshot_emission_durations

    hydro = sorted((p for p in paths if not _is_event_catalog(p)), key=_read_snapshot_time)
    durations = dict(
        zip(hydro, _snapshot_emission_durations([_read_snapshot_time(p) for p in hydro]))
    )
    catalog = None
    for path in paths:
        if _is_event_catalog(path):
            part = _load_catalog(path, args, rad_hydro, observer, manifest)
        else:
            part = _generate_catalog(
                path, args, rad_hydro, observer, manifest, emit_dt=durations.get(path)
            )
        if catalog is None:
            catalog = part
        else:
            catalog.extend(part)
    return catalog


def _flux_diagnostics(image, half_mas):
    """intensity-weighted flux centroid (xc, yc) and marginal FWHM (fwhm_x, fwhm_y) in
    mas, computed from the brightness distribution (paper-style image diagnostics)."""
    n = image.shape[0]
    coord = np.linspace(-half_mas, half_mas, n)  # pixel-center axis [mas]
    prof_x = image.sum(axis=0)  # marginal over y -> longitudinal profile I(x)
    prof_y = image.sum(axis=1)  # marginal over x -> transverse profile I(y)
    if prof_x.sum() <= 0.0:
        return None
    xc = float((coord * prof_x).sum() / prof_x.sum())
    yc = float((coord * prof_y).sum() / prof_y.sum())

    def _fwhm(prof):
        above = coord[prof >= 0.5 * prof.max()]
        return float(above.max() - above.min()) if above.size else 0.0

    return xc, yc, _fwhm(prof_x), _fwhm(prof_y)


def _smooth_image(image, sigma_px):
    """gaussian-smooth a surface-brightness image (zrake+2018 convolve each frame with a small
    kernel) so monte-carlo photon SPECKLE becomes a continuous gradient. sigma in pixels; the
    convolution conserves the total flux, so F_nu is unchanged. <=0 is a no-op."""
    if sigma_px and sigma_px > 0.0:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(np.asarray(image, dtype=float), sigma_px)
    return image


_RAD_TO_MAS = 206_264_806.247_096_36
_JY_CGS = 1.0e-23


def _calibrate_deposit(image, half_width_cm, observer, time_window_day):
    """calibrate the raw DETERMINISTIC-deposit image (sum of delta^p * j_nu * dV * dt_lab per pixel,
    erg/Hz) to surface brightness [mJy/mas^2] + integrated F_nu [mJy]. the deposit is already a
    MONOCHROMATIC (per-Hz) emissivity, so -- unlike the monte-carlo path -- there is NO 1/dnu:
    F_nu = (beamed monochromatic energy) / (4 pi d_L^2 dt_obs)."""
    n_pix = image.shape[0]
    d_l = observer.luminosity_distance_cm()
    d_a = observer.angular_diameter_distance_cm()
    dt_s = time_window_day * 86400.0
    px_cm = 2.0 * half_width_cm / n_pix if half_width_cm > 0.0 else 1.0
    denom = 4.0 * np.pi * d_l * d_l * dt_s
    flux_total_mjy = float(image.sum()) / denom / _JY_CGS * 1.0e3
    px_mas = px_cm / d_a * _RAD_TO_MAS
    surface_brightness = image / (px_mas * px_mas) / denom / _JY_CGS * 1.0e3
    return surface_brightness, flux_total_mjy


def _progress(iterable, total, label):
    """lightweight carriage-return progress for long multi-snapshot loops (no extra deps).
    writes count / % / rate / ETA to stderr so it overwrites in place and never pollutes stdout."""
    import sys
    import time

    t0 = time.time()
    step = max(1, total // 200)
    for ii, item in enumerate(iterable, 1):
        yield item
        if ii == total or ii % step == 0:
            dt = time.time() - t0
            rate = ii / dt if dt > 0 else 0.0
            eta = (total - ii) / rate if rate > 0 else 0.0
            sys.stderr.write(
                f"\r  {label}: {ii}/{total} ({100 * ii / total:3.0f}%)  "
                f"{rate:5.1f}/s  eta {eta:4.0f}s   "
            )
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()


def _deposit_skymap(paths, args, rad_hydro, observer, manifest, obs_time, nhat, frequency):
    """DETERMINISTIC deposition over the snapshot ARRAY: each hydro checkpoint deposits its lab-frame
    emissivity onto a SHARED sky grid via the EATS, accumulated. noise-free (no photon sampling) ->
    a continuous gradient. spherical checkpoints in 1/2/3D; the velocity components ride along so
    a laterally-spreading jet beams correctly. returns (image_2d, half_cm)."""
    from simbi.reader import read_simulation

    from ...afterglow.generate import _read_snapshot_time, _snapshot_emission_durations
    from ...afterglow.inputs import build_fields, build_mesh, build_velocity

    n_pix = args.n_pix
    qscales = manifest.to_qscales()

    # shared field of view [cm]: --fov, else auto-size with ONE cheap MC pass on the mid checkpoint.
    if args.fov is not None:
        half_width_cm = observer.mas_to_length(args.fov)
    else:
        mid = paths[len(paths) // 2]
        cat = _load_catalog(mid, args, rad_hydro, observer, manifest)
        try:
            _, _, half_width_cm = rad_hydro.skymap_from_events(
                cat, nhat, obs_time, time_window=args.time_window, redshift=observer.redshift,
                doppler_power=3.0, n_pix=n_pix, half_width=0.0,
            )
        finally:
            del cat
        if not half_width_cm or half_width_cm <= 0.0:
            raise SystemExit("could not auto-size the deposit grid; pass --fov MAS.")

    # sort snapshots by lab time and weight each by the interval it represents (trapezoid, code units).
    paths = sorted(paths, key=_read_snapshot_time)
    durations = _snapshot_emission_durations([_read_snapshot_time(p) for p in paths])

    print(f"depositing {len(paths)} snapshots onto a {n_pix}x{n_pix} grid (deterministic)...")
    image = np.zeros(n_pix * n_pix)
    for path, dt_code in _progress(zip(paths, durations), len(paths), "depositing"):
        data = read_simulation(path)
        sim_cond = {
            "dt": dt_code if dt_code > 0.0 else data.metadata.dt,
            "theta_obs": 0.0,
            "adiabatic_index": data.metadata.gamma,
            "current_time": data.metadata.time,
            "p": observer.p,
            "z": observer.redshift,
            "eps_e": observer.eps_e,
            "eps_b": observer.eps_b,
            "d_L": observer.luminosity_distance_cm(),
            "nus": [float(frequency)],
        }
        # velocity components ride with the fields so lateral spreading beams correctly.
        fields = build_fields(data) | build_velocity(data)
        img = rad_hydro.skymap_deposit(
            sim_cond, qscales, fields, build_mesh(data), list(nhat),
            float(obs_time), float(args.time_window), float(frequency), observer.redshift,
            float(half_width_cm), n_pix, 2.0,
        )
        image += np.asarray(img)
    return image.reshape(n_pix, n_pix), half_width_cm


def _draw_skymap_on_ax(ax, image, half_mas, log_decades=None, contours=False, diagnostics=False, vmax=None):
    """draw ONE skymap onto a given axes in the paper style: jet axis HORIZONTAL (x), thin
    grey axis lines through the origin, and (optionally) the yellow flux-centroid marker +
    FWHM error bars. returns the imshow handle. the image is transposed so the projected
    symmetry axis (sky-plane e2) runs horizontally, then flipped left<->right so the
    APPROACHING (doppler-beamed, +z) hemisphere sits on the LEFT, matching nedora+2023."""
    img = np.asarray(image).T[:, ::-1]  # jet -> horizontal, approaching side on the left
    extent = [-half_mas, half_mas, -half_mas, half_mas]

    if log_decades is not None:
        peak = img.max() if img.max() > 0 else 1.0
        disp = np.log10(np.where(img > 0, img / peak, 10.0 ** (-2 * log_decades)))
        disp = np.clip(disp, -log_decades, 0.0)
        im = ax.imshow(disp, origin="lower", cmap="magma", extent=extent, vmin=-log_decades, vmax=0.0)
        if contours:
            levels = np.linspace(-log_decades, 0.0, int(2 * log_decades) + 1)[1:-1]
            ax.contour(disp, levels=levels, colors="white", linewidths=0.4, extent=extent, origin="lower")
    else:
        im = ax.imshow(img, origin="lower", cmap="magma", extent=extent, vmax=vmax)
        if contours and img.max() > 0:
            levels = img.max() * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            ax.contour(img, levels=levels, colors="white", linewidths=0.4, extent=extent, origin="lower")

    # thin grey x/z axis lines through the origin.
    ax.axhline(0.0, color="0.6", lw=0.5, ls=":", alpha=0.7)
    ax.axvline(0.0, color="0.6", lw=0.5, ls=":", alpha=0.7)

    if diagnostics:
        diag = _flux_diagnostics(img, half_mas)
        if diag is not None:
            xc, yc, fwhm_x, fwhm_y = diag
            ax.errorbar(
                xc, yc, xerr=0.5 * fwhm_x, yerr=0.5 * fwhm_y, fmt="o", color="yellow",
                ecolor="yellow", elinewidth=1.2, capsize=4, markersize=7, zorder=5,
            )
            print(
                f"  centroid (x, z) = ({xc:.3f}, {yc:.3f}) mas; "
                f"FWHM (x, z) = ({fwhm_x:.3f}, {fwhm_y:.3f}) mas"
            )
    return im


def _plot_skymap(
    image, half_mas, title, save_fig=None, show=False, vmax=None,
    cbar_label="surface brightness [mJy/mas$^2$]",
    log_decades=None, contours=False, diagnostics=False, colorbar=True,
):
    """single skymap with mas axes (x = jet axis). morphology mode (log_decades) drops the
    colorbar by default — for morphology the shape is what matters, not the scale."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    im = _draw_skymap_on_ax(ax, image, half_mas, log_decades, contours, diagnostics, vmax)
    ax.set_xlabel("x [mas]")
    ax.set_ylabel("z [mas]")
    ax.set_title(title)
    if colorbar and log_decades is None:
        fig.colorbar(im, ax=ax, label=cbar_label)
    if save_fig:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"saved figure to {save_fig}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_skymap_panel(images, half_mas_list, titles, log_decades=None, contours=False, diagnostics=False, save_fig=None, show=False):
    """a ROW of skymaps (one per observer angle) on a common mas field of view — the
    multi-angle morphology comparison. no colorbar; shape is the point."""
    import matplotlib.pyplot as plt

    n = len(images)
    fov = max([h for h in half_mas_list if h > 0], default=1.0)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, img, hm, t in zip(axes, images, half_mas_list, titles):
        _draw_skymap_on_ax(ax, img, hm, log_decades, contours, diagnostics)
        ax.set_xlim(-fov, fov)
        ax.set_ylim(-fov, fov)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mas]")
        ax.set_title(t)
    axes[0].set_ylabel("z [mas]")
    fig.tight_layout()
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
        return float(t_arr.min()), float(np.median(t_arr)), float(t_arr.max())
    cw = cw / cw[-1]

    def _at(q):
        return float(t_arr[order[min(np.searchsorted(cw, q), len(order) - 1)]])

    # (lo percentile, flux-weighted MEDIAN/peak, hi percentile)
    return _at(lo), _at(0.5), _at(hi)


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
        help="hydro checkpoint files spanning the blast evolution -- the catalog is built "
        "from the ARRAY (one snapshot is a single emission epoch, not an afterglow)",
    )

    parser.add_argument(
        "--observer",
        default=None,
        help="observer yaml (redshift, luminosity_distance, microphysics); auto-discovered "
        "next to the data, else 10 pc / p=2.5 defaults. NOTE: the catalog is angle-independent "
        "-- the viewing angle is chosen later by skymap/lightcurve",
    )
    parser.add_argument(
        "--scale",
        default="blandford-mckee",
        help="fallback code->cgs scale model when no system.yaml sits next to the input",
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
        help="maximum number of events to generate (split across snapshots)",
    )

    parser.add_argument(
        "--photons-per-cell",
        type=int,
        default=0,
        help="photons per cell (0=auto)",
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
        help="hydro checkpoint(s) OR a saved photon-event catalog; the EATS is integrated "
        "over all inputs (each streamed then discarded — O(bins) memory)",
    )

    parser.add_argument(
        "--observer",
        default=None,
        help="observer yaml (redshift, luminosity_distance, microphysics, frequencies); "
        "auto-discovered next to the data, else 10 pc / p=2.5 defaults",
    )
    parser.add_argument(
        "--scale",
        default="blandford-mckee",
        help="fallback code->cgs scale model when no system.yaml sits next to the "
        "input (the manifest is preferred)",
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
        help="number of observer-time bins",
    )

    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        default=None,
        help="observer-time range [day] (auto 1e-3..1e3 if omitted)",
    )

    parser.add_argument(
        "--max-events", type=int, default=2_000_000,
        help="max photon packets generated PER checkpoint (streamed, then discarded)",
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
        "inputs",
        nargs="+",
        help="one or more checkpoints OR catalogs; multiple are MERGED so the EATS at a "
        "given observer time integrates all epochs (a single checkpoint covers only its "
        "own narrow arrival window)",
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
        default=None,
        help="observer time [day]; if omitted, defaults to the catalog's flux-peak "
        "arrival time (the window is computed and printed either way)",
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
        "--log-decades",
        type=float,
        default=3.0,
        help="show the FREQUENCY-INDEPENDENT log morphology log10(I/I_max) over this many "
        "decades (DEFAULT 3 -- log is the standard afterglow view). use --linear for "
        "absolute mJy/mas^2 on a linear scale.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="plot absolute mJy/mas^2 on a LINEAR scale instead of the default log morphology",
    )
    parser.add_argument(
        "--contours",
        action="store_true",
        help="overlay iso-intensity contour lines on the image",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="overlay the flux centroid (marker) + FWHM (error bars) and print them",
    )
    parser.add_argument(
        "--observer-angles",
        nargs="+",
        type=float,
        default=None,
        help="multiple viewing angles [deg] -> a ROW of morphology panels (e.g., 15 45 75), "
        "the catalog reduced once per angle",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="fixed image half-width [mas] for MULTI-checkpoint streaming (one pass); if "
        "omitted, a two-pass auto sizes it. fixes the shared grid so frames accumulate.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="save skymap data to file",
    )

    parser.add_argument(
        "--method",
        choices=["mc", "deposit"],
        default=None,
        help="reduction method; auto-selected when omitted (deposit for hydro checkpoints, "
        "mc for a saved events catalog). 'deposit': DETERMINISTIC cell deposition "
        "(zrake+2018) -- noise-free, publication-clean images from spherical checkpoints in "
        "1/2/3D, optically-thin synchrotron only. 'mc': monte-carlo photon catalog -- works "
        "on events files OR checkpoints and carries scattering/absorption/polarization, but "
        "needs many photons + smoothing for clean images.",
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=2.1,
        help="gaussian smoothing sigma in PIXELS. DEFAULT 2.1 = zrake+2018's 5-px (300 uas) kernel, "
        "applied to EVERY image -- it absorbs the monte-carlo speckle AND the on-axis azimuthal "
        "tessellation spokes (which zrake calls 'redundant for on-axis observers'). 0 disables.",
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
    """build the canonical photon-events catalog from an ARRAY of snapshots. the same
    yaml spec as skymap/lightcurve drives units + microphysics; the catalog is angle-
    INDEPENDENT (the line of sight is chosen later at reduction)."""
    from ...afterglow.generate import generate_from_files
    from ...afterglow.spec import ObserverParams, SystemManifest

    _report_spec_sources(args.files[0], args.observer)
    observer = ObserverParams.resolve(args.observer, near=args.files[0])
    manifest = SystemManifest.resolve(args.files[0], scale_fallback=args.scale)

    print(f"generating photon events from {len(args.files)} snapshot(s)...")
    generate_from_files(
        files=args.files,
        output=args.output,
        max_events=args.max_events,
        photons_per_cell=args.photons_per_cell,
        eps_e=observer.eps_e,
        eps_b=observer.eps_b,
        p=observer.p,
        theta_obs=0.0,  # catalog is angle-independent; the angle is set at reduction
        z=observer.redshift,
        d_L=observer.luminosity_distance_cm(),
        apply_mcrt=args.mcrt,
        include_scattering=not args.no_scattering,
        qscales=manifest.to_qscales(),
    )
    print(f"saved photon events to {args.output}")


def execute_lightcurve(args: Namespace, remaining: Optional[list] = None) -> None:
    """observer light curve F_nu(t), STREAMED over checkpoints via the cpu_ext catalog path:
    each checkpoint -> in-process synchrotron catalog -> EATS reduce into the time bins ->
    accumulate -> DISCARD the events. memory is O(bins), not O(total events), so it scales to
    many checkpoints. inherits the lab-radius / 2d-revolve / units fixes (one afterglow path)."""
    import h5py

    from ...afterglow.lightcurve import stream_lightcurve
    from ...afterglow.spec import ObserverParams, SystemManifest

    # same yaml-driven spec as `skymap`: the observer yaml (redshift, luminosity_distance,
    # microphysics, frequencies) is the single source of truth, with built-in defaults
    # (10 pc, p=2.5, eps_e=0.1, eps_b=0.01, 1 GHz) when none is found.
    _report_spec_sources(args.files[0], args.observer)
    observer = ObserverParams.resolve(args.observer, near=args.files[0])
    manifest = SystemManifest.resolve(args.files[0], scale_fallback=args.scale)

    freqs = [float(f) for f in observer.frequencies]
    d_l = observer.luminosity_distance_cm()
    micro = {"p": observer.p, "eps_e": observer.eps_e, "eps_b": observer.eps_b}
    if args.time_range:
        time_edges = np.geomspace(args.time_range[0], args.time_range[1], args.n_bins + 1)
    else:
        time_edges = np.geomspace(1e-3, 1e3, args.n_bins + 1)

    print(f"computing light curve, streaming {len(args.files)} input(s)...")
    times, total, freqs_arr = stream_lightcurve(
        list(args.files), manifest.to_qscales(), micro, np.deg2rad(args.observer_angle),
        freqs, observer.redshift, d_l, [float(t) for t in time_edges], max_events=args.max_events,
    )
    freqs = list(freqs_arr)
    fluxes = total.reshape(len(times), len(freqs))[: args.n_bins]
    times = times[: args.n_bins]
    print(f"computed {len(times)} time bins x {len(freqs)} frequencies (streamed)")

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

    _report_spec_sources(args.inputs[0], args.observer)
    observer = ObserverParams.resolve(args.observer, near=args.inputs[0])
    manifest = SystemManifest.resolve(args.inputs[0], scale_fallback=args.scale)
    doppler_power = 4.0 if args.bolometric else 3.0
    multi = len(args.inputs) > 1
    # log morphology is the DEFAULT afterglow view; --linear opts into absolute mJy/mas^2.
    if getattr(args, "linear", False):
        args.log_decades = None

    # method auto-selection: deposit is the default imager for hydro checkpoints (noise-free);
    # a saved events catalog has no cells to deposit, so it routes to the mc reducer.
    method = getattr(args, "method", None)
    if method is None:
        method = "mc" if _is_event_catalog(args.inputs[0]) else "deposit"
        print(f"method: {method} (auto; override with --method)")

    # DETERMINISTIC deposition path (zrake+2018): no photon catalog, no shot noise. operates on
    # the hydro CHECKPOINT array directly, depositing each cell's emissivity onto a shared grid.
    if method == "deposit":
        if _is_event_catalog(args.inputs[0]):
            raise SystemExit(
                "--method deposit needs hydro CHECKPOINTS, not an events file (it deposits cell "
                "emissivity directly). use --method mc for an events file."
            )
        if args.time is None:
            raise SystemExit(
                "--method deposit needs --time (the EATS observer time). run a quick "
                "`--method mc` skymap first — it prints the arrival window + flux peak."
            )
        th = np.deg2rad(args.observer_angle)
        nh = [float(np.sin(th)), 0.0, float(np.cos(th))]
        nu = observer.frequencies[0]
        image, half_width = _deposit_skymap(
            args.inputs, args, rad_hydro, observer, manifest, args.time, nh, nu
        )
        sb, flux_mjy = _calibrate_deposit(image, half_width, observer, args.time_window)
        sb = _smooth_image(sb, args.smooth)
        half_mas = observer.length_to_mas(half_width)
        print(
            f"computed {args.n_pix}x{args.n_pix} DEPOSIT skymap: half_width={half_mas:.3f} mas, "
            f"F_nu={flux_mjy:.4g} mJy at {nu:.2e} Hz (peak {sb.max():.3g} mJy/mas^2)"
        )
        if args.output:
            import h5py

            with h5py.File(args.output, "w") as f:
                f.create_dataset("surface_brightness", data=sb)
                f.attrs["time"] = args.time
                f.attrs["half_width_mas"] = half_mas
                f.attrs["frequency_hz"] = nu
                f.attrs["flux_mjy"] = flux_mjy
                f.attrs["method"] = "deposit"
            print(f"saved skymap to {args.output}")
        if args.plot or args.save_fig:
            title = (
                f"t = {args.time:.3g} day   ({nu:.1e} Hz)"
                if args.log_decades is not None
                else f"t = {args.time:.3g} day   ({nu:.1e} Hz, {flux_mjy:.3g} mJy)"
            )
            _plot_skymap(
                sb, half_mas, title, args.save_fig, args.plot,
                cbar_label="surface brightness [mJy/mas$^2$]",
                log_decades=args.log_decades, contours=args.contours,
                diagnostics=args.diagnostics,
            )
        return

    if multi:
        # STREAM: never merge all checkpoints' events (the memory blowup). the EATS observer
        # time can't be auto-found without a pass over the data, so it is required here.
        if args.time is None:
            raise SystemExit(
                "multi-checkpoint skymap needs --time (the EATS observer time) — find it from a "
                "single-checkpoint run's reported window, or the lightcurve peak."
            )
        obs_time = args.time
        catalog = None
        fov = f"--fov {args.fov:g} mas" if args.fov is not None else "two-pass auto fov"
        print(f"streaming {len(args.inputs)} checkpoints at t={obs_time:.3g} day ({fov})...")
    else:
        catalog = _load_catalog(args.inputs[0], args, rad_hydro, observer, manifest)
        primary = args.observer_angles[0] if args.observer_angles else args.observer_angle
        th0 = np.deg2rad(primary)
        nhat0 = [float(np.sin(th0)), 0.0, float(np.cos(th0))]
        t_lo, t_peak, t_hi = _handle_arrival_window(
            catalog, rad_hydro, manifest, observer, nhat0, doppler_power
        )
        obs_time = args.time if args.time is not None else t_peak
        print(
            f"EATS arrival window: [{t_lo:.3g}, {t_hi:.3g}] day, flux peak ~{t_peak:.3g} day "
            f"-> imaging t={obs_time:.3g} day"
        )
        if not (t_lo <= obs_time <= t_hi):
            print(
                f"  note: t={obs_time:.3g} day is OUTSIDE the window; image will be faint/empty. "
                "merge more checkpoints to reach it, or pick a t in range."
            )

    def _image_at_angle(angle):
        """(image_2d, half_width_cm) at a viewing angle: stream over inputs (multi, O(n_pix^2)
        memory) or reduce the single loaded catalog."""
        th = np.deg2rad(angle)
        nh = [float(np.sin(th)), 0.0, float(np.cos(th))]
        if multi:
            return _stream_skymap(
                args.inputs, args, rad_hydro, observer, manifest, obs_time, nh, doppler_power
            )
        intensity, npx, hw = rad_hydro.skymap_from_events(
            catalog, nh, obs_time, time_window=args.time_window, redshift=observer.redshift,
            doppler_power=doppler_power, n_pix=args.n_pix, half_width=0.0,
            frequency=float(observer.frequencies[0]), frac_bandwidth=0.1,
        )
        return np.asarray(intensity).reshape(npx, npx), hw

    # multi-angle morphology panel.
    if args.observer_angles:
        images, half_mas_list, titles = [], [], []
        for angle in args.observer_angles:
            img, hw = _image_at_angle(angle)
            images.append(_smooth_image(img, args.smooth))
            half_mas_list.append(observer.length_to_mas(hw))
            titles.append(rf"$\theta_{{\rm obs}}$ = {angle:g}$^\circ$")
        print(f"panel: {len(args.observer_angles)} angles at t={obs_time:.3g} day")
        _plot_skymap_panel(
            images, half_mas_list, titles,
            log_decades=args.log_decades if args.log_decades is not None else 2.0,
            contours=args.contours, diagnostics=args.diagnostics,
            save_fig=args.save_fig, show=args.plot,
        )
        return

    image, half_width = _image_at_angle(args.observer_angle)
    n_pix = image.shape[0]
    half_mas = observer.length_to_mas(half_width)

    if image.max() <= 0.0:
        print(
            f"  empty image at t={obs_time:.3g} day: no photons in +/-{args.time_window / 2:g} day. "
            f"the flux-peak time is ~{t_peak:.3g} day (window [{t_lo:.3g}, {t_hi:.3g}]); "
            "pick a --time in range, widen --time-window, or merge more checkpoints."
        )
        return

    from ...afterglow.spec import calibrate_skymap

    nu = observer.frequencies[0]
    sb, flux_mjy = calibrate_skymap(image, half_width, observer, args.time_window, nu)
    # zrake-style gaussian smoothing: turn monte-carlo photon speckle into a gradient.
    sb = _smooth_image(sb, args.smooth)
    print(
        f"computed {n_pix}x{n_pix} skymap: half_width={half_mas:.3f} mas, "
        f"F_nu={flux_mjy:.4g} mJy at {nu:.2e} Hz (peak {sb.max():.3g} mJy/mas^2)"
    )

    if args.output:
        import h5py

        with h5py.File(args.output, "w") as f:
            f.create_dataset("surface_brightness", data=sb)  # mJy/mas^2
            f.create_dataset("intensity", data=image)  # raw beamed energy/cm^2
            f.attrs["time"] = obs_time
            f.attrs["n_pix"] = n_pix
            f.attrs["half_width_cm"] = half_width
            f.attrs["half_width_mas"] = half_mas
            f.attrs["frequency_hz"] = nu
            f.attrs["flux_mjy"] = flux_mjy
        print(f"saved skymap to {args.output}")

    if args.plot or args.save_fig:
        # the log morphology is frequency-INDEPENDENT, so drop the nu/flux label there.
        beaming = "bolometric" if args.bolometric else f"{nu:.1e} Hz"
        title = (
            f"t = {obs_time:.3g} day   ({beaming})"
            if args.log_decades is not None
            else f"t = {obs_time:.3g} day   ({nu:.1e} Hz, {flux_mjy:.3g} mJy)"
        )
        _plot_skymap(
            sb,
            half_mas,
            title,
            args.save_fig,
            args.plot,
            cbar_label="surface brightness [mJy/mas$^2$]",
            log_decades=args.log_decades,
            contours=args.contours,
            diagnostics=args.diagnostics,
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
    # provided epochs).
    if args.t_start is None or args.t_stop is None:
        t0, _, t1 = _handle_arrival_window(
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
            frequency=float(nu),
            frac_bandwidth=0.1,
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
    # paper-style orientation (matches _draw_skymap_on_ax): jet axis horizontal, approaching
    # (doppler-beamed, +z) hemisphere on the LEFT.
    def _orient(sb):
        return np.asarray(sb).T[:, ::-1]

    norm = LogNorm(vmax * 1e-3, vmax) if args.log_color else Normalize(0.0, vmax)
    fig, ax = plt.subplots()
    im = ax.imshow(
        _orient(frames[0][0]),
        origin="lower",
        cmap="inferno",
        norm=norm,
        extent=[-half_max, half_max, -half_max, half_max],
    )
    ax.set_xlim(-half_max, half_max)
    ax.set_ylim(-half_max, half_max)
    ax.set_xlabel("x [mas]")
    ax.set_ylabel("z [mas]")
    fig.colorbar(im, ax=ax, label="surface brightness [mJy/mas$^2$]")
    title = ax.set_title("")

    def update(idx):
        sb, hmm, t, flux_mjy = frames[idx]
        extent = hmm if hmm > 0 else half_max
        im.set_data(_orient(sb))
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

    print(f"computing polarization for observer angle {args.observer_angle} deg...")

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
