import numpy as np
import argparse
import h5py
from numpy.typing import NDArray
from typing import Union
from astropy.cosmology import FlatLambdaCDM
from simbi import compute_num_polar_zones, get_dimensionality, read_file
from astropy import units

cosmo = FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc, Tcmb0=2.725 * units.K, Om0=0.3
)


def vector_magnitude(a: np.ndarray) -> np.ndarray:
    """return magnitude of vector(s)"""
    if a.ndim <= 3:
        return (a.dot(a)) ** 0.5
    else:
        return (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5


def vector_dotproduct(a: np.ndarray, b: np.ndarray) -> float:
    """dot product between vectors or array of vectors"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def read_afterglow_library_data(filename: str) -> dict:
    """
    Reads afterglow data from afterglow library (Zhang and MacFadyen 2009 or van Eerten et al. 2010)
    """
    if filename.endswith(".h5"):
        with h5py.File(filename, "r") as hf:
            nu = hf.get("nu")[:] * units.Hz
            t = hf.get("t")[:] * units.s
            fnu = hf.get("fnu")[:] * 1e26 * units.mJy
            fnu2 = hf.get("fnu2")[:] * 1e26 * units.mJy
    elif filename.endswith(".npz"):
        dat = np.load(filename)
        t = dat["time"] * units.day
        fnu = dat["flux"] * units.mJy
        nu = dat["nu"] * units.Hz

    tday = t.to(units.day)
    data_dict = {}
    data_dict["tday"] = tday
    data_dict["freq"] = nu

    if filename.endswith(".h5"):
        data_dict["fnu"] = {nu_val: fnu[i, :] for i, nu_val in enumerate(nu)}
        data_dict["spectra"] = {tday_val: fnu[:, i] for i, tday_val in enumerate(tday)}

        if "fnu2" in locals():
            data_dict["fnu_pcj"] = {
                nu_val: fnu[i, :] + fnu2[i, :] for i, nu_val in enumerate(nu)
            }
            data_dict["spectra_pcj"] = {
                tday_val: fnu2[:, i] for i, tday_val in enumerate(tday)
            }
    else:
        data_dict["fnu"] = {nu_val: fnu[:, i] for i, nu_val in enumerate(nu)}
        data_dict["spectra"] = {tday_val: fnu[i, :] for i, tday_val in enumerate(tday)}

    return data_dict


def read_simbi_afterglow(filename: str) -> dict:
    """
    Reads afterglow data from simbi output
    """
    with h5py.File(filename, "r") as hf:
        nu = hf.get("nu")[:] * units.Hz
        tday = hf.get("tbins")[:] * units.day
        fnu = hf.get("fnu")[:] * units.mJy

    data_dict = {}
    data_dict["tday"] = tday
    data_dict["freq"] = nu
    data_dict["fnu"] = {nu_val: fnu[i, :] for i, nu_val in enumerate(nu)}
    # data_dict['light_curve_pcj'] = {nu_val: fnu[i, :] + fnu2[i, :] for i, nu_val in enumerate(nu)}
    # data_dict['spectra'] = {tday_val: fnu[:, i] for i, tday_val in enumerate(tday)}
    # data_dict['spectra_pcj'] = {tday_val: fnu2[:, i] for i, tday_val in enumerate(tday)}

    return data_dict


def get_dL(z):
    if z > 0:
        return cosmo.luminosity_distance(z).cgs
    else:
        return 1e28 * units.cm


def calc_rhat(
    theta: NDArray[np.floating], phi: NDArray[np.floating] | float
) -> NDArray:
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def generate_pseudo_mesh(
    args: argparse.Namespace, mesh: dict, full_sphere: bool, full_threed: bool
):
    """
    Generate a real or pseudo 3D mesh based on checkpoint data
    assuming a spherical geometry

    Parameters
    --------------------------
    args: argparser arguments from CLI
    mesh: the mesh data from the checkpoint

    Return
    --------------------------
    None
    """
    ndim = mesh["x1"].ndim
    if "x2" not in mesh:
        theta_min = 0.0
        theta_max = np.pi if full_sphere else 0.5 * np.pi
        ntheta = args.theta_samples or compute_num_polar_zones(
            rmin=mesh["x1"].min(),
            rmax=mesh["x1"].max(),
            nr=mesh["x1"].size,
            theta_bounds=(theta_min, theta_max),
        )
        mesh["x2"] = np.linspace(theta_min, theta_max, ntheta)

    if "x3" not in mesh:
        mesh["x3"] = np.linspace(0.0, 2.0 * np.pi, args.phi_samples)

    if ndim < 3:
        if full_threed:
            mesh["xx2"], mesh["xx3"], mesh["xx1"] = np.meshgrid(
                mesh["x2"], mesh["x3"], mesh["x1"]
            )
        elif ndim < 2:
            mesh["xx1"], mesh["xx2"] = np.meshgrid(mesh["x1"], mesh["x2"])
            mesh["xx3"] = 0
        else:
            mesh["xx1"] = mesh["x1"][:]
            mesh["xx2"] = mesh["x2"][:]
            mesh["xx3"] = 0


def get_tbin_edges(
    args: argparse.Namespace,
    files: Union[list[str], dict[int, list[str]]],
    time_scale: float,
):
    """
    Get the bin edges of the lightcurves based on the checkpoints given

    Parameters:
    -----------------------------------
    files: list of files to draw data from

    Returns:
    -----------------------------------
    tmin, tmax: tuple of time bins in units of days
    """
    at_pole = abs(np.cos(args.theta_obs)) == 1
    ndim = get_dimensionality(files)
    setup_init, mesh_init = read_file(files[+0])[1:]
    setup_final, mesh_final = read_file(files[-1])[1:]

    t_beg = setup_init["time"] * time_scale
    t_end = setup_final["time"] * time_scale

    generate_pseudo_mesh(args, mesh_init, full_sphere=True, full_threed=not at_pole)
    generate_pseudo_mesh(args, mesh_final, full_sphere=True, full_threed=not at_pole)
    rhat = calc_rhat(mesh_init["xx2"], mesh_init["xx3"] * (at_pole ^ 1))

    # Place observer along chosen axis
    theta_obs_rad = args.theta_obs
    theta_obs = theta_obs_rad * np.ones_like(mesh_init["xx1"])
    obs_hat = calc_rhat(theta_obs, 0.0)
    r_dot_nhat = vector_dotproduct(rhat, obs_hat)

    if at_pole:
        theta_slice = np.s_[:, 0]
    else:
        theta_slice = np.s_[0, :, 0]

    t_obs_beg = t_beg - mesh_init["xx1"] * time_scale * r_dot_nhat
    t_obs_end = t_end - mesh_final["xx1"] * time_scale * r_dot_nhat

    tmin = (min(j for j in t_obs_beg[theta_slice] if j > 0)).to(units.day)
    tmax = (max(j for j in t_obs_end[theta_slice] if j > 0)).to(units.day)

    return tmin, tmax
