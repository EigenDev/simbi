# Utility functions for visualization scripts
import h5py
import astropy.constants as const
import matplotlib
import astropy.units as units
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Union, Any, Callable, Optional, no_type_check
from numpy.typing import NDArray
from numpy import float64 as numpy_float
from ..detail.helpers import find_nearest

# FONT SIZES
SMALL_SIZE = 6
DEFAULT_SIZE = 10
BIGGER_SIZE = 12

logically_curvlinear = ["spherical", "planar_cylindrical"]
logically_cartesian = ["cartesian", "axis_cylindrical", "cylindrical"]
# ================================
#   constants of nature
# ================================
R_0 = const.R_sun.cgs
c = const.c.cgs
m = const.M_sun.cgs

rho_scale = m / R_0**3
e_scale = m * c**2
edens_scale = e_scale / R_0**3
time_scale = R_0 / c
mass_scale = m

e_scale_bmk = 1e53 * units.erg
rho_scale_bmk = 1.0 * const.m_p.cgs / units.cm**3
ell_scale = (e_scale_bmk / rho_scale_bmk / const.c.cgs**2) ** (1 / 3)
t_scale = const.c.cgs * ell_scale


def calc_enthalpy(fields: dict[str, NDArray[numpy_float]]) -> Any:
    return 1.0 + fields["p"] * fields["ad_gamma"] / (
        fields["rho"] * (fields["ad_gamma"] - 1.0)
    )


def calc_lorentz_factor(fields: dict[str, NDArray[numpy_float]]) -> Any:
    return (1.0 + fields["gamma_beta"] ** 2) ** 0.5


def calc_beta(fields: dict[str, NDArray[numpy_float]]) -> Any:
    W = calc_lorentz_factor(fields)
    return (1.0 - 1.0 / W**2) ** 0.5


def get_field_str(args: argparse.Namespace) -> Union[str, list[str]]:
    field_map = {
        "rho": r"\rho",
        "D": "D",
        "gamma_beta": r"$\Gamma \beta$",
        "u": r"$\Gamma \beta$",
        "gamma_beta_1": r"$\Gamma \beta_1$",
        "u1": r"$\Gamma \beta_1$",
        "gamma_beta_2": r"$\Gamma \beta_2$",
        "u2": r"$\Gamma \beta_2$",
        "gamma_beta_3": r"$\Gamma \beta_3$",
        "u3": r"$\Gamma \beta_3$",
        "energy": r"\tau",
        "p": r"p",
        "energy_rst": r"$E$",
        "chi": r"$\chi$",
        "chi_dens": r"$\rho \cdot \chi$",
        "T_eV": "T [eV]",
        "temperature": "T",
        "mach": "M",
        "v1": r"$v_1 / v_0$",
        "v": r"$v_1 / v_0$",
        "v2": r"$v_2 / v_0$",
        "v3": r"$v_3 / v_0$",
        "tau-s": r"$\tau_s$",
        "pmag": r"$p_{\rm mag}$",
        "ptot": r"$p_{\rm tot}$",
        "sigma": r"$\sigma$",
    }

    energy_unit = r"\rm erg \ cm^{-3}"
    density_unit = r"\rm g \ cm^{-3}"

    field_str_list = []
    for field in args.fields:
        if field in field_map:
            var = field_map[field]
            if field in ["rho", "D"]:
                if args.units:
                    field_str_list.append(r"${} [{}]$]".format(var, density_unit))
                else:
                    field_str_list.append(r"${}/{}_0$".format(var, var))
            elif field in ["energy", "p"]:
                if args.units:
                    field_str_list.append(r"${} [{}]$".format(var, energy_unit))
                else:
                    field_str_list.append(r"${}/{}_0$".format(var, var))
            elif field == "energy_rst":
                if args.units:
                    field_str_list.append(r"${} \  [{}]$".format(var, energy_unit))
                else:
                    field_str_list.append(r"${} / {}_0$".format(var, var))
            elif field == "temperature":
                field_str_list.append("T [K]" if args.units else "T")
            else:
                field_str_list.append(var)
        elif field in ["b1", "b2", "b3"]:
            field_str_list.append(rf"$B_{field[1]}$")
        else:
            field_str_list.append(rf"${field}$")

    return field_str_list if len(args.fields) > 1 else field_str_list[0]


def unpad(arr: NDArray[numpy_float], pad_width: tuple[tuple[Any, ...], ...]) -> Any:
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return arr[tuple(slices)]


def flatten_fully(x: NDArray[numpy_float]) -> NDArray[numpy_float] | Any:
    if any(dim == 1 for dim in x.shape):
        x = np.vstack(x)  # type: ignore
        if len(x.shape) == 2 and x.shape[0] == 1:
            return x.flat
        return flatten_fully(x)
    else:
        return np.asanyarray(x)


def get_dimensionality(files: Union[list[str], dict[int, list[str]]]) -> int:
    dims = []
    all_equal: Callable[[list[int]], bool] = lambda x: x.count(x[0]) == len(x)
    ndim: int = 0
    if isinstance(files, dict):
        import itertools

        files = list(itertools.chain(*files.values()))

    files = list(filter(bool, files))
    for file in files:
        with h5py.File(file, "r") as hf:
            ds = hf.get("sim_info").attrs
            ndim = ds["dimensions"]
            dims += [ndim]
            if not all_equal(dims):
                raise ValueError(
                    "All simulation files require identical dimensionality"
                )

    return ndim

@no_type_check
def read_file(filename: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with h5py.File(filename, "r") as hf:
        ds = dict(hf.get("sim_info").attrs)
        ndim: int = ds["dimensions"]
        ds.update(
            {k: v.decode("utf-8") for k, v in ds.items() if isinstance(v, np.bytes_)}
        )

        def read_and_flatten(name: str) -> NDArray[numpy_float]:
            return flatten_fully(hf.get(name)[:].reshape(ds["nz"], ds["ny"], ds["nx"]))

        rho = read_and_flatten("rho")
        p = read_and_flatten("p")
        chi = read_and_flatten("chi")
        v = [read_and_flatten(f"v{dim}") for dim in range(1, ndim + 1)]

        padwidth = (ds["spatial_order"] != "pcm") + 1
        npad = tuple((padwidth, padwidth) for _ in range(ndim))

        def unpad_all(arrays: list[NDArray[numpy_float]]) -> list[NDArray[numpy_float]]:
            return [unpad(arr, npad) for arr in arrays]

        rho, p, chi = unpad_all([rho, p, chi])
        v = np.array(unpad_all(v))

        fields = {f"v{i+1}": v[i] for i in range(len(v))}
        fields.update(
            {"rho": rho, "p": p, "chi": chi, "ad_gamma": ds.pop("adiabatic_gamma")}
        )

        vsqr = np.sum(v**2, axis=0)
        if ds["regime"] in ["srhd", "srmhd"]:
            W = (1 + vsqr) ** 0.5 if ds["using_gamma_beta"] else (1 - vsqr) ** (-0.5)
            if ds["using_gamma_beta"]:
                fields.update({f"v{i+1}": v[i] / W for i in range(len(v))})
                vsqr /= W**2

            if ds["regime"] == "srmhd":

                def read_bfield(
                    name: str, shape: tuple[int, ...]
                ) -> NDArray[numpy_float]:
                    return hf.get(name)[:].reshape(shape)

                b1 = read_bfield(
                    "b1",
                    (ds["zactive_zones"], ds["yactive_zones"], ds["xactive_zones"] + 1),
                )
                b2 = read_bfield(
                    "b2",
                    (ds["zactive_zones"], ds["yactive_zones"] + 1, ds["xactive_zones"]),
                )
                b3 = read_bfield(
                    "b3",
                    (ds["zactive_zones"] + 1, ds["yactive_zones"], ds["xactive_zones"]),
                )

                fields.update(
                    {
                        "b1": 0.5 * (b1[..., 1:] + b1[..., :-1]),
                        "b2": 0.5 * (b2[:, 1:, :] + b2[:, :-1, :]),
                        "b3": 0.5 * (b3[1:, :, :] + b3[:-1, :, :]),
                    }
                )

                for dim in range(2, ndim + 1):
                    if f"v{dim}" not in fields:
                        fields[f"v{dim}"] = unpad(read_and_flatten(f"v{dim}"), npad)
        else:
            W = 1

        fields["gamma_beta"] = np.sqrt(vsqr) * W
        fields["W"] = W

        funcs = [
            (
                np.linspace
                if ds.get(f"{x}_cell_spacing", "linear") == "linear"
                else np.geomspace
            )
            for x in ["x1", "x2", "x3"]
        ]
        mesh = {f"x{i+1}v": hf[f"x{i+1}"][:] for i in range(ndim)}

        ds.update(
            {
                "is_cartesian": ds["geometry"] in logically_cartesian,
                "coord_system": ds.pop("geometry"),
                "time": ds.pop("current_time"),
            }
        )

        if ds["x1max"] > mesh["x1v"][-1]:
            mesh["x1v"] = funcs[0](ds["x1min"], ds["x1max"], ds["xactive_zones"] + 1)

    return fields, ds, mesh


def prims2var(fields: dict[str, NDArray[numpy_float]], var: str) -> Any:
    h = calc_enthalpy(fields)
    W = calc_lorentz_factor(fields)
    if var == "D":
        # Lab frame density
        return fields["rho"] * W
    elif var == "S":
        # Lab frame momentum density
        return fields["rho"] * W**2 * calc_enthalpy(fields) * fields["v"]
    elif var == "energy":
        bsquared = 0.0
        vsquared = 0.0
        vdb = 0.0
        # check for bfield
        if "b1" in fields:
            bvec = np.array([fields["b1"], fields["b2"], fields["b3"]])
            vvec = np.array([fields["v1"], fields["v2"], fields["v3"]])
            bsquared = np.sum([b**2 for b in bvec], axis=0)
            vsquared = np.sum([v**2 for v in vvec], axis=0)
            vdb = np.sum([x * y for x, y in zip(bvec, vvec)], axis=0)
        # Energy minus rest mass energy
        return (
            fields["rho"] * h * W**2
            - fields["p"]
            - fields["rho"] * W
            + 0.5 * (bsquared + vsquared * bsquared - vdb**2)
        )
    elif var == "energy_rst":
        bsquared = 0.0
        vsquared = 0.0
        vdb = 0.0
        # check for bfield
        if "b1" in fields:
            bvec = np.array([fields["b1"], fields["b2"], fields["b3"]])
            vvec = np.array([fields["v1"], fields["v2"], fields["v3"]])
            bsquared = np.sum([b**2 for b in bvec], axis=0)
            vsquared = np.sum([v**2 for v in vvec], axis=0)
            vdb = np.sum([x * y for x, y in zip(bvec, vvec)], axis=0)
        # Total Energy
        return (
            fields["rho"] * h * W**2
            - fields["p"]
            + 0.5 * (bsquared + vsquared * bsquared - vdb**2)
        )
    elif var == "temperature":
        a = 4.0 * const.sigma_sb.cgs / c
        T = (3.0 * fields["p"] * edens_scale / a) ** 0.25
        return T
    elif var == "T_eV":
        a = 4.0 * const.sigma_sb.cgs / c
        T = (3.0 * fields["p"] * edens_scale / a) ** 0.25
        T_eV = (const.k_B.cgs * T).to(units.eV)
        return T_eV
    elif var == "chi_dens":
        fields["chi"][fields["chi"] == 0] = 1.0e-10
        return fields["chi"] * fields["rho"] * W
    elif var == "gamma_beta_1":
        return W * fields["v1"]
    elif var == "gamma_beta_2":
        return W * fields["v2"]
    elif var == "gamma_beta_3":
        return W * fields["v3"]
    elif var == "sp_enthalpy":
        # Specific enthalpy
        return h - 1.0
    elif var == "mach":
        beta2 = 1.0 - (1.0 + fields["gamma_beta"] ** 2) ** (-1)
        cs2 = fields["ad_gamma"] * fields["p"] / fields["rho"] / h
        return np.sqrt(beta2 / cs2)
    elif var == "u1":
        return W * fields["v1"]
    elif var == "u2":
        return W * fields["v2"]
    elif var == "u3":
        return W * fields["v3"]
    elif var == "u":
        return fields["gamma_beta"]
    elif var == "tau-s":
        return (1 - 1 / W**2) ** (-0.5)
    elif var == "ptot":
        try:
            bsq = fields["b1"] ** 2 + fields["b2"] ** 2 + fields["b3"] ** 2
            return fields["p"] + 0.5 * bsq
        except KeyError:
            return fields["p"]
    elif var == "pmag":
        try:
            vvec = np.array([fields["v1"], fields["v2"], fields["v3"]])
            lorentzsq = calc_lorentz_factor(fields) ** 2
            bvec = np.array([fields["b1"], fields["b2"], fields["b3"]])
            vdotb = np.sum([x * y for x, y in zip(vvec, bvec)], axis=0)
            bsq = fields["b1"] ** 2 + fields["b2"] ** 2 + fields["b3"] ** 2

            return 0.5 * (bsq / lorentzsq + vdotb**2)
        except KeyError:
            raise KeyError("The simulation data is not from an MHD run")
    elif var == "sigma":
        try:
            bvec = np.array([fields["b1"], fields["b2"], fields["b3"]])
            vvec = np.array([fields["v1"], fields["v2"], fields["v3"]])
            bsquared = np.sum([b**2 for b in bvec], axis=0)
            vsquared = np.sum([v**2 for v in vvec], axis=0)
            vdb = np.sum([x * y for x, y in zip(bvec, vvec)], axis=0)
            sigma = np.sqrt(
                fields["b1"] ** 2 + fields["b2"] ** 2 + fields["b3"] ** 2
            ) / np.sqrt(fields["rho"])
        except KeyError:
            raise KeyError("The simulation date is not from an MHD run")
        return sigma
    else:
        raise NotImplementedError("derived variable {var} not implemented")


def get_colors(
    interval: NDArray[numpy_float],
    cmap: matplotlib.colors.ListedColormap,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> NDArray[Any]:
    """
    Return array of rgba colors for a given matplotlib colormap

    Parameters
    -------------------------
    interval: interval range for colormarp min and max
    cmap: the matplotlib colormap instance
    vmin: minimum for colormap
    vmax: maximum for colormap

    Returns
    -------------------------
    arr: the colormap array generate by the user conditions
    """
    matplotlib.colors.Normalize(vmin, vmax)
    return cmap(interval)


def fill_below_intersec(
    x: NDArray[numpy_float],
    y: NDArray[numpy_float],
    constraint: float,
    color: float,
    axis: str,
) -> None:
    if axis == "x":
        ind: int = find_nearest(x, constraint)[0]
    else:
        ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:], y[ind:], color=color, alpha=0.1, interpolate=True)


def get_file_list(
    inputs: str, sort: bool = False
) -> Union[tuple[list[str], int], tuple[dict[int, list[str]], bool]]:
    from pathlib import Path
    from typing import cast

    files: Union[list[str], dict[int, list[str]]]
    dirs = list(filter(lambda x: Path(x).is_dir(), inputs))
    multidir = len(dirs) > 1

    if multidir:
        files = {
            key: sorted([str(f) for f in Path(fdir).glob("*.h5") if f.is_file()])
            for key, fdir in enumerate(inputs)
        }
    else:
        files = []
        if dirs:
            files = sorted(
                [str(f) for d in dirs for f in Path(d).glob("*.h5") if f.is_file()]
            )
        files += [file for file in filter(lambda x: x not in dirs, inputs)]

    if not isinstance(files, dict):
        # sort by length of strings now
        if sort:
            files.sort(key=len, reverse=False)
        return files, len(files)
    else:
        any(files[key].sort(key=len, reverse=False) for key in files.keys())
        return files, multidir
