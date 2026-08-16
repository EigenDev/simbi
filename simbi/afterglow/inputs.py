# =============================================================================
# inputs.py
#
# stateless construction of afterglow event-generator inputs from a simbi
# checkpoint (SimData). each builder function:
# - reads the relevant primitive/derived fields from the reader
# - maps reader field names to the fixed rust binding contract
# - returns flat row-major float64 arrays in code units
# - derives the dimensionless four-velocity magnitude gamma_beta
# usage:
#  fields = build_fields(data)
#  mesh = build_mesh(data)
#  qscales = build_qscales("blandford-mckee")
# =============================================================================

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .scales import get_scale_model

# code velocity is in units of c, so the code->cgs velocity multiplier is c.
# astropy provides c in cgs (cm/s); kept as a module constant to avoid repeated
# unit conversions in the hot path.
from astropy import constants as _const

_C_CGS = _const.c.cgs.value


def _cell_centers(vertices: NDArray) -> NDArray:
    """midpoints of an ascending vertex array (length n+1 -> n)."""
    verts = np.asarray(vertices, dtype=np.float64)
    return 0.5 * (verts[:-1] + verts[1:])


def _mirror_about_equator(values: NDArray, flip_sign: bool = False) -> NDArray:
    """append the equator-mirrored hemisphere along the POLAR axis.

    a run on theta in [0, pi/2] closed by a REFLECTING equator does not model half a
    system -- it models a whole one that happens to be symmetric about that plane. the
    material at pi - theta is physically present, and an off-axis observer sees it: it
    is the RECEDING side, the counter-jet, the far half of an equatorial ring. the
    imager treats whatever mesh it is handed as the entire sky, so without this the
    emitting volume is silently halved and the receding shell can never appear. that
    shell is exactly what forms the double-ring in an off-axis afterglow image, so its
    absence is not a small error in the flux -- it removes a morphological feature.

    reflection theta -> pi - theta maps v_theta -> -v_theta and leaves v_r, v_phi
    alone; `flip_sign` carries that for the velocity components.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"equatorial mirroring is defined for a 2d (theta, r) field; got "
            f"ndim={arr.ndim}. a 1d run has no polar axis to mirror, and a 3d one "
            f"needs the polar axis identified rather than assumed."
        )
    mirrored = arr[::-1, :]
    return np.concatenate([arr, -mirrored if flip_sign else mirrored], axis=0)


def _assert_mirrorable(data: Any) -> None:
    """refuse to mirror a mesh that does not actually stop at the equator.

    mirroring a run that already spans [0, pi] would DOUBLE-COUNT it, and mirroring one
    truncated at, say, 30 degrees would invent material in a wedge the simulation never
    solved. both produce a plausible-looking image, so this is checked rather than
    trusted to the caller.
    """
    x2v = np.asarray(data.mesh.x2v, dtype=np.float64)
    lo, hi = float(x2v[0]), float(x2v[-1])
    if abs(lo) > 1e-6 or abs(hi - 0.5 * np.pi) > 1e-6:
        raise ValueError(
            f"--mirror-equator needs a polar range of [0, pi/2]; this checkpoint spans "
            f"[{lo:.6f}, {hi:.6f}] rad ([{np.degrees(lo):.2f}, {np.degrees(hi):.2f}] deg). "
            f"a full [0, pi] run is already whole, and a narrower wedge would have "
            f"material invented outside what was solved."
        )


def build_fields(data: Any, mirror_equator: bool = False) -> dict[str, NDArray]:
    """
    build the rust contract `fields` dict from a SimData checkpoint.

    contract:
      "rho", "gamma_beta", "pre" -> flat row-major float64 arrays in code units.

    gamma_beta is the dimensionless four-velocity magnitude |gamma*beta|. the
    simbi reader stores v1/v2/v3 as the 3-velocity (|v| < 1, c = 1); the derived
    field "u" already evaluates |gamma*beta| from those 3-velocities for the
    checkpoint regime, so it is used directly rather than re-deriving here.

    `mirror_equator` appends the theta -> pi - theta hemisphere (see
    `_mirror_about_equator`); these are all SCALARS, so they mirror unsigned.
    """
    if mirror_equator:
        _assert_mirrorable(data)

    def read(name: str) -> NDArray:
        arr = np.asarray(data.get_field(name), dtype=np.float64)
        if mirror_equator:
            arr = _mirror_about_equator(arr)
        return np.ascontiguousarray(arr.ravel())

    return {
        "rho": read("rho"),
        "gamma_beta": read("u"),
        "pre": read("p"),
    }


def build_mesh(data: Any, mirror_equator: bool = False) -> dict[str, Any]:
    """
    build the rust contract `mesh` dict from a SimData checkpoint.

    contract:
      "x1"       -> radial cell-centers in INERTIAL-LAB code length (log ascending),
      "x2"/"x3"  -> optional cell-centers for higher dimensions,
      "data_dim" -> int dimensionality (1 for 1d-spherical bmk).

    the reader returns PHYSICAL radii r_phys = a(t) * r_lab for a homologous
    (moving-mesh) run; the afterglow EATS is an inertial-lab reduction, so the radius
    must be the inertial-lab coordinate r_lab = r_phys / a(t). this is what makes
    r_lab/t <= c (the co-expanding r_phys/t reaches ~9c at late times). a static mesh
    reports a = 1, so the divide is a no-op. only the radius scales; angles do not.
    """
    data_dim = int(data.metadata.dimensions)
    scale_factor = float(getattr(data.mesh, "scale_factor_a", 1.0) or 1.0)

    mesh: dict[str, Any] = {
        "x1": np.ascontiguousarray(_cell_centers(data.mesh.x1v) / scale_factor),
        "data_dim": data_dim,
    }
    if data_dim >= 2:
        theta = _cell_centers(data.mesh.x2v)
        if mirror_equator:
            _assert_mirrorable(data)
            # the mirrored centers are pi - theta reversed, so the joined axis stays
            # ascending across the equator -- which the imager's cell walk requires.
            theta = np.concatenate([theta, (np.pi - theta)[::-1]])
        mesh["x2"] = np.ascontiguousarray(theta)
    if data_dim >= 3:
        mesh["x3"] = np.ascontiguousarray(_cell_centers(data.mesh.x3v))

    return mesh


def build_velocity(data: Any, mirror_equator: bool = False) -> dict[str, NDArray]:
    """
    the three-velocity component arrays ("v1".."vD", units of c) present in a checkpoint,
    flat float64 — the deposit imager uses them to capture lateral spreading (a jet's
    theta-velocity changes the observer-direction doppler). components beyond the
    checkpoint's dimensionality do not exist and are omitted (the imager treats missing
    components as zero, radial flow).
    """
    if mirror_equator:
        _assert_mirrorable(data)
    out: dict[str, NDArray] = {}
    for ax in range(1, int(data.metadata.dimensions) + 1):
        arr = np.asarray(data.get_field(f"v{ax}"), dtype=np.float64)
        if mirror_equator:
            # v_theta (axis 2) is the only component the reflection flips: a parcel
            # moving away from the pole in the north moves away from it in the south
            # too, which is the opposite theta direction. leaving the sign alone would
            # give the mirrored hemisphere a lateral velocity converging on the
            # equator, and the doppler factor -- the whole point of the exercise --
            # would be wrong for exactly the receding material being added.
            arr = _mirror_about_equator(arr, flip_sign=(ax == 2))
        out[f"v{ax}"] = np.ascontiguousarray(arr.ravel())
    return out


def build_qscales(scale_name: str) -> dict[str, float]:
    """
    build the rust contract `qscales` dict (code -> cgs multipliers) from a
    named user scale model.

    contract keys: "time", "pre", "rho", "velocity", "length".
    code velocity is measured in units of c, so the velocity multiplier is c.
    """
    scales = get_scale_model(scale_name)

    return {
        "time": float(scales.time_scale.cgs.value),
        "pre": float(scales.pre_scale.cgs.value),
        "rho": float(scales.rho_scale.cgs.value),
        "velocity": float(_C_CGS),
        "length": float(scales.length_scale.cgs.value),
    }


def build_afterglow_inputs(
    data: Any, scale_name: str
) -> tuple[dict[str, NDArray], dict[str, Any], dict[str, float]]:
    """
    convenience composition returning (fields, mesh, qscales) for a checkpoint.

    pure input -> output; performs no i/o beyond reading the already-loaded
    SimData and the static scale model.
    """
    return build_fields(data), build_mesh(data), build_qscales(scale_name)
