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


def build_fields(data: Any) -> dict[str, NDArray]:
    """
    build the rust contract `fields` dict from a SimData checkpoint.

    contract:
      "rho", "gamma_beta", "pre" -> flat row-major float64 arrays in code units.

    gamma_beta is the dimensionless four-velocity magnitude |gamma*beta|. the
    simbi reader stores v1/v2/v3 as the 3-velocity (|v| < 1, c = 1); the derived
    field "u" already evaluates |gamma*beta| from those 3-velocities for the
    checkpoint regime, so it is used directly rather than re-deriving here.
    """
    rho = np.ascontiguousarray(
        np.asarray(data.get_field("rho"), dtype=np.float64).ravel()
    )
    pre = np.ascontiguousarray(
        np.asarray(data.get_field("p"), dtype=np.float64).ravel()
    )
    gamma_beta = np.ascontiguousarray(
        np.asarray(data.get_field("u"), dtype=np.float64).ravel()
    )

    return {
        "rho": rho,
        "gamma_beta": gamma_beta,
        "pre": pre,
    }


def build_mesh(data: Any) -> dict[str, Any]:
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
        mesh["x2"] = np.ascontiguousarray(_cell_centers(data.mesh.x2v))
    if data_dim >= 3:
        mesh["x3"] = np.ascontiguousarray(_cell_centers(data.mesh.x3v))

    return mesh


def build_velocity(data: Any) -> dict[str, NDArray]:
    """
    the three-velocity component arrays ("v1".."vD", units of c) present in a checkpoint,
    flat float64 — the deposit imager uses them to capture lateral spreading (a jet's
    theta-velocity changes the observer-direction doppler). components beyond the
    checkpoint's dimensionality do not exist and are omitted (the imager treats missing
    components as zero, radial flow).
    """
    out: dict[str, NDArray] = {}
    for ax in range(1, int(data.metadata.dimensions) + 1):
        out[f"v{ax}"] = np.ascontiguousarray(
            np.asarray(data.get_field(f"v{ax}"), dtype=np.float64).ravel()
        )
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
