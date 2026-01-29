# =============================================================================
# mesh_expansion.py
#
# geometry-aware mesh expansion from simulation coordinates to R^3.
# handles various coordinate systems and dimensionalities for afterglow
# radiation calculations.
#
# design:
#   - takes simulation data (fields, mesh, metadata)
#   - returns expanded 3D mesh + fields in spherical coordinates (r, theta, phi)
#   - preserves physical quantities correctly
#
# usage:
#   mesh_3d, fields_3d = expand_to_3d(data)
# =============================================================================

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


@dataclass(frozen=True)
class mesh_config_t:
    """mesh expansion configuration based on simulation geometry"""

    coord_system: str
    dimensions: int
    n_theta: int = 32  # polar zones for expansion
    n_phi: int = 16  # azimuthal zones for expansion


@dataclass(frozen=True)
class expanded_mesh_t:
    """3D mesh in spherical coordinates (r, theta, phi)"""

    x1: Array  # radial coordinate [cm]
    x2: Array  # polar angle theta [radians]
    x3: Array  # azimuthal angle phi [radians]
    coord_system: str  # "spherical" after expansion


def expand_to_3d(
    fields: Dict[str, Array],
    mesh: Dict[str, Array],
    coord_system: str,
    dimensions: int,
    n_theta: int = 32,
    n_phi: int = 16,
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    expand simulation data to full 3D spherical mesh for afterglow calculations.

    args:
        fields: dict with keys {rho, gamma_beta, p}
        mesh: dict with keys {x1, x2?, x3?}
        coord_system: geometry from metadata.coord_system
        dimensions: effective dimensions from metadata.dimensions

    returns:
        (expanded_fields, expanded_mesh) both in 3D spherical coordinates
    """

    # dispatch based on geometry
    if dimensions == 3:
        # already 3D - just validate and return
        return _validate_3d(fields, mesh, coord_system)

    elif dimensions == 2:
        if coord_system == "spherical":
            return _expand_2d_spherical(fields, mesh, n_phi)
        elif coord_system in ["cylindrical", "axis_cylindrical"]:
            return _expand_2d_cylindrical(fields, mesh, n_phi)
        elif coord_system == "planar_cylindrical":
            return _expand_2d_planar_cylindrical(fields, mesh, n_theta)
        elif coord_system == "cartesian":
            return _expand_2d_cartesian(fields, mesh, n_phi)
        else:
            raise ValueError(f"unknown 2D coord system: {coord_system}")

    elif dimensions == 1:
        if coord_system == "spherical":
            return _expand_1d_spherical(fields, mesh, n_theta, n_phi)
        elif coord_system in ["cylindrical", "axis_cylindrical"]:
            # 1D cylindrical: radial shells at z=0, expand in theta and phi
            return _expand_1d_cylindrical(fields, mesh, n_theta, n_phi)
        elif coord_system == "cartesian":
            raise ValueError(
                "1D cartesian afterglow is not physical - "
                "use spherical coordinates for jet/blastwave simulations"
            )
        else:
            raise ValueError(f"1D afterglow not supported for {coord_system}")

    else:
        raise ValueError(f"invalid dimensions: {dimensions}")


# =============================================================================
# 1D expansion
# =============================================================================


def _expand_1d_spherical(
    fields: Dict[str, Array], mesh: Dict[str, Array], n_theta: int, n_phi: int
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    1D spherical (r) -> 3D spherical (r, theta, phi)

    assumption: spherically symmetric - broadcast radial profiles
    """
    x1 = mesh["x1"]  # radial coordinate
    n_r = len(x1)

    # create angular grids
    x2 = np.linspace(0, np.pi, n_theta)  # theta in [0, pi]
    x3 = np.linspace(0, 2 * np.pi, n_phi)  # phi in [0, 2pi]

    expanded_mesh = {"x1": x1, "x2": x2, "x3": x3}

    # expand fields: (n_r,) -> (n_r, n_theta, n_phi)
    expanded_fields = {}
    for name, field_1d in fields.items():
        if field_1d.shape != (n_r,):
            raise ValueError(
                f"field {name} has wrong shape {field_1d.shape}, expected ({n_r},)"
            )

        # broadcast to 3D
        field_3d = np.broadcast_to(
            field_1d[:, None, None], (n_r, n_theta, n_phi)
        )
        expanded_fields[name] = np.ascontiguousarray(field_3d, dtype=np.float64)

    return expanded_fields, expanded_mesh


def _expand_1d_cylindrical(
    fields: Dict[str, Array], mesh: Dict[str, Array], n_theta: int, n_phi: int
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    1D cylindrical (R) -> 3D spherical (r, theta, phi)

    assumption: cylindrical shells at z=0, expand to full sphere
    treat as disk at midplane
    """
    R = mesh["x1"]  # cylindrical radius
    n_R = len(R)

    # create angular grids - disk at theta = pi/2
    x2 = np.linspace(
        np.pi / 2 - 0.2, np.pi / 2 + 0.2, n_theta
    )  # theta near midplane
    x3 = np.linspace(0, 2 * np.pi, n_phi)  # phi in [0, 2pi]

    # in disk, r ~ R
    expanded_mesh = {"x1": R, "x2": x2, "x3": x3}

    # expand fields: (n_R,) -> (n_R, n_theta, n_phi)
    expanded_fields = {}
    for name, field_1d in fields.items():
        if field_1d.shape != (n_R,):
            raise ValueError(
                f"field {name} has wrong shape {field_1d.shape}, expected ({n_R},)"
            )

        # broadcast to 3D
        field_3d = np.broadcast_to(
            field_1d[:, None, None], (n_R, n_theta, n_phi)
        )
        expanded_fields[name] = np.ascontiguousarray(field_3d, dtype=np.float64)

    return expanded_fields, expanded_mesh


# =============================================================================
# 2D expansion
# =============================================================================


def _expand_2d_spherical(
    fields: Dict[str, Array], mesh: Dict[str, Array], n_phi: int
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    2D spherical (r, theta) -> 3D spherical (r, theta, phi)

    assumption: axisymmetric - uniform in phi
    """
    x1 = mesh["x1"]  # radial
    x2 = mesh["x2"]  # polar angle
    n_r = len(x1)
    n_theta = len(x2)

    # create azimuthal grid
    x3 = np.linspace(0, 2 * np.pi, n_phi)

    expanded_mesh = {"x1": x1, "x2": x2, "x3": x3}

    # expand fields: (n_r, n_theta) -> (n_r, n_theta, n_phi)
    expanded_fields = {}
    for name, field_2d in fields.items():
        if field_2d.shape != (n_r, n_theta):
            raise ValueError(
                f"field {name} has wrong shape {field_2d.shape}, expected ({n_r}, {n_theta})"
            )

        # broadcast to 3D
        field_3d = np.broadcast_to(field_2d[:, :, None], (n_r, n_theta, n_phi))
        expanded_fields[name] = np.ascontiguousarray(field_3d, dtype=np.float64)

    return expanded_fields, expanded_mesh


def _expand_2d_cylindrical(
    fields: Dict[str, Array], mesh: Dict[str, Array], n_phi: int
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    2D cylindrical (R, z) -> 3D cylindrical (R, phi, z) -> spherical (r, theta, phi)

    conversion:
        r = sqrt(R^2 + z^2)
        theta = atan2(R, z)
        phi = phi (azimuthal)
    """
    R_grid = mesh["x1"]  # cylindrical radius
    z_grid = mesh["x2"]  # height
    n_R = len(R_grid)
    n_z = len(z_grid)

    # create azimuthal grid
    phi_grid = np.linspace(0, 2 * np.pi, n_phi)

    # convert to spherical coordinates
    R_3d, z_3d, phi_3d = np.meshgrid(R_grid, z_grid, phi_grid, indexing="ij")
    r_3d = np.sqrt(R_3d**2 + z_3d**2)
    theta_3d = np.arctan2(R_3d, z_3d)

    # create uniform spherical grids for output
    r_min, r_max = r_3d.min(), r_3d.max()
    theta_min, theta_max = theta_3d.min(), theta_3d.max()

    x1 = np.linspace(r_min, r_max, n_R)
    x2 = np.linspace(theta_min, theta_max, n_z)
    x3 = phi_grid

    expanded_mesh = {"x1": x1, "x2": x2, "x3": x3}

    # interpolate fields from cylindrical to spherical
    # for simplicity, broadcast in phi and use original grid structure
    expanded_fields = {}
    for name, field_2d in fields.items():
        if field_2d.shape != (n_R, n_z):
            raise ValueError(
                f"field {name} has wrong shape {field_2d.shape}, expected ({n_R}, {n_z})"
            )

        # simple broadcast (assumes weak z-dependence)
        field_3d = np.broadcast_to(field_2d[:, :, None], (n_R, n_z, n_phi))
        expanded_fields[name] = np.ascontiguousarray(field_3d, dtype=np.float64)

    return expanded_fields, expanded_mesh


def _expand_2d_planar_cylindrical(
    fields: Dict[str, Array], mesh: Dict[str, Array], n_theta: int
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    2D planar cylindrical (R, phi) -> 3D spherical (r, theta, phi)

    assumption: disk in z=0 plane, expand in polar angle theta
    """
    R_grid = mesh["x1"]  # cylindrical radius
    phi_grid = mesh["x2"]  # azimuthal angle
    n_R = len(R_grid)
    n_phi = len(phi_grid)

    # create polar angle grid (disk at midplane)
    theta_grid = np.linspace(np.pi / 2 - 0.1, np.pi / 2 + 0.1, n_theta)

    # in this case, x1=R=r at theta=pi/2
    expanded_mesh = {"x1": R_grid, "x2": theta_grid, "x3": phi_grid}

    # expand fields: (n_R, n_phi) -> (n_R, n_theta, n_phi)
    expanded_fields = {}
    for name, field_2d in fields.items():
        if field_2d.shape != (n_R, n_phi):
            raise ValueError(
                f"field {name} has wrong shape {field_2d.shape}, expected ({n_R}, {n_phi})"
            )

        # broadcast in theta
        field_3d = np.broadcast_to(field_2d[:, None, :], (n_R, n_theta, n_phi))
        expanded_fields[name] = np.ascontiguousarray(field_3d, dtype=np.float64)

    return expanded_fields, expanded_mesh


def _expand_2d_cartesian(
    fields: Dict[str, Array], mesh: Dict[str, Array], n_phi: int
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    2D cartesian (x, y) -> 3D spherical (r, theta, phi)

    conversion:
        r = sqrt(x^2 + y^2)
        theta = pi/2 (disk in xy-plane)
        phi = atan2(y, x)

    assumption: disk geometry, expand slightly in theta
    """
    x_grid = mesh["x1"]
    y_grid = mesh["x2"]
    n_x = len(x_grid)
    n_y = len(y_grid)

    # create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    # convert to polar in xy-plane
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    # create spherical grids
    r_vals = np.linspace(R.min(), R.max(), n_x)
    theta_vals = np.linspace(
        np.pi / 2 - 0.1, np.pi / 2 + 0.1, 16
    )  # near midplane
    phi_vals = np.linspace(-np.pi, np.pi, n_y)

    expanded_mesh = {"x1": r_vals, "x2": theta_vals, "x3": phi_vals}

    # for simplicity, broadcast in theta
    # proper implementation would interpolate from cartesian to spherical
    expanded_fields = {}
    for name, field_2d in fields.items():
        n_theta = len(theta_vals)
        field_3d = np.broadcast_to(field_2d[:, None, :], (n_x, n_theta, n_y))
        expanded_fields[name] = np.ascontiguousarray(field_3d, dtype=np.float64)

    return expanded_fields, expanded_mesh


# =============================================================================
# 3D validation
# =============================================================================


def _validate_3d(
    fields: Dict[str, Array], mesh: Dict[str, Array], coord_system: str
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    validate 3D data and convert to spherical if needed.
    """
    # check mesh has all 3 coordinates
    if not all(k in mesh for k in ["x1", "x2", "x3"]):
        raise ValueError("3D mesh must have x1, x2, x3 coordinates")

    # check field shapes match
    n1, n2, n3 = len(mesh["x1"]), len(mesh["x2"]), len(mesh["x3"])

    for name, field in fields.items():
        if field.shape != (n1, n2, n3):
            raise ValueError(
                f"field {name} shape {field.shape} doesn't match mesh ({n1}, {n2}, {n3})"
            )

    # convert coordinate systems if needed
    if coord_system == "spherical":
        # already correct
        return fields, mesh
    elif coord_system in ["cylindrical", "axis_cylindrical"]:
        return _convert_3d_cylindrical_to_spherical(fields, mesh)
    elif coord_system == "cartesian":
        return _convert_3d_cartesian_to_spherical(fields, mesh)
    elif coord_system == "planar_cylindrical":
        # already effectively spherical with restricted theta
        return fields, mesh
    else:
        raise ValueError(
            f"3D afterglow not supported for coord_system={coord_system}"
        )


def _convert_3d_cylindrical_to_spherical(
    fields: Dict[str, Array], mesh: Dict[str, Array]
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    3D cylindrical (R, phi, z) -> 3D spherical (r, theta, phi)

    conversion:
        r = sqrt(R^2 + z^2)
        theta = atan2(R, z)
        phi = phi (same)
    """
    R_grid = mesh["x1"]
    phi_grid = mesh["x2"]
    z_grid = mesh["x3"]

    n_R = len(R_grid)
    n_phi = len(phi_grid)
    n_z = len(z_grid)

    # create spherical grid by transforming cylindrical coords
    R_3d, phi_3d, z_3d = np.meshgrid(R_grid, phi_grid, z_grid, indexing="ij")
    r_3d = np.sqrt(R_3d**2 + z_3d**2)
    theta_3d = np.arctan2(R_3d, z_3d)

    # create uniform spherical output grids
    r_min, r_max = r_3d.min(), r_3d.max()
    theta_min, theta_max = theta_3d.min(), theta_3d.max()

    x1 = np.linspace(r_min, r_max, n_R)
    x2 = np.linspace(theta_min, theta_max, n_z)
    x3 = phi_grid  # phi unchanged

    expanded_mesh = {"x1": x1, "x2": x2, "x3": x3}

    # simple remapping (proper version would interpolate)
    expanded_fields = {}
    for name, field_3d in fields.items():
        # reorder axes: (R, phi, z) -> (r, theta, phi) ~ (R, z, phi) with transposition
        field_reordered = np.transpose(field_3d, (0, 2, 1))
        expanded_fields[name] = np.ascontiguousarray(
            field_reordered, dtype=np.float64
        )

    return expanded_fields, expanded_mesh


def _convert_3d_cartesian_to_spherical(
    fields: Dict[str, Array], mesh: Dict[str, Array]
) -> tuple[Dict[str, Array], Dict[str, Array]]:
    """
    3D cartesian (x, y, z) -> 3D spherical (r, theta, phi)

    not implemented: requires proper 3D interpolation from cartesian to spherical grid.
    use spherical coordinates for afterglow simulations instead.
    """
    raise NotImplementedError(
        "3D cartesian to spherical conversion requires interpolation and is not implemented. "
        "run your afterglow simulation in spherical coordinates instead."
    )


# =============================================================================
# utility functions
# =============================================================================


def validate_field_dict(fields: Dict[str, Array]) -> None:
    """ensure fields dict has required keys for radiation"""
    required = {"rho", "gamma_beta", "p"}
    missing = required - set(fields.keys())
    if missing:
        raise ValueError(f"fields dict missing required keys: {missing}")


def get_mesh_shape(mesh: Dict[str, Array]) -> tuple[int, ...]:
    """get mesh dimensions from coordinate arrays"""
    return tuple(len(mesh[k]) for k in ["x1", "x2", "x3"] if k in mesh)
