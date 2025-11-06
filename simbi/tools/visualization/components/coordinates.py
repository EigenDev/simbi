"""Coordinate transformation utilities for accretion analysis."""

from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
Coords = TypeVar("Coords", bound=NDArray[np.float64])
Vector3D = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def cartesian_to_spherical(x: Coords, y: Coords, z: Coords) -> Vector3D:
    """Convert cartesian (x,y,z) to spherical (r,θ,φ) coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        (r, theta, phi): Spherical coordinates where:
            r = radial distance
            theta = polar angle (0 to π)
            phi = azimuthal angle (0 to 2π)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + np.finfo(float).tiny))  # avoid division by zero
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2π]
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return r, theta, phi


def cartesian_to_cylindrical(x: Coords, y: Coords, z: Coords) -> Vector3D:
    """Convert cartesian (x,y,z) to cylindrical (R,φ,z) coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        (R, phi, z): Cylindrical coordinates where:
            R = radial distance in x-y plane
            phi = azimuthal angle (0 to 2π)
            z = height above x-y plane
    """
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2π]
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return R, phi, z
