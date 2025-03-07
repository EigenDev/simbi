import numpy as np
from dataclasses import dataclass
from typing import Sequence, Any
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StateVector:
    """Immutable state vector container with slots for improved performance

    Using slots:
    - Reduces memory usage by storing attributes in a pre-allocated array
    - Faster attribute access since lookup is array-based not dict-based
    - Prevents adding new attributes after instantiation
    - Works well with frozen dataclasses for immutability
    """

    density: NDArray[np.floating[Any]]
    momentum: Sequence[NDArray[np.floating[Any]]]
    energy: NDArray[np.floating[Any]]
    rho_chi: NDArray[np.floating[Any]]
    mean_magnetic_field: Sequence[NDArray[np.floating[Any]]]

    def __array__(self) -> NDArray[np.floating[Any]]:
        """Enables direct numpy array conversion via np.asarray(state)"""
        if not self.mean_magnetic_field:
            # Pre-allocate for [density, mx, my, mz, E, rho_chi]
            nvars = len(self.momentum) + 3
            result = np.empty((nvars,) + self.density.shape, dtype=np.float64)
            result[0] = self.density
            result[1:-2] = self.momentum
            result[-2] = self.energy
            result[-1] = self.rho_chi
        else:
            # Pre-allocate for [density, mx, my, mz, E, Bx, By, Bz, density_chi]
            nvars = 9
            result = np.empty((nvars,) + self.density.shape, dtype=np.float64)
            result[0] = self.density
            result[1:4] = self.momentum
            result[4] = self.energy
            result[5:-1] = self.mean_magnetic_field
            result[-1] = self.rho_chi
        return result

    # Keep original method for backward compatibility
    def to_numpy(self) -> NDArray[np.floating[Any]]:
        """Convert state to numpy array"""
        return np.asarray(self)
