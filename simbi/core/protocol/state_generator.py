import numpy as np
from typing import Protocol, Sequence, Generator
from numpy.typing import NDArray


class StateGenerator(Protocol):
    """Protocol for state generation strategies"""

    def __call__(
        self, resolution: Sequence[int], bounds: Sequence[Sequence[float]]
    ) -> Generator[NDArray[np.float64], None, None]: ...
