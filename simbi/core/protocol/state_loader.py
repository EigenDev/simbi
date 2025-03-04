import numpy as np
from typing import Protocol
from ...functional.maybe import Maybe
from ..config.initialization import InitializationConfig
from numpy.typing import NDArray

class StateLoader(Protocol):
    """Protocol for state loading strategies"""

    def load(self, config: InitializationConfig) -> Maybe[NDArray[np.float64]]: ...