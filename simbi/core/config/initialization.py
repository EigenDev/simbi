import numpy as np
import itertools
from dataclasses import dataclass
from typing import Sequence, Optional
from ...functional.maybe import Maybe
from numpy.typing import NDArray
from ..protocol import StateGenerator


@dataclass(frozen=True)
class InitializationConfig:
    """Configuration for simulation initialization"""

    initial_primitive_gen: StateGenerator
    resolution: Sequence[int]
    bounds: Sequence[Sequence[float]]
    checkpoint_file: Optional[str] = None

    @staticmethod
    def from_generator(
        inital_primitive_gen: StateGenerator,
        resolution: Sequence[int],
        bounds: Sequence[Sequence[float]],
    ) -> Maybe["InitializationConfig"]:

        return Maybe.of(
            InitializationConfig(
                initial_primitive_gen=inital_primitive_gen,
                resolution=resolution,
                bounds=bounds,
            )
        )

    def evaluate(self, pad_width: int, nvars: int) -> Maybe[NDArray[np.float64]]:
        """Lazy evaluation of initial state"""
        try:
            padded_shape = (nvars,) + tuple(r + 2 * pad_width for r in self.resolution)
            padded_state = np.zeros(padded_shape, dtype=np.float64)

            gen1, gen2 = itertools.tee(
                self.initial_primitive_gen(self.resolution, self.bounds)
            )

            # check that the generator yields the correct number of variables
            nvals = len(next(gen1))
            if not (nvars - 1 <= nvals <= nvars):
                return Maybe.save_failure(
                    f"Initial state generator does not yield {nvars} variables"
                )

            interior = (slice(pad_width, -pad_width),) * len(self.resolution)
            padded_state[:nvals, *interior] = np.array(list(gen2)).T

            # fill the ghost cells at the edges with ones from the interior
            for i, axis in enumerate(self.resolution):
                for j in range(pad_width):
                    padded_state[
                        :nvals,
                        *(
                            slice(j, j + 1) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ] = padded_state[
                        :nvals,
                        *(
                            slice(pad_width, pad_width + 1) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ]

                    padded_state[
                        :nvals,
                        *(
                            slice(-j - 1, -j or None) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ] = padded_state[
                        :nvals,
                        *(
                            slice(-pad_width - 1, -pad_width) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ]

            return Maybe.of(padded_state)
        except Exception as e:
            return Maybe.save_failure(f"Failed to generate initial state: {str(e)}")
