import numpy as np
import itertools
from dataclasses import dataclass
from ..types.typing import (
    InitialStateType,
    MHDStateGenerators,
    PureHydroStateGenerator,
    GasStateGenerator,
)
from typing import Sequence, Optional, Union, Any, Generator, TypeGuard, cast
from ...functional import Maybe, to_iterable
from numpy.typing import NDArray
from pathlib import Path


def is_mhd_generator(gen: InitialStateType) -> TypeGuard[MHDStateGenerators]:
    """Type guard to narrow generator function type for MHD problems"""
    try:
        return isinstance(gen, tuple) and len(gen) == 4
    except:
        return False


def is_pure_hydro_generator(
    gen: InitialStateType,
) -> TypeGuard[PureHydroStateGenerator]:
    """Type guard to narrow generator function type for pure hydro problems"""
    return callable(gen) and not is_mhd_generator(gen)


@dataclass(frozen=True)
class InitializationConfig:
    """Configuration for simulation initialization"""

    initial_primitive_gen: InitialStateType
    resolution: Sequence[int]
    bounds: Union[Sequence[Sequence[float]] | Sequence[float]]
    checkpoint_file: Optional[str | Path] = None

    def evaluate(
        self, pad_width: int, nvars: int
    ) -> Maybe[tuple[NDArray[np.floating[Any]], Sequence[NDArray[np.floating[Any]]]]]:
        """Lazy evaluation of initial state"""
        try:
            mhd = False
            staggered_bfields = []
            if nvars == 9:
                mhd = True

            padded_shape = (nvars,) + tuple(
                r + 2 * pad_width for r in to_iterable(self.resolution)[::-1]
            )
            padded_state = np.zeros(padded_shape, dtype=np.float64)

            if mhd:
                if not is_mhd_generator(self.initial_primitive_gen):
                    return Maybe.save_failure(
                        "Expected a tuple of generators for MHD problems"
                    )

                gens: MHDStateGenerators = self.initial_primitive_gen

                gas, b1_gen, b2_gen, b3_gen = (g() for g in gens)
                gas_gen, dummy_gas_gen = itertools.tee(gas)

                b1_shape = (
                    self.resolution[2],
                    self.resolution[1],
                    self.resolution[0] + 1,
                )
                b2_shape = (
                    self.resolution[2],
                    self.resolution[1] + 1,
                    self.resolution[0],
                )
                b3_shape = (
                    self.resolution[2] + 1,
                    self.resolution[1],
                    self.resolution[0],
                )

                staggered_bfields = [
                    np.fromiter(b1_gen, dtype=float).reshape(b1_shape),
                    np.fromiter(b2_gen, dtype=float).reshape(b2_shape),
                    np.fromiter(b3_gen, dtype=float).reshape(b3_shape),
                ]
            else:
                if not is_pure_hydro_generator(self.initial_primitive_gen):
                    return Maybe.save_failure(
                        "Expected a single generator for non-MHD problems"
                    )

                gen: GasStateGenerator = self.initial_primitive_gen()
                gas_gen, dummy_gas_gen = itertools.tee(gen)

            # check that the generator yields the correct number of variables
            # (they can yield up to the pressure, or to passive scalar if the user chooses)
            n_yielded = len(next(cast(GasStateGenerator, dummy_gas_gen)))
            ngas_vars = nvars if not mhd else 6
            if not (ngas_vars - 1 <= n_yielded <= ngas_vars):
                if not mhd or ngas_vars != 5:
                    return Maybe.save_failure(
                        f"Initial state generator does must yield a number of gas variables between {ngas_vars - 1} and {ngas_vars}"
                    )

            interior = (slice(pad_width, -pad_width),) * len(self.resolution)
            interior_shape = padded_state[:n_yielded, *interior].shape
            padded_state[:n_yielded, *interior] = np.fromiter(
                gas_gen, dtype=(float, n_yielded)
            ).T.reshape(interior_shape)

            # fill the ghost cells at the edges with ones from the interior
            for i, _ in enumerate(self.resolution):
                for j in range(pad_width):
                    padded_state[
                        :n_yielded,
                        *(
                            slice(j, j + 1) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ] = padded_state[
                        :n_yielded,
                        *(
                            slice(pad_width, pad_width + 1) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ]

                    padded_state[
                        :n_yielded,
                        *(
                            slice(-j - 1, -j or None) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ] = padded_state[
                        :n_yielded,
                        *(
                            slice(-pad_width - 1, -pad_width) if k == i else slice(None)
                            for k in range(len(self.resolution))
                        ),
                    ]

            return Maybe.of((padded_state, staggered_bfields))
        except Exception as e:
            return Maybe.save_failure(f"Failed to generate initial state: {str(e)}")
