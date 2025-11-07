"""
Streamlined simulation state initialization using iterator pattern.

This module provides efficient initialization by passing generators directly
to C++, eliminating intermediate NumPy array allocation.
"""

from dataclasses import dataclass
from typing import Callable, Optional, cast

from simbi.core.types import (
    GasStateFunction,
    GasStateGenerator,
    InitialStateType,
    MHDStateGenerators,
    StaggeredBFieldGenerator,
)
from simbi.simulation.base import BaseProblemConfig
from simbi.simulation.state import SimulationStateSpec


def _validate_hydro_generator(
    gen_func: Callable, dimensionality: int, num_samples: int = 3
) -> None:
    """
    Validate hydro generator output.

    NOTE: This will call gen_func() and consume the first `num_samples` values.
    The generator function will be called again at runtime to get a fresh iterator.
    Generator functions MUST be pure (no side effects) and repeatable.

    Expected format:
    - 1D: (rho, vx, p) or (rho, vx, p, chi)
    - 2D: (rho, vx, vy, p) or (rho, vx, vy, p, chi)
    - 3D: (rho, vx, vy, vz, p) or (rho, vx, vy, vz, p, chi)

    Raises:
        ValueError: If generator yields invalid tuples
    """
    gen = gen_func()
    expected_min = dimensionality + 2  # rho + v + p
    expected_max = dimensionality + 3  # + chi

    for i in range(num_samples):
        try:
            value = next(gen)
        except StopIteration:
            raise ValueError(f"Generator exhausted after {i} samples")

        # Must be a sequence
        if not isinstance(value, (tuple, list)):
            raise ValueError(
                f"Generator must yield tuple/list, got {type(value).__name__}"
            )

        # Check length
        if not (expected_min <= len(value) <= expected_max):
            component_names = (
                ["rho", "vx", "p", "chi"]
                if dimensionality == 1
                else ["rho", "vx", "vy", "p", "chi"]
                if dimensionality == 2
                else ["rho", "vx", "vy", "vz", "p", "chi"]
            )
            raise ValueError(
                f"Generator must yield {expected_min}-{expected_max} components "
                f"for {dimensionality}D: {component_names[:expected_max]}\n"
                f"Got {len(value)} components: {value}"
            )

        # Check all numeric
        try:
            [float(v) for v in value]
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"All components must be numeric.\nGot: {value}\nError: {e}"
            )


def _validate_mhd_generator(gas_gen, bfield_gens, num_samples=3):
    """Validate MHD generator output."""
    # validation logic here
    pass


@dataclass
class SimulationState:
    """
    Container for simulation state metadata and generators.

    This is a lightweight container that holds:
    - Generator functions (not arrays!)
    - Configuration metadata
    - FMR hierarchy info (if applicable)

    Memory is owned entirely by C++.
    """

    # Generator functions (not consumed iterators!)
    primitive_gen_func: GasStateFunction
    state_spec: SimulationStateSpec

    # Optional: MHD B-field generators
    bfield_gen_funcs: Optional[list[StaggeredBFieldGenerator]] = None

    # @property
    # def is_mhd(self) -> bool:
    #     """Check if this is an MHD simulation."""
    #     return self.bfield_gen_funcs is not None

    # @property
    # def has_fmr(self) -> bool:
    #     """Check if simulation has FMR enabled."""
    #     return self.state_spec.fmr_enabled

    def to_execution_dict(self) -> dict:
        return self.state_spec.model_dump(
            mode="python",
            exclude={"source_config"},
            by_alias=True,
        )

    def fresh_primitive_iterator(self) -> GasStateGenerator:
        """
        Get a fresh iterator for primitive variables.

        Returns:
            Fresh iterator that can be consumed by C++
        """
        return self.primitive_gen_func()

    def fresh_bfield_iterators(
        self,
    ) -> Optional[list[StaggeredBFieldGenerator]]:
        """
        Get fresh iterators for B-field components.

        Returns:
            List of [Bx, By, Bz] iterators, or None for non-MHD
        """
        if self.bfield_gen_funcs is None:
            return None
        return [gen for gen in self.bfield_gen_funcs]


def is_mhd_generator(gen: InitialStateType) -> bool:
    """
    Check if generator tuple is for MHD simulation.

    Args:
        gen: Generator or tuple of generators

    Returns:
        True if this is an MHD generator tuple (gas + 3 B-fields)
    """
    return not callable(gen) and len(gen) == 4


def initialize_state(config: BaseProblemConfig) -> SimulationState:
    """
    Initialize simulation state from configuration.

    This is now trivial - we just extract the generator functions
    and store them. No arrays are created. C++ will consume the
    iterators directly.

    Args:
        config: The simulation configuration

    Returns:
        Lightweight SimulationState with generator functions

    Example:
        >>> config = SodProblem(resolution=1000, end_time=1.0)
        >>> state = initialize_state(config)
        >>> # State now holds generator functions, ready for C++
        >>> iterator = state.fresh_primitive_iterator()
        >>> # C++ consumes this iterator directly
    """
    state_spec = config.build_state()
    # Get the initial state specification from config
    initial_state = config.initial_primitive_state()

    # Determine if this is MHD or pure hydro
    if is_mhd_generator(initial_state):
        # Unpack MHD generators: (gas, Bx, By, Bz)
        gen_tuple = cast(MHDStateGenerators, initial_state)
        gas_gen_func, bx_gen_func, by_gen_func, bz_gen_func = gen_tuple

        _validate_mhd_generator(
            gas_gen_func, [bx_gen_func, by_gen_func, bz_gen_func]
        )
        # Store all generator functions (NOT called yet)
        bfield_gen_funcs = [bx_gen_func(), by_gen_func(), bz_gen_func()]

        return SimulationState(
            primitive_gen_func=gas_gen_func,
            bfield_gen_funcs=bfield_gen_funcs,
            state_spec=state_spec,
        )
    else:
        # Pure hydro case
        gas_gen_func = cast(GasStateFunction, initial_state)
        _validate_hydro_generator(gas_gen_func, state_spec.dimensionality)

        return SimulationState(
            primitive_gen_func=gas_gen_func,
            bfield_gen_funcs=None,
            state_spec=state_spec,
        )


def load_or_initialize_state(
    config: BaseProblemConfig,
) -> SimulationState:
    """
    Load state from checkpoint if specified, or initialize from scratch.

    Args:
        config: Simulation configuration

    Returns:
        SimulationState (either from checkpoint or fresh initialization)

    Note:
        Checkpoint loading is handled by C++. Python just passes the
        checkpoint path through the config.
    """
    state_spec = config.build_state()
    # Check if resuming from checkpoint
    if hasattr(config, "checkpoint_file") and state_spec.checkpoint_file:
        # For checkpoint resume, we don't need generators
        # C++ will load the state from the checkpoint file
        # We create a dummy state that signals "resume mode"

        # Create a no-op generator (won't be used)
        def dummy_gen() -> GasStateGenerator:
            raise RuntimeError(
                "Generator should not be called when resuming from checkpoint"
            )
            yield

        return SimulationState(
            primitive_gen_func=dummy_gen,
            bfield_gen_funcs=None,
            state_spec=state_spec,
        )

    # Fresh initialization
    return initialize_state(config)


def validate_generator_output(
    config: BaseProblemConfig,
    state_spec: SimulationStateSpec,
    num_samples: int = 10,
) -> tuple[bool, Optional[str]]:
    """
    Validate that generator produces correctly-shaped output.

    This is a diagnostic tool to help users debug their initial_primitive_state
    implementations. It consumes a few values from the generator to check shape.

    Args:
        config: Configuration to validate
        num_samples: Number of samples to test

    Returns:
        (is_valid, error_message)

    Example:
        >>> config = SodProblem(resolution=1000)
        >>> valid, error = validate_generator_output(config)
        >>> if not valid:
        ...     print(f"Generator error: {error}")
    """
    try:
        initial_state = config.initial_primitive_state()

        if is_mhd_generator(initial_state):
            gen_tuple = cast(MHDStateGenerators, initial_state)
            gas_gen_func, bx_gen, by_gen, bz_gen = gen_tuple

            # Test gas generator
            gas_iter = gas_gen_func()
            for i in range(num_samples):
                values = next(gas_iter)

                # Check it's a sequence
                if not hasattr(values, "__len__"):
                    return (
                        False,
                        f"Generator must yield sequences, got {type(values)}",
                    )

                # Check expected number of components
                # For MHD: (rho, vx, vy, vz, p, ...) - at least 5
                if len(values) < 5:
                    return (
                        False,
                        f"MHD generator must yield at least 5 values, got {len(values)}",
                    )

                # Check all values are numeric
                try:
                    [float(v) for v in values]
                except (ValueError, TypeError) as e:
                    return False, f"All values must be numeric: {e}"

            # Test B-field generators
            for name, gen_func in [
                ("Bx", bx_gen),
                ("By", by_gen),
                ("Bz", bz_gen),
            ]:
                b_iter = gen_func()
                for i in range(num_samples):
                    value = next(b_iter)
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        return (
                            False,
                            f"{name} generator must yield numeric values",
                        )

        else:
            # Pure hydro
            gas_gen_func = cast(GasStateFunction, initial_state)
            gas_iter = gas_gen_func()

            for i in range(num_samples):
                values = next(gas_iter)

                if not hasattr(values, "__len__"):
                    return (
                        False,
                        f"Generator must yield sequences, got {type(values)}",
                    )

                # For hydro: (rho, v1, [v2, v3,] p) - at least 3 (1D)
                expected_min = state_spec.dimensionality + 2
                if len(values) < expected_min:
                    return False, (
                        f"Generator must yield at least {expected_min} values "
                        f"for {state_spec.dimensionality}D, got {len(values)}"
                    )

                try:
                    [float(v) for v in values]
                except (ValueError, TypeError) as e:
                    return False, f"All values must be numeric: {e}"

        return True, None

    except StopIteration:
        return False, f"Generator exhausted after {num_samples} samples"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def estimate_generator_size(state_spec: SimulationStateSpec) -> dict[str, int]:
    """
    Estimate number of values the generator should produce.

    Useful for debugging - tells user how many values their generator
    needs to yield.

    Args:
        config: Configuration to analyze

    Returns:
        Dictionary with expected sizes for each generator

    Example:
        >>> config = SodProblem(resolution=1000)
        >>> sizes = estimate_generator_size(config)
        >>> print(f"Your generator must yield {sizes['primitive']} tuples")
    """

    # Get resolution
    if isinstance(state_spec.resolution, int):
        resolution = [state_spec.resolution, 1, 1]
    else:
        resolution = list(state_spec.resolution)

    # Active dimensions
    nx = resolution[0]
    ny = resolution[1] if len(resolution) > 1 else 1
    nz = resolution[2] if len(resolution) > 2 else 1

    # For MHD, all dimensions are active
    if state_spec.is_mhd:
        ny = max(ny, 1)
        nz = max(nz, 1)
        active_resolution = [nx, ny, nz]
    else:
        active_resolution = [r for r in resolution if r > 1]

    # Total cells (without ghost zones)
    total_cells = 1
    for r in active_resolution:
        total_cells *= r

    result = {
        "primitive": total_cells,
        "components_per_tuple": state_spec.nvars,
    }

    # Add B-field sizes for MHD (staggered grids)
    if state_spec.is_mhd:
        result["bx"] = (nx + 1) * ny * nz
        result["by"] = nx * (ny + 1) * nz
        result["bz"] = nx * ny * (nz + 1)

    return result
