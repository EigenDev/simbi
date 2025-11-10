"""
Simulation execution.

This module provides components for running simulations with the Pybind11 backend.
"""

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Sequence, cast

from simbi.core.serialization.executor import SimulationExecutor
from simbi.functional.helpers import print_progress
from simbi.io.logging import logger
from simbi.io.summary import print_simulation_parameters
from simbi.simulation.base import BaseProblemConfig

from .state_init import (
    SimulationState,
    load_or_initialize_state,
    validate_generator_output,
)


@dataclass
class SimulationRunner:
    """
    Manages the execution of a simulation.

    This class orchestrates the initialization, execution, and output
    handling for a simulation.

    Attributes:
        config: The configuration for the simulation
        state: The current simulation state (lightweight - just generators)
    """

    config: BaseProblemConfig
    state: Optional[SimulationState] = None

    def initialize(self) -> "SimulationRunner":
        """
        Initialize the simulation state.

        This is now very lightweight - just extracts generator functions.

        Returns:
            Self for method chaining
        """
        self.state = load_or_initialize_state(self.config)
        self.config = self.state.config
        return self

    def validate(self) -> "SimulationRunner":
        """
        Validate generator output (optional diagnostic step).

        Returns:
            Self for method chaining

        Raises:
            ValueError: If generator produces invalid output
        """
        valid, error = validate_generator_output(self.config)
        if not valid:
            raise ValueError(f"Generator validation failed: {error}")

        logger.info("✓ Generator validation passed")
        return self

    def _configure_backend(
        self, compute_mode: str = "cpu"
    ) -> tuple[Optional[ModuleType], Optional[Sequence[int]]]:
        """
        Configure and load the appropriate backend.

        Args:
            compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')

        Returns:
            Tuple of (backend module, GPU block dimensions if applicable)
        """
        runtime_block_dims: Optional[Sequence[int]] = None

        # Configure block dimensions for GPU
        if compute_mode == "gpu":
            dims = {1: (128, 1, 1), 2: (16, 16, 1), 3: (4, 4, 4)}
            dim = min(3, self.config.dimensionality)
            block_dims = dims[dim]

            # Set environment variables if not already set
            if "BLOCK_X" not in os.environ:
                os.environ["BLOCK_X"] = str(block_dims[0])
                os.environ["GPU_BLOCK_Y"] = str(block_dims[1])
                os.environ["GPU_BLOCK_Z"] = str(block_dims[2])

            runtime_block_dims = (
                int(os.environ.get("BLOCK_X", block_dims[0])),
                int(os.environ.get("GPU_BLOCK_Y", block_dims[1])),
                int(os.environ.get("GPU_BLOCK_Z", block_dims[2])),
            )

        # Import the appropriate module
        lib_mode = "cpu" if compute_mode in ["cpu", "omp"] else "gpu"
        try:
            simulation_module = importlib.import_module(
                f"simbi.libs.{lib_mode}_ext"
            )
            return simulation_module, runtime_block_dims
        except ImportError as e:
            logger.info(f"Error loading simulation backend: {e}")
            logger.info(
                "Running in demo mode - no actual simulation will be executed"
            )
            return None, None

    def run(self, compute_mode: str = "cpu") -> None:
        """
        Run the simulation.

        Args:
            compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')
        """
        if self.state is None:
            self.initialize()
            self.state = cast(SimulationState, self.state)

        # Convert configuration to execution format
        execution_dict = SimulationExecutor.to_execution_dict(self.config)

        # Configure backend
        backend, gpu_block_dims = self._configure_backend(compute_mode)
        if backend is None:
            logger.info("Demo mode: Simulation would execute with parameters:")
            for key, value in sorted(execution_dict.items()):
                logger.info(f"  {key}: {value}")
            return

        print_simulation_parameters(execution_dict, gpu_block_dims)
        print_progress()

        # Get fresh iterators for C++
        prim_iterator = self.state.fresh_primitive_iterator()
        bfield_iterators = self.state.fresh_bfield_iterators() or []

        # Create scale factor functions
        scale_factor = self.config.scale_factor or (lambda t: 1.0)
        scale_factor_derivative = self.config.scale_factor_derivative or (
            lambda t: 0.0
        )

        backend.run_simulation(
            prim_gen=prim_iterator,
            staggered_bfields=bfield_iterators,
            sim_info=execution_dict,
            a=scale_factor,
            adot=scale_factor_derivative,
        )


def run_simulation(
    config: BaseProblemConfig,
    compute_mode: str = "cpu",
    validate: bool = False,
) -> None:
    """
    Run a simulation with the given configuration.

    Args:
        config: The simulation configuration
        compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')
        validate: Whether to validate generator output before running

    Example:
        >>> from simbi.problems import SodProblem
        >>> config = SodProblem(resolution=1000, end_time=0.2)
        >>> run_simulation(config, compute_mode='cpu', validate=True)
    """
    runner = SimulationRunner(config)

    if validate:
        runner.validate()

    runner.run(compute_mode=compute_mode)
