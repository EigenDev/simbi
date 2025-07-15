"""
Simulation execution.

This module provides components for running simulations with the Pybind11 backend.
"""

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Sequence, Optional, cast
from types import ModuleType

from simbi.functional.helpers import print_progress
from simbi.io.summary import print_simulation_parameters

from ..config.base_config import SimbiBaseConfig
from ..serialization.executor import SimulationExecutor
from .state_init import SimulationState, load_or_initialize_state
from ...io.logging import logger


@dataclass
class SimulationRunner:
    """Manages the execution of a simulation.

    This class orchestrates the initialization, execution, and output
    handling for a simulation.

    Attributes:
        config: The configuration for the simulation
        state: The current simulation state
    """

    config: SimbiBaseConfig
    state: Optional[SimulationState] = None

    def initialize(self) -> "SimulationRunner":
        """Initialize the simulation state.

        Returns:
            Self for method chaining
        """
        self.state = load_or_initialize_state(self.config).unwrap()
        self.config = self.state.config
        return self

    def _configure_backend(
        self, compute_mode: str = "cpu"
    ) -> tuple[Optional[ModuleType], Optional[Sequence[int]]]:
        """Configure and load the appropriate backend.

        Args:
            compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')

        Returns:
            The backend module or class
        """
        runtime_block_dims: Optional[Sequence[int]] = None
        # Configure block dimensions for GPU
        if compute_mode == "gpu":
            # Set environment variables for block dimensions based on dimensionality
            dims = {1: (128, 1, 1), 2: (16, 16, 1), 3: (4, 4, 4)}
            dim = min(3, self.config.dimensionality)
            block_dims = dims[dim]

            # if enviornment variables are not set, set them
            if dim == 1 and "BLOCK_X" not in os.environ:
                os.environ["BLOCK_X"] = str(block_dims[0])
                os.environ["GPU_BLOCK_Y"] = "1"
                os.environ["GPU_BLOCK_Z"] = "1"
            elif dim == 2 and "BLOCK_X" not in os.environ:
                os.environ["BLOCK_X"] = str(block_dims[0])
                os.environ["GPU_BLOCK_Y"] = str(block_dims[1])
                os.environ["GPU_BLOCK_Z"] = "1"
            elif dim == 3 and "BLOCK_X" not in os.environ:
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
            simulation_module = importlib.import_module(f"simbi.libs.{lib_mode}_ext")
            return simulation_module, runtime_block_dims
        except ImportError as e:
            logger.info(f"Error loading simulation backend: {e}")
            logger.info("Running in demo mode - no actual simulation will be executed")
            return None, None

    def run(self, compute_mode: str = "cpu") -> None:
        """Run the simulation.

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

        # Print key simulation parameters
        print_simulation_parameters(execution_dict, gpu_block_dims)
        # Give the user a moment to read the parameters
        print_progress()

        # Run the simulation
        if self.state.conserved_state is not None:
            # Reshape for contiguous memory layout.
            # Since the backend expects an array of structs
            # layout, we transpose the conserved state.
            cons_contig = self.state.conserved_state.reshape(
                self.state.conserved_state.shape[0], -1
            ).T
            prim_contig = self.state.primitive_state.reshape(
                self.state.primitive_state.shape[0], -1
            ).T

            # Create scale factor and derivative functions
            a = self.config.scale_factor or (lambda t: 1.0)
            adot = self.config.scale_factor_derivative or (lambda t: 0.0)
            if self.state.staggered_bfields:
                staggered_fields = [b.flat for b in self.state.staggered_bfields[::-1]]
            else:
                staggered_fields = []

            # Execute the simulation
            backend.run_simulation(
                cons_array=cons_contig,
                prim_array=prim_contig,
                staggered_bfields=staggered_fields,
                sim_info=execution_dict,
                a=a,
                adot=adot,
            )
        else:
            logger.info("Error: Simulation state not initialized properly")


def run_simulation(config: SimbiBaseConfig, compute_mode: str = "cpu") -> None:
    """Run a simulation with the given configuration.

    Args:
        config: The simulation configuration
        compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')
    """
    runner = SimulationRunner(config)
    runner.run(compute_mode=compute_mode)
