"""
Simulation execution.

This module provides components for running simulations with the Cython backend.
"""

import importlib
import os
from dataclasses import dataclass
from typing import Any, Optional, cast

from ..config.base_config import SimbiBaseConfig
from ..serialization.executor import SimulationExecutor
from .state_init import SimulationState, load_or_initialize_state


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
        return self

    def _configure_backend(self, compute_mode: str = "cpu") -> Any:
        """Configure and load the appropriate backend.

        Args:
            compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')

        Returns:
            The backend module or class
        """
        # Configure block dimensions for GPU
        if compute_mode == "gpu":
            # Set environment variables for block dimensions based on dimensionality
            dims = {1: (128, 1, 1), 2: (16, 16, 1), 3: (4, 4, 4)}
            dim = min(3, self.config.dimensionality)
            block_dims = dims[dim]

            os.environ["GPU_BLOCK_X"] = str(block_dims[0])
            os.environ["GPU_BLOCK_Y"] = str(block_dims[1])
            os.environ["GPU_BLOCK_Z"] = str(block_dims[2])
            print(f"Using GPU block dimensions: {block_dims}")

        # Import the appropriate module
        lib_mode = "cpu" if compute_mode in ["cpu", "omp"] else "gpu"
        try:
            simulation_module = importlib.import_module(f"simbi.libs.{lib_mode}_ext")
            return simulation_module.SimState
        except ImportError as e:
            print(f"Error loading simulation backend: {e}")
            print("Running in demo mode - no actual simulation will be executed")
            return None

    def run(self, compute_mode: str = "cpu") -> None:
        """Run the simulation.

        Args:
            compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')
        """
        if self.state is None:
            self.initialize()
            self.state = cast(SimulationState, self.state)

        # Convert configuration to execution format
        execution_dict = SimulationExecutor.to_execution_dict(
            self.config, self.state.staggered_bfields
        )

        # Configure backend
        backend = self._configure_backend(compute_mode)
        if backend is None:
            print("Demo mode: Simulation would execute with parameters:")
            for key, value in sorted(execution_dict.items()):
                print(f"  {key}: {value}")
            return

        # Print key simulation parameters
        print(
            f"Running {self.config.dimensionality}D {self.config.regime.value} simulation"
        )
        print(f"Resolution: {execution_dict['resolution']}")
        print(f"CFL number: {execution_dict['cfl_number']}")
        print(f"Output directory: {execution_dict['data_directory']}")
        print(f"Adiabatic: {execution_dict['adiabatic_index']}")
        print(f"Solver: {execution_dict['solver']}")
        print(f"Time step: {execution_dict['temporal_order']}")
        print(f"Spatial order: {execution_dict['spatial_order']}")
        zzz = input("Press Enter to continue or Ctrl+C to abort...")
        # Run the simulation
        if self.state.conserved_state is not None:
            # Reshape for contiguous memory layout
            state_contig = self.state.conserved_state.reshape(
                self.state.conserved_state.shape[0], -1
            )

            # Create scale factor and derivative functions
            a = self.config.scale_factor or (lambda t: 1.0)
            adot = self.config.scale_factor_derivative or (lambda t: 0.0)
            # Execute the simulation
            backend().run(state=state_contig, sim_info=execution_dict, a=a, adot=adot)
        else:
            print("Error: Simulation state not initialized properly")


def run_simulation(config: SimbiBaseConfig, compute_mode: str = "cpu") -> None:
    """Run a simulation with the given configuration.

    Args:
        config: The simulation configuration
        compute_mode: Backend compute mode ('cpu', 'omp', or 'gpu')
    """
    runner = SimulationRunner(config)
    runner.run(compute_mode=compute_mode)
