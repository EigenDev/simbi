from dataclasses import dataclass
from ..config import GPUConfig
from ...io.logging import logger
from ...functional.utilities import compose, pipe
from .builder import SimStateBuilder
from typing import Any
from pathlib import Path
from ..simulation.state_init import SimulationBundle
import numpy as np
import os
import importlib


@dataclass(frozen=True)
class SimulationRunner:
    bundle: SimulationBundle

    def _configure_gpu_environment(self) -> None:
        """Configure GPU environment variables"""
        gpu_config = GPUConfig.from_dimension(self.bundle.mesh_config.dimensionality)
        os.environ["GPU_BLOCK_X"] = str(gpu_config.block_dims[0])
        os.environ["GPU_BLOCK_Y"] = str(gpu_config.block_dims[1])
        os.environ["GPU_BLOCK_Z"] = str(gpu_config.block_dims[2])
        logger.info(f"Using GPU block dimensions: {gpu_config.block_dims})")

    def _setup_compute_environment(self, compute_mode: str) -> Any:
        """Configure compute environment and return execution module"""
        if compute_mode in ["cpu", "omp"]:
            logger.verbose(
                "Using OpenMP multithreading"
                if "USE_OMP" in os.environ
                else "Using STL std::thread multithreading"
            )
        else:
            self._configure_gpu_environment()

        lib_mode = "cpu" if compute_mode in ["cpu", "omp"] else "gpu"
        return getattr(
            importlib.import_module(f".{lib_mode}_ext", package="simbi.libs"),
            "SimState",
        )

    def _prepare_data_directory(self) -> str:
        """Ensure data directory exists"""
        data_path = Path(self.bundle.io_config.data_directory)
        if not data_path.is_dir():
            data_path.mkdir(parents=True)
            logger.info(f"Created data directory: {data_path}")
        return str(Path)

    def _prepare_simulation_state(self) -> dict[str, Any]:
        """Convert SimulationBundle to execution format"""
        # return the cython-compatible state
        return SimStateBuilder.build(self.bundle)

    def _execute_simulation(self, executor: Any, sim_state: dict[str, Any]) -> None:
        """Execute simulation using loaded module"""
        # Reshape state for contiguous memory access
        state_contig = self.bundle.state.reshape(self.bundle.state.shape[0], -1)
        # Execute simulation
        return executor().run(
            state=state_contig,
            sim_info=sim_state,
            a=self.bundle.mesh_config.scale_factor or (lambda t: 1.0),
            adot=self.bundle.mesh_config.scale_factor_derivative or (lambda t: 0.0),
        )

    def run(self, **kwargs) -> None:
        """Run simulation with functional composition"""
        return pipe(
            None,
            lambda _: self._prepare_data_directory(),
            lambda _: self._setup_compute_environment(kwargs["compute_mode"]),
            lambda executor: (executor, self._prepare_simulation_state()),
            lambda args: self._execute_simulation(args[0], args[1]),
        )
