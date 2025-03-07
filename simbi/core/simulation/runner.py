from dataclasses import dataclass
from ..config import GPUConfig
from ...io.logging import logger
from ...functional.utilities import pipe
from ...functional.helpers import tuple_of_tuples, print_progress
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

    def _prepare_simulation_state(self, cli_args: dict[str, Any]) -> dict[str, Any]:
        """Convert SimulationBundle to execution format"""
        # return the cython-compatible state
        return SimStateBuilder.build(self.bundle.copy_from(cli_args))

    def _execute_simulation(self, executor: Any, sim_state: dict[str, Any]) -> None:
        """Execute simulation using loaded module"""
        # Reshape state for contiguous memory access
        state_contig = self.bundle.state.reshape(self.bundle.state.shape[0], -1)
        # Give user a chance to check their params
        print_progress()

        # Execute simulation
        executor().run(
            state=state_contig,
            sim_info=sim_state,
            a=self.bundle.mesh_config.scale_factor or (lambda t: 1.0),
            adot=self.bundle.mesh_config.scale_factor_derivative or (lambda t: 0.0),
        )

    def _print_simulation_parameter_summary(
        self, sim_state: dict[str, Any]
    ) -> dict[str, Any]:
        logger.info("=" * 80)
        logger.info("Simulation Parameters")
        logger.info("=" * 80)

        def format_tuple_of_tuples(param: Any) -> str:
            if tuple_of_tuples(param):
                formatted = tuple(
                    tuple(
                        f"{x:.3f}" if isinstance(x, float) else str(x)
                        for x in inner_tuple
                    )
                    for inner_tuple in param
                )
                return str(formatted).replace("'", "").replace(" ", "")
            else:
                return str(param)

        def format_param(param: Any) -> str:
            """
            Format the parameter for logging.

            Parameters:
                param (Any): The parameter to format.

            Returns:
                str: The formatted parameter as a string.
            """
            if isinstance(param, (float, np.float64)):
                return f"{param:.3f}"
            elif callable(param):
                return f"user-defined {param.__name__} function"
            elif isinstance(param, (list, np.ndarray)):
                if len(param) > 6:
                    return f"user-defined {param.__class__.__name__} terms"
                return [format_param(p) for p in param]  # type: ignore
            elif isinstance(param, tuple):
                return format_tuple_of_tuples(param)

            x = param.decode("utf-8") if isinstance(param, bytes) else str(param)
            if x == "":
                return "None"
            return x

        for key, param in sim_state.items():
            if key not in ["bfield", "staggered_bfields"]:
                val_str = format_param(param)
                logger.info(f"{key.ljust(30, '.')} {val_str}")

        logger.info("=" * 80)

        return sim_state

    def run(self, **cli_args: Any) -> None:
        """Run simulation with functional composition"""
        pipe(
            None,
            lambda _: self._setup_compute_environment(cli_args["compute_mode"]),
            lambda executor: (executor, self._prepare_simulation_state(cli_args)),
            lambda args: (args[0], self._print_simulation_parameter_summary(args[1])),
            lambda args: self._execute_simulation(args[0], args[1]),
        )
