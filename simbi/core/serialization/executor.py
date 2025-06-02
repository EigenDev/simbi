"""
Configuration serialization for simulation execution.

This module provides utilities to convert SimbiBaseConfig objects into
a format suitable for passing to the Cython/C++ backend.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence, Union, Optional
from pathlib import Path
from numpy.typing import NDArray
import numpy as np

from ..config.base_config import SimbiBaseConfig


@dataclass
class SimulationExecutor:
    """Handles conversion of configuration to execution format"""

    @staticmethod
    def prepare_data_directory(config: SimbiBaseConfig) -> None:
        """Ensure data directory exists"""
        data_dir = Path(config.data_directory)
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created data directory: {data_dir}")

    @staticmethod
    def to_execution_dict(
        config: SimbiBaseConfig,
        staggered_bfields: Optional[list[NDArray[np.floating]]] = None,
    ) -> dict[str, Any]:
        """
        Convert configuration to execution dictionary.

        Args:
            config: The simulation configuration
            format: Format of the output dictionary

        Returns:
            Dictionary suitable for passing to backend executor
        """
        # Prepare data directory
        SimulationExecutor.prepare_data_directory(config)

        # Get model fields as dictionary
        model_dict = config.model_dump()

        # ensure data_directory has trailing slash
        if isinstance(model_dict["data_directory"], Path):
            model_dict["data_directory"] = str(model_dict["data_directory"])

        if not model_dict["data_directory"].endswith("/"):
            model_dict["data_directory"] += "/"

        # Add computed fields
        computed_fields = [
            "dimensionality",
            "is_mhd",
            "isothermal",
            "nvars",
            "is_relativistic",
            "mesh_motion",
            "is_homologous",
            "dlogt",
            "_immersed_bodes",  # might be loaded from checkpoint
            "_body_system",  # might be loaded from checkpoint
        ]

        for field in computed_fields:
            if hasattr(config, field):
                if dataclasses.is_dataclass(getattr(config, field)):
                    model_dict[field] = dataclasses.asdict(getattr(config, field))
                else:
                    model_dict[field] = getattr(config, field)

        # Process bounds to separate x1, x2, x3 bounds
        bounds = config.bounds
        effective_dim = config.dimensionality

        # Normalize bounds to 3D
        x1bounds = bounds[0] if len(bounds) > 0 else (0.0, 1.0)
        x2bounds = bounds[1] if len(bounds) > 1 else (0.0, 1.0)
        x3bounds = bounds[2] if len(bounds) > 2 else (0.0, 1.0)

        # Remove bounds from dict and replace with x1bounds, etc.
        model_dict.pop("bounds", None)
        model_dict["x1bounds"] = x1bounds
        model_dict["x2bounds"] = x2bounds
        model_dict["x3bounds"] = x3bounds

        # Process boundary conditions
        bcs = SimulationExecutor._process_boundary_conditions(
            config.boundary_conditions, effective_dim
        )
        model_dict["boundary_conditions"] = bcs

        # Process paths to strings
        for key, value in list(model_dict.items()):
            if isinstance(value, Path):
                model_dict[key] = str(value)

        # Process callbacks to None
        # (can't be serialized, backend should have its own implementations)
        for key, value in list(model_dict.items()):
            if callable(value):
                model_dict[key] = None

        # Normalize resolution to 3D array for backend
        resolution = model_dict["resolution"]
        if isinstance(resolution, int):
            model_dict["resolution"] = [resolution, 1, 1]
        elif len(resolution) == 1:
            model_dict["resolution"] = [resolution[0], 1, 1]
        elif len(resolution) == 2:
            model_dict["resolution"] = [resolution[0], resolution[1], 1]

        if staggered_bfields is not None:
            model_dict["bfield"] = [b.flat for b in staggered_bfields]
        else:
            model_dict["bfield"] = []

        return model_dict

    @staticmethod
    def _process_boundary_conditions(
        boundary_conditions: Union[str, Sequence[str]], effective_dim: int
    ) -> list[str]:
        """Process and normalize boundary conditions"""
        # Handle string case
        if isinstance(boundary_conditions, str):
            # Replicate boundary condition for all faces
            return [boundary_conditions] * (2 * effective_dim)

        # Handle sequence case
        bcs = list(boundary_conditions)
        num_bcs = len(bcs)
        num_faces = 2 * effective_dim

        # Case 1: One BC per dimension (same for inner and outer)
        if num_bcs == effective_dim:
            return [bc for bc in bcs for _ in range(2)]

        # Case 2: One BC for each face
        elif num_bcs == num_faces:
            return bcs

        # Case 3: Single BC for all faces
        elif num_bcs == 1:
            return bcs * num_faces

        else:
            # Default to outflow for unspecified boundaries
            return ["outflow"] * num_faces
