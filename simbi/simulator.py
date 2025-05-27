# Simulator Driver
# Marcus DuPont
# New York University
# 06/10/2020

from .core.config.base_config import SimbiBaseConfig
from .core.simulation.runner import SimulationRunner
from typing import Any


class Hydro:
    """Interface for simbi simulations using the new core."""

    def __init__(self, config: SimbiBaseConfig) -> None:
        """Initialize a hydro simulation from a configuration.

        Args:
            config: A SimbiBaseConfig instance
        """
        self.runner = SimulationRunner(config)
        self.state = None  # Will be initialized when simulate is called

    def simulate(self, compute_mode: str) -> None:
        """Run the simulation with the given CLI arguments."""
        self.runner.run(compute_mode=compute_mode)
