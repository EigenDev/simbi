# Simulator Driver
# Marcus DuPont
# New York University
# 06/10/2020

from .core.config.base_config import BaseConfig
from .core.simulation.runner import SimulationRunner
from typing import Any


class Hydro:
    def __init__(self, config: BaseConfig) -> None:
        bundle = config.to_simulation_bundle().unwrap()
        self.state = bundle.state
        self.runner = SimulationRunner(bundle)

    def simulate(self, **cli_args: dict[str, Any]) -> None:
        self.runner.run(**cli_args)
