from typing import Protocol
from ..config.initialization import InitializationConfig
from ...functional.maybe import Maybe
from ..simulation.state_init import SimulationBundle


class StateInitializer(Protocol):
    """Protocol for initialization strategies"""

    def initialize(self, config: InitializationConfig) -> Maybe[SimulationBundle]: ...
