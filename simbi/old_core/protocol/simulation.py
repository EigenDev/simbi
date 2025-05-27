from typing import Protocol, Any


class SimulationProtocol(Protocol):
    def run(self, **kwargs: Any) -> None: ...
