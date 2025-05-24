from typing import Any, Callable
from ..state.core import VisualizationState


class DataPipeline:
    """Handles data processing pipeline"""

    def __init__(self, state: VisualizationState):
        self.state = state
        self.processors: list[Callable] = []
        self.transformations: dict[str, Any] = {}

    def add_processor(self, processor: Callable) -> None:
        """Add a data processor to the pipeline"""
        self.processors.append(processor)

    def process(self) -> dict[str, Any]:
        """Run the full data processing pipeline"""
        if not self.state.data:
            return {}

        result = {"raw": self.state.data}

        # Run each processor in sequence
        for processor in self.processors:
            processor_result = processor(result, self.state)
            result.update(processor_result)

        self.transformations = result
        return result
