from dataclasses import dataclass


@dataclass(frozen=True)
class GPUConfig:
    """GPU configuration settings"""

    block_dims: tuple[int, int, int]

    @classmethod
    def from_dimension(cls, dim: int) -> "GPUConfig":
        dims = {1: (128, 1, 1), 2: (16, 16, 1), 3: (4, 4, 4)}
        return GPUConfig(block_dims=dims[dim])
