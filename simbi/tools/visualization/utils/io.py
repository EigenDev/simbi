from typing import Any, Generator, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from ....functional import read_file as util_read_file


@dataclass
class SimulationData:
    """Container for simulation data"""

    fields: dict[str, np.ndarray]
    setup: dict[str, Any]
    mesh: dict[str, np.ndarray]
    immersed_bodies: Optional[dict[str, Any]]


class DataManager:
    """Handles data I/O operations"""

    def __init__(self, files: list[str] | str, movie_mode: bool = False):
        self.files = files
        self.file_list, self.frame_count = DataManager.get_file_list(self.files)
        self.movie_mode = movie_mode
        if self.movie_mode:
            self.file_list_iter = iter(self.file_list)

    @staticmethod
    def read_file(path: str) -> SimulationData:
        """Read simulation data from file"""
        fpath = Path(path)
        if not fpath.exists():
            raise FileNotFoundError(f"File {path} not found")

        try:
            fields, setup, mesh, immersed_bodies = util_read_file(path)
            return SimulationData(fields, setup, mesh, immersed_bodies)
        except Exception as e:
            raise IOError(f"Failed to read {path}: {str(e)}")

    @staticmethod
    def get_file_list(files: str, sort: bool = False) -> tuple[list, int]:
        """Get sorted list of files and frame count"""
        if isinstance(files, dict):
            file_list = {k: sorted(v) if sort else v for k, v in files.items()}
            frame_count = len(next(iter(file_list.values())))
        else:
            file_list = sorted(files) if sort else files
            frame_count = len(file_list)

        return file_list, frame_count

    def iter_files(self) -> Generator[SimulationData, None, None]:
        """Iterate over files and yield simulation data"""
        if not self.movie_mode:
            for file in self.file_list:
                yield DataManager.read_file(file)
        else:
            yield DataManager.read_file(next(self.file_list_iter))

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata from the first file"""
        if not self.file_list:
            raise ValueError("No files to read metadata from")

        first_file = self.file_list[0]
        data = DataManager.read_file(first_file)
        return data.setup
