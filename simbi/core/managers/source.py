from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import subprocess
import textwrap
from time import time


@dataclass
class SourceManager:
    """Manages compilation of source code"""

    lib_dir: Path
    config_name: str = ""
    _compile_timestamps: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lib_dir.mkdir(parents=True, exist_ok=True)

    def compile_sources(
        self, class_name: str, sources: dict[str, str | None]
    ) -> dict[str, Path]:
        """Compile source files and return paths to libraries"""
        compiled_libs = {}
        self.config_name = class_name

        # Clear previous timestamps for this config
        self._compile_timestamps = {
            k: v
            for k, v in self._compile_timestamps.items()
            if not k.startswith(f"{class_name}.")
        }

        for source_name, source_code in sources.items():
            if not source_code:
                continue

            cpp_file = self.lib_dir / f"{class_name}.{source_name}.cpp"
            so_file = self.lib_dir / f"{class_name}.{source_name}.so"

            with open(cpp_file, "w+") as f:
                f.write(textwrap.dedent(source_code))

            try:
                subprocess.run(
                    [
                        "c++",
                        "-shared",
                        "-std=c++20",
                        "-O3",
                        "-fPIC",
                        "-o",
                        str(so_file),
                        str(cpp_file),
                    ],
                    check=True,
                )
                # Record compilation timestamp
                self._compile_timestamps[f"{class_name}.{source_name}"] = time()
                compiled_libs[source_name] = so_file
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to compile {source_name}: {e}")

        return compiled_libs

    def get_library_path(self, source_name: str) -> Optional[Path]:
        """Get library path if freshly compiled, None otherwise"""
        lib_key = f"{self.config_name}.{source_name}"
        so_file = self.lib_dir / f"{lib_key}.so"

        # Return None if:
        # 1. Library doesn't exist
        # 2. No compilation timestamp (wasn't compiled this session)
        # 3. Library exists but is from a previous session
        if not so_file.exists() or lib_key not in self._compile_timestamps:
            return None

        return so_file
