#!/usr/bin/env python3
"""
This script handles the building and installation of the Simbi cli code.
"""

import argparse
import json
import logging
import os
import platform as platform_module
import subprocess
import sys
import dataclasses
from dataclasses import dataclass, asdict
from functools import reduce, wraps
from pathlib import Path
from typing import Any, Callable, Final, Generic, Literal, TypeVar, override
from collections.abc import Sequence

# ======================================================================
# Type definitions
# ======================================================================

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
R = TypeVar("R")

CommandFunction = Callable[[argparse.Namespace], int]
Platform = Literal["cuda", "hip", "None"]
BuildType = Literal["release", "debug"]
InstallMode = Literal["default", "develop"]
CppVersion = Literal["c++17", "c++20"]
GPUCompilation = Literal["enabled", "disabled"]

# ======================================================================
# Constants
# ======================================================================

CACHE_FILE: Final[str] = "simbi_build_cache.txt"
GITHUB_TOPLEVEL: Final[str] = "gitrepo_home.txt"
YELLOW: Final[str] = "\033[0;33m"
GREEN: Final[str] = "\033[0;32m"
RED: Final[str] = "\033[0;31m"
RST: Final[str] = "\033[0m"  # No Color
CURRENT_CACHE_VERSION: Final[str] = "1.0"

# Default configuration
DEFAULT_CONFIG: Final[dict[str, str | bool]] = {
    "gpu_compilation": "disabled",
    "progress_bar": True,
    "column_major": False,
    "precision": "double",
    "install_mode": "default",
    "dev_arch": "",
    "build_dir": "build",
    "four_velocity": False,
    "shared_memory": True,
    "cpp_version": "c++20",
    "build_type": "release",
    "gpu_platform": "cuda",
}

# Flag overrides
FLAG_OVERRIDES: Final[dict[str, list[str]]] = {
    "precision": ["--double", "--float"],
    "gpu_compilation": ["--gpu-compilation", "--cpu-compilation"],
    "column_major": ["--row-major", "--column-major"],
    "four_velocity": ["--four-velocity", "--no-four-velocity"],
    "progress_bar": ["--progress-bar", "--no-progress-bar"],
    "shared_memory": ["--shared-memory", "--no-shared-memory"],
    "install_mode": ["develop", "default"],
    "gpu_platform": ["cuda", "hip", "None"],
    "build_type": ["release", "debug"],
    "cpp_version": ["c++17", "c++20"],
}

# Set up logging
logger = logging.getLogger("simbi")

# ======================================================================
# Functional core utilities
# ======================================================================


class Result(Generic[T, E]):
    """A result type that represents either a success or failure outcome."""

    @staticmethod
    def success(value: T) -> "Success[T, E]":
        return Success(value)

    @staticmethod
    def failure(error: E) -> "Failure[T, E]":
        return Failure(error)

    def is_success(self) -> bool:
        raise NotImplementedError

    def is_failure(self) -> bool:
        raise NotImplementedError

    def value_or(self, default: T) -> T:
        raise NotImplementedError

    def error_or(self, default: E) -> E:
        raise NotImplementedError

    def map(self, f: Callable[[T], U]) -> "Result[U, E]":
        raise NotImplementedError

    def map_error(self, f: Callable[[E], U]) -> "Result[T, U]":
        raise NotImplementedError

    def and_then(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        raise NotImplementedError

    def or_else(self, f: Callable[[E], "Result[T, U]"]) -> "Result[T, U]":
        raise NotImplementedError


class Success(Result[T, E]):
    """A successful result with a value."""

    def __init__(self, value: T):
        self._value = value

    @override
    def is_success(self) -> bool:
        return True

    @override
    def is_failure(self) -> bool:
        return False

    @override
    def value_or(self, default: T) -> T:
        return self._value

    @override
    def error_or(self, default: E) -> E:
        return default

    @override
    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        return Success(f(self._value))

    @override
    def map_error(self, f: Callable[[E], U]) -> Result[T, U]:
        return Success(self._value)

    @override
    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return f(self._value)

    @override
    def or_else(self, f: Callable[[E], Result[T, U]]) -> Result[T, E]:
        return self


class Failure(Result[T, E]):
    """A failed result with an error."""

    def __init__(self, error: E):
        self._error = error

    @override
    def is_success(self) -> bool:
        return False

    @override
    def is_failure(self) -> bool:
        return True

    @override
    def value_or(self, default: T) -> T:
        return default

    @override
    def error_or(self, default: E) -> E:
        return self._error

    @override
    def map(self, f: Callable[[T], U]) -> Result[U, E]:
        return Failure(self._error)

    @override
    def map_error(self, f: Callable[[E], U]) -> Result[T, U]:
        return Failure(f(self._error))

    @override
    def and_then(self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return Failure(self._error)

    @override
    def or_else(self, f: Callable[[E], Result[T, U]]) -> Result[T, U]:
        return f(self._error)


def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose functions from right to left."""
    if not funcs:
        return lambda x: x

    def composed_function(x: Any) -> Any:
        result = x
        for f in reversed(funcs):
            result = f(result)
        return result

    return composed_function


def pipe(value: T, *funcs: Callable[[Any], Any]) -> Any:
    """Pipe a value through a series of functions from left to right."""
    return reduce(lambda acc, f: f(acc), funcs, value)


def safe_run(
    func: Callable[..., T], default: T, error_log: str | None = None
) -> Callable[..., T]:
    """Higher-order function to safely run a function with error handling."""

    @wraps(func)
    def safe_func(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if error_log:
                logger.error(f"{error_log}: {e}")
            return default

    return safe_func


def try_sequentially(
    funcs: list[Callable[..., T | None]], fallback: T | None = None
) -> Callable[..., T | None]:
    """Try a sequence of functions until one succeeds."""

    def try_sequence(*args: Any, **kwargs: Any) -> T | None:
        for func in funcs:
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception:
                continue
        return fallback

    return try_sequence


def first_result(
    funcs: list[Callable[..., Result[T, E]]], default_error: E
) -> Callable[..., Result[T, E]]:
    """Try a sequence of functions until one returns a Success, otherwise return Failure with default_error."""

    def try_all(*args: Any, **kwargs: Any) -> Result[T, E]:
        for func in funcs:
            result = func(*args, **kwargs)
            if result.is_success():
                return result
        return Result.failure(default_error)

    return try_all


def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """Simple memoization decorator for functions with hashable arguments."""
    cache: dict[Sequence[Any], T] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = (*args, *(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


# ======================================================================
# Logging setup
# ======================================================================


def setup_logging(verbose: bool = False) -> None:
    """Set up a proper logging system."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure handlers if not already configured
    if not logger.handlers:
        # Create a custom formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Configure file handler
        file_handler = logging.FileHandler("simbi_build.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Add handlers to logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


# ======================================================================
# Configuration management with dataclasses and monads
# ======================================================================


@dataclass
class BuildConfig:
    """Strongly typed configuration for Simbi build process."""

    gpu_compilation: GPUCompilation = "disabled"
    progress_bar: bool = True
    column_major: bool = False
    precision: str = "double"
    install_mode: InstallMode = "default"
    dev_arch: str = ""
    build_dir: str = "build"
    four_velocity: bool = False
    shared_memory: bool = True
    cpp_version: CppVersion = "c++20"
    build_type: BuildType = "release"
    gpu_platform: Platform = "cuda"
    verbose: bool = False
    configure: bool = False
    cli_extras: bool = False
    visual_extras: bool = False
    venv_path: str = ".venv"
    func: CommandFunction | None = None
    _cache_version: str = CURRENT_CACHE_VERSION

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "BuildConfig":
        """Create BuildConfig from argparse namespace."""
        # Extract values from args that match our fields
        fields = {f.name for f in dataclasses.fields(cls)}
        config_dict = {k: v for k, v in vars(args).items() if k in fields}
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildConfig":
        """Create BuildConfig from dictionary."""
        fields = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result.pop("func", None)  # Function can't be serialized
        return result

    def with_updates(self, **kwargs: Any) -> "BuildConfig":
        """Create a new config with updated values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return BuildConfig.from_dict(config_dict)


def read_cache() -> Result[dict[str, Any], str]:
    """Read configuration from cache file."""
    cache_path = Path(CACHE_FILE)

    if not cache_path.exists():
        # Create an empty cache file
        cache_path.touch()
        return Result.failure("Cache file not found")

    try:
        with open(CACHE_FILE, "r") as f:
            cached_vars = json.load(f)

        # Check version
        if cached_vars.get("_cache_version") != CURRENT_CACHE_VERSION:
            return Result.failure(
                f"Cache version mismatch: {cached_vars.get('_cache_version')} vs {CURRENT_CACHE_VERSION}"
            )

        return Result.success(cached_vars)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse cache file {CACHE_FILE}, using defaults")
        return Result.failure("Invalid JSON in cache file")


def write_cache(config: BuildConfig) -> None:
    """Write configuration to cache file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(config.to_dict(), f, indent=4)


def merge_with_cli_args(config: BuildConfig, cli_args: set[str]) -> BuildConfig:
    """Merge cached configuration with command line arguments."""
    # Try to read from cache
    cache_result = read_cache()

    if cache_result.is_failure():
        return config

    cached_vars = cache_result.value_or({})
    if not cached_vars:
        return config

    # Only override with cached values if not specified in CLI
    updates = {}
    for arg, default_value in DEFAULT_CONFIG.items():
        # Skip special args
        if arg in ["verbose", "configure", "func", "cli_extras", "visual_extras"]:
            continue

        # Only use cache if not specified in CLI and matches default
        if getattr(config, arg) == default_value and arg not in cli_args:
            # Check if any flag override was specified
            if arg in FLAG_OVERRIDES and any(
                flag in cli_args for flag in FLAG_OVERRIDES[arg]
            ):
                continue

            # Use cached value if available
            if arg in cached_vars:
                updates[arg] = cached_vars[arg]

    return config.with_updates(**updates)


def validate_config(config: BuildConfig) -> Result[BuildConfig, str]:
    """Validate the build configuration."""
    # Check GPU compilation settings
    if config.gpu_compilation == "enabled":
        if config.gpu_platform == "cuda" and not is_tool("nvcc"):
            logger.warning("CUDA GPU compilation requested but nvcc not found")
            if not confirm("Continue anyway?"):
                return Result.failure("NVCC not found")
        elif config.gpu_platform == "hip" and not is_tool("hipcc"):
            logger.warning("HIP GPU compilation requested but hipcc not found")
            if not confirm("Continue anyway?"):
                return Result.failure("HIPCC not found")

    # Validate build directory
    build_dir = Path(config.build_dir)
    if build_dir.exists() and not build_dir.is_dir():
        logger.warning(f"Build path {build_dir} exists but is not a directory")
        if confirm("Remove the file and create directory?"):
            build_dir.unlink()
            build_dir.mkdir(parents=True)
        else:
            return Result.failure("Build path exists but is not a directory")

    return Result.success(config)


# ======================================================================
# System utilities for subprocess and tool handling
# ======================================================================


def get_tool(name: str) -> str | None:
    """Find a tool on PATH and return its location."""
    from shutil import which

    if name in ["cc", "c++"] and platform_module.system() == "Darwin":
        homebrew = Path("/opt/homebrew/")
        if not homebrew.is_dir():
            logger.warning(
                "No homebrew found. Running Apple's default compiler might raise issues"
            )
            if not confirm("Continue anyway?"):
                sys.exit(0)

    return which(name)


def is_tool(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    return get_tool(name) is not None


def run_subprocess(
    cmd: Sequence[str],
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with better error handling."""
    try:
        logger.debug(f"Running command: {' '.join(cmd)}")

        kwargs: dict[str, Any] = {
            "env": env or os.environ.copy(),
            "check": check,
            "cwd": cwd,
        }

        if capture:
            kwargs.update(
                {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
            )

        result = subprocess.run(cmd, **kwargs)
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        if hasattr(e, "stdout") and e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if hasattr(e, "stderr") and e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        if check:
            raise e
        return subprocess.CompletedProcess(cmd, e.returncode, "", "")


def get_output(command: Sequence[str]) -> str:
    """Run command and return its output as a string."""
    return safe_run(
        lambda: subprocess.check_output(command).decode("utf-8").strip(),
        default="",
        error_log=f"Failed to get output from {command}",
    )()


def confirm(prompt: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{prompt} [y/N] ").lower()
    return response == "y"


# ======================================================================
# Path management utilities
# ======================================================================


def safe_path_operations(simbi_dir: Path) -> None:
    """Create build directory and other paths safely."""
    lib_dir = simbi_dir / "simbi/libs"

    # Create directories if they don't exist
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Create .gitignore for build directories if it doesn't exist
    gitignore_path = simbi_dir / ".gitignore"
    build_patterns = ["build*/", "*.so", "*.egg-info/", "__pycache__/", "*.pyc"]

    try:
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                content = f.read()

            missing_patterns = [p for p in build_patterns if p not in content]
            if missing_patterns:
                with open(gitignore_path, "a") as f:
                    f.write("\n# Automatically added by simbi installer\n")
                    for pattern in missing_patterns:
                        f.write(f"{pattern}\n")
        else:
            with open(gitignore_path, "w") as f:
                f.write("# Automatically generated by simbi installer\n")
                for pattern in build_patterns:
                    f.write(f"{pattern}\n")
    except Exception as e:
        logger.warning(f"Could not update .gitignore: {e}")


def generate_home_locator(simbi_dir: Path) -> None:
    """Generate a file with the path to the git repository."""
    git_home_file = simbi_dir / ("simbi/" + GITHUB_TOPLEVEL)
    if not git_home_file.exists():
        with open(git_home_file, "w") as f:
            f.write(get_output(["git", "rev-parse", "--show-toplevel"]))


def find_project_root() -> Path:
    """Find the root directory of the project."""
    # Try to find by looking for common project files
    current = Path.cwd()

    # Look for indicators
    indicators = ["meson.build", "pyproject.toml", ".git"]

    while current != current.parent:
        for indicator in indicators:
            if (current / indicator).exists():
                return current
        current = current.parent

    # Fallback to current directory
    return Path.cwd()


# ======================================================================
# GPU utilities
# ======================================================================


def suggest_gpu_architectures(platform: str) -> str:
    """Suggest default GPU architectures based on platform."""
    if platform.lower() == "cuda":
        # Common NVIDIA architectures covering the last few generations
        # Pascal (60), Volta (70), Turing (75), Ampere (80,86), Ada Lovelace (89)
        return "60,70,75,80,86,89"
    elif platform.lower() == "hip":
        # Common AMD architectures: Vega (gfx900, gfx906), CDNA (gfx908), RDNA 2 (gfx1030)
        return "gfx900,gfx906,gfx908,gfx1030"
    return ""


def parse_gpu_architectures(arch_str: str, platform: str) -> list[str]:
    """Convert comma-separated architecture string into appropriate format."""
    if not arch_str:
        return []

    archs = [a.strip() for a in arch_str.split(",")]

    # for NVIDIA: ensure all architectures are numeric and have consistent format
    if platform.lower() == "cuda":
        return [a if a.isdigit() else a.replace("sm_", "") for a in archs]

    # for AMD: ensure all architectures start with "gfx" prefix
    elif platform.lower() == "hip":
        return ["gfx" + a if not a.startswith("gfx") else a for a in archs]

    return archs


def generate_gpu_arch_flags(arch_str: str, platform: str) -> str:
    """Generate GPU architecture flags for the specified platform."""
    archs = parse_gpu_architectures(arch_str, platform)

    if not archs:
        return ""

    if platform.lower() == "cuda":
        # for CUDA: generate -gencode flags for each architecture
        return " ".join(
            f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in archs
        )

    elif platform.lower() == "hip":
        # for HIP: generate --offload-arch flags for each architecture
        return " ".join(f"--offload-arch={arch}" for arch in archs)

    return ""


def find_gpu_runtime_dir() -> str:
    """Find GPU runtime directory using multiple strategies."""

    # Define finder functions as closures
    def find_via_nvcc() -> str | None:
        if not (tool := get_tool("nvcc")):
            return None

        which_nvcc = Path(tool)
        # Try to extract from path
        for path in which_nvcc.parents:
            if "cuda" in str(path).lower():
                return str(path)
        return None

    def find_via_nvidia_smi() -> str | None:
        try:
            nvidia_smi = get_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            )

            if nvidia_smi:
                # Check common CUDA locations based on driver being present
                for cuda_path in ["/usr/local/cuda", "/opt/cuda"]:
                    if Path(cuda_path).exists():
                        return cuda_path
        except:
            pass
        return None

    def find_via_hipconfig() -> str | None:
        if not is_tool("hipcc"):
            return None

        try:
            return get_output(["hipconfig", "--rocmpath"])
        except subprocess.CalledProcessError:
            pass
        return None

    def find_via_common_rocm() -> str | None:
        for rocm_path in ["/opt/rocm", "/usr/local/rocm"]:
            if Path(rocm_path).exists():
                return rocm_path
        return None

    # Try strategies in sequence
    finder = try_sequentially(
        [find_via_nvcc, find_via_nvidia_smi, find_via_hipconfig, find_via_common_rocm]
    )

    return finder() or ""


# ======================================================================
# HDF5 utilities
# ======================================================================


def find_hdf5_include() -> str:
    """Find HDF5 include directory using multiple strategies."""

    def try_pkg_config() -> str | None:
        if not is_tool("pkg-config"):
            return None

        try:
            output = get_output(["pkg-config", "--cflags-only-I", "hdf5"])
            if output:
                for part in output.split():
                    if part.startswith("-I"):
                        include_path = part[2:].strip()
                        if Path(include_path).exists():
                            return include_path
        except subprocess.CalledProcessError:
            pass
        return None

    def try_h5py_location() -> str | None:
        try:
            # Check if h5py is installed
            try:
                import h5py

                h5py_path = Path(h5py.__file__).parent
                # Check nearby directories for HDF5 headers
                for parent_level in range(3):
                    parent = h5py_path
                    for _ in range(parent_level):
                        parent = parent.parent

                    include_path = parent / "include"
                    if (include_path / "hdf5.h").exists():
                        return str(include_path)
            except ImportError:
                pass
        except Exception:
            pass
        return None

    def try_common_paths() -> str | None:
        common_paths = [
            # Linux paths
            "/usr/include/hdf5/serial",
            "/usr/include",
            "/usr/local/include",
            # macOS Homebrew paths
            "/opt/homebrew/include",
            "/usr/local/opt/hdf5/include",
            # Windows paths with common prefixes
            "C:/Program Files/HDF_Group/HDF5/include",
            # Conda paths
            f"{sys.prefix}/include",
        ]

        for path in common_paths:
            hdf5_header = Path(path) / "hdf5.h"
            if hdf5_header.exists():
                return path
        return None

    # Try all strategies
    finder = try_sequentially(
        [
            try_pkg_config,
            try_h5py_location,
            try_common_paths,
        ]
    )

    return finder() or ""


# ======================================================================
# Dependency management
# ======================================================================


def check_minimal_dependencies(verbose: bool = False) -> Result[None, str]:
    """Check and install minimal dependencies."""
    # Check Python version
    MIN_PYTHON = (3, 10)
    if sys.version_info < MIN_PYTHON:
        return Result.failure(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later is required"
        )

    # Check for system dependencies
    system_deps = {
        "c++": "C++ compiler",
        "ninja": "Ninja build system",
        "git": "Git version control",
        "pkg-config": "Package configuration tool",
    }

    missing_sys_deps = [
        f"{desc} ({tool})" for tool, desc in system_deps.items() if not is_tool(tool)
    ]

    if missing_sys_deps:
        logger.warning("The following system dependencies are missing:")
        for dep in missing_sys_deps:
            logger.warning(f"  - {dep}")
        logger.warning(
            "Please install these dependencies using your system package manager"
        )
        if not confirm("Continue without these dependencies?"):
            return Result.failure("Missing system dependencies")

    # Check for Python dependencies
    python_deps = [
        "meson",
        "numpy",
        "pybind11",
        "wheel",
    ]

    def check_import(dep: str) -> bool:
        try:
            if dep == "meson":
                __import__("mesonbuild")
            else:
                __import__(dep.split("[")[0])
            return True
        except ImportError:
            return False

    missing_py_deps = [dep for dep in python_deps if not check_import(dep)]

    if missing_py_deps:
        logger.info(
            f"Installing missing Python dependencies: {', '.join(missing_py_deps)}"
        )
        try:
            # Check if using UV
            if is_tool("uv"):
                run_subprocess(
                    ["uv", "pip", "install"] + missing_py_deps,
                    check=True,
                    capture=not verbose,
                )
            else:
                # Fallback to pip
                run_subprocess(
                    [sys.executable, "-m", "pip", "install"] + missing_py_deps,
                    capture=not verbose,
                )

            # Verify installation
            still_missing = [dep for dep in missing_py_deps if not check_import(dep)]
            if still_missing:
                return Result.failure(f"Failed to install: {', '.join(still_missing)}")

        except Exception as e:
            return Result.failure(f"Failed to install dependencies: {e}")

    return Result.success(None)


def is_in_virtualenv() -> bool:
    """Check if the current Python executable is in a virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def activate_virtualenv(venv_path: Path) -> str:
    """Modify the current process environment to activate a virtual environment."""
    venv_path = venv_path.resolve()

    if sys.platform == "win32":
        bin_dir = venv_path / "Scripts"
        lib_dir = venv_path / "Lib" / "site-packages"
    else:
        bin_dir = venv_path / "bin"
        lib_dir = (
            venv_path
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )

    # Update PATH
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"

    # Set VIRTUAL_ENV environment variable
    os.environ["VIRTUAL_ENV"] = str(venv_path)

    # Update Python path
    sys.path.insert(0, str(lib_dir))

    # Update sys.prefix and sys.exec_prefix
    sys.prefix = sys.exec_prefix = str(venv_path)

    logger.info(f"Activated virtual environment at {venv_path}")

    return str(bin_dir / "python")


def get_python_executable(venv_path: Path) -> str:
    """Get the path to the Python executable in a virtual environment."""
    if sys.platform == "win32":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python")


def check_and_setup_environment(config: BuildConfig) -> str | None:
    """
    Check if we're in a virtual environment and set one up if needed.

    Returns the path to the Python executable if we're in a virtual environment,
    or None if we're not or if setting up failed.
    """
    # If we're already in a venv, just return
    if is_in_virtualenv():
        # logger.info("Already in a virtual environment.")
        return None

    # Default venv path
    venv_path = Path(config.venv_path).resolve()

    # Check if an existing venv exists
    if venv_path.exists() and (venv_path / "pyvenv.cfg").exists():
        logger.info(f"Found existing virtual environment at {venv_path}")
        if confirm("Use existing virtual environment?"):
            python_exec = get_python_executable(venv_path)
            activate_virtualenv(venv_path)
            return python_exec

    # Ask to create a new venv
    if confirm("Would you like to create a virtual environment for simbi?"):
        logger.info(f"Creating virtual environment at {venv_path}")
        try:
            if is_tool("uv"):
                logger.info("Using UV to create virtual environment")
                run_subprocess(["uv", "venv", str(venv_path)])
            else:
                logger.info("Using Python's built-in venv module")
                import venv

                venv.create(venv_path, with_pip=True)

            python_exec = get_python_executable(venv_path)
            activate_virtualenv(venv_path)

            return python_exec
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}")

    return None


# ======================================================================
# Build system utilities
# ======================================================================


def configure_build(
    config: BuildConfig, reconfigure: str, hdf5_include: str
) -> list[str]:
    """Create meson configure command."""
    # Check if we need to prompt for GPU architecture
    if config.gpu_compilation == "enabled" and not config.dev_arch:
        suggested_archs = suggest_gpu_architectures(config.gpu_platform)
        logger.info(f"No GPU architecture specified, suggesting: {suggested_archs}")

        if confirm(f"Use suggested architectures ({suggested_archs})?"):
            config = config.with_updates(dev_arch=suggested_archs)
        elif confirm("Would you like to specify GPU architecture(s) now?"):
            arch_example = (
                "e.g. 70,75,80"
                if config.gpu_platform.lower() == "cuda"
                else "e.g. gfx906,gfx908"
            )
            arch_input = input(
                f"Enter comma-separated GPU architecture(s) ({arch_example}): "
            )
            config = config.with_updates(dev_arch=arch_input)
        elif not confirm("Continue without specifying GPU architecture?"):
            sys.exit(1)

    # Generate arch flags if GPU compilation is enabled
    arch_flags = ""
    if config.gpu_compilation == "enabled" and config.dev_arch:
        arch_flags = generate_gpu_arch_flags(config.dev_arch, config.gpu_platform)

    # Create the meson command
    return [
        "meson",
        "setup",
        config.build_dir,
        f"-Dgpu_compilation={config.gpu_compilation}",
        f"-Dcolumn_major={config.column_major}",
        f"-Dprecision={config.precision}",
        f"-Dprofile={config.install_mode}",
        f"-Dgpu_arch={arch_flags}",
        f"-Dfour_velocity={config.four_velocity}",
        f"-Dcpp_std={config.cpp_version}",
        f"-Dbuildtype={config.build_type}",
        reconfigure,
        f"-Dprogress_bar={config.progress_bar}",
        f"-Dhdf5_inc={hdf5_include}",
        f"-Dshared_memory={config.shared_memory}",
    ]


def build_simbi(args: argparse.Namespace, install: bool = False) -> tuple[str, str]:
    """Build the Simbi library."""
    # Convert args to typed config
    config = BuildConfig.from_namespace(args)
    logger.debug(f"Starting build with config: {config}")
    if not install:
        _ = check_and_setup_environment(config)

    # Find project root
    simbi_dir = find_project_root()

    # Check dependencies
    dep_result = check_minimal_dependencies()
    if dep_result.is_failure():
        logger.error(f"Dependency check failed: {dep_result.error_or('Unknown error')}")
        sys.exit(1)

    # Validate configuration
    config_result = validate_config(config)
    if config_result.is_failure():
        logger.error(
            f"Configuration validation failed: {config_result.error_or('Unknown error')}"
        )
        sys.exit(1)

    config = config_result.value_or(config)

    # Prepare the environment
    safe_path_operations(simbi_dir)

    # Check if any args passed to CLI would override the cache args
    cli_args = set(sys.argv[1:])
    config = merge_with_cli_args(config, cli_args)

    # Write updated config to cache
    write_cache(config)

    # Generate home locator file
    generate_home_locator(simbi_dir)

    # Check if build is already configured
    build_configured = (
        run_subprocess(
            ["meson", "introspect", f"{config.build_dir}", "-i", "--targets"],
            capture=True,
            check=False,
        ).returncode
        == 0
    )

    reconfigure_flag = "--reconfigure" if build_configured else ""

    # Set up environment
    simbi_env = os.environ.copy()

    # Set up C compiler
    if "CC" not in simbi_env:
        simbi_env["CC"] = get_tool("cc") or ""
        if simbi_env["CC"]:
            logger.warning(f"C compiler not set, using {simbi_env['CC']}")
        else:
            logger.error("C compiler not found")
            sys.exit(1)

    # Set up C++ compiler
    if "CXX" not in simbi_env:
        simbi_env["CXX"] = get_tool("c++") or ""
        if simbi_env["CXX"]:
            logger.warning(f"C++ compiler not set, using {simbi_env['CXX']}")
        else:
            logger.error("C++ compiler not found")
            sys.exit(1)

    # Find GPU runtime and HDF5 include paths
    hdf5_include = find_hdf5_include()

    # Configure the build
    config_command = configure_build(config, reconfigure_flag, hdf5_include)
    run_subprocess(config_command, env=simbi_env)

    # Create required directories
    build_dir = f"{simbi_dir}/{config.build_dir}"
    egg_dir = f"{simbi_dir}/simbi.egg-info"
    lib_dir = Path(simbi_dir) / "simbi/libs"
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Compile and install if not just configuring
    if not config.configure:
        verbose_args = ["--verbose"] if config.verbose else []

        compile_success = (
            run_subprocess(
                ["meson", "compile"] + verbose_args,
                cwd=f"{config.build_dir}",
                check=False,
            ).returncode
            == 0
        )

        install_success = (
            run_subprocess(
                ["meson", "install"], cwd=f"{config.build_dir}", check=False
            ).returncode
            == 0
        )

        if not (compile_success and install_success):
            logger.error("Build failed")
            sys.exit(1)

    return egg_dir, build_dir


# ======================================================================
# Installation utilities
# ======================================================================


def install_simbi(args: argparse.Namespace) -> int:
    """Install the simbi package."""
    config = BuildConfig.from_namespace(args)

    # First, check if we need to set up a virtual environment
    python_exec = check_and_setup_environment(config)

    # If we got a new Python executable and we're not already using it,
    # we need to re-execute the command in that environment
    if python_exec and python_exec != sys.executable:
        logger.info(f"Re-executing in the virtual environment: {python_exec}")
        cmd = [python_exec, __file__] + sys.argv[1:]
        os.execv(python_exec, cmd)
        # The above call replaces the current process, so we won't reach here

    # Continue with normal installation
    egg_dir, build_dir = build_simbi(args, install=True)

    extras = ""
    if config.visual_extras:
        extras = "[visual]"
    if config.cli_extras:
        extras += "[cli]" if not extras else ",[cli]"

    install_mode = (
        "." + extras if config.install_mode == "default" else "-e" + "." + extras
    )

    # Check if UV is available and use it for installation
    if is_tool("uv"):
        logger.info("UV package manager detected, using UV for installation")
        if python_exec:
            install_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                python_exec,
                install_mode,
            ]
        else:
            install_cmd = ["uv", "pip", "install", install_mode]
    else:
        logger.info("Using standard pip for installation")
        if python_exec:
            install_cmd = [python_exec, "-m", "pip", "install", install_mode]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install", install_mode]

    logger.info(f"Installing simbi with mode: {install_mode}")

    # Use popen + grep to filter out "already satisfied" messages
    try:
        p1 = subprocess.Popen(install_cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(
            ["grep", "-v", "Requirement already satisfied"], stdin=p1.stdout
        )
        if p1.stdout is not None:
            p1.stdout.close()
        p2.communicate()
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        return 1

    # Clean up
    logger.info("Cleaning up build artifacts")
    run_subprocess(["rm", "-rf", f"{egg_dir}", f"{build_dir}"], check=False)

    # Final message
    if python_exec:
        venv_path = Path(config.venv_path).resolve()
        logger.info("Installation complete in virtual environment!")
        logger.info("To use simbi, first activate the virtual environment:")
        if sys.platform == "win32":
            logger.info(f"   {venv_path}\\Scripts\\activate")
        else:
            logger.info(f"   source {venv_path}/bin/activate")
    else:
        logger.info("Installation complete!")

    return 0


def uninstall_simbi(_: argparse.Namespace) -> int:
    """Uninstall the simbi package."""
    logger.info("Uninstalling simbi")

    simbi_dir = find_project_root()
    run_subprocess(
        [sys.executable, "-m", "pip", "uninstall", "-y", "simbi"], check=False
    )

    # Remove compiled extensions
    exts = list((Path(simbi_dir) / "simbi/libs/").glob("*.so"))
    if exts:
        if confirm(f"Remove {len(exts)} compiled extensions?"):
            for ext in exts:
                ext.unlink()
            logger.info("Removed compiled extensions")

    logger.info("Uninstallation complete!")
    return 0


def setup_virtual_environment(args: argparse.Namespace) -> int:
    """Set up a virtual environment for Simbi."""
    config = BuildConfig.from_namespace(args)
    venv_path = Path(config.venv_path).resolve()

    if venv_path.exists():
        if not confirm(f"Virtual environment at {venv_path} already exists. Recreate?"):
            logger.info("Using existing virtual environment")
            return 0

        logger.info(f"Removing existing virtual environment at {venv_path}")
        import shutil

        shutil.rmtree(venv_path, ignore_errors=True)

    logger.info(f"Creating virtual environment at {venv_path}")
    try:
        if is_tool("uv"):
            logger.info("Using UV to create virtual environment")
            run_subprocess(["uv", "venv", str(venv_path)])
            return 0
        else:
            logger.info("Using Python's built-in venv module")
            import venv

            venv.create(venv_path, with_pip=True)
            return 0

    except Exception as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return 1


# ======================================================================
# Command line parsing
# ======================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        "dev.py", description="Tool for building and installing simbi"
    )

    subparsers = parser.add_subparsers(
        help="Commands for building, installing, or uninstalling simbi", dest="command"
    )

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the simbi library")
    _add_build_arguments(build_parser)
    build_parser.set_defaults(func=build_simbi)

    # Install command
    install_parser = subparsers.add_parser(
        "install", help="Build and install the simbi library"
    )
    _add_build_arguments(install_parser)
    install_parser.set_defaults(func=install_simbi)

    # Uninstall command
    uninstall_parser = subparsers.add_parser(
        "uninstall", help="Uninstall the simbi library"
    )
    uninstall_parser.set_defaults(func=uninstall_simbi)

    # Venv command
    venv_parser = subparsers.add_parser("venv", help="Set up a virtual environment")
    venv_parser.set_defaults(func=setup_virtual_environment)

    # Global options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    return parser


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Add build-related arguments to a parser."""
    parser.add_argument(
        "--dev-arch",
        type=str,
        default=DEFAULT_CONFIG["dev_arch"],
        help="GPU architecture for compilation as comma-separated list (e.g. '70,75,80' for NVIDIA or 'gfx900,gfx906' for AMD)",
    )
    parser.add_argument(
        "--verbose",
        action="store_const",
        default=[],
        const=["--verbose"],
        help="Flag for verbose compilation output",
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        default=False,
        help="Flag to only configure the meson build directory without installing",
    )
    parser.add_argument(
        "--install-mode",
        type=str,
        choices=["default", "develop"],
        default=DEFAULT_CONFIG["install_mode"],
        help="Install mode (normal or editable)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=DEFAULT_CONFIG["build_dir"],
        help="Build directory name for meson build",
    )
    parser.add_argument(
        "--cli-extras",
        action="store_true",
        default=False,
        help="Flag to install the optional cli dependencies",
    )
    parser.add_argument(
        "--visual-extras",
        action="store_true",
        default=False,
        help="Flag to install the optional visual dependencies",
    )
    parser.add_argument(
        "--four-velocity",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["four_velocity"],
        help="Flag to set four-velocity as the velocity primitive instead of beta",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["progress_bar"],
        help="Flag to show / hide progress bar",
    )
    parser.add_argument(
        "--shared-memory",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["shared_memory"],
        help="Flag to enable / disable shared memory for gpu builds",
    )
    parser.add_argument(
        "--cpp17",
        action="store_const",
        default="c++20",
        const="c++17",
        dest="cpp_version",
        help="Flag for setting c++ version to c++17 instead of default c++20",
    )
    parser.add_argument(
        "--gpu-platform",
        type=str,
        default=DEFAULT_CONFIG["gpu_platform"],
        choices=["cuda", "hip", "None"],
        help="Flag to set the gpu platform for compilation",
    )
    parser.add_argument(
        "--create-venv",
        choices=["yes", "no", "ask"],
        default="ask",
        help="Create a dedicated virtual environment for simbi (yes/no/ask)",
    )

    # Mutually exclusive options
    compile_type = parser.add_mutually_exclusive_group()
    compile_type.add_argument(
        "--gpu-compilation",
        action="store_const",
        dest="gpu_compilation",
        const="enabled",
        help="Enable GPU compilation",
    )
    compile_type.add_argument(
        "--cpu-compilation",
        action="store_const",
        dest="gpu_compilation",
        const="disabled",
        help="Disable GPU compilation",
    )

    memory_layout = parser.add_mutually_exclusive_group()
    memory_layout.add_argument(
        "--row-major",
        action="store_const",
        dest="column_major",
        const=False,
        help="Use row-major memory layout",
    )
    memory_layout.add_argument(
        "--column-major",
        action="store_const",
        dest="column_major",
        const=True,
        help="Use column-major memory layout",
    )

    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--double",
        action="store_const",
        dest="precision",
        const="double",
        help="Use double precision",
    )
    precision.add_argument(
        "--float",
        action="store_const",
        dest="precision",
        const="float",
        help="Use single precision",
    )
    parser.set_defaults(
        precision="double",
        column_major=False,
        gpu_compilation="disabled",
        build_type="release",
    )


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse command-line arguments."""
    parser = create_parser()
    args = parser.parse_args()

    # Check if a command was specified
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    return parser, args


# ======================================================================
# Main entry point
# ======================================================================


def main() -> int:
    """Main entry point."""
    # Set up logging first
    setup_logging(verbose="--verbose" in sys.argv or "-v" in sys.argv)

    # Parse arguments
    _, args = parse_arguments()

    # Execute the appropriate function
    try:
        # Set up logging first
        setup_logging(verbose="--verbose" in sys.argv or "-v" in sys.argv)

        # Parse arguments
        _, args = parse_arguments()

        # Execute the appropriate function
        return args.func(args)
    except KeyboardInterrupt:
        logger.error("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
