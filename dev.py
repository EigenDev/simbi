import argparse
import sys
import subprocess
import json
import os
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Any, Union
from functools import reduce, partial
import itertools

# Constants
CACHE_FILE = "simbi_build_cache.txt"
GITHUB_TOPLEVEL = "gitrepo_home.txt"
YELLOW = "\033[0;33m"
RST = "\033[0m"  # No Color
CURRENT_CACHE_VERSION = "1.0"

# Default configuration
DEFAULT_CONFIG = {
    "gpu_compilation": "disabled",
    "progress_bar": True,
    "column_major": False,
    "precision": "double",
    "install_mode": "default",
    "dev_arch": 0,
    "build_dir": "build",
    "four_velocity": False,
    "shared_memory": True,
    "cpp_version": "c++20",
    "build_type": "release",
    "gpu_platform": "cuda",
}

# Flag overrides
FLAG_OVERRIDES = {
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

# =================================================================
# Functional Utilities
# =================================================================


def compose(*funcs):
    """Compose functions from right to left."""
    if not funcs:
        return lambda x: x

    def composed_function(x):
        result = x
        for f in reversed(funcs):
            result = f(result)
        return result

    return composed_function


def safe_run(func, default=None, error_log=None):
    """Higher-order function to safely run a function with error handling."""

    def safe_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if error_log:
                logger.error(f"{error_log}: {str(e)}")
            return default

    return safe_func


def try_sequentially(funcs, fallback=None):
    """Try a sequence of functions until one succeeds."""

    def try_sequence(*args, **kwargs):
        for func in funcs:
            try:
                result = func(*args, **kwargs)
                if result:
                    return result
            except Exception:
                continue
        return fallback

    return try_sequence


# =================================================================
# Core Utilities
# =================================================================


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


def get_tool(name: str) -> Optional[str]:
    """Find a tool on PATH and return its location."""
    from shutil import which
    import platform

    if name in ["cc", "c++"] and platform.system() == "Darwin":
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
    cmd: List[str], env=None, check=True, capture=False, cwd=None
) -> subprocess.CompletedProcess:
    """Run a subprocess with better error handling."""
    try:
        logger.debug(f"Running command: {' '.join(cmd)}")

        kwargs = {"env": env or os.environ.copy(), "check": check, "cwd": cwd}

        if capture:
            kwargs.update(
                {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
            )

        result = subprocess.run(cmd, **kwargs)
        return result

    except subprocess.SubprocessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        if hasattr(e, "stdout") and e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if hasattr(e, "stderr") and e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        if check:
            raise
        return e


def get_output(command: List[str]) -> str:
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


# =================================================================
# File System Operations
# =================================================================


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
                    f.write("\n# Automatically added by dev.py\n")
                    for pattern in missing_patterns:
                        f.write(f"{pattern}\n")
        else:
            with open(gitignore_path, "w") as f:
                f.write("# Automatically generated by dev.py\n")
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


# =================================================================
# Dependency Management
# =================================================================


def find_hdf5_include() -> str:
    """Robust method to find HDF5 include directory."""
    # Try different methods to find HDF5 include path
    strategies = [
        # Strategy 1: pkg-config
        lambda: subprocess.check_output(
            ["pkg-config", "--cflags", "hdf5"], stderr=subprocess.DEVNULL
        )
        .decode("utf-8")
        .split(),
        # Strategy 2: h5cc
        lambda: subprocess.check_output(["h5cc", "-show"], stderr=subprocess.DEVNULL)
        .decode("utf-8")
        .split(),
        # Strategy 3: Common paths
        lambda: check_common_hdf5_paths(),
    ]

    # Try pkg-config
    try:
        h5pkg = strategies[0]()
        include_dirs = [
            include_dir[2:] for include_dir in h5pkg if include_dir.startswith("-I")
        ]
        if include_dirs:
            return " ".join(include_dirs)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Try h5cc
    try:
        h5cc_show = strategies[1]()
        lib_dirs = [lib_dir[2:] for lib_dir in h5cc_show if lib_dir.startswith("-L")]
        if lib_dirs:
            hdf5_libpath = Path(" ".join(lib_dirs))
            include_dir = str(hdf5_libpath.parents[0] / "include")
            if Path(include_dir).exists():
                return include_dir
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Try common paths
    common_path = strategies[2]()
    if common_path:
        return common_path

    logger.warning("Could not find HDF5 include path")
    return ""


def check_common_hdf5_paths() -> str:
    """Check common paths where HDF5 headers might be found."""
    common_paths = [
        "/usr/include/hdf5",
        "/usr/local/include/hdf5",
        "/opt/homebrew/include/hdf5",
        "/usr/include/x86_64-linux-gnu/hdf5/serial",
        "/usr/include",
    ]

    # First check direct paths
    for path in common_paths:
        if Path(path).exists():
            return path

    # Then try to find hdf5.h anywhere in system paths
    try:
        result = (
            subprocess.check_output(
                ["find", "/usr", "-name", "hdf5.h"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )

        if result:
            # Return the directory containing hdf5.h
            return str(Path(result.split("\n")[0]).parent)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return ""


def find_gpu_runtime_dir() -> str:
    """Find GPU runtime directory more robustly."""

    # Strategy 1: Find CUDA via nvcc
    def find_via_nvcc():
        if not is_tool("nvcc"):
            return None

        which_nvcc = Path(get_tool("nvcc"))
        # Try to extract from path
        for path in which_nvcc.parents:
            if "cuda" in str(path).lower():
                return str(path)
        return None

    # Strategy 2: Find CUDA via nvidia-smi
    def find_via_nvidia_smi():
        try:
            nvidia_smi = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )

            if nvidia_smi:
                # Check common CUDA locations based on driver being present
                for cuda_path in ["/usr/local/cuda", "/opt/cuda"]:
                    if Path(cuda_path).exists():
                        return cuda_path
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    # Strategy 3: Find HIP via hipconfig
    def find_via_hipconfig():
        if not is_tool("hipcc"):
            return None

        try:
            return get_output(["hipconfig", "--rocmpath"])
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    # Strategy 4: Check common ROCm paths
    def find_via_common_rocm():
        for rocm_path in ["/opt/rocm", "/usr/local/rocm"]:
            if Path(rocm_path).exists():
                return rocm_path
        return None

    # Try strategies in sequence
    finder = try_sequentially(
        [find_via_nvcc, find_via_nvidia_smi, find_via_hipconfig, find_via_common_rocm]
    )

    result = finder()
    return result or ""


def check_minimal_dependencies() -> None:
    """Check and install minimal dependencies with better error handling."""
    MIN_PYTHON = (3, 10)
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
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
            sys.exit(1)

    # Check for Python dependencies
    python_deps = ["mesonbuild", "numpy", "cython", "cogapp"]

    def check_import(dep: str) -> bool:
        try:
            __import__(dep.split("[")[0] if "[" in dep else dep)
            return True
        except ImportError:
            return False

    missing_py_deps = [dep for dep in python_deps if not check_import(dep)]

    if missing_py_deps:
        logger.info(
            f"Installing missing Python dependencies: {', '.join(missing_py_deps)}"
        )
        try:
            run_subprocess([sys.executable, "-m", "pip", "install"] + missing_py_deps)
        except subprocess.SubprocessError:
            logger.warning("Failed to install some Python dependencies")
            if not confirm("Continue anyway?"):
                sys.exit(1)


# =================================================================
# Configuration Management
# =================================================================


def read_from_cache() -> Optional[Dict[str, Any]]:
    """Read configuration from cache file."""
    cached_args = Path(CACHE_FILE)
    if cached_args.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                cached_vars = json.load(f)
                return check_cache_compatibility(cached_vars)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse cache file {CACHE_FILE}, using defaults")
            return None
    else:
        with open(CACHE_FILE, "w+"):
            pass
        return None


def check_cache_compatibility(cached_vars: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Check if cached configuration is compatible with current version."""
    if not cached_vars:
        return None

    # Check the version field in the cache
    cache_version = cached_vars.get("_cache_version", "0.0")

    if cache_version != CURRENT_CACHE_VERSION:
        logger.warning(
            f"Cache version mismatch ({cache_version} vs {CURRENT_CACHE_VERSION})"
        )
        if confirm("Clear cache and use defaults?"):
            new_cache = {"_cache_version": CURRENT_CACHE_VERSION}
            with open(CACHE_FILE, "w") as f:
                json.dump(new_cache, f, indent=4)
            return new_cache

    return cached_vars


def write_to_cache(args: argparse.Namespace) -> None:
    """Write configuration to cache file."""
    details = vars(args).copy()
    details["build_dir"] = str(Path(details["build_dir"]).resolve())
    details.pop("func", None)
    details.pop("verbose", None)
    details.pop("configure", None)
    details.pop("extras", None)

    # Add cache version
    details["_cache_version"] = CURRENT_CACHE_VERSION

    with open(CACHE_FILE, "w") as f:
        json.dump(details, f, indent=4)


def merge_cached_config(
    args: argparse.Namespace, cli_args: List[str]
) -> argparse.Namespace:
    """Merge cached configuration with command line arguments."""
    cached_vars = read_from_cache()
    if not cached_vars:
        return args

    # Only override with cached values if not specified in CLI
    for arg, default_value in DEFAULT_CONFIG.items():
        # Skip special args
        if arg in [
            "verbose",
            "configure",
            "func",
            "extras",
            "cpp_version",
            "build_type",
        ]:
            continue

        # Only use cache if not specified in CLI
        if getattr(args, arg) == default_value and arg not in cli_args:
            # Check if any flag override was specified
            if arg in FLAG_OVERRIDES and any(
                flag in cli_args for flag in FLAG_OVERRIDES[arg]
            ):
                continue

            # Use cached value if available
            if arg in cached_vars:
                setattr(args, arg, cached_vars[arg])

    return args


def generate_build_options(args: argparse.Namespace) -> None:
    """Generate build options from configuration."""
    try:
        run_subprocess(
            [
                "cog",
                "-d",
                "-o",
                "build_options.hpp.in",
                "code_gen/build_options.hpp.in.cog",
            ],
            env={
                **os.environ,
                "GPU_ENABLED": str(args.gpu_compilation == "enabled"),
                "GPU_PLATFORM": args.gpu_platform.upper(),
            },
            check=True,
            capture=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("cog failed to generate build_options.hpp.in")
        if hasattr(e, "stderr"):
            logger.error(e.stderr)
        sys.exit(1)


def validate_configuration(args: argparse.Namespace) -> argparse.Namespace:
    """Validate the build configuration settings."""
    # Check GPU compilation settings
    if args.gpu_compilation == "enabled":
        if args.gpu_platform == "cuda" and not is_tool("nvcc"):
            logger.warning("CUDA GPU compilation requested but nvcc not found")
            if not confirm("Continue anyway?"):
                sys.exit(1)
        elif args.gpu_platform == "hip" and not is_tool("hipcc"):
            logger.warning("HIP GPU compilation requested but hipcc not found")
            if not confirm("Continue anyway?"):
                sys.exit(1)

    # Validate build directory
    build_dir = Path(args.build_dir)
    if build_dir.exists() and not build_dir.is_dir():
        logger.warning(f"Build path {build_dir} exists but is not a directory")
        if confirm("Remove the file and create directory?"):
            build_dir.unlink()
            build_dir.mkdir(parents=True)
        else:
            sys.exit(1)

    return args


def configure(
    args: argparse.Namespace, reconfigure: str, hdf5_include: str, gpu_include: str
) -> List[str]:
    """Create meson configure command."""
    if args.gpu_compilation == "enabled" and (
        args.dev_arch is None or args.dev_arch == 0
    ):
        logger.warning("No GPU architecture specified")
        if not confirm("Continue anyway?"):
            if not confirm("Would you like to set the gpu architecure now?"):
                sys.exit(1)
            else:
                args.dev_arch = int(
                    input(
                        "Please enter the gpu compute capability (omitting decimals): "
                    )
                )

    command = [
        "meson",
        "setup",
        args.build_dir,
        f"-Dgpu_compilation={args.gpu_compilation}",
        f"-Dcolumn_major={args.column_major}",
        f"-Dprecision={args.precision}",
        f"-Dprofile={args.install_mode}",
        f"-Dgpu_arch={args.dev_arch}",
        f"-Dfour_velocity={args.four_velocity}",
        f"-Dcpp_std={args.cpp_version}",
        f"-Dbuildtype={args.build_type}",
        reconfigure,
        f"-Dprogress_bar={args.progress_bar}",
        f"-Dshared_memory={args.shared_memory}",
    ]
    return command


# =================================================================
# Command Line Interface
# =================================================================


def parse_the_arguments() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        "Parser for building and installing simbi with meson"
    )
    subparsers = parser.add_subparsers(
        help="sub-commands that build / install / uninstall the code"
    )

    build_parser = subparsers.add_parser("build", add_help=False)
    build_parser.set_defaults(func=build_simbi)
    build_parser.add_argument(
        "--dev-arch",
        type=int,
        default=0,
        help="SM architecture specification for gpu compilation",
    )
    build_parser.add_argument(
        "--verbose",
        "-v",
        action="store_const",
        default=[],
        const=["--verbose"],
        help="flag for verbose compilation output",
    )
    build_parser.add_argument(
        "--configure",
        action="store_true",
        default=False,
        help="flag to only configure the meson build directory without installing",
    )
    build_parser.add_argument(
        "--install-mode",
        type=str,
        choices=["default", "develop"],
        default="default",
        help="install mode (normal or editable)",
    )
    build_parser.add_argument(
        "--build-dir",
        type=str,
        default="build",
        help="build directory name for meson build",
    )
    build_parser.add_argument(
        "--extras",
        action="store_true",
        default=False,
        help="flag to install the optional dependencies",
    )
    build_parser.add_argument(
        "--four-velocity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="flag to set four-velocity as the velocity primitive instead of beta",
    )
    build_parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flag to show / hide progress bar",
    )
    build_parser.add_argument(
        "--shared-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flag to enable / disable shared memory for gpu builds",
    )
    build_parser.add_argument(
        "--cpp17",
        action="store_const",
        default="c++20",
        const="c++17",
        dest="cpp_version",
        help="flag for setting c++ version to c++17 instead of default c++20",
    )
    build_parser.add_argument(
        "--gpu-platform",
        type=str,
        default="cuda",
        choices=["cuda", "hip", "None"],
        help="flag to set the gpu platform for compilation",
    )

    compile_type = build_parser.add_mutually_exclusive_group()
    compile_type.add_argument(
        "--gpu-compilation",
        action="store_const",
        dest="gpu_compilation",
        const="enabled",
    )
    compile_type.add_argument(
        "--cpu-compilation",
        action="store_const",
        dest="gpu_compilation",
        const="disabled",
    )

    build_type = build_parser.add_mutually_exclusive_group()
    build_type.add_argument(
        "--release", action="store_const", dest="build_type", const="release"
    )
    build_type.add_argument(
        "--debug", action="store_const", dest="build_type", const="debug"
    )

    precision = build_parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--double", action="store_const", dest="precision", const="double"
    )
    precision.add_argument(
        "--float", action="store_const", dest="precision", const="single"
    )

    major = build_parser.add_mutually_exclusive_group()
    major.add_argument(
        "--row-major", action="store_const", dest="column_major", const=False
    )
    major.add_argument(
        "--column-major", action="store_const", dest="column_major", const=True
    )

    build_parser.set_defaults(
        precision="double",
        column_major=False,
        gpu_compilation="disabled",
        build_type="release",
    )

    install_parser = subparsers.add_parser(
        "install", help="install simbi", parents=[build_parser]
    )
    install_parser.set_defaults(func=install_simbi)

    unbuild_parser = subparsers.add_parser("uninstall", help="uninstall simbi")
    unbuild_parser.set_defaults(func=uninstall_simbi)

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    return parser, args


# =================================================================
# Build Operations
# =================================================================


def build_simbi(args: argparse.Namespace) -> tuple[str]:
    """Build the simbi package."""
    simbi_dir = Path().resolve()

    # Initialize logging
    setup_logging(bool(args.verbose))
    logger.debug(f"Starting build with args: {args}")

    # Check dependencies and validate config
    check_minimal_dependencies()
    args = validate_configuration(args)

    # Generate build options
    generate_build_options(args)

    # Prepare the environment
    safe_path_operations(simbi_dir)

    # Check if any args passed to the CLI exist that would override the cache args
    cli_args = set(sys.argv[1:])
    args = merge_cached_config(args, cli_args)

    # Write updated config to cache
    write_to_cache(args)

    # Generate home locator file
    generate_home_locator(simbi_dir=simbi_dir)

    # Check if build is already configured
    build_configured = (
        run_subprocess(
            ["meson", "introspect", f"{args.build_dir}", "-i", "--targets"],
            capture=True,
            check=False,
        ).returncode
        == 0
    )

    reconfigure_flag = "--reconfigure" if build_configured else ""

    # Set up environment
    simbi_env = os.environ.copy()
    if "CC" not in simbi_env:
        simbi_env["CC"] = get_tool("cc") or ""
        if simbi_env["CC"]:
            logger.warning(f"C compiler not set, using {simbi_env['CC']}")
        else:
            logger.error("C compiler not found")
            sys.exit(1)

    if "CXX" not in simbi_env:
        simbi_env["CXX"] = get_tool("c++") or ""
        if simbi_env["CXX"]:
            logger.warning(f"C++ compiler not set, using {simbi_env['CXX']}")
        else:
            logger.error("C++ compiler not found")
            sys.exit(1)

    # Find GPU runtime and HDF5 include paths
    gpu_runtime_dir = find_gpu_runtime_dir()
    gpu_include = f"{gpu_runtime_dir}/include" if gpu_runtime_dir else ""
    hdf5_include = find_hdf5_include()

    # Configure the build
    config_command = configure(args, reconfigure_flag, hdf5_include, gpu_include)
    run_subprocess(config_command, env=simbi_env)

    # Create required directories
    build_dir = f"{simbi_dir}/{args.build_dir}"
    egg_dir = f"{simbi_dir}/simbi.egg-info"
    lib_dir = Path(simbi_dir / "simbi/libs")
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Compile and install if not just configuring
    if not args.configure:
        compile_success = (
            run_subprocess(
                ["meson", "compile"] + args.verbose,
                cwd=f"{args.build_dir}",
                check=False,
            ).returncode
            == 0
        )

        install_success = (
            run_subprocess(
                ["meson", "install"], cwd=f"{args.build_dir}", check=False
            ).returncode
            == 0
        )

        if not (compile_success and install_success):
            logger.error("Build failed")
            sys.exit(1)

    return egg_dir, build_dir


def install_simbi(args: argparse.Namespace) -> None:
    """Install the simbi package."""
    egg_dir, build_dir = build_simbi(args)

    extras = "" if not args.extras else "[extras]"
    install_mode = (
        "." + extras if args.install_mode == "default" else "-e" + "." + extras
    )

    # Install with pip
    logger.info(f"Installing simbi with mode: {install_mode}")

    # Use popen + grep to filter out "already satisfied" messages
    p1 = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", install_mode], stdout=subprocess.PIPE
    )
    p2 = subprocess.Popen(
        ["grep", "-v", "Requirement already satisfied"], stdin=p1.stdout
    )
    p1.stdout.close()
    p2.communicate()

    # Clean up
    logger.info("Cleaning up build artifacts")
    run_subprocess(["rm", "-rf", f"{egg_dir}", f"{build_dir}"], check=False)
    logger.info("Installation complete!")


def uninstall_simbi(args: argparse.Namespace) -> None:
    """Uninstall the simbi package."""
    logger.info("Uninstalling simbi")

    simbi_dir = Path().resolve()
    run_subprocess(
        [sys.executable, "-m", "pip", "uninstall", "-y", "simbi"], check=False
    )

    # Remove compiled extensions
    exts = list(Path(simbi_dir / "simbi/libs/").glob("*.so"))
    if exts:
        if confirm(f"Remove {len(exts)} compiled extensions?"):
            for ext in exts:
                ext.unlink()
            logger.info("Removed compiled extensions")

    logger.info("Uninstallation complete!")


# =================================================================
# Main Entry Point
# =================================================================


def main() -> int:
    """Main entry point."""
    # Setup basic logging first
    setup_logging()

    try:
        _, args = parse_the_arguments()
        args.func(args)
        return 0
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
