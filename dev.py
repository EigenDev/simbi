import argparse
import sys
import subprocess
import json
import os
from pathlib import Path
from typing import Optional

# Constants
CACHE_FILE = "simbi_build_cache.txt"
GITHUB_TOPLEVEL = "gitrepo_home.txt"
YELLOW = "\033[0;33m"
RST = "\033[0m"  # No Color

# Default configuration
DEFAULT_CONFIG = {
    "gpu_compilation": "disabled",
    "progress_bar": True,
    "column_major": False,
    "precision": 'double',
    "install_mode": "default",
    "dev_arch": 86,
    "build_dir": "builddir",
    "four_velocity": False,
    "shared_memory": True,
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
}

def get_tool(name: str) -> Optional[str]:
    from shutil import which
    import platform

    if name in ["cc", "c++"] and platform.system() == "Darwin":
        homebrew = Path("/opt/homebrew/")
        if not homebrew.is_dir():
            print(f"{YELLOW}WRN{RST} no homebrew found. Running Apple's default compiler might raise issues")
            cont = input("Continue anyway? [y/N]")
            if cont.lower() != "y":
                sys.exit(0)
    return which(name)

def is_tool(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    return get_tool(name) is not None

def read_from_cache() -> Optional[dict[str, str]]:
    cached_args = Path(CACHE_FILE)
    if cached_args.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    else:
        with open(CACHE_FILE, "w+"):
            pass
        return None

def check_minimal_dependencies() -> None:
    MIN_PYTHON = (3, 10)
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later is required")

    dependencies = ["mesonbuild", "numpy", "cython", "ninja"]
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError as e:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)

def write_to_cache(args: argparse.Namespace) -> None:
    details = vars(args)
    details["build_dir"] = str(Path(details["build_dir"]).resolve())
    details.pop("func", None)
    with open(CACHE_FILE, "w") as f:
        json.dump(details, f, indent=4)

def get_output(command: list[str]) -> str:
    return subprocess.check_output(command).decode("utf-8").strip()

def configure(args: argparse.Namespace, reconfigure: str, hdf5_include: str, gpu_include: str) -> list[str]:
    command = [
        "meson", "setup", args.build_dir,
        f"-Dgpu_compilation={args.gpu_compilation}",
        f"-Dhdf5_include_dir={hdf5_include}",
        f"-Dgpu_include_dir={gpu_include}",
        f"-Dcolumn_major={args.column_major}",
        f"-Dprecision={args.precision}",
        f"-Dprofile={args.install_mode}",
        f"-Dgpu_arch={args.dev_arch}",
        f"-Dfour_velocity={args.four_velocity}",
        f"-Dcpp_std={args.cpp_version}",
        f"-Dbuildtype={args.build_type}",
        reconfigure,
        f"-Dprogress_bar={args.progress_bar}",
        f"-Dshared_memory={args.shared_memory}"
    ]
    return command

def generate_home_locator(simbi_dir: str) -> None:
    git_home_file = Path(simbi_dir) / ("simbi/" + GITHUB_TOPLEVEL)
    if not git_home_file.exists():
        with open(git_home_file, "w") as f:
            f.write(get_output(["git", "rev-parse", "--show-toplevel"]))

def parse_the_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser("Parser for building and installing simbi with meson")
    subparsers = parser.add_subparsers(help="sub-commands that build / install / uninstall the code")

    build_parser = subparsers.add_parser("build", add_help=False)
    build_parser.set_defaults(func=build_simbi)
    build_parser.add_argument("--dev-arch", type=int, default=86, help="SM architecture specification for gpu compilation")
    build_parser.add_argument("--verbose", "-v", action="store_const", default=[], const=["--verbose"], help="flag for verbose compilation output")
    build_parser.add_argument("--configure", action="store_true", default=False, help="flag to only configure the meson build directory without installing")
    build_parser.add_argument("--install-mode", type=str, choices=["default", "develop"], default="default", help="install mode (normal or editable)")
    build_parser.add_argument("--build-dir", type=str, default="builddir", help="build directory name for meson build")
    build_parser.add_argument("--extras", action="store_true", default=False, help="flag to install the optional dependencies")
    build_parser.add_argument("--four-velocity", action=argparse.BooleanOptionalAction, default=False, help="flag to set four-velocity as the velocity primitive instead of beta")
    build_parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True, help="flag to show / hide progress bar")
    build_parser.add_argument("--shared-memory", action=argparse.BooleanOptionalAction, default=True, help="flag to enable / disable shared memory for gpu builds")
    build_parser.add_argument("--cpp17", action="store_const", default="c++20", const="c++17", dest="cpp_version", help="flag for setting c++ version to c++17 instead of default c++20")

    compile_type = build_parser.add_mutually_exclusive_group()
    compile_type.add_argument("--gpu-compilation", action="store_const", dest="gpu_compilation", const="enabled")
    compile_type.add_argument("--cpu-compilation", action="store_const", dest="gpu_compilation", const="disabled")

    build_type = build_parser.add_mutually_exclusive_group()
    build_type.add_argument("--release", action="store_const", dest="build_type", const="release")
    build_type.add_argument("--debug", action="store_const", dest="build_type", const="debug")

    precision = build_parser.add_mutually_exclusive_group()
    precision.add_argument("--double", action="store_const", dest="precision", const='double')
    precision.add_argument("--float", action="store_const", dest="precision", const='single')

    major = build_parser.add_mutually_exclusive_group()
    major.add_argument("--row-major", action="store_const", dest="column_major", const=False)
    major.add_argument("--column-major", action="store_const", dest="column_major", const=True)

    build_parser.set_defaults(precision='double', column_major=False, gpu_compilation="disabled", build_type="release")

    install_parser = subparsers.add_parser("install", help="install simbi", parents=[build_parser])
    install_parser.set_defaults(func=install_simbi)

    unbuild_parser = subparsers.add_parser("uninstall", help="uninstall simbi")
    unbuild_parser.set_defaults(func=uninstall_simbi)

    return parser, parser.parse_args(args=None if sys.argv[1:] else ["--help"])

def build_simbi(args: argparse.Namespace) -> tuple[str]:
    simbi_dir = Path().resolve()
    if args.build_dir == "build":
        raise argparse.ArgumentError(args.builddir, "please choose a different build name other than 'build'")

    check_minimal_dependencies()
    simbi_env = os.environ.copy()

    # Check if any args passed to the CLI exist that would override the cache args
    cli_args = sys.argv[1:]
    if cached_vars := read_from_cache():
        for arg in vars(args):
            if arg in ["verbose", "configure", "func", "extras", "cpp_version", "build_type"]:
                continue

            if getattr(args, arg) == DEFAULT_CONFIG[arg]:
                if arg in cli_args:
                    continue

                if arg in FLAG_OVERRIDES.keys() and any(x in FLAG_OVERRIDES[arg] for x in cli_args):
                    continue
                else:
                    setattr(args, arg, cached_vars[arg])

    generate_home_locator(simbi_dir=simbi_dir)
    write_to_cache(args)

    build_configured = subprocess.run(
        ["meson", "introspect", f"{args.build_dir}", "-i", "--targets"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0

    reconfigure_flag = "--reconfigure" if build_configured else ""
    if "CC" not in simbi_env:
        simbi_env["CC"] = get_tool("cc")
        print(f"{YELLOW}WRN{RST}: C compiler not set")
        print(f"Using symbolic link {simbi_env['CC']} as default")

    if "CXX" not in simbi_env:
        simbi_env["CXX"] = get_tool("c++")
        print(f"{YELLOW}WRN{RST}: C++ compiler not set")
        print(f"Using symbolic link {simbi_env['CXX']} as default")

    gpu_runtime_dir = ""
    if is_tool("nvcc"):
        which_cuda = Path(get_tool("nvcc"))
        gpu_runtime_dir = " ".join(str(path.parent) for path in which_cuda.parents if "cuda" in str(path.parent))
    elif is_tool("hipcc"):
        gpu_runtime_dir = get_output(["hipconfig", "--rocmpath"])

    try:
        gpu_include = f"{gpu_runtime_dir.split()[0]}/include"
    except IndexError:
        gpu_include = ""

    h5pkg = get_output(["pkg-config", "--cflags", "hdf5"]).split()
    hdf5_include = " ".join(include_dir[2:] for include_dir in filter(lambda x: x.startswith("-I"), h5pkg))

    if not hdf5_include:
        h5cc_show = get_output(["h5cc", "-show"]).split()
        hdf5_libpath = Path(" ".join(lib_dir[2:] for lib_dir in filter(lambda x: x.startswith("-L"), h5cc_show)))
        hdf5_include = hdf5_libpath.parents[0].resolve() / "include"

    config_command = configure(args, reconfigure_flag, hdf5_include, gpu_include)
    subprocess.run(config_command, env=simbi_env, check=True)

    build_dir = f"{simbi_dir}/build"
    egg_dir = f"{simbi_dir}/simbi.egg-info"
    lib_dir = Path(simbi_dir / "simbi/libs")
    lib_dir.mkdir(parents=True, exist_ok=True)

    if not args.configure:
        compile_child = subprocess.Popen(["meson", "compile"] + args.verbose, cwd=f"{args.build_dir}").wait()
        install_child = subprocess.Popen(["meson", "install"], cwd=f"{args.build_dir}").wait()

        if compile_child == install_child == 0:
            return egg_dir, build_dir
        else:
            raise subprocess.CalledProcessError("Error occurred during build")
    else:
        return egg_dir, build_dir

def install_simbi(args: argparse.Namespace) -> None:
    egg_dir, build_dir = build_simbi(args)
    extras = "" if not args.extras else "[extras]"
    install_mode = "." + extras if args.install_mode == "default" else "-e" + "." + extras

    p1 = subprocess.Popen([sys.executable, "-m", "pip", "install", install_mode], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(("grep", "-v", "Requirement already satisfied"), stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()[0]

    subprocess.run(["rm", "-rf", f"{egg_dir}", f"{build_dir}"], check=True)

def uninstall_simbi(args: argparse.Namespace) -> None:
    simbi_dir = Path().resolve()
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "simbi"], check=True)
    exts = [str(ext) for ext in Path(simbi_dir / "simbi/libs/").glob("*.so")]
    if exts:
        subprocess.run(["rm", "-ri", *exts], check=True)

def main() -> int:
    _, args = parse_the_arguments()
    args.func(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())