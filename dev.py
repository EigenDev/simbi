import argparse
import sys
import subprocess
import json
import os
from pathlib import Path
from typing import Optional

cache_file = "simbi_build_cache.txt"
github_toplevel = "gitrepo_home.txt"
default = {}
default["gpu_compilation"] = "disabled"
default["column_major"] = False
default["float_precision"] = False
default["install_mode"] = "default"
default["dev_arch"] = 86
default["build_dir"] = "builddir"
default["four_velocity"] = False

YELLOW = "\033[0;33m"
RST = "\033[0m"  # No Color

flag_overrides = {}
flag_overrides["float_precision"] = ["--double", "--float"]
flag_overrides["gpu_compilation"] = ["--gpu-compilation", "--cpu-compilation"]
flag_overrides["column_major"] = ["--row-major", "--column-major"]
flag_overrides["four_velocity"] = ["--four-velocity", "--no-four-velocity"]
flag_overrides["install_mode"] = ["develop", "default"]


def get_tool(name: str) -> Optional[str]:
    import platform 
    from shutil import which
    if name in ['cc', 'c++']:
        if platform.system() == 'Darwin':
            comps: list[str]
            homebrew = Path('/opt/homebrew/opt/')
            if not homebrew:
                raise FileExistsError("Homebrew should be installed for Mac downloads")
            #search for gcc in homebrew channel
            if name == 'cc':
                comps = [str(x) for x in Path(f'{homebrew}/gcc/bin/').glob('gcc*')]
            elif name == 'c++':
                comps = [str(x) for x in Path(f'{homebrew}/gcc/bin/').glob('g++*')]
            # no gcc? ok, search for LLVM's clang    
            if not comps:
                if name == 'cc':
                    comps = [str(x) for x in Path(f'{homebrew}/llvm/bin').glob('clang*')]
                elif name == 'c++':
                    comps  = [str(x) for x in Path(f'{homebrew}/llvm/bin').glob('clang++*')]
            
            return min(comps, key=len)
        else:
            return which(name)
    
    return which(name)


def is_tool(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    return get_tool(name) is not None


def read_from_cache() -> Optional[dict[str, str]]:
    cached_args = Path(cache_file)
    if cached_args.exists():
        with open(cache_file, "r") as f:
            data = f.read()
        return json.loads(data)
    else:
        with open(cache_file, "w+"):
            ...
        return None


def check_minimal_dependencies() -> None:
    MIN_PYTHON = (3, 10)
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError("Python {}.{} or later is required".format(*MIN_PYTHON))

    try:
        import mesonbuild
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "meson"], check=True)

    try:
        import numpy
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy"], check=True)

    try:
        import cython
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "cython"], check=True)

    try:
        import ninja
    except ImportError:
        if not is_tool("make"):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "ninja"], check=True
            )


def write_to_cache(args: argparse.Namespace) -> None:
    details = vars(args)
    details["build_dir"] = str(Path(details["build_dir"]).resolve())
    details.pop("func")
    with open(cache_file, "w") as f:
        f.write(json.dumps(details, indent=4))


def get_output(command: str) -> str:
    return (
        subprocess.Popen(command, stdout=subprocess.PIPE)
        .stdout.read()
        .decode("utf-8")
        .strip()
    )


def configure(
    args: argparse.Namespace, reconfigure: str, hdf5_include: str, gpu_include: str
) -> list[str]:
    command = f"""meson setup {args.build_dir} -Dgpu_compilation={args.gpu_compilation}  
    -Dhdf5_include_dir={hdf5_include} -Dgpu_include_dir={gpu_include} \
    -Dcolumn_major={args.column_major} -Dfloat_precision={args.float_precision} \
    -Dprofile={args.install_mode} -Dgpu_arch={args.dev_arch} -Dfour_velocity={args.four_velocity} \
    -Dcpp_std={args.cpp_version} -Dbuildtype={args.build_type} {reconfigure}""".split()
    return command


def generate_home_locator(simbi_dir: str) -> None:
    git_home_file = Path(simbi_dir) / ("simbi/" + github_toplevel)
    if git_home_file.exists():
        return

    with open(str(git_home_file), "w") as f:
        f.write(get_output("git rev-parse --show-toplevel".split()))


def parse_the_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser("Parser for installing simbi with meson")
    subparsers = parser.add_subparsers(
        help="sub-commands that install / uninstall the code"
    )
    install_parser = subparsers.add_parser("install", help="install simbi")
    install_parser.set_defaults(func=install_simbi)
    uninstall_parser = subparsers.add_parser("uninstall", help="uninstall simbi")
    uninstall_parser.set_defaults(func=uninstall_simbi)
    install_parser.add_argument(
        "--dev-arch",
        help="SM architecture specification for gpu compilation",
        type=int,
        default=86,
    )
    install_parser.add_argument(
        "--verbose",
        "-v",
        help="flag for verbose compilation output",
        action="store_const",
        default=[],
        const=["--verbose"],
    )
    install_parser.add_argument(
        "--configure",
        help="flag to only configure the meson build directory without installing",
        action="store_true",
        default=False,
    )
    install_parser.add_argument(
        "--install-mode",
        help="install mode (normal or editable)",
        default="default",
        type=str,
        choices=["default", "develop"],
    )
    install_parser.add_argument(
        "--build-dir",
        help="build directory name for meson build",
        type=str,
        default="builddir",
    )
    install_parser.add_argument(
        "--extras",
        help="flag to install the optional dependencies",
        action="store_true",
        default=False,
    )
    install_parser.add_argument(
        "--four-velocity",
        help="flag to set four-velocity as the velocity primitive instead of beta",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    install_parser.add_argument(
        "--cpp17",
        help="flag for setting c++ version to c++17 instead of default c++20",
        action="store_const",
        default="c++20",
        const="c++17",
        dest="cpp_version",
    )
    compile_type = install_parser.add_mutually_exclusive_group()
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
    build_type = install_parser.add_mutually_exclusive_group()
    build_type.add_argument(
        "--release", action="store_const", dest="build_type", const="release"
    )
    build_type.add_argument(
        "--debug", action="store_const", dest="build_type", const="debug"
    )
    precision = install_parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--double", action="store_const", dest="float_precision", const=False
    )
    precision.add_argument(
        "--float", action="store_const", dest="float_precision", const=True
    )
    major = install_parser.add_mutually_exclusive_group()
    major.add_argument(
        "--row-major", action="store_const", dest="column_major", const=False
    )
    major.add_argument(
        "--column-major", action="store_const", dest="column_major", const=True
    )
    install_parser.set_defaults(
        float_precision=False,
        column_major=False,
        gpu_compilation="disabled",
        build_type="release",
    )

    return parser, parser.parse_args(args=None if sys.argv[1:] else ["--help"])


def install_simbi(args: argparse.Namespace) -> None:
    simbi_dir = Path().resolve()
    if args.build_dir == "build":
        raise argparse.ArgumentError(
            args.builddir, "please choose a different build name other than 'build'"
        )

    check_minimal_dependencies()
    simbi_env = os.environ.copy()
    # Check if any args passed to the cli exist that would override the cache args
    cli_args = sys.argv[1:]
    if cached_vars := read_from_cache():
        for arg in vars(args):
            if arg in [
                "verbose",
                "configure",
                "func",
                "extras",
                "cpp_version",
                "build_type",
            ]:
                continue

            if getattr(args, arg) == default[arg]:
                if arg in cli_args:
                    continue

                if arg in flag_overrides.keys() and any(
                    x in flag_overrides[arg] for x in cli_args
                ):
                    continue
                else:
                    setattr(args, arg, cached_vars[arg])
    generate_home_locator(simbi_dir=simbi_dir)
    write_to_cache(args)

    build_configured = (
        subprocess.run(
            ["meson", "introspect", f"{args.build_dir}", "-i", "--targets"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )

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
        gpu_runtime_dir = " ".join(
            [
                str(path.parent)
                for path in which_cuda.parents
                if "cuda" in str(path.parent)
            ]
        )
    elif is_tool("hipcc"):
        gpu_runtime_dir = get_output(["hipconfig", "--rocmpath"])

    try:
        gpu_include = f"{gpu_runtime_dir.split()[0]}/include"
    except IndexError:
        gpu_include = "" 
    
    h5cc_show = get_output(["h5cc", "-show"]).split()
    hdf5_include = " ".join(
        [
            include_dir[2:]
            for include_dir in filter(lambda x: x.startswith("-I"), h5cc_show)
        ]
    )

    if not hdf5_include:
        hdf5_libpath = Path(
            " ".join(
                [
                    lib_dir[2:]
                    for lib_dir in filter(lambda x: x.startswith("-L"), h5cc_show)
                ]
            )
        )
        hdf5_include = hdf5_libpath.parents[0].resolve() / "include"

    config_command = configure(args, reconfigure_flag, hdf5_include, gpu_include)
    subprocess.run(config_command, env=simbi_env, check=True)
    if not args.configure:
        extras = "" if not args.extras else "[extras]"
        install_mode = (
            "." + extras if args.install_mode == "default" else "-e" + "." + extras
        )
        build_dir = f"{simbi_dir}/build"
        egg_dir = f"{simbi_dir}/simbi.egg-info"
        lib_dir = Path(simbi_dir / "simbi/libs")
        lib_dir.mkdir(parents=True, exist_ok=True)
        compile_child = subprocess.Popen(
            ["meson", "compile"] + args.verbose, cwd=f"{args.build_dir}"
        ).wait()
        install_child = subprocess.Popen(
            ["meson", "install"], cwd=f"{args.build_dir}"
        ).wait()
        if compile_child == install_child == 0:
            subprocess.Popen(
                [sys.executable, "-m", "pip", "install", install_mode]
            ).wait()
            subprocess.run(["rm", "-rf", f"{egg_dir}", f"{build_dir}"], check=True)


def uninstall_simbi(args: argparse.Namespace) -> None:
    simbi_dir = Path().resolve()
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "simbi"], check=True)
    try:
        exts = [str(ext) for ext in Path(simbi_dir / "simbi/libs/").glob("*.so")]
        subprocess.run(["rm", "-r", *exts], check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        print(f"{err} {err.stderr.decode('utf8')}")

def main() -> int:
    _, args = parse_the_arguments()
    args.func(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())
