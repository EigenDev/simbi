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
default["progress_bar"] = True
default["column_major"] = False
default["precision"] = 'double'
default["install_mode"] = "default"
default["dev_arch"] = 86
default["build_dir"] = "builddir"
default["four_velocity"] = False
default["shared_memory"] = True

YELLOW = "\033[0;33m"
RST = "\033[0m"  # No Color

flag_overrides = {}
flag_overrides["precision"] = ["--double", "--float"]
flag_overrides["gpu_compilation"] = ["--gpu-compilation", "--cpu-compilation"]
flag_overrides["column_major"] = ["--row-major", "--column-major"]
flag_overrides["four_velocity"] = ["--four-velocity", "--no-four-velocity"]
flag_overrides["progress_bar"] = ["--progress-bar", "--no-progress-bar"]
flag_overrides["shared_memory"] = ["--shared-memory", "--no-shared-memory"]
flag_overrides["install_mode"] = ["develop", "default"]


def get_tool(name: str) -> Optional[str]:
    import platform
    from shutil import which

    if name in ["cc", "c++"]:
        if platform.system() == "Darwin":
            homebrew = Path("/opt/homebrew/")
            if not homebrew.is_dir():
                print(
                    f"{YELLOW}WRN{RST}no homebrew found. running Apple's default compiler might raise issues"
                )
                cont = input("Continue anyway? [y/N]")
                if cont == "N":
                    sys.exit(0)
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
    -Dcolumn_major={args.column_major} -Dprecision={args.precision} \
    -Dprofile={args.install_mode} -Dgpu_arch={args.dev_arch} -Dfour_velocity={args.four_velocity} \
    -Dcpp_std={args.cpp_version} -Dbuildtype={args.build_type} {reconfigure} \
    -Dprogress_bar={args.progress_bar} -Dshared_memory={args.shared_memory}""".split()
    return command


def generate_home_locator(simbi_dir: str) -> None:
    git_home_file = Path(simbi_dir) / ("simbi/" + github_toplevel)
    if git_home_file.exists():
        return

    with open(str(git_home_file), "w") as f:
        f.write(get_output("git rev-parse --show-toplevel".split()))


def parse_the_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser("Parser for building and installing simbi with meson")
    subparsers = parser.add_subparsers(
        help="sub-commands that build / install / uninstall the code"
    )
    build_parser = subparsers.add_parser("build", add_help=False)
    build_parser.set_defaults(func=build_simbi)
    unbuild_parser = subparsers.add_parser("uninstall", help="uninstall simbi")
    unbuild_parser.set_defaults(func=uninstall_simbi)
    build_parser.add_argument(
        "--dev-arch",
        help="SM architecture specification for gpu compilation",
        type=int,
        default=86,
    )
    build_parser.add_argument(
        "--verbose",
        "-v",
        help="flag for verbose compilation output",
        action="store_const",
        default=[],
        const=["--verbose"],
    )
    build_parser.add_argument(
        "--configure",
        help="flag to only configure the meson build directory without installing",
        action="store_true",
        default=False,
    )
    build_parser.add_argument(
        "--install-mode",
        help="install mode (normal or editable)",
        default="default",
        type=str,
        choices=["default", "develop"],
    )
    build_parser.add_argument(
        "--build-dir",
        help="build directory name for meson build",
        type=str,
        default="builddir",
    )
    build_parser.add_argument(
        "--extras",
        help="flag to install the optional dependencies",
        action="store_true",
        default=False,
    )
    build_parser.add_argument(
        "--four-velocity",
        help="flag to set four-velocity as the velocity primitive instead of beta",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    build_parser.add_argument(
        "--progress-bar",
        help="flag to show / hide progress bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    build_parser.add_argument(
        "--shared-memory",
        help="flag to enable / disable shared memory for gpu builds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    build_parser.add_argument(
        "--cpp17",
        help="flag for setting c++ version to c++17 instead of default c++20",
        action="store_const",
        default="c++20",
        const="c++17",
        dest="cpp_version",
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
        "--double", action="store_const", dest="precision", const='double'
    )
    precision.add_argument(
        "--float", action="store_const", dest="precision", const='single'
    )
    major = build_parser.add_mutually_exclusive_group()
    major.add_argument(
        "--row-major", action="store_const", dest="column_major", const=False
    )
    major.add_argument(
        "--column-major", action="store_const", dest="column_major", const=True
    )
    build_parser.set_defaults(
        precision='double',
        column_major=False,
        gpu_compilation="disabled",
        build_type="release",
    )
    install_parser = subparsers.add_parser("install", help="install simbi", parents=[build_parser])
    install_parser.set_defaults(func=install_simbi)

    return parser, parser.parse_args(args=None if sys.argv[1:] else ["--help"])

def build_simbi(args: argparse.Namespace) -> tuple[str]:
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

    # h5cc_show = get_output(["h5cc", "-show"]).split()
    h5pkg = get_output(["pkg-config", "--cflags", "hdf5"]).split()
    hdf5_include = " ".join(
        [include_dir[2:] for include_dir in filter(lambda x: x.startswith("-I"), h5pkg)]
    )

    # if not hdf5_include:
    #     hdf5_libpath = Path(
    #         " ".join(
    #             [
    #                 lib_dir[2:]
    #                 for lib_dir in filter(lambda x: x.startswith("-L"), h5cc_show)
    #             ]
    #         )
    #     )
    #     hdf5_include = hdf5_libpath.parents[0].resolve() / "include"

    config_command = configure(args, reconfigure_flag, hdf5_include, gpu_include)
    subprocess.run(config_command, env=simbi_env, check=True)
    build_dir = f"{simbi_dir}/build"
    egg_dir = f"{simbi_dir}/simbi.egg-info"
    if not args.configure:
        extras = "" if not args.extras else "[extras]"
        install_mode = (
            "." + extras if args.install_mode == "default" else "-e" + "." + extras
        )
        lib_dir = Path(simbi_dir / "simbi/libs")
        lib_dir.mkdir(parents=True, exist_ok=True)
        compile_child = subprocess.Popen(
            ["meson", "compile"] + args.verbose, cwd=f"{args.build_dir}"
        ).wait()
        install_child = subprocess.Popen(
            ["meson", "install"], cwd=f"{args.build_dir}"
        ).wait()
        
        if compile_child == install_child == 0:
            return egg_dir, build_dir
        else:
            raise subprocess.CalledProcessError("Error ocurred during build")
        
    return egg_dir, build_dir
    
def install_simbi(args: argparse.Namespace) -> None:
    egg_dir, build_dir = build_simbi(args)
    p1 = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", args.install_mode],
        stdout=subprocess.PIPE,
    )
    p2 = subprocess.Popen(
        ("grep", "-v", "Requirement already satisfied"),
        stdin=p1.stdout,
    )
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
