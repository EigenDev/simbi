#!/usr/bin/env python3
"""
simbi build and installation wrapper
wraps meson commands with simplified interface
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    """run subprocess with error handling"""
    try:
        return subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print(
            f"error: command failed: {' '.join(str(c) for c in cmd)}", file=sys.stderr
        )
        sys.exit(e.returncode)


def build_command(args):
    """build the project"""
    build_dir = Path(args.build_dir)

    # reconfigure if needed
    if args.reconfigure or not (build_dir / "build.ninja").exists():
        setup_cmd = ["meson", "setup", str(build_dir)]

        if args.gpu:
            setup_cmd.append("-Dgpu_compilation=enabled")

        if args.precision:
            setup_cmd.append(f"-Dprecision={args.precision}")

        if args.column_major:
            setup_cmd.append("-Dcolumn_major=true")

        if args.four_velocity:
            setup_cmd.append("-Dfour_velocity=true")

        if args.build_tests:
            setup_cmd.append("-Dbuild_tests=true")

        if args.reconfigure:
            setup_cmd.append("--reconfigure")

        run(setup_cmd)

    # compile
    if not args.configure_only:
        compile_cmd = ["meson", "compile", "-C", str(build_dir)]
        if args.verbose:
            compile_cmd.append("--verbose")
        run(compile_cmd)


def install_command(args):
    """build and install the project"""
    # build first
    build_command(args)

    if not args.configure_only:
        # meson install
        run(["meson", "install", "-C", args.build_dir])

        # pip install
        extras = []
        if args.cli_extras:
            extras.append("cli")
        if args.visual_extras:
            extras.append("visual")

        pip_target = "."
        if extras:
            pip_target += "[" + ",".join(extras) + "]"

        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if args.editable:
            pip_cmd.append("-e")
        pip_cmd.append(pip_target)

        run(pip_cmd)


def uninstall_command(args):
    """uninstall the project"""
    run([sys.executable, "-m", "pip", "uninstall", "-y", "simbi"])

    # clean build artifacts
    lib_dir = Path("simbi/libs")
    if lib_dir.exists():
        for ext in lib_dir.glob("*.so"):
            ext.unlink()
            print(f"removed {ext}")


def clean_command(args):
    """clean build directory"""
    build_dir = Path(args.build_dir)
    if build_dir.exists():
        import shutil

        shutil.rmtree(build_dir)
        print(f"removed {build_dir}")


def main():
    parser = argparse.ArgumentParser(description="simbi build tool")
    parser.add_argument("--build-dir", default="build", help="build directory")
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build command
    build_parser = subparsers.add_parser("build", help="build the project")
    build_parser.add_argument(
        "--gpu", action="store_true", help="enable gpu compilation"
    )
    build_parser.add_argument(
        "--precision", choices=["single", "double"], help="floating point precision"
    )
    build_parser.add_argument(
        "--column-major", action="store_true", help="use column-major layout"
    )
    build_parser.add_argument(
        "--four-velocity", action="store_true", help="use four-velocity primitive"
    )
    build_parser.add_argument("--build-tests", action="store_true", help="build tests")
    build_parser.add_argument(
        "--reconfigure", action="store_true", help="force reconfiguration"
    )
    build_parser.add_argument(
        "--configure-only", action="store_true", help="configure without compiling"
    )
    build_parser.set_defaults(func=build_command)

    # install command
    install_parser = subparsers.add_parser("install", help="build and install")
    install_parser.add_argument(
        "--gpu", action="store_true", help="enable gpu compilation"
    )
    install_parser.add_argument(
        "--precision", choices=["single", "double"], help="floating point precision"
    )
    install_parser.add_argument(
        "--column-major", action="store_true", help="use column-major layout"
    )
    install_parser.add_argument(
        "--four-velocity", action="store_true", help="use four-velocity primitive"
    )
    install_parser.add_argument(
        "--build-tests", action="store_true", help="build tests"
    )
    install_parser.add_argument(
        "--reconfigure", action="store_true", help="force reconfiguration"
    )
    install_parser.add_argument(
        "--configure-only", action="store_true", help="configure without compiling"
    )
    install_parser.add_argument(
        "--editable", "-e", action="store_true", help="editable install"
    )
    install_parser.add_argument(
        "--cli-extras", action="store_true", help="install cli extras"
    )
    install_parser.add_argument(
        "--visual-extras", action="store_true", help="install visual extras"
    )
    install_parser.set_defaults(func=install_command)

    # uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="uninstall the project")
    uninstall_parser.set_defaults(func=uninstall_command)

    # clean command
    clean_parser = subparsers.add_parser("clean", help="clean build directory")
    clean_parser.set_defaults(func=clean_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
