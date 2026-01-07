#!/usr/bin/env python3
"""
simbi build and installation wrapper
wraps meson commands with simplified interface
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, **kwargs):
    """run subprocess with error handling and diagnostics"""
    verbose = kwargs.pop('verbose', False)
    timeout = kwargs.pop('timeout', None)

    if verbose:
        print(f"executing: {' '.join(str(c) for c in cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout, **kwargs)
        duration = time.time() - start_time
        if verbose:
            print(f"command completed in {duration:.1f}s")
        return result
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"error: command timeout after {duration:.1f}s: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        sys.exit(124)  # standard timeout exit code
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"error: command failed after {duration:.1f}s: {' '.join(str(c) for c in cmd)}", file=sys.stderr)

        # provide intelligent error hints
        if 'nvcc' in str(cmd) or 'hipcc' in str(cmd):
            print("hint: gpu compilation failed. try:", file=sys.stderr)
            print("  - reducing parallel jobs: --gpu-jobs 1", file=sys.stderr)
            print("  - checking gpu driver installation", file=sys.stderr)
            print("  - verifying device architecture: --device-arch", file=sys.stderr)
        elif 'meson' in str(cmd):
            print("hint: build configuration failed. try:", file=sys.stderr)
            print("  - clean rebuild: ./dev.py clean && ./dev.py build", file=sys.stderr)
            print("  - check dependencies with: meson setup --wipe", file=sys.stderr)

        sys.exit(e.returncode)


def detect_system_capabilities():
    """detect system capabilities for optimal build configuration"""
    capabilities = {
        'cpu_count': os.cpu_count() or 4,
        'memory_gb': 8,  # fallback
        'gpu_backend': None,
        'fast_linker': None,
    }

    # detect memory
    try:
        if Path('/proc/meminfo').exists():
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        capabilities['memory_gb'] = max(1, kb // 1024 // 1024)
                        break
    except (IOError, ValueError):
        pass

    # detect gpu backend
    if shutil.which('nvcc'):
        capabilities['gpu_backend'] = 'cuda'
    elif shutil.which('hipcc'):
        capabilities['gpu_backend'] = 'hip'

    # detect fast linker
    for linker in ['mold', 'lld', 'gold']:
        if shutil.which(linker):
            capabilities['fast_linker'] = linker
            break

    return capabilities


def get_optimal_gpu_jobs(capabilities, user_override=None):
    """determine gpu job count based on system capabilities"""
    if user_override:
        return user_override

    cpu_count = capabilities['cpu_count']
    memory_gb = capabilities['memory_gb']

    # memory-aware approach for gpu compilation
    if memory_gb >= 32:
        return min(cpu_count // 2, 8)
    elif memory_gb >= 16:
        return min(cpu_count // 3, 4)
    elif memory_gb >= 8:
        return min(cpu_count // 4, 2)
    else:
        return 1


def build_command(args):
    """build system with automatic optimization"""
    build_dir = Path(args.build_dir)

    # detect system capabilities for optimization
    capabilities = detect_system_capabilities()

    if args.verbose:
        print("system capabilities:")
        print(f"  cpu cores: {capabilities['cpu_count']}")
        print(f"  memory: {capabilities['memory_gb']} GB")
        print(f"  gpu backend: {capabilities['gpu_backend'] or 'none'}")
        print(f"  fast linker: {capabilities['fast_linker'] or 'system default'}")
        print()

    # meson configuration with defaults
    needs_reconfigure = (
        args.reconfigure or
        not (build_dir / "build.ninja").exists() or
        (args.gpu and not (build_dir / "meson-info/intro-dependencies.json").exists())
    )

    if needs_reconfigure:
        setup_cmd = ["meson", "setup", str(build_dir)]

        # gpu configuration with validation
        if args.gpu:
            if not capabilities['gpu_backend']:
                print("error: gpu compilation requested but no gpu backend available", file=sys.stderr)
                print("install cuda toolkit (nvcc) or rocm (hipcc)", file=sys.stderr)
                sys.exit(1)

            setup_cmd.append("-Dgpu_compilation=enabled")
            print(f"enabling gpu compilation with {capabilities['gpu_backend']} backend")

        # device architecture with intelligent defaults
        if args.device_arch:
            setup_cmd.append(f"-Ddevice_arch={args.device_arch}")
        elif args.gpu and capabilities['gpu_backend'] == 'cuda':
            # auto-detect cuda architecture if possible
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    compute_cap = result.stdout.strip().split('\n')[0]
                    arch = f"sm_{compute_cap.replace('.', '')}"
                    setup_cmd.append(f"-Ddevice_arch={arch}")
                    print(f"auto-detected device architecture: {arch}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # physics configuration
        if args.precision:
            setup_cmd.append(f"-Dprecision={args.precision}")

        if args.column_major:
            setup_cmd.append("-Dcolumn_major=true")

        if args.four_velocity:
            setup_cmd.append("-Dfour_velocity=true")

        # development options
        if args.build_tests:
            setup_cmd.append("-Dbuild_tests=true")

        # build optimization based on system capabilities
        if capabilities['memory_gb'] >= 16:
            setup_cmd.append("-Db_lto=true")  # enable lto for high memory systems

        if args.reconfigure:
            setup_cmd.append("--reconfigure")

        run(setup_cmd, verbose=args.verbose, timeout=300)

    # compilation with parallelization
    if not args.configure_only:
        compile_cmd = ["meson", "compile", "-C", str(build_dir)]
        compile_env = os.environ.copy()

        # intelligent job scheduling
        if args.gpu:
            gpu_jobs = get_optimal_gpu_jobs(capabilities, args.gpu_jobs)
            compile_cmd.extend(["-j", str(gpu_jobs)])
            print(f"gpu build: using {gpu_jobs} parallel jobs")

            # gpu-specific environment optimization
            if capabilities['gpu_backend'] == 'cuda':
                compile_env.update({
                    'CUDA_CACHE_DISABLE': '1',
                    'CUDA_CACHE_MAXSIZE': str(min(1024, capabilities['memory_gb'] * 128)),
                })

                # memory-constrained systems get additional limits
                if capabilities['memory_gb'] < 16:
                    compile_env['NVCC_PREPEND_FLAGS'] = '--memory-limit-mb=1536'

        else:
            # cpu build parallelism
            cpu_jobs = min(capabilities['cpu_count'], capabilities['memory_gb'] * 2)
            compile_cmd.extend(["-j", str(cpu_jobs)])
            if args.verbose:
                print(f"cpu build: using {cpu_jobs} parallel jobs")

        # verbose output for debugging
        if args.verbose:
            compile_cmd.append("--verbose")

        # build timeout based on system and compilation type
        timeout = 600 if args.gpu else 300  # longer timeout for gpu builds
        if capabilities['memory_gb'] < 8:
            timeout *= 2  # longer for low memory systems

        print(f"starting {'gpu' if args.gpu else 'cpu'} compilation...")
        start_time = time.time()
        run(compile_cmd, env=compile_env, verbose=args.verbose, timeout=timeout)

        build_time = time.time() - start_time
        print(f"compilation completed in {build_time:.1f}s")


def install_command(args):
    """build and installation with validation"""
    # build first with timing
    print("=== build phase ===")
    build_start = time.time()
    build_command(args)
    build_time = time.time() - build_start

    if not args.configure_only:
        print("\n=== install phase ===")
        install_start = time.time()

        # meson install with gpu environment
        install_env = os.environ.copy()
        if args.gpu:
            install_env['CUDA_CACHE_DISABLE'] = '1'

        run(["meson", "install", "-C", args.build_dir], env=install_env, verbose=args.verbose)

        # pip install with validation
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

        run(pip_cmd, verbose=args.verbose)

        install_time = time.time() - install_start
        total_time = time.time() - build_start

        print("\n=== installation summary ===")
        print(f"build time: {build_time:.1f}s")
        print(f"install time: {install_time:.1f}s")
        print(f"total time: {total_time:.1f}s")

        # validate installation
        try:
            result = subprocess.run([sys.executable, "-c", "import simbi; print('installation verified')"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("installation validation: success")
            else:
                print("warning: installation validation failed", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("warning: installation validation timeout", file=sys.stderr)


def uninstall_command(args):
    """uninstallation with cleanup"""
    print("=== uninstalling simbi ===")

    # pip uninstall
    run([sys.executable, "-m", "pip", "uninstall", "-y", "simbi"], verbose=args.verbose)

    # clean build artifacts
    artifacts_removed = 0
    lib_dir = Path("simbi/libs")
    if lib_dir.exists():
        for ext in lib_dir.glob("*.so"):
            ext.unlink()
            print(f"removed {ext}")
            artifacts_removed += 1

    # clean python cache
    for cache_dir in Path(".").rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            artifacts_removed += 1
            if args.verbose:
                print(f"removed cache: {cache_dir}")

    print(f"cleanup complete: {artifacts_removed} artifacts removed")


def clean_command(args):
    """build cleanup"""
    cleaned_items = 0

    # clean build directory
    build_dir = Path(args.build_dir)
    if build_dir.exists():
        if args.verbose:
            print(f"removing build directory: {build_dir}")
        shutil.rmtree(build_dir)
        print(f"removed build directory: {build_dir}")
        cleaned_items += 1

    # clean generated files if requested
    if args.all:
        # remove compiled extensions
        lib_dir = Path("simbi/libs")
        if lib_dir.exists():
            for ext in lib_dir.glob("*.so"):
                ext.unlink()
                print(f"removed extension: {ext}")
                cleaned_items += 1

        # remove python cache
        for cache_dir in Path(".").rglob("__pycache__"):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
                cleaned_items += 1
                if args.verbose:
                    print(f"removed cache: {cache_dir}")

        # remove temporary gpu files
        for temp_file in Path(".").rglob("gpu_unified.*"):
            if temp_file.is_file():
                temp_file.unlink()
                print(f"removed temp file: {temp_file}")
                cleaned_items += 1

    if cleaned_items == 0:
        print("nothing to clean")
    else:
        print(f"cleanup complete: {cleaned_items} items removed")


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
        "--device-arch", help="GPU device architecture (e.g., sm_75, gfx906)"
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
    build_parser.add_argument(
        "--gpu-jobs", type=int, help="max parallel jobs for gpu compilation (auto-detected if not specified)"
    )
    build_parser.add_argument(
        "--timeout", type=int, help="build timeout in seconds"
    )
    build_parser.set_defaults(func=build_command)

    # install command
    install_parser = subparsers.add_parser("install", help="build and install")
    install_parser.add_argument(
        "--gpu", action="store_true", help="enable gpu compilation"
    )
    install_parser.add_argument(
        "--device-arch", help="GPU device architecture (e.g., sm_75, gfx906)"
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
        "--gpu-jobs", type=int, help="max parallel jobs for gpu compilation (auto-detected if not specified)"
    )
    install_parser.add_argument(
        "--timeout", type=int, help="build timeout in seconds"
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
    clean_parser = subparsers.add_parser("clean", help="clean build directory and artifacts")
    clean_parser.add_argument(
        "--all", action="store_true", help="clean all generated files including extensions and cache"
    )
    clean_parser.set_defaults(func=clean_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
