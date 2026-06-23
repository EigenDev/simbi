#!/usr/bin/env python3
# =============================================================================
# dev.py
#
# simbi build/install wrapper: dev.py -> meson -> cargo.
# the rust hydro backend (rust_src/crates/symbi-py) is the default. meson wraps
# cargo via scripts/build_rust_backend.py and installs the cdylib as
# simbi/libs/cpu_ext<EXT_SUFFIX> (+ rad_hydro from symbi-rad-py); the unchanged
# `simbi.libs.cpu_ext` import then loads rust with no python-side change.
# the legacy c++ backend stays available behind `--backend cpp` (disabled by
# default) and is the ONLY path that consumes the gpu / precision / layout flags.
# usage:
#  ./dev.py install -e              # build rust + editable python install
#  ./dev.py build                   # compile rust cdylib only
#  ./dev.py build --features cuda   # rust gpu build (or: --gpu)
#  ./dev.py build --backend cpp --gpu --device-arch sm_86
#  ./dev.py clean --all             # wipe meson build + cargo target + extensions
# =============================================================================

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUST_SRC = Path("rust_src")


def meson() -> list:
    """resolve the meson launcher.

    prefer the meson sitting next to the interpreter running this script (a venv's
    `bin/meson`), so meson's python — and therefore the EXT_SUFFIX the rust
    custom_target bakes into the extension filename — matches the python we install
    into. fall back to PATH, then to the meson module in this interpreter.
    """
    sibling = Path(sys.executable).parent / "meson"
    if sibling.exists():
        return [str(sibling)]
    found = shutil.which("meson")
    if found:
        return [found]
    return [sys.executable, "-m", "mesonbuild.mesonmain"]


def run(cmd, **kwargs):
    """run subprocess with error handling and diagnostics"""
    verbose = kwargs.pop('verbose', False)
    timeout = kwargs.pop('timeout', None)

    if verbose:
        print(f"executing: {' '.join(str(c) for c in cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout, **kwargs)
        if verbose:
            print(f"command completed in {time.time() - start_time:.1f}s")
        return result
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"error: command timeout after {duration:.1f}s: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        sys.exit(124)  # standard timeout exit code
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"error: command failed after {duration:.1f}s: {' '.join(str(c) for c in cmd)}", file=sys.stderr)

        # intelligent error hints
        if 'cargo' in str(cmd):
            print("hint: cargo build failed. try:", file=sys.stderr)
            print("  - clean rust target: ./dev.py clean --all", file=sys.stderr)
            print("  - build the crate directly: cargo build --release -p symbi-py --manifest-path rust_src/Cargo.toml", file=sys.stderr)
        elif 'nvcc' in str(cmd) or 'hipcc' in str(cmd):
            print("hint: gpu compilation failed. try:", file=sys.stderr)
            print("  - reducing parallel jobs: --gpu-jobs 1", file=sys.stderr)
            print("  - verifying device architecture: --device-arch", file=sys.stderr)
        elif 'meson' in str(cmd):
            print("hint: build configuration failed. try:", file=sys.stderr)
            print("  - clean rebuild: ./dev.py clean && ./dev.py build", file=sys.stderr)

        sys.exit(e.returncode)


def venv_env(base=None):
    """augment the environment so child meson/uv processes resolve THIS
    interpreter's venv python.

    meson's `find_installation('python3')` and uv both honor `VIRTUAL_ENV`;
    without it meson picks the system python3 and bakes the WRONG EXT_SUFFIX into
    the extension filename (e.g. cpython-314 when the venv is 3.13), so the import
    silently fails. a no-op when dev.py runs outside a venv.
    """
    env = dict(base if base is not None else os.environ)
    if sys.prefix != sys.base_prefix:  # inside a venv
        bin_dir = str(Path(sys.executable).parent)
        env["VIRTUAL_ENV"] = sys.prefix
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def pip_install(target, editable, verbose):
    """install the python package into the active venv, preferring uv over pip
    (uv-managed venvs ship no pip module)."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable]
    else:
        cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(target)
    run(cmd, env=venv_env(), verbose=verbose)


def detect_system_capabilities():
    """detect system capabilities for the c++ gpu build path"""
    capabilities = {
        'cpu_count': os.cpu_count() or 4,
        'memory_gb': 8,  # fallback
        'gpu_backend': None,
    }

    # detect memory (linux /proc, darwin sysctl)
    try:
        if Path('/proc/meminfo').exists():
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        capabilities['memory_gb'] = max(1, kb // 1024 // 1024)
                        break
        elif sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                capabilities['memory_gb'] = max(1, int(out.stdout.strip()) // (1024 ** 3))
    except (IOError, ValueError, subprocess.SubprocessError):
        pass

    # detect gpu backend (c++ path only)
    if shutil.which('nvcc'):
        capabilities['gpu_backend'] = 'cuda'
    elif shutil.which('hipcc'):
        capabilities['gpu_backend'] = 'hip'

    return capabilities


def get_optimal_gpu_jobs(capabilities, user_override=None):
    """determine gpu job count based on system capabilities"""
    if user_override:
        return user_override

    cpu_count = capabilities['cpu_count']
    memory_gb = capabilities['memory_gb']
    if memory_gb >= 32:
        return min(cpu_count // 2, 8)
    if memory_gb >= 16:
        return min(cpu_count // 3, 4)
    if memory_gb >= 8:
        return min(cpu_count // 4, 2)
    return 1


def rust_features(args):
    """resolve the cargo feature string for the rust backend"""
    feats = [f.strip() for f in (args.features or "").split(",") if f.strip()]
    if args.gpu and 'cuda' not in feats:
        feats.append('cuda')
    return ",".join(feats)


def meson_setup_cmd(args, capabilities):
    """assemble the `meson setup` option list for the selected backend"""
    setup_cmd = [*meson(), "setup", str(args.build_dir), f"-Dhydro_backend={args.backend}"]

    if args.backend == "rust":
        # the rust cdylib is built --release by cargo; the only knob is the cargo
        # feature set (gpu -> cuda, plus any user --features). precision / layout /
        # unified-memory are c++-only and are intentionally NOT forwarded.
        feats = rust_features(args)
        if feats:
            setup_cmd.append(f"-Dcargo_features={feats}")
            print(f"rust features: {feats}")
        if args.gpu and not shutil.which("cargo"):
            print("error: rust gpu build requested but cargo not found", file=sys.stderr)
            sys.exit(1)
        for ignored in ("precision", "column_major", "four_velocity", "unified_memory", "linker"):
            val = getattr(args, ignored, None)
            if val and val not in ("auto",):
                print(f"note: --{ignored.replace('_', '-')} is c++-only; ignored for the rust backend")
        return setup_cmd

    # ---- c++ backend (legacy, disabled by default) ----
    if args.gpu:
        if not capabilities['gpu_backend']:
            print("error: gpu compilation requested but no gpu backend available", file=sys.stderr)
            print("install cuda toolkit (nvcc) or rocm (hipcc)", file=sys.stderr)
            sys.exit(1)
        setup_cmd.append("-Dgpu_compilation=enabled")
        print(f"enabling gpu compilation with {capabilities['gpu_backend']} backend")

    if args.device_arch:
        setup_cmd.append(f"-Ddevice_arch={args.device_arch}")
    elif args.gpu and capabilities['gpu_backend'] == 'cuda':
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                compute_cap = result.stdout.strip().split('\n')[0]
                arch = f"sm_{compute_cap.replace('.', '')}"
                setup_cmd.append(f"-Ddevice_arch={arch}")
                print(f"auto-detected device architecture: {arch}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if args.linker and args.linker != "auto":
        setup_cmd.append(f"-Dc_link_args=-fuse-ld={args.linker}")
        setup_cmd.append(f"-Dcpp_link_args=-fuse-ld={args.linker}")
        print(f"using {args.linker} linker")
    if args.precision:
        setup_cmd.append(f"-Dprecision={args.precision}")
    if args.column_major:
        setup_cmd.append("-Dcolumn_major=true")
    if args.four_velocity:
        setup_cmd.append("-Dfour_velocity=true")
    if args.unified_memory:
        setup_cmd.append("-Dunified_memory=true")
        print("enabling CUDA unified memory")
    if args.build_tests:
        setup_cmd.append("-Dbuild_tests=true")
    return setup_cmd


def build_command(args):
    """configure (meson) + compile (ninja -> cargo for rust)"""
    build_dir = Path(args.build_dir)
    capabilities = detect_system_capabilities()

    if args.backend == "rust" and not shutil.which("cargo"):
        print("error: cargo not found — install the rust toolchain (https://rustup.rs)", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"backend: {args.backend}")
        print(f"  cpu cores: {capabilities['cpu_count']}")
        print(f"  memory: {capabilities['memory_gb']} GB")
        if args.backend == "cpp":
            print(f"  gpu backend: {capabilities['gpu_backend'] or 'none'}")
        print()

    # reconfigure when forced, when the build dir is unconfigured, or when the
    # build dir holds a DIFFERENT backend than requested (a no-op compile would
    # otherwise build the stale backend).
    needs_reconfigure = (
        args.reconfigure
        or not (build_dir / "build.ninja").exists()
        or _configured_backend(build_dir) not in (None, args.backend)
    )

    if needs_reconfigure:
        setup_cmd = meson_setup_cmd(args, capabilities)
        if (build_dir / "build.ninja").exists():
            setup_cmd.append("--reconfigure")
        run(setup_cmd, env=venv_env(), verbose=args.verbose, timeout=300)

    if args.configure_only:
        return

    compile_cmd = [*meson(), "compile", "-C", str(build_dir)]
    compile_env = venv_env()

    # job scheduling matters only for the c++ gpu path; cargo manages its own
    # parallelism, so the rust path leaves -j to ninja's single custom_target.
    if args.backend == "cpp" and args.gpu:
        gpu_jobs = get_optimal_gpu_jobs(capabilities, args.gpu_jobs)
        compile_cmd.extend(["-j", str(gpu_jobs)])
        print(f"gpu build: using {gpu_jobs} parallel jobs")
        if capabilities['gpu_backend'] == 'cuda':
            compile_env['CUDA_CACHE_DISABLE'] = '1'
            if capabilities['memory_gb'] < 16:
                compile_env['NVCC_PREPEND_FLAGS'] = '--memory-limit-mb=1536'

    if args.verbose:
        compile_cmd.append("--verbose")

    timeout = args.timeout or (1800 if (args.backend == "cpp" and args.gpu) else 600)
    label = "c++ gpu" if (args.backend == "cpp" and args.gpu) else f"{args.backend}"
    print(f"starting {label} compilation...")
    start_time = time.time()
    run(compile_cmd, env=compile_env, verbose=args.verbose, timeout=timeout)
    print(f"compilation completed in {time.time() - start_time:.1f}s")


def _configured_backend(build_dir):
    """read the hydro_backend a build dir was configured with, or None"""
    try:
        import json
        opts = json.loads((build_dir / "meson-info" / "intro-buildoptions.json").read_text())
        for opt in opts:
            if opt.get("name") == "hydro_backend":
                return opt.get("value")
    except (IOError, ValueError, KeyError):
        pass
    return None


def install_command(args):
    """build + meson install (cdylib -> simbi/libs) + pip install"""
    print("=== build phase ===")
    build_start = time.time()
    build_command(args)
    build_time = time.time() - build_start

    if args.configure_only:
        return

    print("\n=== install phase ===")
    install_start = time.time()

    install_env = venv_env()
    if args.backend == "cpp" and args.gpu:
        install_env['CUDA_CACHE_DISABLE'] = '1'
    run([*meson(), "install", "-C", str(args.build_dir)], env=install_env, verbose=args.verbose)

    extras = []
    if args.cli_extras:
        extras.append("cli")
    if args.visual_extras:
        extras.append("visual")
    pip_target = "." + (f"[{','.join(extras)}]" if extras else "")
    pip_install(pip_target, args.editable, args.verbose)

    install_time = time.time() - install_start
    print("\n=== installation summary ===")
    print(f"build time: {build_time:.1f}s")
    print(f"install time: {install_time:.1f}s")
    print(f"total time: {time.time() - build_start:.1f}s")

    # validate installation
    try:
        result = subprocess.run([sys.executable, "-c", "import simbi; print('installation verified')"],
                                capture_output=True, text=True, timeout=20)
        print("installation validation:", "success" if result.returncode == 0 else "FAILED")
        if result.returncode != 0 and args.verbose:
            print(result.stderr, file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("warning: installation validation timeout", file=sys.stderr)


def uninstall_command(args):
    """pip uninstall + extension/cache cleanup"""
    print("=== uninstalling simbi ===")
    run([sys.executable, "-m", "pip", "uninstall", "-y", "simbi"], verbose=args.verbose)

    artifacts_removed = 0
    lib_dir = Path("simbi/libs")
    if lib_dir.exists():
        for ext in list(lib_dir.glob("*.so")) + list(lib_dir.glob("*.dylib")) + list(lib_dir.glob("*.pyd")):
            ext.unlink()
            print(f"removed {ext}")
            artifacts_removed += 1

    for cache_dir in Path(".").rglob("__pycache__"):
        if cache_dir.is_dir() and ".venv" not in cache_dir.parts:
            shutil.rmtree(cache_dir)
            artifacts_removed += 1
            if args.verbose:
                print(f"removed cache: {cache_dir}")

    print(f"cleanup complete: {artifacts_removed} artifacts removed")


def clean_command(args):
    """remove the meson build dir; with --all also the cargo target + extensions"""
    cleaned_items = 0

    build_dir = Path(args.build_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"removed build directory: {build_dir}")
        cleaned_items += 1

    if args.all:
        # cargo's own target tree (the rust incremental cache)
        if (RUST_SRC / "Cargo.toml").exists() and shutil.which("cargo"):
            run(["cargo", "clean", "--manifest-path", str(RUST_SRC / "Cargo.toml")], verbose=args.verbose)
            print("ran cargo clean")
            cleaned_items += 1
        elif (RUST_SRC / "target").exists():
            shutil.rmtree(RUST_SRC / "target")
            print(f"removed {RUST_SRC / 'target'}")
            cleaned_items += 1

        # compiled extensions
        lib_dir = Path("simbi/libs")
        if lib_dir.exists():
            for ext in list(lib_dir.glob("*.so")) + list(lib_dir.glob("*.dylib")) + list(lib_dir.glob("*.pyd")):
                ext.unlink()
                print(f"removed extension: {ext}")
                cleaned_items += 1

        for cache_dir in Path(".").rglob("__pycache__"):
            if cache_dir.is_dir() and ".venv" not in cache_dir.parts:
                shutil.rmtree(cache_dir)
                cleaned_items += 1

    if cleaned_items == 0:
        print("nothing to clean")
    else:
        print(f"cleanup complete: {cleaned_items} items removed")


def _add_build_args(p):
    """build/install share the same configuration flags"""
    p.add_argument("--backend", choices=["rust", "cpp"], default="rust",
                   help="hydro backend (default: rust; cpp is the legacy pybind11 path)")
    p.add_argument("--features", default="",
                   help="comma-separated cargo features for the rust backend (e.g. cuda)")
    p.add_argument("--gpu", action="store_true",
                   help="gpu build: rust -> cargo 'cuda' feature; cpp -> meson gpu_compilation")
    p.add_argument("--device-arch", help="[cpp] GPU device architecture (e.g., sm_75, gfx906)")
    p.add_argument("--linker", choices=["auto", "mold", "lld", "gold", "bfd"], default="auto",
                   help="[cpp] linker (auto=system default)")
    p.add_argument("--precision", choices=["single", "double"], help="[cpp] floating point precision")
    p.add_argument("--column-major", action="store_true", help="[cpp] column-major layout")
    p.add_argument("--four-velocity", action="store_true", help="[cpp] four-velocity primitive")
    p.add_argument("--unified-memory", action="store_true", help="[cpp] CUDA unified memory")
    p.add_argument("--build-tests", action="store_true", help="[cpp] build tests")
    p.add_argument("--reconfigure", action="store_true", help="force meson reconfiguration")
    p.add_argument("--configure-only", action="store_true", help="configure without compiling")
    p.add_argument("--gpu-jobs", type=int, help="[cpp] max parallel jobs for gpu compilation")
    p.add_argument("--timeout", type=int, help="build timeout in seconds")


def main():
    parser = argparse.ArgumentParser(description="simbi build tool (dev -> meson -> cargo)")
    parser.add_argument("--build-dir", default="build", help="meson build directory")
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build the project")
    _add_build_args(build_parser)
    build_parser.set_defaults(func=build_command)

    install_parser = subparsers.add_parser("install", help="build and install")
    _add_build_args(install_parser)
    install_parser.add_argument("--editable", "-e", action="store_true", help="editable pip install")
    install_parser.add_argument("--cli-extras", action="store_true", help="install cli extras")
    install_parser.add_argument("--visual-extras", action="store_true", help="install visual extras")
    install_parser.set_defaults(func=install_command)

    uninstall_parser = subparsers.add_parser("uninstall", help="uninstall the project")
    uninstall_parser.set_defaults(func=uninstall_command)

    clean_parser = subparsers.add_parser("clean", help="clean build directory and artifacts")
    clean_parser.add_argument("--all", action="store_true",
                              help="also remove cargo target, compiled extensions, and python cache")
    clean_parser.set_defaults(func=clean_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
