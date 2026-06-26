#!/usr/bin/env python3
# =============================================================================
# dev.py
#
# simbi build/install wrapper around MATURIN (which drives cargo directly).
# the rust backend (src/crates/symbi-py) builds the pyo3 extension installed
# as simbi/libs/cpu_ext; the unchanged `simbi.libs.cpu_ext` import loads it.
# gpu is a cargo FEATURE: `--gpu` -> `--features cuda` (kernels JIT via NVRTC at
# runtime; build.rs links libcuda/libnvrtc — no nvcc/meson needed).
# usage:
#  ./dev.py install            # maturin develop --release (editable build + install)
#  ./dev.py build              # maturin build --release (wheel in src/target/wheels)
#  (no-uv path: `pip install -e .` builds editable via the maturin backend directly)
#  ./dev.py build --gpu        # rust gpu build (cargo 'cuda' feature)
#  ./dev.py clean [--all]      # remove extensions; --all also runs cargo clean
# =============================================================================

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path("src")


def maturin() -> list:
    """resolve maturin: prefer the venv's bin, else PATH, else `python -m maturin`."""
    sibling = Path(sys.executable).parent / "maturin"
    if sibling.exists():
        return [str(sibling)]
    found = shutil.which("maturin")
    if found:
        return [found]
    return [sys.executable, "-m", "maturin"]


def venv_env() -> dict:
    """ensure child maturin/cargo resolve THIS interpreter's venv (maturin installs
    the extension into VIRTUAL_ENV). a no-op outside a venv."""
    env = dict(os.environ)
    if sys.prefix != sys.base_prefix:
        bin_dir = str(Path(sys.executable).parent)
        env["VIRTUAL_ENV"] = sys.prefix
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def run(cmd, **kwargs) -> None:
    verbose = kwargs.pop("verbose", False)
    if verbose:
        print("executing:", " ".join(map(str, cmd)))
    start = time.time()
    try:
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        print(
            f"error: command failed after {time.time() - start:.1f}s: "
            f"{' '.join(map(str, cmd))}",
            file=sys.stderr,
        )
        if "cargo" in str(cmd) or "maturin" in str(cmd):
            print(
                "hint: ensure the rust toolchain (https://rustup.rs) and maturin are installed "
                "(`uv sync`, or `pip install maturin`); or skip dev.py entirely with "
                "`pip install -e .` (builds via the maturin backend). then `./dev.py clean --all`.",
                file=sys.stderr,
            )
        sys.exit(exc.returncode)


def _features(args) -> str:
    feats = [f.strip() for f in (args.features or "").split(",") if f.strip()]
    if getattr(args, "gpu", False) and "cuda" not in feats:
        feats.append("cuda")
    return ",".join(feats)


def _common(args) -> list:
    cmd: list = []
    feats = _features(args)
    if feats:
        cmd += ["--features", feats]
        print(f"rust features: {feats}")
    # the cuda build defines the `gpu_ext` pymodule (not `cpu_ext`); override
    # maturin's pyproject module-name so the dylib installs as `simbi.libs.gpu_ext`
    # and coexists with the cpu backend instead of overwriting it.
    if "cuda" in feats.split(","):
        cmd += ["--config", 'tool.maturin.module-name="simbi.libs.gpu_ext"']
    if args.verbose:
        cmd.append("-v")
    return cmd


def _require_cargo() -> None:
    if not shutil.which("cargo"):
        print(
            "error: cargo not found — install the rust toolchain (https://rustup.rs)",
            file=sys.stderr,
        )
        sys.exit(1)


def build_command(args) -> None:
    """maturin build --release -> a wheel under src/target/wheels."""
    _require_cargo()
    print("building wheel (maturin -> cargo)...")
    start = time.time()
    run(
        [*maturin(), "build", "--release", *_common(args)],
        env=venv_env(),
        verbose=args.verbose,
    )
    print(
        f"build completed in {time.time() - start:.1f}s (wheel in src/target/wheels/)"
    )


def install_command(args) -> None:
    """maturin develop --release: build the extension into simbi/libs + editable install."""
    _require_cargo()
    cmd = [*maturin(), "develop", "--release", *_common(args)]
    extras = []
    if args.cli_extras:
        extras.append("cli")
    if args.visual_extras:
        extras.append("visual")
    if extras:
        cmd += ["--extras", ",".join(extras)]
    print("building + installing (maturin develop, editable)...")
    start = time.time()
    run(cmd, env=venv_env(), verbose=args.verbose)
    print(f"done in {time.time() - start:.1f}s")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import simbi; print('verified', simbi.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(
            "validation:",
            result.stdout.strip() if result.returncode == 0 else "FAILED",
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("warning: validation timeout", file=sys.stderr)


def _remove_extensions() -> int:
    lib_dir = Path("simbi/libs")
    removed = 0
    if lib_dir.exists():
        for ext in (
            list(lib_dir.glob("*.so"))
            + list(lib_dir.glob("*.dylib"))
            + list(lib_dir.glob("*.pyd"))
        ):
            ext.unlink()
            print(f"removed {ext}")
            removed += 1
    return removed


def uninstall_command(args) -> None:
    if shutil.which("uv"):
        run(
            ["uv", "pip", "uninstall", "simbi"],
            env=venv_env(),
            verbose=args.verbose,
        )
    else:
        run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "simbi"],
            verbose=args.verbose,
        )
    _remove_extensions()


def clean_command(args) -> None:
    removed = _remove_extensions()
    if args.all:
        if (SRC / "Cargo.toml").exists() and shutil.which("cargo"):
            run(
                [
                    "cargo",
                    "clean",
                    "--manifest-path",
                    str(SRC / "Cargo.toml"),
                ],
                verbose=args.verbose,
            )
            print("ran cargo clean")
            removed += 1
        for cache in Path(".").rglob("__pycache__"):
            if cache.is_dir() and ".venv" not in cache.parts:
                shutil.rmtree(cache)
                removed += 1
    print(
        f"cleanup complete: {removed} items removed"
        if removed
        else "nothing to clean"
    )


def _add_build_args(p) -> None:
    p.add_argument(
        "--features",
        default="",
        help="comma-separated cargo features (e.g. cuda)",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="gpu build -> cargo 'cuda' feature (NVRTC JIT)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="simbi build tool (dev -> maturin -> cargo)"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    sub = parser.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("build", help="build a wheel")
    _add_build_args(bp)
    bp.set_defaults(func=build_command)

    ip = sub.add_parser(
        "install", help="build + editable install (maturin develop)"
    )
    _add_build_args(ip)
    ip.add_argument(
        "--editable",
        "-e",
        action="store_true",
        help="(default; maturin develop is editable)",
    )
    ip.add_argument(
        "--cli-extras", action="store_true", help="install cli extras"
    )
    ip.add_argument(
        "--visual-extras", action="store_true", help="install visual extras"
    )
    ip.set_defaults(func=install_command)

    up = sub.add_parser("uninstall", help="uninstall the project")
    up.set_defaults(func=uninstall_command)

    cp = sub.add_parser(
        "clean", help="remove compiled extensions; --all also cargo clean"
    )
    cp.add_argument(
        "--all",
        action="store_true",
        help="also run cargo clean + drop python cache",
    )
    cp.set_defaults(func=clean_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
