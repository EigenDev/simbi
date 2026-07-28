#!/usr/bin/env python3
# =============================================================================
# dev.py
#
# simbi build/install wrapper around MATURIN (which drives cargo directly).
# the rust backend (src/crates/symbi-py) builds the pyo3 extension installed
# as simbi/libs/cpu_ext; the unchanged `simbi.libs.cpu_ext` import loads it.
# gpu is a cargo FEATURE: `--cuda` -> nvidia (`--features cuda`, NVRTC JIT,
# links libcuda/libnvrtc); `--hip` -> amd (`--features hip`, hipRTC JIT, links
# libamdhip64/libhiprtc under ROCM_PATH). no nvcc/hipcc/meson needed. both produce
# the gpu_ext extension, which coexists with cpu_ext.
# usage:
#  ./dev.py install            # maturin develop --release (editable build + install)
#  ./dev.py build              # maturin build --release (wheel in src/target/wheels)
#  (no-uv path: `pip install -e .` builds editable via the maturin backend directly)
#  ./dev.py build --cuda       # nvidia gpu build (cargo 'cuda' feature)
#  ./dev.py install --hip      # amd gpu build (cargo 'hip' feature, ROCm)
#  ./dev.py clean [--all]      # cargo clean; --all also removes python caches
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


def target_python() -> str:
    """the interpreter maturin installs the extension into.

    maturin targets VIRTUAL_ENV, which is not necessarily the interpreter running this
    script — invoking `./dev.py` picks whatever the shebang resolves to. validating the
    install with the wrong one reports a working build as a failure (typically a missing
    h5py, which only the project venv carries).
    """
    venv = os.environ.get("VIRTUAL_ENV") or (
        sys.prefix if sys.prefix != sys.base_prefix else None
    )
    if venv:
        for candidate in (Path(venv) / "bin" / "python", Path(venv) / "Scripts" / "python.exe"):
            if candidate.exists():
                return str(candidate)
    return sys.executable


def _fast_linker_flag():
    """the fastest link-time-only linker installed, as a rustc `-C link-arg`, or None. mold
    and lld cut the final cdylib link (a heavily-monomorphized artifact) from minutes to
    seconds with byte-identical output — the linker never touches codegen, so there is no
    runtime-quality tradeoff. None when only the system default (bfd/gold) is present keeps
    the build portable. needs a `cc` that understands `-fuse-ld` (gcc >= 12 or clang)."""
    for binary, fuse in (("mold", "mold"), ("ld.lld", "lld"), ("lld", "lld")):
        if shutil.which(binary):
            return f"-Clink-arg=-fuse-ld={fuse}"
    return None


def _build_jobs() -> int:
    """parallel rustc jobs, capped so a wide-but-RAM-poor machine is not OOM-killed: the
    monomorphized crates here peak near 4 GB per rustc, so bound jobs by total_ram / 4 as
    well as by core count. cargo otherwise defaults its job count to the cores alone."""
    cores = os.cpu_count() or 1
    try:
        total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        ram_cap = max(1, int(total_gb // 4))
    except (ValueError, OSError, AttributeError):
        ram_cap = cores
    return max(1, min(cores, ram_cap))


_TUNING_ANNOUNCED = False


def venv_env() -> dict:
    """ensure child maturin/cargo resolve THIS interpreter's venv (maturin installs the
    extension into VIRTUAL_ENV; a no-op outside a venv), plus opportunistic build-speed
    tuning: a fast linker when one is installed and a RAM-aware job cap. both degrade to the
    stock behavior when the tool / headroom is absent, and defer to any RUSTFLAGS or
    CARGO_BUILD_JOBS the caller already set. set SIMBI_NO_BUILD_TUNING=1 to opt out."""
    global _TUNING_ANNOUNCED
    env = dict(os.environ)
    if sys.prefix != sys.base_prefix:
        bin_dir = str(Path(sys.executable).parent)
        env["VIRTUAL_ENV"] = sys.prefix
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

    if env.get("SIMBI_NO_BUILD_TUNING") == "1":
        return env

    notes = []
    linker = _fast_linker_flag()
    rustflags = env.get("RUSTFLAGS", "")
    if linker is not None and "fuse-ld" not in rustflags:
        env["RUSTFLAGS"] = f"{rustflags} {linker}".strip()
        notes.append(f"linker={linker.rsplit('=', 1)[1]}")
    if "CARGO_BUILD_JOBS" not in env:
        jobs = _build_jobs()
        env["CARGO_BUILD_JOBS"] = str(jobs)
        notes.append(f"jobs={jobs}/{os.cpu_count() or 1} cores")
    if notes and not _TUNING_ANNOUNCED:
        print(f"build tuning: {', '.join(notes)} (SIMBI_NO_BUILD_TUNING=1 to disable)")
        _TUNING_ANNOUNCED = True
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
                "`pip install -e .` (builds via the maturin backend). then `./dev.py clean`.",
                file=sys.stderr,
            )
        sys.exit(exc.returncode)


def _features(args) -> str:
    feats = [f.strip() for f in (args.features or "").split(",") if f.strip()]
    # `--cuda` -> the nvidia backend; `--hip` -> the amd backend.
    # mutually exclusive: the rust crate compile_error!s if both are set.
    if getattr(args, "cuda", False) and "cuda" not in feats:
        feats.append("cuda")
    if getattr(args, "hip", False) and "hip" not in feats:
        feats.append("hip")
    return ",".join(feats)


def _common(args) -> list:
    cmd: list = []
    feats = _features(args)
    if feats:
        cmd += ["--features", feats]
        print(f"rust features: {feats}")
    if args.verbose:
        cmd.append("-v")
    return cmd


def _is_gpu_build(args) -> bool:
    # any gpu backend (cuda or hip) produces the gpu_ext extension.
    feats = set(_features(args).split(","))
    return bool(feats & {"cuda", "hip"})


def _finalize_gpu_ext() -> None:
    """rename the cuda extension `cpu_ext` -> `gpu_ext` so the file name matches
    its `PyInit_gpu_ext` symbol.

    maturin derives BOTH the dylib name and the expected init symbol from the
    crate's `[lib] name` (`cpu_ext`); it ignores `module-name` for the symbol. the
    cuda build defines the pymodule as `gpu_ext` (see symbi-py/src/lib.rs), so the
    installed `cpu_ext.<suffix>.so` actually exports `PyInit_gpu_ext`. renaming the
    file to `gpu_ext.<suffix>.so` makes name and symbol agree, lets the cpu and gpu
    backends coexist, and matches the `simbi.libs.gpu_ext` import in runner.py.
    """
    lib_dir = Path("simbi/libs")
    renamed = 0
    for src in lib_dir.glob("cpu_ext.*.so"):
        dst = src.with_name(src.name.replace("cpu_ext", "gpu_ext", 1))
        src.replace(dst)
        print(f"gpu build: renamed {src.name} -> {dst.name}")
        renamed += 1
    if renamed == 0:
        print("warning: no cpu_ext dylib found to rename to gpu_ext", file=sys.stderr)


def _stash_cpu_ext() -> list:
    """move any existing cpu_ext dylib aside before a cuda build. the cuda build
    writes to the cpu_ext filename (the crate `[lib] name`) before `_finalize_gpu_ext`
    renames it to gpu_ext, so without this it would clobber an existing cpu backend.
    returns a list of (stashed_path, original_path) to restore afterwards."""
    lib_dir = Path("simbi/libs")
    stashed = []
    for src in lib_dir.glob("cpu_ext.*.so"):
        bak = src.with_name(src.name + ".cpubak")
        src.replace(bak)
        stashed.append((bak, src))
    return stashed


def _restore_cpu_ext(stashed: list) -> None:
    """put the stashed cpu_ext dylib back beside the freshly-built gpu_ext so the two
    backends coexist. the cuda build's cpu_ext was renamed to gpu_ext, so the cpu_ext
    filename is free again."""
    for bak, dst in stashed:
        if dst.exists():
            # a cpu_ext already exists (the rename did not run, e.g., build failed before
            # finalize); keep what is there and drop the stash.
            bak.unlink()
        else:
            bak.replace(dst)
            print(f"gpu build: preserved existing {dst.name} beside gpu_ext")


def _require_cargo() -> None:
    if not shutil.which("cargo"):
        print(
            "error: cargo not found — install the rust toolchain (https://rustup.rs)",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_env(args) -> dict:
    """the child build environment: `venv_env` plus, under `--lean`, LTO disabled. the final
    symbi-py cdylib's thin-LTO codegen is one rustc process whose peak memory exceeds a typical
    login-node cgroup cap (the SIGKILL). GPU hot loops are JIT-compiled kernels (nvrtc / hiprtc),
    NOT this cdylib, so dropping cdylib LTO costs a GPU run ~nothing while slashing build memory;
    a memory-rich node can omit `--lean` and keep LTO. defers to a caller-set override."""
    env = venv_env()
    if getattr(args, "lean", False):
        env.setdefault("CARGO_PROFILE_RELEASE_LTO", "false")
        print("lean build: LTO disabled (lower peak memory; negligible GPU-run cost)")
    jobs = getattr(args, "jobs", None)
    if jobs is not None:
        # explicit override of the auto-cap: a shared login node kills a many-core compile, so
        # `--jobs 4` keeps rustc under the policer's CPU threshold (slower, but it finishes).
        env["CARGO_BUILD_JOBS"] = str(jobs)
        print(f"build jobs: {jobs} (explicit --jobs)")
    return env


def build_command(args) -> None:
    """maturin build --release -> a wheel under src/target/wheels."""
    _require_cargo()
    print("building wheel (maturin -> cargo)...")
    start = time.time()
    run(
        [*maturin(), "build", "--release", *_common(args)],
        env=_build_env(args),
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
    gpu = _is_gpu_build(args)
    # `--with-cpu` on a gpu build first compiles the plain cpu_ext (no gpu feature), so both
    # backends coexist after one command. the fresh cpu_ext.so is picked up by the stash below
    # and restored beside gpu_ext. no effect on a cpu-only build (cpu_ext is already the target).
    # note: this recompiles the shared `symbi` crate under a second feature-set — full build cost,
    # not incremental — so it is opt-in.
    if gpu and getattr(args, "with_cpu", False):
        print("companion cpu build (--with-cpu): building cpu_ext first...")
        cpu_cmd = [*maturin(), "develop", "--release"]
        if args.verbose:
            cpu_cmd.append("-v")
        run(cpu_cmd, env=_build_env(args), verbose=args.verbose)
    # the cuda build writes to the cpu_ext filename before it is renamed to gpu_ext,
    # so stash an existing cpu_ext first and restore it after. this lets the cpu and
    # gpu extensions live side by side.
    stashed = _stash_cpu_ext() if gpu else []
    try:
        run(cmd, env=_build_env(args), verbose=args.verbose)
        if gpu:
            _finalize_gpu_ext()
    finally:
        if gpu:
            _restore_cpu_ext(stashed)
    print(f"done in {time.time() - start:.1f}s")
    # validate with the interpreter maturin installed INTO, not the one running this
    # script: they differ whenever dev.py is invoked directly rather than through the
    # venv, and importing from the wrong one reports a good build as a failure.
    python = target_python()
    try:
        result = subprocess.run(
            [
                python,
                "-c",
                "import simbi; print('verified', simbi.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("validation:", result.stdout.strip())
        else:
            print(f"validation: FAILED (under {python})")
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
    _require_cargo()
    run(
        [
            "cargo",
            "clean",
            "--manifest-path",
            str(SRC / "Cargo.toml"),
        ],
        verbose=args.verbose,
    )
    print("removed rust build artifacts; installed extensions preserved")

    removed = 1
    if args.all:
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
        help="comma-separated cargo features (e.g., cuda)",
    )
    p.add_argument(
        "--lean",
        action="store_true",
        help="disable release LTO to cut peak build memory + time. costs a GPU run ~nothing "
        "(kernels are JIT'd). pair with --jobs on a policed login node",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="cap parallel rustc jobs (sets CARGO_BUILD_JOBS). shared login nodes (e.g. "
        "Princeton Della) KILL compiles that use many cores -- build there with a few (e.g. "
        "--jobs 4 --lean), or omit for the RAM/core auto-cap on a machine you own",
    )
    p.add_argument(
        "--cuda",
        action="store_true",
        help="nvidia gpu build -> cargo 'cuda' feature (NVRTC JIT)",
    )
    p.add_argument(
        "--hip",
        action="store_true",
        help="amd gpu build -> cargo 'hip' feature (hipRTC JIT, ROCm). set ROCM_PATH / "
        "SYMBI_HIP_ARCH if rocm is not at /opt/rocm or arch auto-detect is insufficient",
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
    ip.add_argument(
        "--with-cpu",
        action="store_true",
        help="on a gpu build, also build cpu_ext so both backends coexist (recompiles the "
        "shared crate under a second feature-set -> full build cost; no-op without --cuda/--hip)",
    )
    ip.set_defaults(func=install_command)

    up = sub.add_parser("uninstall", help="uninstall the project")
    up.set_defaults(func=uninstall_command)

    cp = sub.add_parser(
        "clean", help="remove cargo build artifacts while preserving installed extensions"
    )
    cp.add_argument(
        "--all",
        action="store_true",
        help="also remove repository python caches",
    )
    cp.set_defaults(func=clean_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
