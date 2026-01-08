#!/usr/bin/env python3
# =============================================================================
# gpu_compiler.py
#
# gpu compilation strategies with error handling and optimization
# supports multiple backends: cuda, hip, sycl with automatic fallbacks
# provides job scheduling and memory management
# =============================================================================

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class GPUBackend(Enum):
    """supported gpu backends"""
    CUDA = "cuda"
    HIP = "hip"
    SYCL = "sycl"


class CompilationStrategy(Enum):
    """compilation strategies for different scenarios"""
    UNIFIED = "unified"      # single file compilation (fastest)
    BATCHED = "batched"      # small batches (memory constrained)
    INDIVIDUAL = "individual" # per-file compilation (debugging)


@dataclass
class CompilerConfig:
    """gpu compiler configuration"""
    backend: GPUBackend
    strategy: CompilationStrategy
    device_arch: Optional[str] = None
    max_jobs: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    debug_mode: bool = False
    enable_profiling: bool = False
    optimization_level: int = 3
    extra_flags: List[str] = None

    def __post_init__(self):
        if self.extra_flags is None:
            self.extra_flags = []


@dataclass
class CompilationResult:
    """compilation result with timing and diagnostics"""
    success: bool
    duration: float
    memory_peak_mb: Optional[float] = None
    output_files: List[Path] = None
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.warnings is None:
            self.warnings = []


class GPUCompiler:
    """gpu compiler with multiple strategies and error handling"""

    def __init__(self, config: CompilerConfig):
        self.config = config
        self.temp_dir: Optional[Path] = None
        self._validate_environment()

    def _validate_environment(self):
        """validate gpu compilation environment"""
        if self.config.backend == GPUBackend.CUDA:
            if not self._find_cuda_compiler():
                raise RuntimeError("cuda compiler not found")
        elif self.config.backend == GPUBackend.HIP:
            if not self._find_hip_compiler():
                raise RuntimeError("hip compiler not found")
        elif self.config.backend == GPUBackend.SYCL:
            if not self._find_sycl_compiler():
                raise RuntimeError("sycl compiler not found")

    def _find_cuda_compiler(self) -> bool:
        """find cuda compiler and validate installation"""
        nvcc_path = shutil.which('nvcc')
        if not nvcc_path:
            return False

        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _find_hip_compiler(self) -> bool:
        """find hip compiler and validate installation"""
        hipcc_path = shutil.which('hipcc')
        if not hipcc_path:
            return False

        try:
            result = subprocess.run(
                ['hipcc', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _find_sycl_compiler(self) -> bool:
        """find sycl compiler and validate installation"""
        # check for intel dpcpp or other sycl compilers
        compilers = ['dpcpp', 'icpx', 'clang++']
        for compiler in compilers:
            if shutil.which(compiler):
                try:
                    result = subprocess.run(
                        [compiler, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0 and ('sycl' in result.stdout.lower() or 'dpcpp' in result.stdout.lower()):
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
        return False

    def _get_optimal_job_count(self) -> int:
        """determine optimal parallel job count based on memory and cpu"""
        if self.config.max_jobs:
            return self.config.max_jobs

        cpu_count = os.cpu_count() or 4

        # get available memory
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemAvailable:'):
                        available_mb = int(line.split()[1]) // 1024
                        break
                else:
                    available_mb = 4096  # fallback
        except (FileNotFoundError, ValueError):
            available_mb = 4096  # fallback for non-linux systems

        # estimate memory usage per job
        memory_per_job = self.config.memory_limit_mb or 2048
        max_jobs_by_memory = max(1, available_mb // memory_per_job)

        # conservative approach for gpu compilation
        if self.config.strategy == CompilationStrategy.UNIFIED:
            return 1  # single job for unified compilation
        elif self.config.strategy == CompilationStrategy.BATCHED:
            return min(cpu_count // 2, max_jobs_by_memory)
        else:  # individual
            return min(cpu_count // 3, max_jobs_by_memory)

    def _get_base_compiler_flags(self) -> List[str]:
        """get base compiler flags for backend"""
        if self.config.backend == GPUBackend.CUDA:
            flags = [
                '-x', 'cu',
                '--extended-lambda',
                '--expt-relaxed-constexpr',
                '-std=c++20',
                '--compiler-options', '-fPIC,-fopenmp',
                f'-O{self.config.optimization_level}',
            ]

            if self.config.device_arch:
                flags.extend(['-arch', self.config.device_arch])

            if self.config.debug_mode:
                flags.extend(['-g', '-G'])
            else:
                flags.extend(['--compiler-options', '-DNDEBUG'])

            if self.config.enable_profiling:
                flags.append('-lineinfo')

            # memory optimization flags
            if self.config.memory_limit_mb:
                flags.append(f'--memory-limit-mb={self.config.memory_limit_mb}')

        elif self.config.backend == GPUBackend.HIP:
            flags = [
                '-x', 'hip',
                '-std=c++20',
                '-fPIC',
                '-fopenmp',
                f'-O{self.config.optimization_level}',
            ]

            if self.config.device_arch:
                flags.extend(['--offload-arch', self.config.device_arch])

            if self.config.debug_mode:
                flags.append('-g')
            else:
                flags.append('-DNDEBUG')

        elif self.config.backend == GPUBackend.SYCL:
            flags = [
                '-fsycl',
                '-std=c++20',
                '-fPIC',
                '-fopenmp',
                f'-O{self.config.optimization_level}',
            ]

            if self.config.debug_mode:
                flags.append('-g')
            else:
                flags.append('-DNDEBUG')

        # add extra user flags
        flags.extend(self.config.extra_flags)

        return flags

    def _run_compiler_command(
        self,
        cmd: List[str],
        input_files: List[Path],
        output_file: Path,
        timeout: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """run compiler command with timing and error handling"""
        start_time = time.time()

        try:
            # set up environment
            env = os.environ.copy()
            if self.config.backend == GPUBackend.CUDA and self.config.memory_limit_mb:
                env['CUDA_CACHE_DISABLE'] = '1'
                env['CUDA_CACHE_MAXSIZE'] = str(self.config.memory_limit_mb * 1024 * 1024)

            # run compilation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                return True, result.stdout, duration
            else:
                error_msg = f"compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
                return False, error_msg, duration

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return False, f"compilation timeout after {duration:.1f}s", duration
        except Exception as e:
            duration = time.time() - start_time
            return False, f"compilation error: {str(e)}", duration

    def compile_unified(
        self,
        source_files: List[Path],
        output_file: Path,
        include_dirs: List[Path] = None,
        defines: List[str] = None
    ) -> CompilationResult:
        """compile using unified strategy (single compilation unit)"""
        if not source_files:
            return CompilationResult(
                success=False,
                duration=0.0,
                error_message="no source files provided"
            )

        # create temporary unified source file
        self.temp_dir = Path(tempfile.mkdtemp(prefix='gpu_compile_'))
        unified_source = self.temp_dir / 'unified.cu'

        try:
            # generate unified source
            from .generate_gpu_unified import generate_unified_source

            source_file_strs = [str(f) for f in source_files]
            success = generate_unified_source(
                source_files=source_file_strs,
                output_file=str(unified_source),
                base_dir=str(source_files[0].parent.parent),
                extra_defines=defines or []
            )

            if not success:
                return CompilationResult(
                    success=False,
                    duration=0.0,
                    error_message="failed to generate unified source"
                )

            # compile unified source
            compiler = 'nvcc' if self.config.backend == GPUBackend.CUDA else 'hipcc'
            cmd = [compiler] + self._get_base_compiler_flags()

            if include_dirs:
                for inc_dir in include_dirs:
                    cmd.extend(['-I', str(inc_dir)])

            if defines:
                for define in defines:
                    cmd.extend(['-D', define])

            cmd.extend(['-c', str(unified_source), '-o', str(output_file)])

            success, output, duration = self._run_compiler_command(
                cmd=cmd,
                input_files=[unified_source],
                output_file=output_file
            )

            return CompilationResult(
                success=success,
                duration=duration,
                output_files=[output_file] if success else [],
                error_message=output if not success else None
            )

        finally:
            # cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None

    def compile_batched(
        self,
        source_files: List[Path],
        output_dir: Path,
        include_dirs: List[Path] = None,
        defines: List[str] = None,
        batch_size: int = 4
    ) -> CompilationResult:
        """compile using batched strategy (small groups)"""
        if not source_files:
            return CompilationResult(
                success=False,
                duration=0.0,
                error_message="no source files provided"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        total_duration = 0.0
        output_files = []
        all_warnings = []

        # split into batches
        batches = [
            source_files[i:i + batch_size]
            for i in range(0, len(source_files), batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            batch_result = self.compile_unified(
                source_files=batch,
                output_file=output_dir / f'batch_{batch_idx}.o',
                include_dirs=include_dirs,
                defines=defines
            )

            total_duration += batch_result.duration

            if not batch_result.success:
                return CompilationResult(
                    success=False,
                    duration=total_duration,
                    error_message=f"batch {batch_idx} failed: {batch_result.error_message}"
                )

            output_files.extend(batch_result.output_files)
            all_warnings.extend(batch_result.warnings)

        return CompilationResult(
            success=True,
            duration=total_duration,
            output_files=output_files,
            warnings=all_warnings
        )

    def compile_individual(
        self,
        source_files: List[Path],
        output_dir: Path,
        include_dirs: List[Path] = None,
        defines: List[str] = None
    ) -> CompilationResult:
        """compile using individual strategy (one file at a time)"""
        if not source_files:
            return CompilationResult(
                success=False,
                duration=0.0,
                error_message="no source files provided"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        total_duration = 0.0
        output_files = []
        all_warnings = []

        compiler = 'nvcc' if self.config.backend == GPUBackend.CUDA else 'hipcc'
        base_flags = self._get_base_compiler_flags()

        for src_file in source_files:
            output_file = output_dir / f'{src_file.stem}.o'

            cmd = [compiler] + base_flags

            if include_dirs:
                for inc_dir in include_dirs:
                    cmd.extend(['-I', str(inc_dir)])

            if defines:
                for define in defines:
                    cmd.extend(['-D', define])

            cmd.extend(['-c', str(src_file), '-o', str(output_file)])

            success, output, duration = self._run_compiler_command(
                cmd=cmd,
                input_files=[src_file],
                output_file=output_file
            )

            total_duration += duration

            if not success:
                return CompilationResult(
                    success=False,
                    duration=total_duration,
                    error_message=f"failed to compile {src_file}: {output}"
                )

            output_files.append(output_file)

        return CompilationResult(
            success=True,
            duration=total_duration,
            output_files=output_files,
            warnings=all_warnings
        )

    def compile(
        self,
        source_files: List[Path],
        output_path: Path,
        include_dirs: List[Path] = None,
        defines: List[str] = None
    ) -> CompilationResult:
        """compile using configured strategy"""
        if self.config.strategy == CompilationStrategy.UNIFIED:
            return self.compile_unified(
                source_files=source_files,
                output_file=output_path,
                include_dirs=include_dirs,
                defines=defines
            )
        elif self.config.strategy == CompilationStrategy.BATCHED:
            return self.compile_batched(
                source_files=source_files,
                output_dir=output_path,
                include_dirs=include_dirs,
                defines=defines
            )
        elif self.config.strategy == CompilationStrategy.INDIVIDUAL:
            return self.compile_individual(
                source_files=source_files,
                output_dir=output_path,
                include_dirs=include_dirs,
                defines=defines
            )
        else:
            return CompilationResult(
                success=False,
                duration=0.0,
                error_message=f"unknown compilation strategy: {self.config.strategy}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


def detect_gpu_backend() -> Optional[GPUBackend]:
    """automatically detect available gpu backend"""
    # prefer cuda if available
    if shutil.which('nvcc'):
        return GPUBackend.CUDA

    # fallback to hip
    if shutil.which('hipcc'):
        return GPUBackend.HIP

    # fallback to sycl
    compilers = ['dpcpp', 'icpx']
    for compiler in compilers:
        if shutil.which(compiler):
            return GPUBackend.SYCL

    return None


def get_recommended_strategy(
    num_sources: int,
    available_memory_mb: Optional[int] = None
) -> CompilationStrategy:
    """recommend compilation strategy based on constraints"""
    if available_memory_mb and available_memory_mb < 8192:
        # low memory system
        if num_sources <= 5:
            return CompilationStrategy.UNIFIED
        else:
            return CompilationStrategy.BATCHED
    else:
        # normal memory system
        if num_sources <= 15:
            return CompilationStrategy.UNIFIED
        else:
            return CompilationStrategy.BATCHED


def main():
    """command line interface for gpu compiler"""
    import argparse

    parser = argparse.ArgumentParser(description="gpu compiler with multiple strategies")
    parser.add_argument("source_files", nargs="+", help="source files to compile")
    parser.add_argument("-o", "--output", required=True, help="output file or directory")
    parser.add_argument("--backend", choices=[b.value for b in GPUBackend], help="gpu backend")
    parser.add_argument("--strategy", choices=[s.value for s in CompilationStrategy], help="compilation strategy")
    parser.add_argument("--device-arch", help="target device architecture")
    parser.add_argument("--max-jobs", type=int, help="maximum parallel jobs")
    parser.add_argument("--memory-limit", type=int, help="memory limit in MB")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--profile", action="store_true", help="enable profiling")
    parser.add_argument("-I", "--include", action="append", dest="includes", help="include directories")
    parser.add_argument("-D", "--define", action="append", dest="defines", help="preprocessor defines")
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    args = parser.parse_args()

    # detect backend if not specified
    backend = GPUBackend(args.backend) if args.backend else detect_gpu_backend()
    if not backend:
        print("error: no gpu backend available", file=sys.stderr)
        return 1

    # recommend strategy if not specified
    source_files = [Path(f) for f in args.source_files]
    strategy = CompilationStrategy(args.strategy) if args.strategy else get_recommended_strategy(len(source_files))

    # create compiler config
    config = CompilerConfig(
        backend=backend,
        strategy=strategy,
        device_arch=args.device_arch,
        max_jobs=args.max_jobs,
        memory_limit_mb=args.memory_limit,
        debug_mode=args.debug,
        enable_profiling=args.profile
    )

    if args.verbose:
        print(f"backend: {backend.value}")
        print(f"strategy: {strategy.value}")
        print(f"sources: {len(source_files)}")

    # compile
    try:
        with GPUCompiler(config) as compiler:
            result = compiler.compile(
                source_files=source_files,
                output_path=Path(args.output),
                include_dirs=[Path(inc) for inc in (args.includes or [])],
                defines=args.defines
            )

            if result.success:
                print(f"compilation successful in {result.duration:.1f}s")
                if args.verbose and result.output_files:
                    print(f"output files: {[str(f) for f in result.output_files]}")
                return 0
            else:
                print(f"compilation failed after {result.duration:.1f}s", file=sys.stderr)
                print(result.error_message, file=sys.stderr)
                return 1

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
