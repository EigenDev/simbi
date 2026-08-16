# =============================================================================
# checkpoint_utils.py
#
# general utilities for checkpoint file handling across viz system.
# filters invalid checkpoints, handles globs, sorts by timestep.
#
# usage:
#   from simbi.viz.utility.checkpoint_utils import glob_checkpoints
#   files = glob_checkpoints("data/sim/")  # auto-filters and sorts
# =============================================================================

import re
from pathlib import Path
from typing import Sequence


def filter_checkpoint_files(
    files: Sequence[str | Path], verbose: bool = True
) -> list[Path]:
    """
    filter out interrupted/crashed checkpoints.

    excludes files with 'interrupted' or 'crashed' in filename.
    these represent incomplete simulation states.

    args:
        files: list of checkpoint file paths
        verbose: if True, print exclusion count

    returns:
        filtered list of valid checkpoint files

    example:
        files = Path("data/sim").glob("*.h5")
        valid = filter_checkpoint_files(files)
        # excludes: foo.interrupted.h5, bar.crashed.h5
    """
    valid_files = []
    excluded_count = 0

    for filepath in files:
        path = Path(filepath)
        name_lower = path.name.lower()

        if "interrupted" in name_lower or "crashed" in name_lower:
            excluded_count += 1
            continue

        valid_files.append(path)

    if excluded_count > 0 and verbose:
        print(f"excluded {excluded_count} interrupted/crashed checkpoint(s)")

    return valid_files


def extract_timestep(filename: str | Path) -> float:
    """
    extract timestep number from checkpoint filename for sorting.

    handles various naming conventions:
    - 128.chkpt.0042.h5 -> 42.0
    - 128.chkpt.1_000_000.h5 -> 1000000.0
    - sim_t100.5.h5 -> 100.5
    - checkpoint_001.h5 -> 1.0

    args:
        filename: checkpoint filename

    returns:
        extracted timestep as float (0.0 if not found)

    example:
        files.sort(key=lambda f: extract_timestep(f.name))
    """
    name = Path(filename).name

    # try standard simbi format: chkpt.nnnn.h5 (supports underscore separators)
    match = re.search(r"chkpt\.(\d[\d_]*(?:\.\d[\d_]*)?)", name)
    if match:
        return float(match.group(1).replace("_", ""))

    # try generic: any sequence of digits (possibly with decimal/underscores)
    match = re.search(r"(\d[\d_]*(?:\.\d[\d_]*)?)", name)
    if match:
        return float(match.group(1).replace("_", ""))

    return 0.0


def _checkpoint_sort_key(filename: str | Path) -> tuple[int, float]:
    """
    ordering key for a checkpoint series: a 'final' checkpoint sorts after every
    timestepped one so it lands last, regardless of any digits in its name.

    the first tuple element is the final-marker flag (0 before 1); the second is
    the extracted timestep, which orders multiple finals among themselves.
    """
    is_final = 1 if "final" in Path(filename).name.lower() else 0
    return (is_final, extract_timestep(filename))


def glob_checkpoints(
    path: str | Path, pattern: str = "*.chkpt.*.h5", filter_invalid: bool = True
) -> list[Path]:
    """
    smart checkpoint discovery from directory, file, or glob pattern.

    handles:
    - directory: globs all checkpoints in dir
    - single file: returns as list
    - glob pattern: expands pattern
    - auto-sorts by timestep, with any 'final' checkpoint placed last
    - optionally filters interrupted/crashed

    args:
        path: directory, file, or glob pattern
        pattern: glob pattern to use for directories
        filter_invalid: if True, exclude interrupted/crashed files

    returns:
        sorted list of checkpoint file paths

    example:
        # from directory
        files = glob_checkpoints("data/sim/")

        # from pattern
        files = glob_checkpoints("data/sim/*.h5")

        # from single file
        files = glob_checkpoints("data/sim/checkpoint.h5")
    """
    input_path = Path(path)

    # handle directory
    if input_path.is_dir():
        checkpoint_files = sorted(
            input_path.glob(pattern), key=lambda f: _checkpoint_sort_key(f.name)
        )

    # handle glob pattern (has wildcard)
    elif "*" in str(path):
        checkpoint_files = sorted(
            Path(".").glob(str(path)), key=lambda f: _checkpoint_sort_key(f.name)
        )

    # handle single file
    elif input_path.is_file():
        checkpoint_files = [input_path]

    # path doesn't exist
    else:
        checkpoint_files = []

    # filter invalid if requested
    if filter_invalid and checkpoint_files:
        checkpoint_files = filter_checkpoint_files(
            checkpoint_files, verbose=True
        )

    return checkpoint_files


def validate_checkpoint_series(files: Sequence[Path]) -> tuple[bool, str]:
    """
    validate that checkpoint files form a valid time series.

    checks:
    - at least one file
    - files are sorted by timestep
    - no duplicate timesteps

    args:
        files: list of checkpoint file paths

    returns:
        (is_valid, error_message)

    example:
        files = glob_checkpoints("data/sim/")
        is_valid, error = validate_checkpoint_series(files)
        if not is_valid:
            print(f"Invalid series: {error}")
    """
    if not files:
        return False, "no checkpoint files found"

    if len(files) == 1:
        return True, ""

    # extract timesteps
    timesteps = [extract_timestep(f.name) for f in files]

    # check for duplicates
    if len(set(timesteps)) != len(timesteps):
        return False, "duplicate timesteps detected"

    # check sorting
    if timesteps != sorted(timesteps):
        return False, "files are not sorted by timestep"

    return True, ""


def get_checkpoint_info(filepath: Path) -> dict:
    """
    extract metadata from checkpoint filename.

    returns dict with:
    - timestep: extracted timestep number
    - resolution: if in filename (e.g., 128.chkpt.*.h5)
    - is_interrupted: True if interrupted/crashed
    - is_valid: True if not interrupted/crashed

    args:
        filepath: checkpoint file path

    returns:
        dict with metadata

    example:
        info = get_checkpoint_info(Path("128.chkpt.0042.h5"))
        # {'timestep': 42.0, 'resolution': 128, 'is_interrupted': False, ...}
    """
    name = filepath.name.lower()

    # extract timestep
    timestep = extract_timestep(filepath)

    # check for resolution prefix (e.g., 128.chkpt.*.h5)
    resolution = None
    res_match = re.match(r"(\d+)\.chkpt", name)
    if res_match:
        resolution = int(res_match.group(1))

    # check validity
    is_interrupted = "interrupted" in name or "crashed" in name
    is_valid = not is_interrupted

    return {
        "timestep": timestep,
        "resolution": resolution,
        "is_interrupted": is_interrupted,
        "is_valid": is_valid,
        "filename": filepath.name,
        "path": filepath,
    }


__all__ = [
    "filter_checkpoint_files",
    "extract_timestep",
    "glob_checkpoints",
    "validate_checkpoint_series",
    "get_checkpoint_info",
]
