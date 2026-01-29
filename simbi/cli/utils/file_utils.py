# =============================================================================
# file_utils.py
#
# shared file handling utilities for CLI commands.
# provides glob_files action for accepting files or directories.
# =============================================================================
import argparse
from pathlib import Path


def glob_checkpoints(
    directory: str | Path, filter_invalid: bool = True
) -> list[Path]:
    """
    find all checkpoint files in directory.

    args:
        directory: path to search
        filter_invalid: skip files that don't match checkpoint pattern

    returns:
        sorted list of checkpoint paths
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")

    # common checkpoint patterns
    patterns = ["checkpoint_*.h5", "chkpt_*.h5", "*.chk", "*.h5"]

    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))

    # sort by name (typically includes timestep)
    files = sorted(set(files))

    if filter_invalid:
        # filter out non-checkpoint files (e.g., output.h5, events.h5)
        files = [
            f
            for f in files
            if any(p in f.name for p in ["checkpoint", "chkpt", ".chk"])
        ]

    return files


class glob_files(argparse.Action):
    """
    argparse action that accepts files or directories.
    directories are expanded to checkpoint files.

    usage:
        parser.add_argument("files", nargs="+", action=glob_files)

    example:
        # explicit files
        simbi afterglow generate checkpoint_0001.h5 checkpoint_0002.h5

        # directory (auto-glob)
        simbi afterglow generate /path/to/checkpoints/

        # mixed
        simbi afterglow generate /path/to/dir1/ /path/to/checkpoint_0050.h5
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str | None = None,
    ) -> None:
        files: list[Path] = []

        try:
            if values:
                for value in values:
                    path = Path(value)

                    if path.is_dir():
                        # expand directory to checkpoint files
                        dir_files = glob_checkpoints(path, filter_invalid=True)
                        if not dir_files:
                            parser.error(
                                f"no checkpoint files found in directory: {path}"
                            )
                        files.extend(dir_files)
                    elif path.is_file():
                        # explicit file
                        files.append(path)
                    else:
                        parser.error(f"path does not exist: {path}")

        except (ValueError, OSError) as ex:
            parser.error(f"file handling error: {ex}")

        # remove duplicates, maintain order
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        setattr(namespace, self.dest, unique_files)
