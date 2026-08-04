#!/usr/bin/env python3
# =============================================================================
# checkpoint_state.py
#
# resolves the authoritative checkpoint in one run directory from embedded
# simulation time and iteration metadata. corrupt or incomplete files are
# ignored. status filenames classify completion; modification time is never
# consulted.
#
# usage:
#  python scripts/checkpoint_state.py data/run
# =============================================================================
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass(frozen=True)
class CheckpointState:
    path: Path
    time: float
    iteration: int
    outcome: str


# tie-break precedence when two checkpoints share an evolution point. a run whose last
# scheduled dump lands exactly on the end time writes two files at the SAME time and
# iteration -- the dump, which reads as still running, and the authoritative terminator --
# and ordering on the pair alone resolves that arbitrarily, by directory order. a terminal
# status is never the arbitrary answer: a crash outranks a completion so a failure is never
# reported as success, and both outrank a checkpoint that merely exists.
OUTCOME_PRECEDENCE = {"crashed": 3, "completed": 2, "interrupted": 1, "running": 0}


def checkpoint_outcome(path: Path) -> str:
    name = path.name.lower()
    if ".final" in name:
        return "completed"
    if ".crashed" in name:
        return "crashed"
    if ".interrupted" in name:
        return "interrupted"
    return "running"


def read_checkpoint_state(path: Path) -> CheckpointState | None:
    try:
        with h5py.File(path, "r") as checkpoint:
            metadata = checkpoint["metadata"]
            time = float(metadata.attrs["time"])
            iteration = int(metadata.attrs["iteration"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return CheckpointState(path, time, iteration, checkpoint_outcome(path))


def latest_checkpoint(directory: Path) -> CheckpointState | None:
    states = (
        state
        for path in directory.glob("*chkpt*.h5")
        if (state := read_checkpoint_state(path)) is not None
    )
    return max(
        states,
        key=lambda state: (
            state.time,
            state.iteration,
            OUTCOME_PRECEDENCE[state.outcome],
        ),
        default=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="print OUTCOME and PATH for the latest valid checkpoint"
    )
    parser.add_argument("data_directory", type=Path)
    args = parser.parse_args()
    state = latest_checkpoint(args.data_directory)
    if state is None:
        print("missing")
        return 0
    print(f"{state.outcome}\t{state.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
