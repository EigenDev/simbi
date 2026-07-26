# =============================================================================
# test_checkpoint_state.py
#
# metadata-authoritative cluster restart checkpoint discovery.
# =============================================================================

from pathlib import Path

import h5py

from scripts.checkpoint_state import latest_checkpoint


def write_checkpoint(path: Path, time: float, iteration: int) -> None:
    with h5py.File(path, "w") as checkpoint:
        metadata = checkpoint.create_group("metadata")
        metadata.attrs["time"] = time
        metadata.attrs["iteration"] = iteration


def test_latest_checkpoint_uses_simulation_time_not_mtime(tmp_path: Path) -> None:
    newer_state = tmp_path / "run.chkpt.010.h5"
    older_state = tmp_path / "run.chkpt.999.h5"
    write_checkpoint(newer_state, 10.0, 100)
    write_checkpoint(older_state, 2.0, 20)
    older_state.touch()

    state = latest_checkpoint(tmp_path)

    assert state is not None
    assert state.path == newer_state
    assert state.time == 10.0


def test_corrupt_partial_checkpoint_is_ignored(tmp_path: Path) -> None:
    valid = tmp_path / "run.chkpt.interrupted.h5"
    write_checkpoint(valid, 4.0, 40)
    (tmp_path / "run.chkpt.final.h5").write_bytes(b"incomplete")

    state = latest_checkpoint(tmp_path)

    assert state is not None
    assert state.path == valid
    assert state.outcome == "interrupted"


def test_outcome_belongs_to_latest_state_only(tmp_path: Path) -> None:
    write_checkpoint(tmp_path / "old.chkpt.crashed.h5", 1.0, 10)
    current = tmp_path / "current.chkpt.interrupted.h5"
    write_checkpoint(current, 3.0, 30)

    state = latest_checkpoint(tmp_path)

    assert state is not None
    assert state.path == current
    assert state.outcome == "interrupted"
