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


def test_a_terminator_outranks_a_dump_at_the_same_evolution_point(tmp_path: Path) -> None:
    # a run clamps its last step to land exactly on the end time, so the final scheduled dump
    # and the authoritative terminator carry identical time AND iteration. ranking on that
    # pair alone picks between them by directory order, and picking the dump reports a
    # finished run as still running -- which makes a restart chain resubmit forever, every
    # link exiting cleanly the moment it sees there is nothing left to integrate.
    dump = tmp_path / "run.chkpt.040_000.h5"
    terminator = tmp_path / "run.chkpt.final.h5"
    write_checkpoint(dump, 40.0, 212207)
    write_checkpoint(terminator, 40.0, 212207)

    state = latest_checkpoint(tmp_path)

    assert state is not None
    assert state.outcome == "completed"
    assert state.path == terminator


def test_a_crash_at_the_same_point_is_not_reported_as_a_completion(tmp_path: Path) -> None:
    # a failure masked as success burns the rest of an allocation reproducing it silently.
    write_checkpoint(tmp_path / "run.chkpt.final.h5", 40.0, 212207)
    write_checkpoint(tmp_path / "run.chkpt.crashed.h5", 40.0, 212207)

    state = latest_checkpoint(tmp_path)

    assert state is not None
    assert state.outcome == "crashed"


def test_a_later_dump_still_beats_an_earlier_terminator(tmp_path: Path) -> None:
    # status breaks TIES only. a resumed run whose schedule was extended past a previous
    # end time carries a stale terminator, and treating that as authoritative would refuse
    # to continue a run that genuinely has integrating left to do.
    write_checkpoint(tmp_path / "run.chkpt.final.h5", 40.0, 212207)
    resumed = tmp_path / "run.chkpt.045_000.h5"
    write_checkpoint(resumed, 45.0, 238000)

    state = latest_checkpoint(tmp_path)

    assert state is not None
    assert state.outcome == "running"
    assert state.path == resumed
