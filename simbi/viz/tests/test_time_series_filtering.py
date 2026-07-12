# =============================================================================
# test_time_series_filtering.py
#
# --tmin/--tmax/--stride select a subset of the checkpoint series. the filter
# must actually run (it was advertised but had no caller, so the flags vanished
# into the plotter **kwargs and every file was plotted).
# =============================================================================

from pathlib import Path

from simbi.viz.cli import filter_files


def _files(*indices: int) -> list[Path]:
    return [Path(f"128.chkpt.{ii:04d}.h5") for ii in indices]


def test_stride_keeps_every_nth_file() -> None:
    files = _files(0, 1, 2, 3, 4, 5)
    kept = filter_files(files, stride=2)
    assert [extract(f) for f in kept] == [0, 2, 4]


def test_tmin_drops_earlier_timesteps() -> None:
    files = _files(0, 5, 10, 15)
    kept = filter_files(files, tmin=10.0)
    assert [extract(f) for f in kept] == [10, 15]


def test_tmax_drops_later_timesteps() -> None:
    files = _files(0, 5, 10, 15)
    kept = filter_files(files, tmax=10.0)
    assert [extract(f) for f in kept] == [0, 5, 10]


def test_range_and_stride_compose() -> None:
    files = _files(0, 2, 4, 6, 8, 10)
    kept = filter_files(files, tmin=2.0, tmax=8.0, stride=2)
    # window {2,4,6,8} then every 2nd -> {2, 6}
    assert [extract(f) for f in kept] == [2, 6]


def test_no_filters_returns_all() -> None:
    files = _files(0, 1, 2)
    assert filter_files(files) == files


def extract(path: Path) -> int:
    # local mirror of the checkpoint index encoded in the fixture names
    return int(path.name.split(".")[2])
