# =============================================================================
# test_checkpoint_naming.py
#
# checkpoint filenames encode <zones>.chkpt.<time>.h5 with the time in the
# problem's natural unit (time / time_unit), the decimal rendered as an
# underscore. an arbitrary user time-unit label (tbondi, tdynamical, tjet, ...)
# becomes a filename segment, sanitized to a safe path component. requires the
# built cpu_ext backend; skipped in its absence.
# =============================================================================
import os
import tempfile

import pytest

from simbi.simulation import runner

needs_backend = pytest.mark.skipif(
    runner._load_backend("cpu") is None, reason="rust cpu_ext backend not built"
)


def _run(data_dir, **overrides):
    from simbi_configs.examples.sod import SodProblem

    p = SodProblem(
        end_time=0.05, checkpoint_interval=1.0, data_directory=data_dir, **overrides
    )
    runner.run(p, compute_mode="cpu")
    return sorted(f for f in os.listdir(data_dir) if "chkpt" in f)


@needs_backend
def test_default_names_are_zones_dot_chkpt_dot_time() -> None:
    with tempfile.TemporaryDirectory() as d:
        files = _run(d + "/")
        # 1d resolution 100, IC at t=0 -> "000_000", no unit segment.
        assert "100.chkpt.000_000.h5" in files


@needs_backend
def test_per_axis_resolution_tag_2d() -> None:
    # the name tag is the per-axis resolution joined by 'x', not the zone count.
    import tempfile

    from simbi_configs.examples.kh import KelvinHelmholtz

    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        p = KelvinHelmholtz(
            resolution=(128, 96), end_time=0.02, checkpoint_interval=1.0,
            data_directory=d,
        )
        runner.run(p, compute_mode="cpu")
        files = [f for f in os.listdir(d) if "chkpt" in f]
        assert "128x96.chkpt.000_000.h5" in files


@needs_backend
def test_custom_label_becomes_a_filename_segment() -> None:
    with tempfile.TemporaryDirectory() as d:
        files = _run(d + "/", time_unit=0.05, time_unit_label="tbondi")
        assert any(f.endswith(".tbondi.h5") for f in files)
        assert "100.chkpt.000_000.tbondi.h5" in files


@needs_backend
def test_hostile_label_is_sanitized_in_filename() -> None:
    with tempfile.TemporaryDirectory() as d:
        files = _run(d + "/", time_unit=0.05, time_unit_label="t/ff (free-fall)")
        # slashes/spaces/parens stripped -> a valid single path component.
        assert any(f.endswith(".tfffreefall.h5") for f in files)
        assert all("/" not in os.path.basename(f) for f in files)
