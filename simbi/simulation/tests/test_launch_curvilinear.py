# =============================================================================
# test_launch_curvilinear.py
#
# the curvilinear charts through `simbi launch`. every tile carries the whole
# grid's origin, spacing and maps and is placed by the index of its first
# cell, so its cells sit bitwise where the single grid puts them. the
# spherical pulse on a log-radial shell spanning both poles, and the
# magnetized rotor on the (r, phi) disk plane with a periodic azimuth, each
# equal the single-grid run on four workers, fresh and with the correction
# fired against a cut; the spherical run also resumes onto three unevenly
# owned tiles, so the log map is rebuilt from the problem on restart.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_curvilinear.py
# =============================================================================

from pathlib import Path

import h5py
import pytest

from simbi_configs.examples.newtonian.nmhd_rotor_cyl import NewtonianRotorCyl
from simbi_configs.examples.newtonian.spherical_pulse import SphericalPulse

from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    STEPS,
    _assert_same_state,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation


def _script_of(cls: type) -> str:
    return str(Path(cls.__module__.replace(".", "/") + ".py"))


def _assert_same_mesh(a: Path, b: Path, label: str) -> None:
    """the mesh datasets of a checkpoint (the coordinate faces and centers, the spacing and
    origin) equal the single grid's exactly: a launched run writes the whole grid's mesh."""
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        names: list[str] = []
        fb["level_0/mesh"].visit(lambda n: names.append(n) if isinstance(fb["level_0/mesh"][n], h5py.Dataset) else None)
        assert names, f"{label}: the reference carries no mesh datasets"
        for name in names:
            x, y = fa["level_0/mesh"][name][()], fb["level_0/mesh"][name][()]
            assert x.shape == y.shape and (x == y).all(), f"{label}: mesh/{name} differs"
        for key, value in fb["level_0/mesh"].attrs.items():
            assert (fa["level_0/mesh"].attrs[key] == value).all() if hasattr(value, "shape") and value.shape else fa["level_0/mesh"].attrs[key] == value, f"{label}: mesh attr {key}"


PULSE_SCRIPT = _script_of(SphericalPulse)
ROTOR_SCRIPT = _script_of(NewtonianRotorCyl)


def test_a_spherical_log_radial_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    resolution = (32, 32)
    reference = _script_reference(SphericalPulse.__module__, "SphericalPulse", resolution, tmp_path / "single", STEPS)
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2))
    launched = _script_launch(PULSE_SCRIPT, resolution, tmp_path / "four", four, STEPS, workers=4)
    _assert_same_state(launched, reference, "four workers on the spherical shell")
    _assert_same_mesh(launched, reference, "four workers on the spherical shell")
    # the flagged cell (15, 31) sits against the radial cut at 16 and the theta = pi pole
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:1:15:31"}
    clean = _script_launch(PULSE_SCRIPT, resolution, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(PULSE_SCRIPT, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(PULSE_SCRIPT, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    _assert_same_state(corrected, flagged, "four workers, the correction against the cut at the pole")


def test_a_spherical_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path) -> None:
    resolution = (32, 32)
    reference = _script_reference(SphericalPulse.__module__, "SphericalPulse", resolution, tmp_path / "single", STEPS)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    partial = _script_launch(PULSE_SCRIPT, resolution, tmp_path / "first", two, STEPS // 2, workers=2)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    # the cap is the absolute root iteration: the checkpoint's steps and the rest
    resumed = _script_launch(PULSE_SCRIPT, resolution, tmp_path / "three", three, STEPS, workers=3, checkpoint=partial)
    _assert_same_state(resumed, reference, "spherical shell resumed on three workers")


def test_a_magnetized_disk_plane_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    resolution = (32, 32, 1)
    env = {"SIMBI_FABRIC_AUDIT": "all"}
    reference = _script_reference(NewtonianRotorCyl.__module__, "NewtonianRotorCyl", resolution, tmp_path / "single", MHD_STEPS)
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2))
    launched = _script_launch(ROTOR_SCRIPT, resolution, tmp_path / "four", four, MHD_STEPS, workers=4, env=env)
    # the faces of the launched run equal the single grid's, whose constrained transport keeps
    # the disk plane's div B at round-off
    _assert_same_state(launched, reference, "four workers on the disk plane")
    _assert_same_mesh(launched, reference, "four workers on the disk plane")
    # the rotor's edge sits at r = 0.3, radial cell 7 of the 32 spanning [0.1, 1.0]; a radial cut
    # there puts the flagged cell (6, 0) against the cut inside the rotor's gradient and on the
    # periodic azimuth seam. away from the rotor the flow is uniform and a first-order redo
    # changes nothing, so the flag would leave the run unchanged
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    at_the_edge = _layout(tmp_path / "edge.toml", workers=4, cuts=[[7], [16]])
    inject = dict(env, SIMBI_FABRIC_INJECT="trouble:3:0:6:0")
    clean = _script_launch(ROTOR_SCRIPT, resolution, tmp_path / "clean", one, MHD_STEPS, workers=1, env=env)
    flagged = _script_launch(ROTOR_SCRIPT, resolution, tmp_path / "flagged", one, MHD_STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(ROTOR_SCRIPT, resolution, tmp_path / "corrected", at_the_edge, MHD_STEPS, workers=4, env=inject)
    _assert_same_state(corrected, flagged, "four workers, the correction against the cut at the seam")
