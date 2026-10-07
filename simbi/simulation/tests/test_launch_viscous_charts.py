# =============================================================================
# test_launch_viscous_charts.py
#
# the constant-nu viscous transport through `simbi launch` on the curvilinear
# grids and in three dimensions, for every regime the single path carries it
# in: newtonian and isothermal hydro on a uniform-radius spherical (r, theta)
# shell spanning both poles, newtonian and isothermal hydro, newtonian and
# isothermal MHD on the (r, phi) disk plane, and all four on a 3d cartesian
# periodic box; newtonian and isothermal hydro on the (r, z) plane with the axis
# on the grid, the 3d spherical shell and the 3d cylindrical annulus; newtonian
# and isothermal MHD on the (r, z) plane and the 2d spherical shell. each case on four workers equals the single-grid run bit for
# bit, fresh and with the correction fired against a cut; the viscosity acts,
# the inviscid launch differing from the viscous one; a run checkpointed
# halfway resumes to the uninterrupted single-grid state on one worker and
# onto three unevenly owned tiles from a four-worker checkpoint. a viscosity on
# a relativistic regime, which carries no viscous operator, is refused at
# startup by name rather than dropped.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_viscous_charts.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests.test_launch import (
    STEPS,
    _assert_same_state,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

N = "simbi_configs.examples.newtonian"
ISO = "simbi_configs.examples.isothermal"
# the spherical flag (15, 31) sits against the radial cut at 16 and the theta = pi pole. on the
# disk plane the core's edge at r = 0.3 falls in radial cell 7 of the 32 spanning [0.1, 1.0]; a
# radial cut there puts the flagged cell (6, 0) against the cut inside the core's gradient and on
# the periodic seam. in 3d the flagged cell (7, 3, 7) sits against the x and z cuts at 8
SPHERE = dict(resolution=(32, 32), cuts=[[16], [16]], trouble="trouble:3:1:15:31", restart=[[10, 21], []])
DISK = dict(resolution=(32, 32), cuts=[[7], [16]], trouble="trouble:3:0:6:0", restart=[[10, 21], []])
ROTOR = dict(DISK, resolution=(32, 32, 1))
BOX = dict(resolution=(16, 16, 16), cuts=[[8], [], [8]], trouble="trouble:3:0:7:3:7", restart=[[5, 11], [], []])
RZ = dict(resolution=(32, 32), cuts=[[16], [16]], trouble="trouble:3:0:15:15", restart=[[10, 21], []])
SHELL_3D = dict(resolution=(16, 16, 16), cuts=[[], [8], [8]], trouble="trouble:3:0:5:0:7", restart=[[], [], [6, 10]])
ANNULUS = BOX
JET = dict(ROTOR)
MHD_SHELL = dict(SPHERE, resolution=(32, 32, 1))
CASES = {
    "newtonian_spherical": (f"{N}.viscous_spherical_pulse", "ViscousSphericalPulse", SPHERE),
    "isothermal_spherical": (f"{ISO}.viscous_isothermal_spherical_pulse", "ViscousIsothermalSphericalPulse", SPHERE),
    "newtonian_disk": (f"{N}.viscous_disk_cyl", "ViscousDiskCyl", DISK),
    "isothermal_disk": (f"{ISO}.viscous_isothermal_disk_cyl", "ViscousIsothermalDiskCyl", DISK),
    "nmhd_disk": (f"{N}.viscous_nmhd_rotor_cyl", "ViscousNewtonianRotorCyl", ROTOR),
    "imhd_disk": (f"{ISO}.viscous_imhd_rotor_cyl", "ViscousIsothermalRotorCyl", ROTOR),
    "newtonian_3d": (f"{N}.viscous_shear_3d", "ViscousShear3D", BOX),
    "isothermal_3d": (f"{ISO}.viscous_isothermal_shear_3d", "ViscousIsothermalShear3D", BOX),
    "nmhd_3d": (f"{N}.viscous_nmhd_orszag_tang_3d", "ViscousNewtonianOrszagTang3D", BOX),
    "imhd_3d": (f"{ISO}.viscous_imhd_orszag_tang_3d", "ViscousIsothermalOrszagTang3D", BOX),
    "newtonian_rz": (f"{N}.viscous_rz_pulse", "ViscousRzPulse", RZ),
    "isothermal_rz": (f"{ISO}.viscous_isothermal_rz_pulse", "ViscousIsothermalRzPulse", RZ),
    "newtonian_shell_3d": (f"{N}.viscous_spherical_pulse_3d", "ViscousSphericalPulse3D", SHELL_3D),
    "isothermal_shell_3d": (f"{ISO}.viscous_isothermal_spherical_pulse_3d", "ViscousIsothermalSphericalPulse3D", SHELL_3D),
    "newtonian_annulus": (f"{N}.viscous_cylindrical_pulse_3d", "ViscousCylindricalPulse3D", ANNULUS),
    "isothermal_annulus": (f"{ISO}.viscous_isothermal_cylindrical_pulse_3d", "ViscousIsothermalCylindricalPulse3D", ANNULUS),
    "nmhd_rz": (f"{N}.viscous_nmhd_jet_rz", "ViscousNewtonianJetRz", JET),
    "imhd_rz": (f"{ISO}.viscous_imhd_jet_rz", "ViscousIsothermalJetRz", JET),
    "nmhd_shell": (f"{N}.viscous_nmhd_spherical_field", "ViscousNewtonianSphericalField", MHD_SHELL),
    "imhd_shell": (f"{ISO}.viscous_imhd_spherical_field", "ViscousIsothermalSphericalField", MHD_SHELL),
}


def _one(path: Path, case: dict) -> Path:
    return _layout(path, workers=1, cuts=[[] for _ in case["cuts"]])


@pytest.mark.parametrize("name", list(CASES))
def test_a_viscous_launch_matches_the_single_grid_run(tmp_path: Path, name: str) -> None:
    module, cls, case = CASES[name]
    script = module.replace(".", "/") + ".py"
    resolution = case["resolution"]
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS)
    four = _layout(tmp_path / "four.toml", workers=4, cuts=case["cuts"])
    launched = _script_launch(script, resolution, tmp_path / "four", four, STEPS, workers=4)
    _assert_same_state(launched, reference, f"{name}: four workers")
    one = _one(tmp_path / "one.toml", case)
    inviscid = _script_launch(script, resolution, tmp_path / "inviscid", one, STEPS, workers=1, flags=["--nu", "0.0"])
    _assert_flag_changed(inviscid, launched)
    inject = {"SIMBI_FABRIC_INJECT": case["trouble"]}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, STEPS, workers=1)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, STEPS, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, STEPS, workers=4, env=inject)
    _assert_same_state(corrected, flagged, f"{name}: four workers, the correction against the cut")


@pytest.mark.parametrize("name", list(CASES))
def test_a_viscous_launch_resumed_onto_another_partition_matches(tmp_path: Path, name: str) -> None:
    module, cls, case = CASES[name]
    script = module.replace(".", "/") + ".py"
    resolution = case["resolution"]
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS)
    one = _one(tmp_path / "one.toml", case)
    partial_one = _script_launch(script, resolution, tmp_path / "half", one, STEPS // 2, workers=1)
    with h5py.File(partial_one, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # `max_steps` caps the absolute root iteration: each resumed launch steps from its checkpoint
    # at written_at to STEPS
    resumed_one = _script_launch(script, resolution, tmp_path / "resumed_one", one, STEPS, workers=1, checkpoint=partial_one)
    _assert_same_state(resumed_one, reference, f"{name}: resumed on one worker against the uninterrupted run")
    four = _layout(tmp_path / "four.toml", workers=4, cuts=case["cuts"])
    partial = _script_launch(script, resolution, tmp_path / "first", four, STEPS // 2, workers=4)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=case["restart"])
    resumed = _script_launch(script, resolution, tmp_path / "three", three, STEPS, workers=3, checkpoint=partial)
    _assert_same_state(resumed, reference, f"{name}: resumed onto three uneven workers against the uninterrupted run")



VISCOUS_BLAST = """
from pydantic import computed_field
from simbi_configs.examples.srhd.synge_blast import SyngeBlast

class ViscousSyngeBlast(SyngeBlast):
    @computed_field
    @property
    def viscosity(self) -> float:
        return 0.01
"""


def test_viscosity_on_a_relativistic_regime_is_refused_by_name(tmp_path: Path) -> None:
    script = tmp_path / "viscous_synge_blast.py"
    script.write_text(VISCOUS_BLAST)
    layout = _layout(tmp_path / "two.toml", workers=2, cuts=[[8], []])
    cmd = [
        sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
        "--resolution", "16,16", "--data-directory", str(tmp_path / "refused"), "--checkpoint-interval", "1e9",
    ]
    env = {k: v for k, v in os.environ.items() if k != "SIMBI_LAUNCH_EXPERIMENTAL"}
    env["SIMBI_LAUNCH_EXPERIMENTAL"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), env.get("PYTHONPATH", "")])
    # the opt-in admits every ungated combination, so the refusal comes from the regime itself
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "viscous transport is not built for the rhd regime" in flat, done.stderr
    assert not list((tmp_path / "refused").glob("*.h5")), "a checkpoint was published"
