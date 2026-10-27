# =============================================================================
# test_launch_grid_matrix.py
#
# the curvilinear grids through `simbi launch` for the regimes whose launch
# admission covers them: newtonian hydro on the axisymmetric (r, z) plane with
# the axis on the grid, on the (r, phi) disk and on a 3d cylindrical annulus;
# special-relativistic hydro on the 2d and 3d spherical shells, the (r, z)
# plane, the (r, phi) disk and the 3d annulus; newtonian MHD on the (r, z)
# plane and on the 2d spherical shell under both edge EMFs; isothermal hydro on
# the (r, z) plane, the 3d shell and the 3d annulus; isothermal MHD on the (r, z)
# plane and the 2d shell under both edge EMFs; newtonian and isothermal MHD on
# the 3d shell spanning both poles, its azimuth cut, and on a 3d annulus
# periodic on every axis, its radius cut, under both edge EMFs. each case on four
# workers equals the single-grid run bit for bit, every staggered face included,
# fresh and with the correction fired against a cut (the flag asserted to change
# the run on one worker), and a run checkpointed halfway on four workers resumes
# onto three unevenly owned tiles to the uninterrupted single-grid state. the
# annulus wraps its radius: the first and last radial faces are two different
# faces the curl gives different values, which the interface audit leaves to
# their own tiles while it audits the interior cut.
# the single-grid references and the launches run on the session's backend
# and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_grid_matrix.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import h5py
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    STEPS,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_walls import _assert_flag_changed

pytestmark = pytest.mark.simulation

N = "simbi_configs.examples.newtonian"
R = "simbi_configs.examples.srhd"
ISO = "simbi_configs.examples.isothermal"
# the flagged cells sit in the initial gradient against a cut: on the (r, z) plane the pulse at
# (0.4, 0.5) spans radial cells 9..16 of 32, cut at 16; on the disk and the (r, z) jet the core's
# edge at r = 0.3 falls in radial cell 7 of the 32 spanning [0.1, 1.0], cut there; on the annulus
# the pulse at r = 0.5 is radial cell 7 of 16, cut at 8; on the shells the flags follow the
# newtonian spherical gates
RZ = dict(resolution=(32, 32), cuts=[[16], [16]], trouble="trouble:3:0:15:15", restart=[[10, 21], []], steps=STEPS)
DISK = dict(resolution=(32, 32), cuts=[[7], [16]], trouble="trouble:3:0:6:0", restart=[[10, 21], []], steps=STEPS)
ANNULUS = dict(resolution=(16, 16, 16), cuts=[[8], [], [8]], trouble="trouble:3:0:7:3:7", restart=[[5, 11], [], []], steps=STEPS)
SHELL = dict(resolution=(32, 32), cuts=[[16], [16]], trouble="trouble:3:1:15:31", restart=[[10, 21], []], steps=STEPS)
SHELL_3D = dict(resolution=(16, 16, 16), cuts=[[], [8], [8]], trouble="trouble:3:0:5:0:7", restart=[[], [], [6, 10]], steps=STEPS)
JET = dict(resolution=(32, 32, 1), cuts=[[7], [16]], trouble="trouble:3:0:6:0", restart=[[10, 21], []], steps=MHD_STEPS)
MHD_SHELL = dict(SHELL, resolution=(32, 32, 1), steps=MHD_STEPS)
UCT = dict(flags=["--ct-method", "uct"], fields="ct_method='uct'")
# the pole rows (theta cells 0 and 7 of 8) cut into two azimuth slabs, so every axis ring spans
# two tiles; the flag sits on the north pole row against the azimuth cut at 8
POLAR_SHELL = dict(resolution=(8, 8, 16), cuts=[[], [4], [8]], trouble="trouble:3:0:3:0:7", restart=[[], [], [5, 11]], steps=MHD_STEPS)
CASES = {
    "newtonian_rz": (f"{N}.rz_pulse", "RzPulse", RZ),
    "newtonian_disk": (f"{N}.viscous_disk_cyl", "ViscousDiskCyl", dict(DISK, flags=["--nu", "0.0"], fields="nu=0.0")),
    "newtonian_annulus": (f"{N}.cylindrical_pulse_3d", "CylindricalPulse3D", ANNULUS),
    "rhd_shell": (f"{R}.rhd_spherical_pulse", "RhdSphericalPulse", SHELL),
    "rhd_shell_3d": (f"{R}.rhd_spherical_pulse_3d", "RhdSphericalPulse3D", SHELL_3D),
    "rhd_rz": (f"{R}.rhd_rz_pulse", "RhdRzPulse", RZ),
    "rhd_disk": (f"{R}.rhd_disk_cyl", "RhdDiskCyl", DISK),
    "rhd_annulus": (f"{R}.rhd_cylindrical_pulse_3d", "RhdCylindricalPulse3D", ANNULUS),
    "nmhd_rz": (f"{N}.nmhd_jet_rz", "NewtonianJetRz", JET),
    "nmhd_rz_uct": (f"{N}.nmhd_jet_rz", "NewtonianJetRz", dict(JET, **UCT)),
    "nmhd_shell": (f"{N}.nmhd_spherical_field", "NewtonianSphericalField", MHD_SHELL),
    "nmhd_shell_uct": (f"{N}.nmhd_spherical_field", "NewtonianSphericalField", dict(MHD_SHELL, **UCT)),
    "nmhd_shell_3d": (f"{N}.nmhd_polar_shell_3d", "NewtonianPolarShell3D", POLAR_SHELL),
    "nmhd_shell_3d_uct": (f"{N}.nmhd_polar_shell_3d", "NewtonianPolarShell3D", dict(POLAR_SHELL, **UCT)),
    "imhd_shell_3d": (f"{ISO}.imhd_polar_shell_3d", "IsothermalPolarShell3D", POLAR_SHELL),
    "imhd_shell_3d_uct": (f"{ISO}.imhd_polar_shell_3d", "IsothermalPolarShell3D", dict(POLAR_SHELL, **UCT)),
    "isothermal_rz": (f"{ISO}.isothermal_rz_pulse", "IsothermalRzPulse", RZ),
    "isothermal_shell_3d": (f"{ISO}.isothermal_spherical_pulse_3d", "IsothermalSphericalPulse3D", SHELL_3D),
    "isothermal_annulus": (f"{ISO}.isothermal_cylindrical_pulse_3d", "IsothermalCylindricalPulse3D", ANNULUS),
    "imhd_rz": (f"{ISO}.imhd_jet_rz", "IsothermalJetRz", JET),
    "imhd_rz_uct": (f"{ISO}.imhd_jet_rz", "IsothermalJetRz", dict(JET, **UCT)),
    "imhd_shell": (f"{ISO}.imhd_spherical_field", "IsothermalSphericalField", MHD_SHELL),
    "imhd_shell_uct": (f"{ISO}.imhd_spherical_field", "IsothermalSphericalField", dict(MHD_SHELL, **UCT)),
}


def _case(name: str) -> tuple[str, str, str, dict]:
    module, cls, case = CASES[name]
    return module, cls, module.replace(".", "/") + ".py", case


@pytest.mark.parametrize("name", list(CASES))
def test_a_launch_matches_the_single_grid_run_fresh_and_corrected(tmp_path: Path, name: str) -> None:
    module, cls, script, case = _case(name)
    resolution, steps, flags = case["resolution"], case["steps"], case.get("flags", [])
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps, fields=case.get("fields", ""))
    four = _layout(tmp_path / "four-layout.toml", workers=4, cuts=case["cuts"])
    launched = _script_launch(script, resolution, tmp_path / "four", four, steps, workers=4, flags=flags)
    ph.same_checkpoint(launched, reference, f"{name}: four workers")
    one = _layout(tmp_path / "one-layout.toml", workers=1, cuts=[[] for _ in case["cuts"]])
    inject = {"SIMBI_FABRIC_INJECT": case["trouble"]}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, steps, workers=1, flags=flags)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, steps, workers=1, env=inject, flags=flags)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, steps, workers=4, env=inject, flags=flags)
    ph.same_checkpoint(corrected, flagged, f"{name}: four workers, the correction against the cut")


@pytest.mark.parametrize("name", list(CASES))
def test_a_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path, name: str) -> None:
    module, cls, script, case = _case(name)
    resolution, steps, flags = case["resolution"], case["steps"], case.get("flags", [])
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps, fields=case.get("fields", ""))
    four = _layout(tmp_path / "four-layout.toml", workers=4, cuts=case["cuts"])
    partial = _script_launch(script, resolution, tmp_path / "first", four, steps // 2, workers=4, flags=flags)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < steps
    three = _layout(tmp_path / "three-layout.toml", workers=3, cuts=case["restart"])
    resumed = _script_launch(script, resolution, tmp_path / "three", three, steps, workers=3, checkpoint=partial, flags=flags)
    ph.same_checkpoint(resumed, reference, f"{name}: resumed onto three uneven workers against the uninterrupted run")




MHD_ANNULUS = """
import math
from typing import Annotated
from simbi import ProblemParam
from simbi.types import CoordSystem
from simbi_configs.examples.newtonian.nmhd_orszag_tang_3d import NewtonianOrszagTang3D

class OrszagTangAnnulus(NewtonianOrszagTang3D):
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="coordinate system")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.1, 1.0), (0.0, 2.0 * math.pi), (0.0, 1.0)], description="bounds")]
"""
IMHD_ANNULUS = MHD_ANNULUS.replace("newtonian.nmhd_orszag_tang_3d import NewtonianOrszagTang3D", "isothermal.imhd_orszag_tang_3d import IsothermalOrszagTang3D").replace(
    "(NewtonianOrszagTang3D)", "(IsothermalOrszagTang3D)"
)
# the radius (8 cells over [0.1, 1.0]) cut at 4 and wrapped; the flag sits against the radial cut
ANNULUS_MHD = dict(resolution=(8, 8, 8), cuts=[[4], [], [4]], trouble="trouble:3:0:3:2:3", restart=[[3, 5], [], []], steps=MHD_STEPS)


@pytest.fixture(params=["nmhd", "imhd"])
def annulus(request, tmp_path: Path, monkeypatch) -> tuple[str, str]:
    """the periodic annulus as an importable module and as a launch script, newtonian and isothermal"""
    source = tmp_path / "src"
    source.mkdir()
    (source / "orszag_tang_annulus.py").write_text(MHD_ANNULUS if request.param == "nmhd" else IMHD_ANNULUS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(source), str(Path.cwd()), os.environ.get("PYTHONPATH", "")]))
    return "orszag_tang_annulus", str(source / "orszag_tang_annulus.py")


def _wrap_faces_differ(path: Path) -> bool:
    with h5py.File(path, "r") as f:
        b1 = f["level_0/partition_0/hydro/magnetic/B1/data"][()]
    return bool((b1[..., 0] != b1[..., -1]).any())


@pytest.mark.parametrize("ct", ["contact", "uct"])
def test_mhd_on_an_annulus_wrapped_in_radius_matches_fresh_corrected_and_resumed(tmp_path: Path, annulus, ct: str) -> None:
    module, script = annulus
    case, flags = ANNULUS_MHD, ["--ct-method", ct]
    resolution, steps = case["resolution"], case["steps"]
    reference = _script_reference(module, "OrszagTangAnnulus", resolution, tmp_path / "single", steps, fields=f"ct_method={ct!r}")
    assert _wrap_faces_differ(reference), "the first and last radial faces agree; the wrap is indistinguishable from one face"
    audit = {"SIMBI_FABRIC_AUDIT": "all"}
    four = _layout(tmp_path / "four-layout.toml", workers=4, cuts=case["cuts"])
    launched = _script_launch(script, resolution, tmp_path / "four", four, steps, workers=4, env=audit, flags=flags)
    ph.same_checkpoint(launched, reference, f"annulus {ct}: four workers auditing every exchange")
    one = _layout(tmp_path / "one-layout.toml", workers=1, cuts=[[], [], []])
    inject = {"SIMBI_FABRIC_INJECT": case["trouble"]}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, steps, workers=1, flags=flags)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, steps, workers=1, env=inject, flags=flags)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, steps, workers=4, env=dict(inject, **audit), flags=flags)
    ph.same_checkpoint(corrected, flagged, f"annulus {ct}: four workers, the correction against the radial cut")
    partial = _script_launch(script, resolution, tmp_path / "first", four, steps // 2, workers=4, flags=flags)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < steps
    three = _layout(tmp_path / "three-layout.toml", workers=3, cuts=case["restart"])
    resumed = _script_launch(script, resolution, tmp_path / "three", three, steps, workers=3, checkpoint=partial, env=audit, flags=flags)
    ph.same_checkpoint(resumed, reference, f"annulus {ct}: resumed onto three uneven workers")
