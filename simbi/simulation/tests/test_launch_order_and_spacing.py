# =============================================================================
# test_launch_order_and_spacing.py
#
# first-order reconstruction and mapped radial or cartesian spacing through
# `simbi launch` beyond newtonian hydro on a uniform cartesian grid, each on
# four workers against the single-grid run bit for bit and resumed onto three
# unevenly owned tiles:
# - pcm (the plm kernel at a zero slope) with isothermal hydro, relativistic
#   hydro, newtonian and isothermal MHD; the pcm run differs from the plm run.
#   a first-order redo equals the pcm update, so no correction arm applies;
# - a geometric axis with isothermal and relativistic hydro on a cartesian
#   grid, a geometric radius on the newtonian, relativistic and isothermal
#   spherical shells, and a geometric or logarithmic radius under newtonian and
#   isothermal MHD on the (r, phi) disk; each mapped run differs from the uniform
#   one, and the correction fired against a cut matches.
# the mapped problems are the examples with their spacing set by a subclass
# written for the test. ppm outside newtonian hydro on a flat cartesian grid is
# refused by name, even under the experimental opt-in.
# the single-grid references and the launches run on the session's backend
# and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_order_and_spacing.py
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
PCM = dict(flags=["--reconstruction", "pcm"], fields="reconstruction='pcm'")
PLANE = dict(cuts=[[16], [16]], restart=[[10, 21], []])
# (module, class, resolution, steps, layout)
ORDER_CASES = {
    "isothermal": (f"{ISO}.isothermal_channel", "IsothermalChannel", (32, 32), STEPS, PLANE),
    "rhd": (f"{R}.synge_blast", "SyngeBlast", (32, 32), STEPS, PLANE),
    "nmhd": (f"{N}.nmhd_orszag_tang", "NewtonianOrszagTang", (32, 32, 1), MHD_STEPS, PLANE),
    "imhd": (f"{ISO}.imhd_orszag_tang", "IsothermalOrszagTang", (32, 32, 1), MHD_STEPS, PLANE),
}
# (module, class, resolution, steps, the spacing lines of the subclass, extra reference fields, layout)
SHELL = dict(cuts=[[16], [16]], trouble="trouble:3:1:15:31", restart=[[10, 21], []])
DISK = dict(cuts=[[7], [16]], trouble="trouble:3:0:6:0", restart=[[10, 21], []])
GEOMETRIC_Y = ["x2_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.GEOMETRIC, description='y spacing')]",
               "x2_spacing_ratio: Annotated[float, ProblemParam(1.04, description='y width ratio')]"]
GEOMETRIC_R = ["x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.GEOMETRIC, description='r spacing')]",
               "x1_spacing_ratio: Annotated[float, ProblemParam(1.03, description='r width ratio')]"]
LOG_R = ["x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LOG, description='r spacing')]"]
SPACING_CASES = {
    "isothermal_cartesian": (f"{ISO}.isothermal_channel", "IsothermalChannel", (32, 32), STEPS, GEOMETRIC_Y, "",
                             dict(cuts=[[16], [16]], trouble="trouble:3:0:15:0", restart=[[10, 21], []])),
    "rhd_cartesian": (f"{R}.synge_blast", "SyngeBlast", (32, 32), STEPS, GEOMETRIC_Y, "",
                      dict(cuts=[[16], [16]], trouble="trouble:3:0:15:15", restart=[[10, 21], []])),
    "newtonian_shell": (f"{N}.spherical_pulse", "SphericalPulse", (32, 32), STEPS, GEOMETRIC_R, "", SHELL),
    "rhd_shell": (f"{R}.rhd_spherical_pulse", "RhdSphericalPulse", (32, 32), STEPS, GEOMETRIC_R, "", SHELL),
    "isothermal_shell": (f"{ISO}.isothermal_spherical_pulse", "IsothermalSphericalPulse", (32, 32), STEPS,
                         GEOMETRIC_R, "", SHELL),
    "nmhd_disk_geometric": (f"{N}.nmhd_rotor_cyl", "NewtonianRotorCyl", (32, 32, 1), MHD_STEPS,
                            GEOMETRIC_R + ["br: Annotated[float, ProblemParam(0.1, description='b_r')]",
                                           "bphi: Annotated[float, ProblemParam(0.05, description='b_phi')]"],
                            "", DISK),
    "nmhd_disk_log": (f"{N}.nmhd_rotor_cyl", "NewtonianRotorCyl", (32, 32, 1), MHD_STEPS,
                      LOG_R + ["br: Annotated[float, ProblemParam(0.1, description='b_r')]",
                               "bphi: Annotated[float, ProblemParam(0.05, description='b_phi')]"],
                      "", DISK),
    "imhd_disk_geometric": (f"{ISO}.imhd_rotor_cyl", "IsothermalRotorCyl", (32, 32, 1), MHD_STEPS, GEOMETRIC_R, "",
                            DISK),
}


def _script(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _mapped_module(tmp_path: Path, module: str, cls: str, lines: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    """a module in tmp_path subclassing `cls` with the spacing `lines`, importable by the reference
    and the launched workers"""
    name = f"mapped_{cls.lower()}"
    body = "\n".join(f"    {line}" for line in lines)
    (tmp_path / f"{name}.py").write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi.types import CellSpacing\n"
        f"from {module} import {cls}\n\n\n"
        f"class Mapped{cls}({cls}):\n{body}\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return name, f"Mapped{cls}", str(tmp_path / f"{name}.py")


def _resume(tmp_path: Path, script: str, module: str, cls: str, resolution, steps: int, layout: dict, flags=None, fields="") -> None:
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps, fields=fields)
    four = _layout(tmp_path / "four-layout.toml", workers=4, cuts=layout["cuts"])
    partial = _script_launch(script, resolution, tmp_path / "first", four, steps // 2, workers=4, flags=flags)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < steps
    three = _layout(tmp_path / "three-layout.toml", workers=3, cuts=layout["restart"])
    resumed = _script_launch(script, resolution, tmp_path / "three", three, steps, workers=3, checkpoint=partial, flags=flags)
    ph.same_checkpoint(resumed, reference, "resumed onto three uneven workers against the uninterrupted run")


@pytest.mark.parametrize("name", list(ORDER_CASES))
def test_a_first_order_launch_matches_the_single_grid_run(tmp_path: Path, name: str) -> None:
    module, cls, resolution, steps, layout = ORDER_CASES[name]
    script = _script(module)
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps, fields=PCM["fields"])
    four = _layout(tmp_path / "four-layout.toml", workers=4, cuts=layout["cuts"])
    launched = _script_launch(script, resolution, tmp_path / "four", four, steps, workers=4, flags=PCM["flags"])
    ph.same_checkpoint(launched, reference, f"{name}: four workers under pcm")
    one = _layout(tmp_path / "one-layout.toml", workers=1, cuts=[[] for _ in layout["cuts"]])
    plm = _script_launch(script, resolution, tmp_path / "plm", one, steps, workers=1)
    _assert_flag_changed(plm, launched)
    _resume(tmp_path / "resume", script, module, cls, resolution, steps, layout, flags=PCM["flags"], fields=PCM["fields"])


@pytest.mark.parametrize("name", list(SPACING_CASES))
def test_a_mapped_launch_matches_the_single_grid_run_fresh_corrected_and_resumed(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    base_module, base_cls, resolution, steps, lines, _fields, layout = SPACING_CASES[name]
    module, cls, script = _mapped_module(tmp_path, base_module, base_cls, lines, monkeypatch)
    reference = _script_reference(module, cls, resolution, tmp_path / "single", steps)
    four = _layout(tmp_path / "four-layout.toml", workers=4, cuts=layout["cuts"])
    launched = _script_launch(script, resolution, tmp_path / "four", four, steps, workers=4)
    ph.same_checkpoint(launched, reference, f"{name}: four workers on the mapped grid")
    one = _layout(tmp_path / "one-layout.toml", workers=1, cuts=[[] for _ in layout["cuts"]])
    uniform = _script_launch(_script(base_module), resolution, tmp_path / "uniform", one, steps, workers=1)
    _assert_flag_changed(uniform, launched)
    inject = {"SIMBI_FABRIC_INJECT": layout["trouble"]}
    clean = _script_launch(script, resolution, tmp_path / "clean", one, steps, workers=1)
    flagged = _script_launch(script, resolution, tmp_path / "flagged", one, steps, workers=1, env=inject)
    _assert_flag_changed(clean, flagged)
    corrected = _script_launch(script, resolution, tmp_path / "corrected", four, steps, workers=4, env=inject)
    ph.same_checkpoint(corrected, flagged, f"{name}: four workers, the correction against the cut")
    _resume(tmp_path / "resume", script, module, cls, resolution, steps, layout)


def test_ppm_outside_newtonian_cartesian_hydro_is_refused_by_name(tmp_path: Path) -> None:
    # the isothermal, magnetized and relativistic tile builders carry no ppm kernel; a request is
    # refused at startup rather than run as plm
    module, cls, resolution, steps, layout = ORDER_CASES["isothermal"]
    args = ["--max-steps", "1", "--resolution", ",".join(str(r) for r in resolution), "--checkpoint-interval", "1e9", "--reconstruction", "ppm"]
    done = ph.launch(_script(module), tmp_path / "out", 2, cuts=[[16], []], args=args, env=ph.environment(opt_in=True), timeout=600, expect_ok=False)
    assert done.returncode != 0, f"the launch ran a ppm request:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "ppm reconstruction is wired for the newtonian (adiabatic ideal-gas) regime only; got 'isothermal'" in flat, done.stderr
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"
