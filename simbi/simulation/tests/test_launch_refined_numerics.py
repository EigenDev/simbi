# =============================================================================
# test_launch_refined_numerics.py
#
# the numerics a refined launch carries on both levels of every tile, through
# `simbi launch` against the single-process refined run, both levels compared
# bit for bit: first-order space (pcm), ppm with the convergence-gated
# flattening, a constant viscosity, a user source and a gradient wall in
# newtonian hydro, and a resistivity in 2.5D newtonian MHD under the contact
# and the uct edge EMF. each variant runs fresh on four workers whose ragged
# cuts cross the fine box, and resumed onto the four from a two-worker
# checkpoint written halfway; each is shown to change the single-process run,
# so the equality is a statement about the setting; the pcm reference on its
# own is the single-process refined pcm run, which takes the plm prolongation.
# periodic boundaries: the kelvin-helmholtz layers on a 64 x 64 root and the
# magnetized blast made periodic under both edge EMFs, the root wrapped across
# its seam by the launch's plans, fresh and resumed. a patch on a periodic face
# and an alpha viscosity with refinement are refused by name.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_refined_numerics.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_refined import RESOLUTION, STEPS, _assert_same_levels, _levels

pytestmark = pytest.mark.simulation

HYDRO = ("simbi_configs.examples.newtonian.refined_blast", "RefinedBlast")
MHD = ("simbi_configs.examples.newtonian.refined_mhd_blast", "RefinedMhdBlast")
PERIODIC = "boundary_conditions: Annotated[list, ProblemParam([BoundaryCondition.PERIODIC], description='bc')]"
# a shear v_y = 0.2 sin(2 pi x) across the x seam, so the wrap carries a state the outflow copy
# does not: the periodic arms are compared against the same sheared run with outflow walls
SHEAR = [
    "def initial_primitive_state(self):",
    "    gas, *faces = super().initial_primitive_state()",
    "    nx = self.resolution[0]",
    "    def sheared():",
    "        for k, s in enumerate(gas()):",
    "            x = -0.5 + ((k % nx) + 0.5) / nx",
    "            yield (s[0], s[1], s[2] + 0.2 * math.sin(2.0 * math.pi * x), *s[3:])",
    "    return (sheared, *faces)",
]
# class-body lines of each variant's subclass; the flags a launch passes and the constructor fields
# the reference passes stay empty, since the subclass carries the setting
VARIANTS = {
    "pcm": (HYDRO, ["reconstruction: Annotated[Reconstruction, ProblemParam(Reconstruction.PCM, description='r')]"]),
    "ppm-flattened": (
        HYDRO,
        [
            "reconstruction: Annotated[Reconstruction, ProblemParam(Reconstruction.PPM, description='r')]",
            "ppm_flatten_onset: Annotated[float, ProblemParam(0.05, description='onset')]",
            "ppm_flatten_full: Annotated[float, ProblemParam(0.5, description='full')]",
        ],
    ),
    "viscous": (HYDRO, ["@computed_field", "@property", "def viscosity(self) -> float:", "    return 0.02"]),
    "sourced": (
        HYDRO,
        ["@property", "def source_expressions(self):", "    return [expr.force([0.3, -0.2], dim=2)]"],
    ),
    "gradient-wall": (
        HYDRO,
        ["boundary_conditions: Annotated[list, ProblemParam([Neumann(rho=0.1)], description='walls')]"],
    ),
    "periodic-mhd-contact": (MHD, [PERIODIC, *SHEAR]),
    "periodic-mhd-uct": (MHD, [PERIODIC, "ct_method: Annotated[CtMethod, ProblemParam(CtMethod.UCT, description='edge emf')]", *SHEAR]),
    "resistive-contact": (MHD, ["@computed_field", "@property", "def resistivity(self) -> float:", "    return 0.01"]),
    "resistive-uct": (
        MHD,
        [
            "ct_method: Annotated[CtMethod, ProblemParam(CtMethod.UCT, description='edge emf')]",
            "@computed_field",
            "@property",
            "def resistivity(self) -> float:",
            "    return 0.01",
        ],
    ),
}


def _variant_module(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch, lines=None) -> tuple[str, str, str]:
    """(module, class, script) of a subclass of the variant's base carrying its setting (or the
    class-body `lines` given), importable by the reference and by the launched workers"""
    (module, cls), variant_lines = VARIANTS[name]
    stem = "variant_" + name.replace("-", "_") + ("" if lines is None else "_baseline")
    lines = variant_lines if lines is None else lines
    body = "\n".join(f"    {line}" for line in lines)
    (tmp_path / f"{stem}.py").write_text(
        "import math\n"
        "from typing import Annotated\n"
        "from pydantic import computed_field\n"
        "import simbi.expression as expr\n"
        "from simbi import Neumann, ProblemParam\n"
        "from simbi.types import BoundaryCondition, CtMethod, Reconstruction\n"
        f"from {module} import {cls}\n\n\n"
        f"class Variant({cls}):\n{body}\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return stem, "Variant", str(tmp_path / f"{stem}.py")


def _differs(a: Path, b: Path) -> bool:
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        xs, ys = _levels(fa), _levels(fb)
    return any(not np.array_equal(x, ys[k]) for k, x in xs.items() if k in ys)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_a_refined_launch_carries_the_setting_fresh_and_resumed(tmp_path: Path, name: str, monkeypatch) -> None:
    module, cls, script = _variant_module(tmp_path, name, monkeypatch)
    (base_module, base_cls), lines = VARIANTS[name]
    reference = _script_reference(module, cls, RESOLUTION, tmp_path / "single", STEPS)
    if PERIODIC in lines:
        # the baseline is the same variant with the periodic walls taken out
        base_module, base_cls, _ = _variant_module(tmp_path, name, monkeypatch, [l for l in lines if l != PERIODIC])
    base = _script_reference(base_module, base_cls, RESOLUTION, tmp_path / "base", STEPS)
    assert _differs(reference, base), f"{name}: the setting left the single-process run unchanged; the gate is vacuous"
    four = _layout(tmp_path / "four.toml", workers=4, cuts=[[13], [18]])
    launched = _script_launch(script, RESOLUTION, tmp_path / "four", four, STEPS, workers=4)
    _assert_same_levels(launched, reference, f"{name}: four workers")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    first = _script_launch(script, RESOLUTION, tmp_path / "first", two, STEPS // 2, workers=2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    resumed = _script_launch(script, RESOLUTION, tmp_path / "resumed", four, STEPS - written_at, workers=4, checkpoint=first)
    _assert_same_levels(resumed, reference, f"{name}: resumed onto four workers")


KH = ("simbi_configs.examples.newtonian.refined_kh", "RefinedKelvinHelmholtz")
# the fine box spans root cells 4..60 in x and 8..56 in y of the 64 x 64 root, four cells clear of
# the seam; the ragged cuts cross it on both axes
KH_RESOLUTION = (64, 64)


def test_a_periodic_refined_launch_matches_fresh_and_resumed(tmp_path: Path) -> None:
    module, cls = KH
    script = module.replace(".", "/") + ".py"
    reference = _script_reference(module, cls, KH_RESOLUTION, tmp_path / "single", STEPS)
    walled = _script_reference(module, cls, KH_RESOLUTION, tmp_path / "walled", STEPS, fields="boundary_conditions='outflow'")
    assert _differs(reference, walled), "the periodic seam left the kh run unchanged; the wrap is never exercised"
    four = _layout(tmp_path / "four.toml", workers=4, cuts=[[26], [36]])
    launched = _script_launch(script, KH_RESOLUTION, tmp_path / "four", four, STEPS, workers=4)
    _assert_same_levels(launched, reference, "periodic kh: four workers")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    first = _script_launch(script, KH_RESOLUTION, tmp_path / "first", two, STEPS // 2, workers=2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    resumed = _script_launch(script, KH_RESOLUTION, tmp_path / "resumed", four, STEPS - written_at, workers=4, checkpoint=first)
    _assert_same_levels(resumed, reference, "periodic kh: resumed onto four workers")


def test_a_refined_patch_on_a_periodic_face_is_refused_by_name(tmp_path: Path) -> None:
    script = tmp_path / "seam_blast.py"
    script.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi.types import BoundaryCondition\n"
        "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n"
        "class SeamBlast(RefinedBlast):\n"
        "    boundary_conditions: Annotated[list, ProblemParam([BoundaryCondition.PERIODIC], description='bc')]\n"
        "    refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.5, 0.0, -0.25, 0.25]], description='r')]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", "32,32,1", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "a refined patch on the periodic face of axis 0 with mesh refinement is outside the distributed first release" in flat, done.stderr
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


def test_an_alpha_viscosity_with_refinement_is_refused_by_name(tmp_path: Path) -> None:
    script = tmp_path / "alpha_blast.py"
    script.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n"
        "class AlphaBlast(RefinedBlast):\n"
        "    viscosity_alpha: Annotated[float, ProblemParam(0.01, description='alpha')]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", "32,32,1", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "an alpha viscosity with mesh refinement is outside the distributed first release" in flat, done.stderr
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"
