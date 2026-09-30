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
# its seam by the launch's plans, fresh and resumed. relativistic hydro: the
# synge blast with a fine box over its hot disk, under the synge and the ideal
# gas, shown to differ, fresh and resumed. isothermal hydro (the drifting blob)
# and isothermal MHD (the walled orszag-tang vortex, under contact and uct,
# shown to differ), each with a fine box over the central half, fresh and
# resumed. 3D: the refined spherical blast in newtonian hydro and the extruded
# orszag-tang vortex in newtonian MHD (periodic, the fine box inside the domain,
# under contact and uct), fresh on eight workers whose ragged cuts meet at a
# corner inside the box and resumed onto them from two. a patch on a periodic face
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


RHD = ("simbi_configs.examples.srhd.refined_synge_blast", "RefinedSyngeBlast")
RHD_RESOLUTION = (32, 32)


def _ideal_rhd_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    """the refined synge blast under the ideal gas at gamma = 4/3, importable by the reference and
    by the launched workers"""
    module, cls = RHD
    (tmp_path / "ideal_rhd.py").write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi.types.input import Eos\n"
        f"from {module} import {cls}\n\n\n"
        f"class IdealRhd({cls}):\n"
        "    eos: Annotated[Eos, ProblemParam(Eos.IDEAL, description='the ideal gas')]\n"
        "    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description='adiabatic index')]\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return "ideal_rhd", "IdealRhd", str(tmp_path / "ideal_rhd.py")


@pytest.mark.parametrize("eos", ["synge", "ideal"])
def test_a_relativistic_refined_launch_matches_fresh_and_resumed(tmp_path: Path, eos: str, monkeypatch) -> None:
    synge_module, synge_cls = RHD
    ideal_module, ideal_cls, ideal_script = _ideal_rhd_module(tmp_path, monkeypatch)
    arms = {"synge": (synge_module, synge_cls, synge_module.replace(".", "/") + ".py"), "ideal": (ideal_module, ideal_cls, ideal_script)}
    module, cls, script = arms[eos]
    other_module, other_cls, _ = arms["ideal" if eos == "synge" else "synge"]
    reference = _script_reference(module, cls, RHD_RESOLUTION, tmp_path / "single", STEPS)
    contrast = _script_reference(other_module, other_cls, RHD_RESOLUTION, tmp_path / "contrast", STEPS)
    assert _differs(reference, contrast), "the equation of state left the refined run unchanged; the gate is vacuous"
    four = _layout(tmp_path / "four.toml", workers=4, cuts=[[13], [18]])
    launched = _script_launch(script, RHD_RESOLUTION, tmp_path / "four", four, STEPS, workers=4)
    _assert_same_levels(launched, reference, f"rhd {eos}: four workers")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    first = _script_launch(script, RHD_RESOLUTION, tmp_path / "first", two, STEPS // 2, workers=2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    resumed = _script_launch(script, RHD_RESOLUTION, tmp_path / "resumed", four, STEPS - written_at, workers=4, checkpoint=first)
    _assert_same_levels(resumed, reference, f"rhd {eos}: resumed onto four workers")


ISO = "simbi_configs.examples.isothermal"
# (module, class, resolution, launch flags, reference fields)
ISOTHERMAL = {
    "isothermal": (f"{ISO}.refined_isothermal_blob", "RefinedIsothermalBlob", (32, 32), [], ""),
    "imhd-contact": (f"{ISO}.refined_imhd_orszag_tang", "RefinedIsothermalOrszagTang", (32, 32, 1), [], ""),
    "imhd-uct": (f"{ISO}.refined_imhd_orszag_tang", "RefinedIsothermalOrszagTang", (32, 32, 1), ["--ct-method", "uct"], "ct_method='uct'"),
}


@pytest.mark.parametrize("name", list(ISOTHERMAL))
def test_an_isothermal_refined_launch_matches_fresh_and_resumed(tmp_path: Path, name: str) -> None:
    module, cls, resolution, flags, fields = ISOTHERMAL[name]
    script = module.replace(".", "/") + ".py"
    reference = _script_reference(module, cls, resolution, tmp_path / "single", STEPS, fields=fields)
    if fields:
        contact = _script_reference(module, cls, resolution, tmp_path / "contact", STEPS)
        assert _differs(reference, contact), f"{name}: the edge EMF left the refined run unchanged; the gate is vacuous"
    four = _layout(tmp_path / "four.toml", workers=4, cuts=[[13], [18]])
    launched = _script_launch(script, resolution, tmp_path / "four", four, STEPS, workers=4, flags=flags)
    _assert_same_levels(launched, reference, f"{name}: four workers")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    first = _script_launch(script, resolution, tmp_path / "first", two, STEPS // 2, workers=2, flags=flags)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    resumed = _script_launch(script, resolution, tmp_path / "resumed", four, STEPS - written_at, workers=4, checkpoint=first)
    _assert_same_levels(resumed, reference, f"{name}: resumed onto four workers")


# the fine box spans root cells 6..18 on every axis of the 24^3 root; the cuts sit at least four
# root cells from its faces and meet at a corner inside it
RESOLUTION_3D = (24, 24, 24)
CUTS_3D = [[11], [13], [10]]
NMHD_3D = [
    "regime: Annotated[Regime, ProblemParam(Regime.NMHD, description='physics regime')]",
    "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]], description='r')]",
]
THREE_D = {
    "hydro": (("simbi_configs.examples.newtonian.refined_blast_3d", "RefinedBlast3D"), None),
    "nmhd-contact": (("simbi_configs.examples.newtonian.refined_mhd_3d", "RefinedMhd3D"), NMHD_3D),
    "nmhd-uct": (("simbi_configs.examples.newtonian.refined_mhd_3d", "RefinedMhd3D"), NMHD_3D + ["ct_method: Annotated[CtMethod, ProblemParam(CtMethod.UCT, description='edge emf')]"]),
}


@pytest.mark.parametrize("name", list(THREE_D))
def test_a_3d_refined_launch_matches_fresh_and_resumed(tmp_path: Path, name: str, monkeypatch) -> None:
    (module, cls), lines = THREE_D[name]
    script = module.replace(".", "/") + ".py"
    if lines is not None:
        stem = "three_d_" + name.replace("-", "_")
        (tmp_path / f"{stem}.py").write_text(
            "from typing import Annotated\n"
            "from simbi import ProblemParam\n"
            "from simbi.types import CtMethod, Regime\n"
            f"from {module} import {cls}\n\n\n"
            f"class Variant({cls}):\n" + "\n".join(f"    {line}" for line in lines) + "\n"
        )
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
        module, cls, script = stem, "Variant", str(tmp_path / f"{stem}.py")
    reference = _script_reference(module, cls, RESOLUTION_3D, tmp_path / "single", STEPS)
    if name == "nmhd-uct":
        contact = _script_reference(module, cls, RESOLUTION_3D, tmp_path / "contact", STEPS, fields="ct_method='contact'")
        assert _differs(reference, contact), "the edge EMF left the 3d refined run unchanged; the gate is vacuous"
    eight = _layout(tmp_path / "eight.toml", workers=8, cuts=CUTS_3D)
    launched = _script_launch(script, RESOLUTION_3D, tmp_path / "eight", eight, STEPS, workers=8)
    _assert_same_levels(launched, reference, f"3d {name}: eight workers")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1, 1))
    first = _script_launch(script, RESOLUTION_3D, tmp_path / "first", two, STEPS // 2, workers=2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    resumed = _script_launch(script, RESOLUTION_3D, tmp_path / "resumed", eight, STEPS - written_at, workers=8, checkpoint=first)
    _assert_same_levels(resumed, reference, f"3d {name}: resumed onto eight workers")


def test_the_primed_root_reaches_the_halos_before_the_first_stage(tmp_path: Path, monkeypatch) -> None:
    # the priming restricts the seeded fine level into each tile's covered root cells and faces
    # after the prime exchange; a uct edge EMF differences those faces across a cut in the first
    # stage, so a halo left at its pre-restriction value flips the sign of a one-ulp jump there.
    # one z cut through the box, every exchange audited, one step
    (module, cls), lines = THREE_D["nmhd-uct"]
    stem = "primed_uct"
    (tmp_path / f"{stem}.py").write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi.types import CtMethod, Regime\n"
        f"from {module} import {cls}\n\n\n"
        f"class Variant({cls}):\n" + "\n".join(f"    {line}" for line in lines) + "\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    script = str(tmp_path / f"{stem}.py")
    reference = _script_reference(stem, "Variant", RESOLUTION_3D, tmp_path / "single", 1)
    # the premise: the priming changes root faces within the halo band of the cut at z = 10
    unrefined = _script_reference(stem, "Variant", RESOLUTION_3D, tmp_path / "unrefined", 1, fields="refinement_enabled=False")
    initial = lambda out: sorted(out.glob("*000_000*.h5"))[0]
    with h5py.File(initial(tmp_path / "single"), "r") as fa, h5py.File(initial(tmp_path / "unrefined"), "r") as fb:
        magnetic = "level_0/partition_0/hydro/magnetic"
        moved = [k for k in fa[magnetic] if not np.array_equal(fa[f"{magnetic}/{k}/data"][8:12], fb[f"{magnetic}/{k}/data"][8:12])]
    assert moved, "the priming changed no root face near the cut; the gate is vacuous"
    two = _layout(tmp_path / "two.toml", workers=2, cuts=[[], [], [10]])
    launched = _script_launch(script, RESOLUTION_3D, tmp_path / "two", two, 1, workers=2, env={"SIMBI_FABRIC_AUDIT": "all"})
    _assert_same_levels(launched, reference, "one z cut through the box, every exchange audited")


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
