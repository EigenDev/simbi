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
# resumed. 3D: the refined spherical blast in newtonian, relativistic and
# isothermal hydro and the extruded orszag-tang vortex in newtonian,
# relativistic and isothermal MHD (periodic, the fine box inside the domain,
# under contact and uct, shown to differ), fresh on eight workers whose ragged cuts meet at a
# corner inside the box and resumed onto them from two. a patch on a periodic face
# of a cut axis is refused by name. a nested magnetized run whose first box spans
# the uncut periodic z axis, its second box inside, launches on two and four
# uneven workers cut through both boxes as the single hierarchy runs; a nested
# magnetized box on a periodic face is refused by name. a nested orszag-tang vortex, whose field
# changes sign across the second box's faces, launches on two and four workers
# cut clear of the second box and cut through it, fresh and resumed, and the 3D
# vortex under two nested boxes on two workers cut through both, as the single
# hierarchy runs. nested levels:
# the refined blast with a second box inside the first, every level compared,
# fresh on two and on four ragged workers and resumed onto four from two; a
# depth whose phase table overflows the wire byte is refused by name before any
# session is built. nested MHD under contact and uct: each nested level's face
# halo prolongation reads its parent's faces from a stencil gathered across the
# parent's cuts at the parent's last stage, same gates. a late rejection on the deepest level (the
# last stage of its last substep in a root step, after every earlier substep of
# that step completed) rolls every worker back, those holding no tile of that
# level included: every published checkpoint equals the clean run capped to the
# accepted timestep on every level, its faces and its clocks.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_refined_numerics.py
# =============================================================================

import os
import re
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
    resumed = _script_launch(script, RESOLUTION, tmp_path / "resumed", four, STEPS, workers=4, checkpoint=first)
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
    resumed = _script_launch(script, KH_RESOLUTION, tmp_path / "resumed", four, STEPS, workers=4, checkpoint=first)
    _assert_same_levels(resumed, reference, "periodic kh: resumed onto four workers")


def test_a_patch_spanning_an_uncut_periodic_axis_launches_as_the_single_grid_runs(tmp_path: Path, monkeypatch) -> None:
    # the kh fine box widened over the whole periodic x axis: with the workers cutting y alone every
    # tile holds the whole x axis and its wrap; with cuts across x the fine level wraps across the
    # workers at the seam
    stem = "spanning_kh"
    (tmp_path / f"{stem}.py").write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        f"from {KH[0]} import {KH[1]}\n\n\n"
        f"class Spanning({KH[1]}):\n"
        "    refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.5, 0.5, -0.375, 0.375]], description='r')]\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    script = str(tmp_path / f"{stem}.py")
    reference = _script_reference(stem, "Spanning", KH_RESOLUTION, tmp_path / "single", STEPS)
    walled = _script_reference(stem, "Spanning", KH_RESOLUTION, tmp_path / "walled", STEPS, fields="boundary_conditions='outflow'")
    assert _differs(reference, walled), "the periodic seam left the spanning kh run unchanged; the wrap is never exercised"
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, cuts=[[], [36]])),
        ("three uneven workers", 3, _layout(tmp_path / "three.toml", workers=3, cuts=[[], [30, 60]])),
        # the middle tile, y cells 20..41, lies wholly under the patch spanning x
        ("a worker's tile wholly under the patch", 3, _layout(tmp_path / "under.toml", workers=3, cuts=[[], [20, 41]])),
        # cuts across the periodic x axis itself: the patch has no face along x, and its level
        # wraps across the workers at the seam as the root does
        ("two workers cut across the periodic axis", 2, _layout(tmp_path / "across.toml", workers=2, cuts=[[26], []])),
        ("four uneven workers cut along both axes", 4, _layout(tmp_path / "both.toml", workers=4, cuts=[[20], [36]])),
    ]:
        launched = _script_launch(script, KH_RESOLUTION, tmp_path / label.replace(" ", "_"), layout, STEPS, workers=workers)
        _assert_same_levels(launched, reference, f"spanning kh: {label}")


@pytest.mark.parametrize("name", ["viscous", "ito2"])
def test_a_patch_spanning_a_cut_periodic_axis_launches_the_corner_reading_operators_as_the_single_grid_runs(tmp_path: Path, monkeypatch, name: str) -> None:
    # the viscous cross derivatives and the continuous tracers' coefficient interpolation read the
    # fine halo cells where the seam meets a coarse-fine face; the workers cut across the seam
    from simbi.simulation.tests.test_decomposed_mhd import _population
    from simbi.simulation.tests.test_refined_decomposed_numerics import SPANNING_KH

    stem = "spanning_" + SPANNING_KH[name][0]
    (tmp_path / f"{stem}.py").write_text(
        "from typing import Annotated\n"
        "from pydantic import computed_field\n"
        "from simbi import ProblemParam\n"
        "from simbi.types import TracerScheme\n"
        f"from {KH[0]} import {KH[1]} as RefinedKelvinHelmholtz\n\n\n"
        + SPANNING_KH[name][1].replace("class Variant(RefinedKelvinHelmholtz):", "class Spanning(RefinedKelvinHelmholtz):\n    refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.5, 0.5, -0.375, 0.375]], description='r')]\n")
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    script = str(tmp_path / f"{stem}.py")
    reference = _script_reference(stem, "Spanning", KH_RESOLUTION, tmp_path / "single", STEPS)
    layout = _layout(tmp_path / "across.toml", workers=2, cuts=[[26], []])
    launched = _script_launch(script, KH_RESOLUTION, tmp_path / "across", layout, STEPS, workers=2)
    _assert_same_levels(launched, reference, f"{name} spanning kh on two workers cut across the periodic axis")
    if name == "ito2":
        want, got = _population(reference, "continuous_tracers"), _population(launched, "continuous_tracers")
        levels = want["owner"].astype(np.uint64) >> np.uint64(56)
        assert 1 in set(levels.tolist()), "no tracer lives on the fine level; the interpolation there is never read"
        assert got.keys() == want.keys(), "the tracer fields differ"
        for field, values in want.items():
            np.testing.assert_array_equal(got[field], values, err_msg=f"launched continuous_tracers/{field}")


def test_a_magnetized_patch_spanning_an_uncut_periodic_axis_launches_as_the_single_grid_runs(tmp_path: Path, monkeypatch) -> None:
    # the vortex box spans the whole periodic z axis with a flow along z through the seam; the
    # workers cut x alone, so the root's parent face stencil reads past the z faces at their
    # periodic image on each worker's own tile
    from simbi.simulation.tests.test_decomposed_mhd import SEAM_FLOW_3D

    module, cls = VORTEX_3D
    stem = "spanning_vortex"
    (tmp_path / f"{stem}.py").write_text(
        f"from {module} import {cls}\n\n\nclass Spanning({cls}):\n" + "\n".join(f"    {line}" for line in SEAM_FLOW_3D) + "\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    script = str(tmp_path / f"{stem}.py")
    reference = _script_reference(stem, "Spanning", RESOLUTION_3D, tmp_path / "single", STEPS)
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, shape=(2, 1, 1))),
        ("three uneven workers", 3, _layout(tmp_path / "three.toml", workers=3, cuts=[[12, 22], [], []])),
        # cuts across the periodic z axis itself: the patch has no face along z, and its level
        # wraps across the workers at the seam as the root does
        ("two workers cut across the periodic axis", 2, _layout(tmp_path / "across.toml", workers=2, cuts=[[], [], [12]])),
        ("four uneven workers cut across it and x", 4, _layout(tmp_path / "both.toml", workers=4, cuts=[[11], [], [9]])),
    ]:
        launched = _script_launch(script, RESOLUTION_3D, tmp_path / label.replace(" ", "_"), layout, STEPS, workers=workers)
        _assert_same_levels(launched, reference, f"spanning vortex: {label}")


# a second box inside the vortex box, clear of the periodic z faces
NESTED_INSIDE_SPAN = [
    "refinement_max_levels: Annotated[int, ProblemParam(3, description='levels')]",
    "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.0, 1.0], [0.375, 0.625, 0.375, 0.625, 0.375, 0.625]], description='r')]",
    "refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratios')]",
]


def _spanning_variant(tmp_path: Path, monkeypatch, stem: str, lines: list[str]) -> str:
    module, cls = VORTEX_3D
    (tmp_path / f"{stem}.py").write_text(
        "from typing import Annotated\nfrom simbi import ProblemParam\n"
        f"from {module} import {cls}\n\n\nclass Spanning({cls}):\n" + "\n".join(f"    {line}" for line in lines) + "\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return str(tmp_path / f"{stem}.py")


def test_a_nested_magnetized_run_spanning_an_uncut_periodic_axis_launches_as_the_single_hierarchy_runs(tmp_path: Path, monkeypatch) -> None:
    # the first box spans the whole periodic z axis with a flow along z through the seam, the second
    # sits inside it clear of the z faces; the workers cut x (two) and x and y (four), through both
    # boxes, leaving z whole on every worker
    from simbi.simulation.tests.test_decomposed_mhd import SEAM_FLOW_3D

    stem = "nested_spanning_vortex"
    script = _spanning_variant(tmp_path, monkeypatch, stem, [*NESTED_INSIDE_SPAN, *SEAM_FLOW_3D])
    reference = _script_reference(stem, "Spanning", RESOLUTION_3D, tmp_path / "single", STEPS)
    with h5py.File(reference, "r") as f:
        assert "level_2" in f, "the single hierarchy published no second refined level"
        h = int(f["level_1"]["mesh"].attrs["halo_width"])
        fine = f["level_1/conserved/den"][()][h:-h, h:-h, h:-h]
    assert np.ptp(fine.mean(axis=(1, 2))) > 0.0, "the first fine level is uniform along z; its wrap carries nothing"
    for label, workers, layout in [
        ("two workers", 2, _layout(tmp_path / "two.toml", workers=2, shape=(2, 1, 1))),
        ("four uneven workers", 4, _layout(tmp_path / "four.toml", workers=4, cuts=[[11], [13], []])),
    ]:
        launched = _script_launch(script, RESOLUTION_3D, tmp_path / label.replace(" ", "_"), layout, STEPS, workers=workers)
        _assert_same_levels(launched, reference, f"nested spanning vortex: {label}")


def test_a_nested_magnetized_patch_on_a_periodic_face_is_refused_on_launch(tmp_path: Path, monkeypatch) -> None:
    # the second box reaches the periodic z face of the domain
    lines = [NESTED_INSIDE_SPAN[0], "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.0, 1.0], [0.375, 0.625, 0.375, 0.625, 0.0, 0.5]], description='r')]", NESTED_INSIDE_SPAN[2]]
    script = _spanning_variant(tmp_path, monkeypatch, "nested_on_face", lines)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", "1",
           "--resolution", ",".join(str(r) for r in RESOLUTION_3D), "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    flat = " ".join((done.stdout + done.stderr).split())
    assert done.returncode != 0, "a nested magnetized patch on a periodic face launched"
    assert "a nested magnetized patch on the periodic face of axis 2 with mesh refinement is outside the distributed first release" in flat, flat[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a refused launch published a checkpoint"


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
    resumed = _script_launch(script, RHD_RESOLUTION, tmp_path / "resumed", four, STEPS, workers=4, checkpoint=first)
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
    resumed = _script_launch(script, resolution, tmp_path / "resumed", four, STEPS, workers=4, checkpoint=first)
    _assert_same_levels(resumed, reference, f"{name}: resumed onto four workers")


# the fine box spans root cells 6..18 on every axis of the 24^3 root; the cuts sit at least four
# root cells from its faces and meet at a corner inside it
RESOLUTION_3D = (24, 24, 24)
CUTS_3D = [[11], [13], [10]]
NMHD_3D = [
    "regime: Annotated[Regime, ProblemParam(Regime.NMHD, description='physics regime')]",
    "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]], description='r')]",
]
UCT_3D = "ct_method: Annotated[CtMethod, ProblemParam(CtMethod.UCT, description='edge emf')]"
# the isothermal closure keeps density and velocity: the pressure column of each state is dropped
ISOTHERMAL_STATE = [
    "sound_speed: Annotated[float, ProblemParam(1.0, description='isothermal sound speed')]",
    "def initial_primitive_state(self):",
    "    state = super().initial_primitive_state()",
    "    gas, faces = (state[0], state[1:]) if isinstance(state, tuple) else (state, ())",
    "    def isothermal():",
    "        for row in gas():",
    "            yield row[:-1]",
    "    return (isothermal, *faces) if faces else isothermal",
]
BLAST_3D = ("simbi_configs.examples.newtonian.refined_blast_3d", "RefinedBlast3D")
VORTEX_3D = ("simbi_configs.examples.newtonian.refined_mhd_3d", "RefinedMhd3D")
INTERIOR_BOX_3D = "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]], description='r')]"


def _regime(name: str) -> str:
    return f"regime: Annotated[Regime, ProblemParam(Regime.{name}, description='physics regime')]"


THREE_D = {
    "hydro": (BLAST_3D, None),
    "rhd": (BLAST_3D, [_regime("RHD")]),
    "isothermal": (BLAST_3D, [_regime("ISOTHERMAL"), *ISOTHERMAL_STATE]),
    "rmhd-contact": (VORTEX_3D, [INTERIOR_BOX_3D]),
    "rmhd-uct": (VORTEX_3D, [INTERIOR_BOX_3D, UCT_3D]),
    "imhd-contact": (VORTEX_3D, [_regime("IMHD"), INTERIOR_BOX_3D, *ISOTHERMAL_STATE]),
    "imhd-uct": (VORTEX_3D, [_regime("IMHD"), INTERIOR_BOX_3D, UCT_3D, *ISOTHERMAL_STATE]),
    "nmhd-contact": (("simbi_configs.examples.newtonian.refined_mhd_3d", "RefinedMhd3D"), NMHD_3D),
    "nmhd-uct": (VORTEX_3D, NMHD_3D + [UCT_3D]),
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
    if name.endswith("-uct"):
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
    resumed = _script_launch(script, RESOLUTION_3D, tmp_path / "resumed", eight, STEPS, workers=8, checkpoint=first)
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


# the second box spans root cells 12..20, level-1 cells 24..40; the cuts keep every patch face at
# least four cells of its parent level from a cut
NESTED_LINES = [
    "refinement_max_levels: Annotated[int, ProblemParam(3, description='levels')]",
    "refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.25, 0.25, -0.25, 0.25], [-0.125, 0.125, -0.125, 0.125]], description='r')]",
    "refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratios')]",
]


def _all_levels(path: Path) -> dict[str, np.ndarray]:
    """every cell and face dataset of every level, the cells trimmed of their halo"""
    out = {}
    with h5py.File(path, "r") as f:
        for level_name in sorted(k for k in f if k.startswith("level_")):
            level = f[level_name]
            ng = int(level["mesh"].attrs["halo_width"])
            names: list[str] = []
            level.visit(lambda name, names=names, level=level: names.append(name) if isinstance(level[name], h5py.Dataset) else None)
            for name in names:
                if name.startswith("mesh/") or name.startswith("partition_0/owned") or "/domain/" in name:
                    continue
                data = level[name][()]
                # the mesh and ownership records are excluded by name above; every remaining
                # array is a field of the grid's own dimension, a 1d grid's included
                if data.ndim == 0:
                    continue
                out[f"{level_name}/{name}"] = data if "/magnetic/" in name else data[tuple(slice(ng, -ng) for _ in range(data.ndim))]
            for key in ("time", "iteration"):
                out[f"{level_name}@{key}"] = np.array(level.attrs[key])
    return out


def _assert_same_all_levels(a: Path, b: Path, label: str) -> None:
    xs, ys = _all_levels(a), _all_levels(b)
    assert xs.keys() == ys.keys(), f"{label}: dataset sets differ: {sorted(xs.keys() ^ ys.keys())}"
    assert any(k.startswith("level_2/") for k in xs), f"{label}: no second refined level was written"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _nested_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base: tuple[str, str] = HYDRO, extra: list[str] | None = None) -> tuple[str, str, str]:
    module, cls = base
    lines = NESTED_LINES + (extra or [])
    (tmp_path / "nested.py").write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi.types import CtMethod\n"
        f"from {module} import {cls}\n\n\n"
        f"class Nested({cls}):\n" + "\n".join(f"    {line}" for line in lines) + "\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return "nested", "Nested", str(tmp_path / "nested.py")


# the nested arms: hydro, and MHD under both edge EMFs (each nested level's face prolongation
# reading its parent's faces from a stencil gathered across the parent's cuts)
NESTED_ARMS = {
    "hydro": (HYDRO, []),
    "mhd-contact": (MHD, []),
    "mhd-uct": (MHD, [UCT_3D]),
}


@pytest.mark.parametrize("arm", list(NESTED_ARMS))
def test_nested_levels_on_a_refined_launch_match_fresh_and_resumed(tmp_path: Path, monkeypatch, arm: str) -> None:
    base, extra = NESTED_ARMS[arm]
    # the second box spans root cells 12..20 on each axis inside the first (8..24); every layout
    # cuts through both boxes
    two_cuts, four_cuts = {"shape": (2, 1)}, {"cuts": [[15], [17]]}
    module, cls, script = _nested_module(tmp_path, monkeypatch, base=base, extra=extra)
    reference = _script_reference(module, cls, RESOLUTION, tmp_path / "single", STEPS)
    if arm == "mhd-uct":
        contact = _script_reference(module, cls, RESOLUTION, tmp_path / "contact", STEPS, fields="ct_method='contact'")
        assert _differs(reference, contact), "the edge EMF left the nested run unchanged; the gate is vacuous"
    for tag, workers, kwargs in [("two", 2, two_cuts), ("four", 4, four_cuts)]:
        layout = _layout(tmp_path / f"{tag}.toml", workers=workers, **kwargs)
        launched = _script_launch(script, RESOLUTION, tmp_path / tag, layout, STEPS, workers=workers)
        _assert_same_all_levels(launched, reference, f"three levels on {workers} workers")
    two = _layout(tmp_path / "two_first.toml", workers=2, **two_cuts)
    first = _script_launch(script, RESOLUTION, tmp_path / "first", two, STEPS // 2, workers=2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
        assert "level_2" in f, "the half-run checkpoint carries no second refined level"
    assert 0 < written_at < STEPS
    four = _layout(tmp_path / "four_resumed.toml", workers=4, **four_cuts)
    resumed = _script_launch(script, RESOLUTION, tmp_path / "resumed", four, STEPS, workers=4, checkpoint=first)
    _assert_same_all_levels(resumed, reference, "three levels resumed onto four workers")


# the orszag-tang vortex under two nested boxes: its field changes sign across the second box's
# faces, so the nested coarse-fine corrections there carry a current a uniform field would not
OT_NESTED = [
    "refinement_enabled: Annotated[bool, ProblemParam(True, description='refined')]",
    "refinement_max_levels: Annotated[int, ProblemParam(3, description='levels')]",
    "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75], [0.375, 0.625, 0.375, 0.625]], description='r')]",
    "refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratios')]",
]
OT = ("simbi_configs.examples.newtonian.nmhd_orszag_tang", "NewtonianOrszagTang")


def _ot_nested(tmp_path: Path, monkeypatch) -> str:
    module, cls = OT
    (tmp_path / "ot_nested.py").write_text(
        "from typing import Annotated\nfrom simbi import ProblemParam\n"
        f"from {module} import {cls}\n\n\nclass OtNested({cls}):\n" + "\n".join(f"    {line}" for line in OT_NESTED) + "\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return str(tmp_path / "ot_nested.py")


def test_a_nested_vortex_launches_as_the_single_hierarchy_runs_with_cuts_clear_of_the_second_box(tmp_path: Path, monkeypatch) -> None:
    script = _ot_nested(tmp_path, monkeypatch)
    reference = _script_reference("ot_nested", "OtNested", RESOLUTION, tmp_path / "single", STEPS)
    for tag, workers, kwargs in [("two", 2, {"cuts": [[10], []]}), ("four", 4, {"cuts": [[10], [22]]})]:
        layout = _layout(tmp_path / f"{tag}.toml", workers=workers, **kwargs)
        launched = _script_launch(script, RESOLUTION, tmp_path / tag, layout, STEPS, workers=workers)
        _assert_same_all_levels(launched, reference, f"nested vortex on {workers} workers")


def test_a_nested_vortex_cut_through_the_second_box_launches_as_the_single_hierarchy_runs(tmp_path: Path, monkeypatch) -> None:
    # the second box's faces beside a cut prolong from the first level's faces and cell field
    # past the cut, which cross with the first level's conserved halos before the seed; a seed
    # from the conserved halos alone left the tangential field beside the cut on its stale
    # values (4.4e-3 on the second level's B_y after one step of the vortex field at rest)
    script = _ot_nested(tmp_path, monkeypatch)
    reference = _script_reference("ot_nested", "OtNested", RESOLUTION, tmp_path / "single", STEPS)
    for tag, workers, kwargs in [("two", 2, {"shape": (2, 1)}), ("four", 4, {"cuts": [[15], [17]]})]:
        layout = _layout(tmp_path / f"{tag}.toml", workers=workers, **kwargs)
        launched = _script_launch(script, RESOLUTION, tmp_path / tag, layout, STEPS, workers=workers)
        _assert_same_all_levels(launched, reference, f"nested vortex cut through the second box on {workers} workers")
    two = _layout(tmp_path / "two_first.toml", workers=2, shape=(2, 1))
    first = _script_launch(script, RESOLUTION, tmp_path / "first", two, STEPS // 2, workers=2)
    four = _layout(tmp_path / "four_resumed.toml", workers=4, cuts=[[15], [17]])
    resumed = _script_launch(script, RESOLUTION, tmp_path / "resumed", four, STEPS, workers=4, checkpoint=first)
    _assert_same_all_levels(resumed, reference, "nested vortex resumed onto four workers cut through the second box")


def test_a_3d_nested_vortex_cut_through_the_second_box_launches_as_the_single_hierarchy_runs(tmp_path: Path, monkeypatch) -> None:
    # the extruded vortex under a box over the central half and a second box inside it; the x cut
    # at root cell 12 passes through both
    module, cls = VORTEX_3D
    (tmp_path / "nested3d.py").write_text(
        "from typing import Annotated\nfrom simbi import ProblemParam\n"
        f"from {module} import {cls}\n\n\nclass Nested3d({cls}):\n"
        "    refinement_max_levels: Annotated[int, ProblemParam(3, description='levels')]\n"
        "    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75], [0.375, 0.625, 0.375, 0.625, 0.375, 0.625]], description='r')]\n"
        "    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratios')]\n"
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    reference = _script_reference("nested3d", "Nested3d", RESOLUTION_3D, tmp_path / "single", STEPS)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1, 1))
    launched = _script_launch(str(tmp_path / "nested3d.py"), RESOLUTION_3D, tmp_path / "two", layout, STEPS, workers=2)
    _assert_same_all_levels(launched, reference, "3d nested vortex cut through the second box on two workers")


def _refused_launch(tmp_path: Path, script: str, reason: str) -> None:
    layout = _layout(tmp_path / "two_refused.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", "1",
           "--resolution", "32,32,1", "--data-directory", str(tmp_path / "refused"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert reason in flat, done.stderr
    assert not list((tmp_path / "refused").glob("*.h5")), "a checkpoint was published"


def test_a_depth_past_the_phase_byte_is_refused_by_name(tmp_path: Path, monkeypatch) -> None:
    regions = [[-0.25 / 2**k, 0.25 / 2**k, -0.25 / 2**k, 0.25 / 2**k] for k in range(5)]
    extra = [
        f"refinement_max_levels: Annotated[int, ProblemParam(6, description='levels')]",
        f"refinement_regions: Annotated[list[list[float]], ProblemParam({regions!r}, description='r')]",
        "refinement_ratios: Annotated[list[int], ProblemParam([2, 2, 2, 2, 2], description='ratios')]",
    ]
    _, _, script = _nested_module(tmp_path, monkeypatch, extra=extra)
    _refused_launch(tmp_path, script, "6 levels, the root included (phase table of")


# the deepest box spans root cells 16..20 on both axes, inside the root tile above both cuts at 12:
# three of the four workers hold no tile of level 2
LATE_LINES = [
    "refinement_max_levels: Annotated[int, ProblemParam(3, description='levels')]",
    "refinement_regions: Annotated[list[list[float]], ProblemParam([[-0.25, 0.25, -0.25, 0.25], [0.0, 0.125, 0.0, 0.125]], description='r')]",
    "refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratios')]",
]
LATE_STEP = 2
# level 2 takes four substeps per root step; the last stage of the last one (rk2)
LATE_REJECTION = f"reject_level:3:{LATE_STEP}:2:3:1"
REJECTIONS = re.compile(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)")




def _per_step_launch(script: str, out: Path, layout: Path, workers: int, inject: str | None, steps: int = STEPS) -> tuple[dict[int, Path], set[int]]:
    """the run with a checkpoint at every step's time, by iteration, the final file under the key
    -1, and every worker's rejection count. checkpoints are named by their time to three decimals,
    so two steps whose times round alike publish one file: the iterations a run keeps depend on its
    timesteps, and a comparison takes those both runs kept"""
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out), "--checkpoint-interval", "1e-9"]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    files: dict[int, Path] = {}
    for path in out.glob("*.h5"):
        if "final" in path.name:
            files[-1] = path
            continue
        with h5py.File(path, "r") as f:
            files[int(f["level_0"].attrs["iteration"])] = path
    assert -1 in files, "no final checkpoint was published"
    return files, {int(r) for r in REJECTIONS.findall(done.stderr)}


def _assert_same_published(a: Path, b: Path, label: str) -> None:
    xs, ys = _all_levels(a), _all_levels(b)
    assert xs.keys() == ys.keys(), f"{label}: dataset sets differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for level in (k for k in fa if k.startswith("level_")):
            assert fa[level].attrs["dt"] == fb[level].attrs["dt"], f"{label}: {level} dt"


@pytest.mark.parametrize("arm", list(NESTED_ARMS))
def test_a_late_rejection_on_the_deepest_level_rolls_every_worker_back(tmp_path: Path, monkeypatch, arm: str) -> None:
    base, extra = NESTED_ARMS[arm]
    _, _, script = _nested_module(tmp_path, monkeypatch, base=base, extra=LATE_LINES + extra)
    four = _layout(tmp_path / "four.toml", workers=4, cuts=[[12], [12]])
    # the rejected step's accepted timestep, from the run stopped right after it
    stopped, counts = _per_step_launch(script, tmp_path / "stopped", four, 4, LATE_REJECTION, steps=LATE_STEP + 1)
    assert counts == {1}, f"the deepest-level rejection was not counted once on every worker: {counts}"
    with h5py.File(stopped[-1], "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == LATE_STEP + 1
        accepted = float(f["level_0"].attrs["dt"])
        assert "level_2" in f, "the rejecting run wrote no second refined level"
    cap = f"cap:{LATE_STEP}:{int(np.float64(accepted).view(np.uint64))}"
    # the cap touches the capped step's timestep alone: stopped right after it, the capped control
    # took exactly the cap, below the clean timestep, and equals the rejecting run there
    capped_stop, capped_counts = _per_step_launch(script, tmp_path / "capped_stop", four, 4, cap, steps=LATE_STEP + 1)
    clean_stop, _ = _per_step_launch(script, tmp_path / "clean_stop", four, 4, None, steps=LATE_STEP + 1)
    assert capped_counts == {0}, "the capped control rejected a step"
    with h5py.File(capped_stop[-1], "r") as fp, h5py.File(clean_stop[-1], "r") as fc:
        assert float(fp["level_0"].attrs["dt"]) == accepted, "the capped step took another timestep than the cap"
        assert accepted < float(fc["level_0"].attrs["dt"]), "the cap is no smaller than the clean timestep"
    _assert_same_published(stopped[-1], capped_stop[-1], "stopped after the rejected step")
    # the full runs: every checkpoint both published, and the final one
    rejecting, counts = _per_step_launch(script, tmp_path / "rejecting", four, 4, LATE_REJECTION)
    assert counts == {1}, f"the deepest-level rejection was not counted once on every worker: {counts}"
    capped, capped_counts = _per_step_launch(script, tmp_path / "capped", four, 4, cap)
    clean, _ = _per_step_launch(script, tmp_path / "clean", four, 4, None)
    assert capped_counts == {0}, "the capped control rejected a step"
    assert _differs(clean[-1], capped[-1]), "the cap left the run unchanged; the rejection never reduced the timestep"
    shared = sorted(set(rejecting) & set(capped))
    assert -1 in shared and any(i > LATE_STEP for i in shared), f"the runs share no checkpoint after the rejected step: {shared}"
    for iteration in shared:
        _assert_same_published(rejecting[iteration], capped[iteration], f"iteration {iteration}")
    # before the capped step, the cap leaves the run alone
    before = [i for i in sorted(set(clean) & set(capped)) if 0 <= i <= LATE_STEP]
    assert before, "no checkpoint before the capped step was published by both controls"
    for iteration in before:
        _assert_same_published(clean[iteration], capped[iteration], f"the cap changed iteration {iteration}")
