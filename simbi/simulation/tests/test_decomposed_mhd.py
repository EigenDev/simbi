# =============================================================================
# test_decomposed_mhd.py
#
# magnetized gas on host tiles against the single grid. the newtonian,
# relativistic and isothermal orszag-tang vortices, whose field varies along
# both axes, run on three uneven tiles along x and on six uneven tiles cut along
# both axes, and publish the single grid's gas, cell field and face field bit
# for bit. the refined newtonian and relativistic magnetized blasts and the
# refined isothermal orszag-tang vortex, each with one fine box, run on two
# tiles, three uneven tiles and six tiles cut along both axes through the box,
# fresh and resumed from a checkpoint written on two tiles, and publish the
# single hierarchy's every level bit for bit. the refined 3d vortex (newtonian
# and isothermal under the contact EMF, relativistic under the upwind EMF) runs
# on eight tiles, and the isothermal vortex with a second nested box on two and
# four tiles, both cut through the boxes, every level bit for bit. the refined
# blast carrying a discrete or a second-order ito population on two and three
# uneven tiles publishes the single hierarchy's population as well. a locally
# isothermal refined run on tiles is refused by name before any output.
#
# run: pytest -m simulation simbi/simulation/tests/test_decomposed_mhd.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 4
VORTICES = {
    "NewtonianOrszagTang": "simbi_configs.examples.newtonian.nmhd_orszag_tang",
    "IsothermalOrszagTang": "simbi_configs.examples.isothermal.imhd_orszag_tang",
    "OrszagTang": "simbi_configs.examples.srmhd.rmhd_orszag_tang",
}
# uneven tiles of the 32 x 32 grid: three along x, and three by two
UNEVEN = {3: [[10, 21], []], 6: [[10, 21], [13]]}


REFINED = {
    "RefinedMhdBlast": "simbi_configs.examples.newtonian.refined_mhd_blast",
    "RefinedRmhdBlast": "simbi_configs.examples.srmhd.refined_rmhd_blast",
    "RefinedIsothermalOrszagTang": "simbi_configs.examples.isothermal.refined_imhd_orszag_tang",
}
# the fine box covers root cells 8..24 of 32 on both axes; every cut lies four or more cells from
# its faces
REFINED_LAYOUTS = {2: None, 3: [[12, 20], []], 6: [[12, 20], [14]]}


def _raw(problem: str, out: Path, gpus: int, decompose=None, steps: int = STEPS, checkpoint: Path | None = None) -> subprocess.CompletedProcess:
    extra = f", decompose={decompose!r}" if decompose is not None else ""
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    module = {**VORTICES, **REFINED}[problem]
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {problem}\n"
        f"runner.run({problem}(resolution=(32, 32, 1), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}{resume}), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)


def _run(problem: str, out: Path, gpus: int, decompose=None, steps: int = STEPS, checkpoint: Path | None = None) -> Path:
    done = _raw(problem, out, gpus, decompose, steps, checkpoint)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    levels = sorted({name.split("/")[0] for name in ys if name.startswith("level_") and "/" in name})
    for level in levels:
        faces = [name for name in ys if name.startswith(f"{level}/") and "/magnetic/B" in name]
        assert len(faces) >= 2 and f"{level}/conserved/b1" in ys, f"{label}: {level} publishes no face and cell field: {sorted(ys)}"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("problem", list(VORTICES))
@pytest.mark.parametrize("gpus", list(UNEVEN))
def test_uneven_tiles_carry_the_single_grid_vortex(tmp_path: Path, problem: str, gpus: int) -> None:
    want = _run(problem, tmp_path / "single", 1)
    _assert_same(_run(problem, tmp_path / "tiles", gpus, UNEVEN[gpus]), want, f"{problem} on {gpus} uneven tiles")


@pytest.mark.parametrize("problem", list(REFINED))
def test_refined_tiles_carry_the_single_hierarchy(tmp_path: Path, problem: str) -> None:
    want = _run(problem, tmp_path / "single", 1)
    levels = {name.split("/")[0] for name in _all_levels(want)}
    assert "level_1" in levels, f"{problem}: the single hierarchy published no fine level: {sorted(levels)}"
    for gpus, decompose in REFINED_LAYOUTS.items():
        _assert_same(_run(problem, tmp_path / f"tiles{gpus}", gpus, decompose), want, f"{problem} on {gpus} refined tiles")


def test_refined_tiles_resume_onto_another_partition(tmp_path: Path) -> None:
    want = _run("RefinedMhdBlast", tmp_path / "single", 1)
    first = _run("RefinedMhdBlast", tmp_path / "first", 2, steps=STEPS // 2)
    # a resumed run on tiles takes `max_steps` further steps
    resumed = _run("RefinedMhdBlast", tmp_path / "resumed", 3, REFINED_LAYOUTS[3], steps=STEPS - STEPS // 2, checkpoint=first)
    _assert_same(resumed, want, "resumed onto three uneven refined tiles")


def test_a_locally_isothermal_refined_run_on_tiles_is_refused(tmp_path: Path) -> None:
    script = tmp_path / "local.py"
    script.write_text(
        "from typing import Annotated\n"
        "from pydantic import computed_field\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.isothermal.refined_imhd_orszag_tang import RefinedIsothermalOrszagTang\n\n\n"
        "class LocalRefined(RefinedIsothermalOrszagTang):\n"
        "    @computed_field\n"
        "    @property\n"
        "    def locally_isothermal(self) -> bool:\n"
        "        return True\n\n"
        "    def initial_primitive_state(self):\n"
        "        gas, *fields = super().initial_primitive_state()\n\n"
        "        def local():\n"
        "            for row in gas():\n"
        "                yield (*row, 1.0)\n\n"
        "        return (local, *fields)\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1", PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    out = tmp_path / "out"
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from local import LocalRefined\n"
        "runner.run(LocalRefined(resolution=(32, 32, 1), gpus=2, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode='cpu', max_steps=1)\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode != 0, "refined tiles ran a locally isothermal closure"
    flat = " ".join(done.stderr.split())
    assert "a locally isothermal closure with mesh refinement and gpus > 1 awaits the whole-patch closure" in flat, done.stderr[-1500:]
    assert not list(out.glob("*.h5")), "output was written before the refusal"


VORTEX_3D = "from simbi_configs.examples.newtonian.refined_mhd_3d import RefinedMhd3D as Base"
INTERIOR_BOX_3D = "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]], description='r')]"
ISOTHERMAL_STATE = [
    "sound_speed: Annotated[float, ProblemParam(1.0, description='isothermal sound speed')]",
    "def initial_primitive_state(self):",
    "    gas, *faces = super().initial_primitive_state()",
    "    def isothermal():",
    "        for row in gas():",
    "            yield row[:-1]",
    "    return (isothermal, *faces)",
]
VARIANTS_3D = {
    "nmhd-contact": [INTERIOR_BOX_3D, "regime: Annotated[Regime, ProblemParam(Regime.NMHD, description='r')]"],
    "imhd-contact": [INTERIOR_BOX_3D, "regime: Annotated[Regime, ProblemParam(Regime.IMHD, description='r')]", *ISOTHERMAL_STATE],
    "rmhd-uct": [INTERIOR_BOX_3D, "ct_method: Annotated[CtMethod, ProblemParam(CtMethod.UCT, description='e')]"],
}
NESTED = [
    "refinement_max_levels: Annotated[int, ProblemParam(3, description='levels')]",
    "refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75], [0.375, 0.625, 0.375, 0.625]], description='r')]",
    "refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratios')]",
]


def _variant(tmp_path: Path, base: str, lines: list[str]) -> Path:
    (tmp_path / "variant.py").write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi.types import CtMethod, Regime\n"
        f"{base}\n\n\n"
        "class Variant(Base):\n" + "\n".join(f"    {line}" for line in lines) + "\n"
    )
    return tmp_path


def _run_variant(where: Path, out: Path, resolution: tuple[int, ...], gpus: int, decompose=None) -> Path:
    extra = f", decompose={decompose!r}" if decompose is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from variant import Variant\n"
        f"runner.run(Variant(resolution={resolution!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}), "
        f"compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1", PYTHONPATH=os.pathsep.join([str(where), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=1800, env=env)
    assert done.returncode == 0, f"variant on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


@pytest.mark.parametrize("name", list(VARIANTS_3D))
def test_refined_3d_tiles_carry_the_single_hierarchy(tmp_path: Path, name: str) -> None:
    where = _variant(tmp_path, VORTEX_3D, VARIANTS_3D[name])
    resolution = (24, 24, 24)
    want = _run_variant(where, tmp_path / "single", resolution, 1)
    got = _run_variant(where, tmp_path / "tiles", resolution, 8, [[11], [13], [10]])
    _assert_same(got, want, f"3d {name} on eight refined tiles")


def test_nested_refined_tiles_carry_the_single_hierarchy(tmp_path: Path) -> None:
    where = _variant(tmp_path, "from simbi_configs.examples.isothermal.refined_imhd_orszag_tang import RefinedIsothermalOrszagTang as Base", NESTED)
    resolution = (32, 32, 1)
    want = _run_variant(where, tmp_path / "single", resolution, 1)
    assert "level_2" in {name.split("/")[0] for name in _all_levels(want)}, "the single hierarchy published no second refined level"
    for gpus, decompose in ((2, None), (4, [[15], [17]])):
        got = _run_variant(where, tmp_path / f"tiles{gpus}", resolution, gpus, decompose)
        _assert_same(got, want, f"nested isothermal vortex on {gpus} refined tiles")


TRACED = {
    "discrete": ("tracers", []),
    "ito2": ("continuous_tracers", ["@computed_field", "@property", "def tracer_scheme(self) -> TracerScheme:", "    return TracerScheme.ITO2"]),
}


def _population(path: Path, group: str) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        g = f[group]
        order = np.argsort(g["id"][:], kind="stable")
        out = {f"attr/{k}": np.asarray(v) for k, v in g.attrs.items()}
        for name, ds in g.items():
            data = ds[:]
            out[name] = data[order] if data.shape[:1] == (len(order),) else data
    return out


@pytest.mark.parametrize("scheme", list(TRACED))
def test_refined_magnetized_tiles_carry_the_single_hierarchy_population(tmp_path: Path, scheme: str) -> None:
    group, lines = TRACED[scheme]
    (tmp_path / "variant.py").write_text(
        "from pydantic import computed_field\n"
        "from simbi.types import TracerScheme\n"
        "from simbi_configs.examples.newtonian.refined_mhd_blast import RefinedMhdBlast as Base\n\n\n"
        "class Variant(Base):\n"
        "    @computed_field\n"
        "    @property\n"
        "    def n_tracers(self) -> int:\n"
        "        return 400\n"
        + "".join(f"\n    {line}" for line in lines)
        + "\n"
    )
    resolution = (32, 32, 1)
    want = _run_variant(tmp_path, tmp_path / "single", resolution, 1)
    population = _population(want, group)
    levels = population["owner"].astype(np.uint64) >> np.uint64(56)
    assert {0, 1} <= set(levels.tolist()), f"{scheme}: the records do not span both levels; the comparison is vacuous"
    for gpus, decompose in ((2, None), (3, [[12, 20], []])):
        got = _run_variant(tmp_path, tmp_path / f"tiles{gpus}", resolution, gpus, decompose)
        _assert_same(got, want, f"traced blast ({scheme}) on {gpus} refined tiles")
        tiled = _population(got, group)
        assert tiled.keys() == population.keys(), f"{scheme}: tracer fields differ"
        for name, values in population.items():
            np.testing.assert_array_equal(tiled[name], values, err_msg=f"{scheme} on {gpus} tiles: {group}/{name}")
