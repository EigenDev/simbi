# =============================================================================
# test_alpha_viscosity.py
#
# the shakura-sunyaev alpha viscosity (nu = alpha c_s^2 / Omega_K) around the
# roche primary:
# - adiabatic: a configured alpha evolves away from the inviscid run on one
#   device. its stability bound multiplies the largest p/rho by the slowest
#   orbit: every tile takes the slowest orbit at the whole grid's farthest
#   corner, and the minimum over the tiles' steps then holds the largest p/rho
#   of any tile, so two tiles and three unequal tiles take the single grid's
#   steps and reach its state bit for bit (the bound binds the step there).
#   refined tiles record every level's whole extent, so two and three unequal
#   refined tiles take the single hierarchy's steps and reach every level of
#   its state bit for bit;
# - isothermal: the bound depends on the orbital radius alone, and two and
#   three unequal host tiles reach the single-grid state bit for bit;
# - refined isothermal: two and three unequal refined tiles reach every level
#   of the single hierarchy's state bit for bit, alpha changing the run;
# - newtonian MHD, an orszag-tang vortex around a gravitating centre: alpha
#   changes the run on one device, and two and three unequal host tiles reach
#   the single grid's state and clock bit for bit;
# - isothermal MHD carries no pressure for the local sound speed: alpha is
#   refused by name before any output.
#
# run: pytest -m simulation simbi/simulation/tests/test_alpha_viscosity.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.simulation

VARIANTS = '''
from typing import Annotated, Optional

from pydantic import computed_field

from simbi import ProblemParam
from simbi.types import Regime, Solver
from simbi.types.bodies import BondedAssembly
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig
from simbi_configs.examples.ibm.roche_passage import RochePassage
from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang


class AlphaRoche(RochePassage):
    viscosity_alpha: Annotated[float, ProblemParam(0.0, description="alpha")]


class AlphaPrimary(AlphaRoche):
    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        return None


class RefinedAlphaPrimary(AlphaPrimary):
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-2.0, 2.0, -2.0, 2.0]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]


class IsothermalAlphaPrimary(AlphaRoche):
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
    # isothermal hydro evaluates the hlle flux
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="hlle")]
    sound_speed: Annotated[float, ProblemParam(0.3, description="sound speed")]

    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        return None

    def initial_primitive_state(self):
        adiabatic = super().initial_primitive_state()

        def rows():
            for row in adiabatic():
                yield tuple(row[:-1])

        return rows


class RefinedIsothermalAlphaPrimary(IsothermalAlphaPrimary):
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[-2.0, 2.0, -2.0, 2.0]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]


class AlphaNewtonianOrszagTang(NewtonianOrszagTang):
    viscosity_alpha: Annotated[float, ProblemParam(0.0, description="alpha")]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # the central mass the alpha law reads its keplerian frequency from
        center = tuple(0.5 * (lo + hi) for lo, hi in self.bounds[:2])
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.GRAVITATIONAL,
                mass=1e-3,
                radius=0.05,
                position=center,
                velocity=(0.0, 0.0),
                gravitational=GravitationalProperties(softening_length=0.05),
            )
        ]


class AlphaIsothermalOrszagTang(IsothermalOrszagTang):
    viscosity_alpha: Annotated[float, ProblemParam(0.1, description="alpha")]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # the central mass the alpha law reads its keplerian frequency from
        center = tuple(0.5 * (lo + hi) for lo, hi in self.bounds[:2])
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.GRAVITATIONAL,
                mass=1e-3,
                radius=0.05,
                position=center,
                velocity=(0.0, 0.0),
                gravitational=GravitationalProperties(softening_length=0.05),
            )
        ]
'''
RESOLUTION = (64, 64)
STEPS = 5


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path


def _run(problem: str, out: Path, gpus: int, extra: str = "", resolution: tuple = RESOLUTION) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}), "
        f"compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)


def _density(out: Path) -> tuple[np.ndarray, float]:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    with h5py.File(files[0], "r") as f:
        level = f["level_0"]
        ng = int(level["mesh"].attrs["halo_width"])
        return level["conserved/den"][()][ng:-ng, ng:-ng], float(level.attrs["time"])


# three tiles cut unequally along x (20, 21 and 23 cells of 64)
UNEQUAL = {3: ", decompose=[[20, 41], []]"}


def _alpha_run(out: Path, gpus: int, alpha: float, problem: str = "AlphaRoche") -> tuple[np.ndarray, float]:
    done = _run(problem, out, gpus, extra=f", viscosity_alpha={alpha!r}" + UNEQUAL.get(gpus, ""))
    assert done.returncode == 0, f"{problem} with alpha = {alpha} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    return _density(out)


@pytest.mark.parametrize("problem", ["AlphaRoche", "IsothermalAlphaPrimary"])
def test_alpha_viscosity_acts_on_one_device(tmp_path: Path, variants: Path, problem: str) -> None:
    viscous, _ = _alpha_run(tmp_path / "viscous", 1, 0.1, problem)
    inviscid, _ = _alpha_run(tmp_path / "inviscid", 1, 0.0, problem)
    assert np.abs(viscous - inviscid).max() > 0.0, "the configured alpha left the run inviscid"


@pytest.mark.parametrize("gpus", [2, 3], ids=["two-tiles", "three-unequal-tiles"])
def test_isothermal_tiles_carry_the_alpha_viscosity(tmp_path: Path, variants: Path, gpus: int) -> None:
    single, t_single = _alpha_run(tmp_path / "single", 1, 0.1, "IsothermalAlphaPrimary")
    tiles, t_tiles = _alpha_run(tmp_path / "tiles", gpus, 0.1, "IsothermalAlphaPrimary")
    assert t_tiles == t_single
    np.testing.assert_array_equal(tiles, single)


@pytest.mark.parametrize("gpus", [2, 3], ids=["two-tiles", "three-unequal-tiles"])
def test_adiabatic_tiles_take_the_single_grid_alpha_steps(tmp_path: Path, variants: Path, gpus: int) -> None:
    single, t_single = _alpha_run(tmp_path / "single", 1, 0.1, "AlphaPrimary")
    _, t_inviscid = _alpha_run(tmp_path / "inviscid", 1, 0.0, "AlphaPrimary")
    assert t_single < t_inviscid, "the alpha bound never limited the step; the gate is vacuous"
    tiles, t_tiles = _alpha_run(tmp_path / "tiles", gpus, 0.1, "AlphaPrimary")
    assert t_tiles == t_single, f"{gpus} tiles reached t = {t_tiles!r} against {t_single!r}"
    np.testing.assert_array_equal(tiles, single)


# the fine patch [-2, 2] spans parent cells 24..40 of 64: three unequal refined tiles (20, 12 and
# 32 cells) cut the patch at 32, clear of its faces by more than the prolongation stencil's reach
REFINED_UNEQUAL = {3: ", decompose=[[20, 32], []]"}


def _levels(out: Path) -> dict[str, np.ndarray]:
    from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return _all_levels(files[0])


@pytest.mark.parametrize("gpus", [2, 3], ids=["two-tiles", "three-unequal-tiles"])
def test_refined_tiles_take_the_single_hierarchy_alpha_steps(tmp_path: Path, variants: Path, gpus: int) -> None:
    def run(out: Path, devices: int, alpha: float) -> dict[str, np.ndarray]:
        done = _run("RefinedAlphaPrimary", out, devices, extra=f", viscosity_alpha={alpha!r}" + REFINED_UNEQUAL.get(devices, ""))
        assert done.returncode == 0, f"refined alpha = {alpha} on {devices} device(s) failed:\n{done.stderr[-3000:]}"
        return _levels(out)

    single, inviscid = run(tmp_path / "single", 1, 0.1), run(tmp_path / "inviscid", 1, 0.0)
    assert "level_1/conserved/den" in single, "the single hierarchy published no fine level; the refined arm is vacuous"
    assert single["level_0@time"] < inviscid["level_0@time"], "the alpha bound never limited the step; the gate is vacuous"
    tiles = run(tmp_path / "tiles", gpus, 0.1)
    assert tiles.keys() == single.keys(), f"datasets {sorted(tiles.keys() ^ single.keys())} differ"
    for name, x in single.items():
        np.testing.assert_array_equal(tiles[name], x, err_msg=f"{gpus} refined tiles: {name}")


def test_alpha_on_isothermal_mhd_is_refused(tmp_path: Path, variants: Path) -> None:
    done = _run("AlphaIsothermalOrszagTang", tmp_path / "imhd", 1, resolution=(32, 32, 1))
    assert done.returncode != 0
    assert "alpha-disk viscosity on the imhd regime" in done.stderr.replace("\n", " "), done.stderr[-1500:]
    assert not list((tmp_path / "imhd").glob("*.h5")), "output was written before the refusal"


@pytest.mark.parametrize("gpus", [2, 3], ids=["two-tiles", "three-unequal-tiles"])
def test_refined_isothermal_tiles_carry_the_single_hierarchy_alpha(tmp_path: Path, variants: Path, gpus: int) -> None:
    def run(out: Path, devices: int, alpha: float) -> dict[str, np.ndarray]:
        done = _run("RefinedIsothermalAlphaPrimary", out, devices, extra=f", viscosity_alpha={alpha!r}" + REFINED_UNEQUAL.get(devices, ""))
        assert done.returncode == 0, f"refined isothermal alpha = {alpha} on {devices} device(s) failed:\n{done.stderr[-3000:]}"
        return _levels(out)

    single, inviscid = run(tmp_path / "single", 1, 0.1), run(tmp_path / "inviscid", 1, 0.0)
    assert "level_1/conserved/den" in single, "the single hierarchy published no fine level; the refined arm is vacuous"
    assert any(not np.array_equal(single[k], inviscid[k]) for k in single if "conserved" in k), "the alpha left the refined run inviscid"
    tiles = run(tmp_path / "tiles", gpus, 0.1)
    assert tiles.keys() == single.keys(), f"datasets {sorted(tiles.keys() ^ single.keys())} differ"
    for name, x in single.items():
        np.testing.assert_array_equal(tiles[name], x, err_msg=f"{gpus} refined isothermal tiles: {name}")


MHD_RESOLUTION = (32, 32, 1)
# three unequal host tiles of the magnetized vortex: 10, 11 and 11 cells along x
MHD_UNEQUAL = {3: ", decompose=[[10, 21], []]"}


@pytest.mark.parametrize("gpus", [2, 3], ids=["two-tiles", "three-unequal-tiles"])
def test_magnetized_tiles_carry_the_single_grid_alpha(tmp_path: Path, variants: Path, gpus: int) -> None:
    def run(out: Path, devices: int, alpha: float) -> dict[str, np.ndarray]:
        done = _run("AlphaNewtonianOrszagTang", out, devices, extra=f", viscosity_alpha={alpha!r}" + MHD_UNEQUAL.get(devices, ""), resolution=MHD_RESOLUTION)
        assert done.returncode == 0, f"nmhd alpha = {alpha} on {devices} device(s) failed:\n{done.stderr[-3000:]}"
        return _levels(out)

    single, inviscid = run(tmp_path / "single", 1, 0.1), run(tmp_path / "inviscid", 1, 0.0)
    assert any(not np.array_equal(single[k], inviscid[k]) for k in single if "conserved" in k), "the alpha left the magnetized run inviscid"
    tiles = run(tmp_path / "tiles", gpus, 0.1)
    assert tiles.keys() == single.keys(), f"datasets {sorted(tiles.keys() ^ single.keys())} differ"
    for name, x in single.items():
        np.testing.assert_array_equal(tiles[name], x, err_msg=f"{gpus} magnetized tiles: {name}")
