# =============================================================================
# test_isothermal_tile_settings.py
#
# the settings an isothermal run carries on one device and on host tiles:
# - a passive scalar: the single grid publishes the transported dye, and two
#   tiles publish the single grid's dye and gas bit for bit;
# - a shaped rigid wall and a bonded fragment cluster: two tiles reach the
#   single grid's state (the wall bit for bit; the cluster within the
#   reassociation bound of its cross-tile force sum);
# - neumann and robin walls, on the isothermal and the adiabatic tiles alike:
#   two tiles reach the single grid's state bit for bit;
# - a locally isothermal closure on isothermal MHD tiles, periodic and with
#   outflow faces: two tiles reach the single grid's state bit for bit, and the
#   closure changes the run.
# a declared perturbation on isothermal tiles is gated with the other drivers
# in test_tiled_perturbation.
# every arm runs as a fresh process, the tiled ones as host tiles on a cpu
# build. the isothermal wind-tunnel and rubble arms are the adiabatic examples
# with the pressure dropped from their rows and their inflow.
#
# run: pytest -m simulation simbi/simulation/tests/test_isothermal_tile_settings.py
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
import importlib
import math
from functools import partial
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi.types import BoundaryCondition, Regime, Solver
from simbi_configs.examples.ibm.rubble_wind import RubbleWind
from simbi_configs.examples.ibm.wind_tunnel import WindTunnel
from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang
from simbi_configs.examples.isothermal.isothermal_channel import IsothermalChannel
from simbi_configs.examples.isothermal.isothermal_gradient_box import IsothermalGradientBox
from simbi_configs.examples.newtonian.gradient_box import GradientBox


def _isothermal(base):
    class Isothermal(base):
        regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
        # isothermal hydro evaluates the hlle flux
        solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="hlle")]
        sound_speed: Annotated[float, ProblemParam(3.0, description="sound speed")]

        @computed_field
        @property
        def bx1_inner_expressions(self) -> dict:
            freestream = importlib.import_module(base.__module__).RHO_INF
            return expr.boundary([freestream, self.freestream_velocity(), 0.0], dim=2)

        def initial_primitive_state(self):
            adiabatic = super().initial_primitive_state()

            def rows():
                for row in adiabatic():
                    yield tuple(row[:-1])

            return rows

    return Isothermal


IsoWindTunnel = _isothermal(WindTunnel)
IsoRubbleWind = _isothermal(RubbleWind)


class DyedChannel(IsothermalChannel):
    def passive_scalar(self):
        def dye():
            ni, nj = self.resolution
            for _jj in range(nj):
                for ii in range(ni):
                    yield 1.0 if ii < ni // 2 else 0.0

        return dye()


class LocallyIsothermalOrszagTang(IsothermalOrszagTang):
    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        return True

    def initial_primitive_state(self):
        gas, *faces = super().initial_primitive_state()

        def rows():
            for row in gas():
                yield (*row, row[0] * (1.0 + 0.1 * math.sin(row[1])))

        return (rows, *faces)


class OutflowLocallyIsothermalOrszagTang(LocallyIsothermalOrszagTang):
    """the closure on a box with outflow faces: the ghosts continue it from the nearest cell"""

    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="outflow")
    ]
'''


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path


def _run(problem: str, resolution: tuple, out: Path, gpus: int, steps: int) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)


def _final(problem: str, resolution: tuple, out: Path, gpus: int, steps: int = 5) -> Path:
    done = _run(problem, resolution, out, gpus, steps)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _cells(path: Path) -> dict[str, np.ndarray]:
    """every level-0 cell dataset over the interior, by name"""
    out = {}
    with h5py.File(path, "r") as f:
        level = f["level_0"]
        ng = int(level["mesh"].attrs["halo_width"])
        names: list[str] = []
        level.visit(lambda name: names.append(name) if isinstance(level[name], h5py.Dataset) else None)
        for name in names:
            data = level[name][()]
            if (name.startswith("conserved/") or name.startswith("partition_0/hydro/primitives/")) and data.ndim == 2:
                out[name] = data[ng:-ng, ng:-ng]
        out["clock"] = np.array([level.attrs["time"], level.attrs["iteration"]])
    return out


def _assert_bitwise(got: Path, want: Path, label: str) -> None:
    xs, ys = _cells(got), _cells(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(xs)} against {sorted(ys)}"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def test_an_isothermal_dye_is_transported_on_one_device_and_on_tiles(tmp_path: Path, variants: Path) -> None:
    resolution = (32, 32)
    single = _final("DyedChannel", resolution, tmp_path / "single", 1)
    cells = _cells(single)
    chi = [name for name in cells if name.endswith("chi")]
    assert "conserved/chi" in chi, f"the isothermal single grid published no dye: {sorted(cells)}"
    seeded = np.where(np.arange(resolution[0]) < resolution[0] // 2, 1.0, 0.0)[None, :]
    concentration = next(cells[name] for name in chi if name.startswith("partition_0"))
    assert not np.array_equal(concentration, np.broadcast_to(seeded, concentration.shape)), "the dye never moved"
    _assert_bitwise(_final("DyedChannel", resolution, tmp_path / "tiles", 2), single, "two isothermal tiles carrying a dye")


def test_isothermal_tiles_carry_a_shaped_wall(tmp_path: Path, variants: Path) -> None:
    single = _final("IsoWindTunnel", (64, 32), tmp_path / "single", 1)
    _assert_bitwise(_final("IsoWindTunnel", (64, 32), tmp_path / "tiles", 2), single, "two isothermal tiles around a shaped wall")


def test_isothermal_tiles_carry_a_bonded_cluster(tmp_path: Path, variants: Path) -> None:
    want = _cells(_final("IsoRubbleWind", (64, 32), tmp_path / "single", 1))
    got = _cells(_final("IsoRubbleWind", (64, 32), tmp_path / "tiles", 2))
    assert got.keys() == want.keys()
    np.testing.assert_array_equal(got.pop("clock"), want.pop("clock"))
    # the fragments' hydrodynamic force is a sum of two tile partials against the single grid's
    # one sum over the same cells: the reassociation moves it by at most n u sum|terms| (n = 2048
    # cells, u = 2^-53, about 2.3e-13 of the summed magnitude), and the fragment kinematics carry
    # that into the gas over five steps; 1e-12 of each field's scale bounds it.
    for name, x in want.items():
        scale = max(np.abs(x).max(), np.abs(got[name]).max(), 1e-300)
        worst = np.abs(got[name] - x).max()
        assert worst <= 1e-12 * scale, f"{name}: two tiles differ by {worst:.3e} against scale {scale:.3e}"


@pytest.mark.parametrize("problem", ["IsothermalGradientBox", "GradientBox"])
def test_tiles_carry_neumann_and_robin_walls(tmp_path: Path, variants: Path, problem: str) -> None:
    single = _final(problem, (32, 32), tmp_path / "single", 1, steps=4)
    _assert_bitwise(_final(problem, (32, 32), tmp_path / "tiles", 2, steps=4), single, f"{problem} on two tiles")


@pytest.mark.parametrize("problem", ["LocallyIsothermalOrszagTang", "OutflowLocallyIsothermalOrszagTang"])
def test_isothermal_mhd_tiles_carry_the_local_closure(tmp_path: Path, variants: Path, problem: str) -> None:
    resolution = (32, 32, 1)
    single = _final(problem, resolution, tmp_path / "single", 1)
    uniform = _cells(_final("IsothermalOrszagTang", resolution, tmp_path / "uniform", 1))
    assert not np.array_equal(_cells(single)["conserved/den"], uniform["conserved/den"]), f"{problem}: the closure left the run unchanged"
    _assert_bitwise(_final(problem, resolution, tmp_path / "tiles", 2), single, f"{problem} on two tiles")
