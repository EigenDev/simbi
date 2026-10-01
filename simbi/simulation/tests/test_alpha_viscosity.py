# =============================================================================
# test_alpha_viscosity.py
#
# the shakura-sunyaev alpha viscosity (nu = alpha c_s^2 / Omega_K) around the
# roche primary:
# - adiabatic: a configured alpha evolves away from the inviscid run on one
#   device; on several devices it is refused by name, since its stability bound
#   multiplies whole-grid maxima of p/rho and of the orbital radius that a
#   tile's own maxima do not reproduce;
# - isothermal: the bound depends on the orbital radius alone, and two host
#   tiles reach the single-grid state bit for bit;
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
from simbi.types import Regime
from simbi.types.bodies import BondedAssembly
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig
from simbi_configs.examples.ibm.roche_passage import RochePassage
from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang


class AlphaRoche(RochePassage):
    viscosity_alpha: Annotated[float, ProblemParam(0.0, description="alpha")]


class IsothermalAlphaPrimary(AlphaRoche):
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
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


def _alpha_run(out: Path, gpus: int, alpha: float, problem: str = "AlphaRoche") -> tuple[np.ndarray, float]:
    done = _run(problem, out, gpus, extra=f", viscosity_alpha={alpha!r}")
    assert done.returncode == 0, f"{problem} with alpha = {alpha} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    return _density(out)


@pytest.mark.parametrize("problem", ["AlphaRoche", "IsothermalAlphaPrimary"])
def test_alpha_viscosity_acts_on_one_device(tmp_path: Path, variants: Path, problem: str) -> None:
    viscous, _ = _alpha_run(tmp_path / "viscous", 1, 0.1, problem)
    inviscid, _ = _alpha_run(tmp_path / "inviscid", 1, 0.0, problem)
    assert np.abs(viscous - inviscid).max() > 0.0, "the configured alpha left the run inviscid"


def test_two_isothermal_tiles_carry_the_alpha_viscosity(tmp_path: Path, variants: Path) -> None:
    single, t_single = _alpha_run(tmp_path / "single", 1, 0.1, "IsothermalAlphaPrimary")
    tiles, t_tiles = _alpha_run(tmp_path / "tiles", 2, 0.1, "IsothermalAlphaPrimary")
    assert t_tiles == t_single
    np.testing.assert_array_equal(tiles, single)


def test_adiabatic_alpha_on_two_tiles_is_refused(tmp_path: Path, variants: Path) -> None:
    done = _run("AlphaRoche", tmp_path / "tiles", 2, extra=", viscosity_alpha=0.1")
    assert done.returncode != 0
    assert "alpha-disk viscosity on the newtonian regime with several devices" in done.stderr.replace("\n", " "), done.stderr[-1500:]
    assert not list((tmp_path / "tiles").glob("*.h5")), "output was written before the refusal"


def test_alpha_on_isothermal_mhd_is_refused(tmp_path: Path, variants: Path) -> None:
    done = _run("AlphaIsothermalOrszagTang", tmp_path / "imhd", 1, resolution=(32, 32, 1))
    assert done.returncode != 0
    assert "alpha-disk viscosity on the imhd regime" in done.stderr.replace("\n", " "), done.stderr[-1500:]
    assert not list((tmp_path / "imhd").glob("*.h5")), "output was written before the refusal"
