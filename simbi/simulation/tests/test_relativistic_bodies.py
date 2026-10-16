# =============================================================================
# test_relativistic_bodies.py
#
# the relativistic regimes carry no immersed-body operator: their kernel sets
# have no body source, feedback or penalization fit for relativistic conserved
# variables (rhd leaves the gas untouched; rmhd met the newtonian
# penalization). a declared body is refused by name before any output on one
# device, on host tiles and through launch:
# - a rigid wall in a v = 0.5 rhd wind (flat spacetime);
# - a light gravitating body on the excised kerr-schild rhd box (curved);
# - an accreting sink in the rmhd orszag-tang vortex (flat).
# the horizon ledger, the one body these regimes book, is asked for by
# `horizon_ledger` (test_horizon_ledger, test_launch_horizon_ledger); the
# option off an excised kerr-schild box is refused by name.
#
# run: pytest -m simulation simbi/simulation/tests/test_relativistic_bodies.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

VARIANTS = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi.types.bodies import AccretionProperties, BodyCapability, GravitationalProperties, ImmersedBodyConfig
from simbi_configs.examples.grhd.gr_bondi_cartesian import GrBondiCartesian
from simbi_configs.examples.ibm.wind_tunnel import WindTunnel
from simbi_configs.examples.srmhd.rmhd_orszag_tang import OrszagTang


class RhdWall(WindTunnel):
    """the wind tunnel's rigid wall in a v = 0.5 relativistic freestream"""

    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="regime")]

    def freestream_velocity(self) -> float:
        return 0.5


class RhdLedgerOffTheBox(RhdWall):
    """a flat relativistic wind asking for a horizon ledger it has no horizon for"""

    horizon_ledger: Annotated[bool, ProblemParam(True, description="ledger")]

    @property
    def immersed_bodies(self):
        return []


class GrBodyOnTheBox(GrBondiCartesian):
    """the excised kerr-schild bondi box with a light gravitating body"""

    @property
    def immersed_bodies(self):
        ll = self.domain_radius * self._r_bondi()
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=1e-6, radius=0.0, position=(0.5 * ll, 0.5 * ll, 0.5 * ll),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.1 * ll))]


class RmhdSink(OrszagTang):
    """an accreting sink at the centre of the relativistic orszag-tang vortex"""

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL, mass=1e-3, radius=0.05,
                                   position=(0.5, 0.5, 0.0), velocity=(0.0, 0.0, 0.0),
                                   gravitational=GravitationalProperties(softening_length=0.02),
                                   accretion=AccretionProperties(accretion_radius=0.05))]
'''
BODY_REASON = "immersed bodies on the {} regime are outside the supported physics: no relativistic body operator exists"
LEDGER_REASON = "horizon_ledger books the flux through the horizon's diagnostic shell on an excised cartesian kerr-schild box"
# (problem, resolution, extra constructor fields, launch flags, cuts across two workers, regime)
CASES = {
    "rhd-wall-flat": ("RhdWall", (64, 32), "", [], [[32], []], "rhd"),
    "rhd-body-curved": ("GrBodyOnTheBox", (24, 24, 24), "domain_radius=0.4, excision_radius=1.6, ",
                        ["--domain-radius", "0.4", "--excision-radius", "1.6"], [[12], [], []], "rhd"),
    "rmhd-sink-flat": ("RmhdSink", (32, 32, 1), "", [], [[16], []], "rmhd"),
}


@pytest.fixture
def variants(tmp_path: Path) -> Path:
    (tmp_path / "relativistic_variants.py").write_text(VARIANTS)
    return tmp_path


def _env(root: Path, tiles: bool = False) -> dict[str, str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if tiles:
        env["SYMBI_GPU_OVERSUBSCRIBE"] = "1"
    return env


def _in_process(root: Path, problem: str, resolution: tuple, fields: str, out: Path, gpus: int, cuts: list) -> subprocess.CompletedProcess:
    decompose = f", decompose={cuts!r}" if gpus > 1 else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from relativistic_variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, {fields}gpus={gpus}{decompose}, data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps=1)\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=_env(root, gpus > 1))


def _launched(root: Path, problem: str, resolution: tuple, flags: list, out: Path, cuts: list) -> subprocess.CompletedProcess:
    script = root / f"launched_{problem.lower()}.py"
    script.write_text(f"from relativistic_variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    layout = _layout(root / f"{problem}.toml", workers=2, cuts=cuts)
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", ",".join(str(r) for r in resolution), *flags, "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env(root))


def _assert_refused(done: subprocess.CompletedProcess, out: Path, reason: str, label: str) -> None:
    flat = " ".join((done.stdout + done.stderr).split())
    assert done.returncode != 0, f"{label}: the run went ahead"
    assert reason in flat, f"{label}: refusal not named:\n{flat[-2000:]}"
    assert not list(out.glob("*.h5")), f"{label}: a refused run published a checkpoint"


@pytest.mark.parametrize("case", list(CASES))
def test_a_body_on_a_relativistic_regime_is_refused_by_name_on_every_driver(variants: Path, case: str) -> None:
    problem, resolution, fields, flags, cuts, regime = CASES[case]
    reason = BODY_REASON.format(regime)
    for label, gpus in (("one device", 1), ("two host tiles", 2)):
        out = variants / f"{case}-{gpus}"
        _assert_refused(_in_process(variants, problem, resolution, fields, out, gpus, cuts), out, reason, f"{case} on {label}")
    out = variants / f"{case}-launch"
    _assert_refused(_launched(variants, problem, resolution, flags, out, cuts), out, reason, f"{case} through launch")


def test_a_horizon_ledger_off_an_excised_kerr_schild_box_is_refused_by_name(variants: Path) -> None:
    out = variants / "ledger-off-box"
    _assert_refused(_in_process(variants, "RhdLedgerOffTheBox", (64, 32), "", out, 1, [[32], []]), out, LEDGER_REASON, "a flat ledger")
