# =============================================================================
# test_swirl_target_admission.py
#
# a declared stationary target supplies density, one velocity per grid axis
# and the pressure; a swirl run (the out-of-plane momentum carried on a 2D
# kerr-schild grid) holds one velocity more than its grid has axes, so no
# declared target describes its state. the run is refused by name before any
# output, on one device and on two host tiles alike, where a target read into
# the swirl layout would put the pressure in the azimuthal velocity.
#
# run: pytest -m simulation simbi/simulation/tests/test_swirl_target_admission.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.simulation

PROBE = '''
import os
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.grhd.gr_fishbone_moncrief import GrFishboneMoncrief


class SwirlTarget(GrFishboneMoncrief):
    nr: Annotated[int, ProblemParam(24, cli=True, description="radial resolution")]
    npolar: Annotated[int, ProblemParam(16, cli=True, description="polar resolution")]
    gpus: Annotated[int, ProblemParam(int(os.environ.get("SWIRL_GPUS", "1")), cli=True, description="devices")]

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return expr.equilibrium([1.0, 0.0, 0.0, 1.0e-3], dim=2)
'''

REASON = "a declared stationary target on a swirl run"


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "swirl_target.py"
    path.write_text(PROBE)
    return path


def _run(probe: Path, out: Path, gpus: int) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(probe.parent)!r})\n"
        "from simbi.simulation import runner\n"
        "from swirl_target import SwirlTarget\n"
        "runner.run(SwirlTarget(data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps=2)\n"
    )
    env = dict(os.environ, SWIRL_GPUS=str(gpus), SYMBI_GPU_OVERSUBSCRIBE="1")
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=env)


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "host-tiles"])
def test_a_stationary_target_on_a_swirl_run_is_refused_by_name(tmp_path: Path, probe: Path, gpus: int) -> None:
    out = tmp_path / "out"
    done = _run(probe, out, gpus)
    flat = " ".join((done.stdout + done.stderr).split())
    assert done.returncode != 0, f"{gpus} device(s): a swirl run with a declared target ran:\n{flat[-1500:]}"
    assert REASON in flat, f"{gpus} device(s): the refusal is not named:\n{flat[-1500:]}"
    assert "panicked" not in flat, f"{gpus} device(s): refused by a panic rather than at admission"
    assert not list(out.glob("*.h5")), f"{gpus} device(s): a refused run published a checkpoint"
