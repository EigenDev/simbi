# =============================================================================
# test_horizon_with_bodies.py
#
# an excised kerr-schild box carrying an immersed body also carries the
# horizon body, whose ledger books the accretion through the diagnostic shell
# around the excised core. its shape slot follows the user bodies', so the
# unrefined run with a light gravitating body evolves and publishes; with
# mesh refinement the ledger would book through the finest level's share of
# the shell alone, and the run is refused by name before any output, on one
# device and on host tiles.
#
# run: pytest -m simulation simbi/simulation/tests/test_horizon_with_bodies.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.simulation

FIXTURE = '''
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig
from simbi_configs.examples.grhd.gr_bondi_cartesian import GrBondiCartesian


class ExcisedRefinedWithBody(GrBondiCartesian):
    """the excised kerr-schild bondi box with a fine patch off the horizon and a light gravitating body"""

    def setup(self) -> None:
        super().setup()
        ll = self.bounds[0][1]
        self.refinement_enabled = True
        self.refinement_max_levels = 2
        self.refinement_ratios = [2]
        self.refinement_regions = [[ll / 3.0, 0.75 * ll] * 3]

    @property
    def immersed_bodies(self):
        ll = self.domain_radius * self._r_bondi()
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=1e-6, radius=0.0, position=(0.5 * ll, 0.5 * ll, 0.5 * ll),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.1 * ll))]


class ExcisedWithBody(ExcisedRefinedWithBody):
    def setup(self) -> None:
        super().setup()
        self.refinement_enabled = False
'''


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "excised_refined.py"
    path.write_text(FIXTURE)
    return path


def _run(fixture: Path, cls: str, out: Path, gpus: int) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixture.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from excised_refined import {cls} as P\n"
        f"runner.run(P(resolution=(24, 24, 24), domain_radius=0.4, excision_radius=1.6, gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps=2)\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))


@pytest.mark.parametrize("gpus", [1, 2])
def test_an_excised_run_with_a_body_carries_the_horizon_body(tmp_path: Path, fixture: Path, gpus: int) -> None:
    out = tmp_path / "out"
    done = _run(fixture, "ExcisedWithBody", out, gpus)
    assert done.returncode == 0, f"the excised run with a body on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    assert list(out.glob("*final*.h5")), "the run published no final checkpoint"


@pytest.mark.parametrize("gpus", [1, 2])
def test_a_refined_excised_run_with_a_body_is_refused_by_name(tmp_path: Path, fixture: Path, gpus: int) -> None:
    out = tmp_path / "out"
    done = _run(fixture, "ExcisedRefinedWithBody", out, gpus)
    assert done.returncode != 0, f"the refined excised run with a body ran on {gpus} device(s)"
    assert "books the horizon accretion through its finest level's share of the diagnostic shell alone" in " ".join(done.stderr.split()), done.stderr[-1500:]
    assert not list(out.glob("*.h5")), "output was written before the refusal"
