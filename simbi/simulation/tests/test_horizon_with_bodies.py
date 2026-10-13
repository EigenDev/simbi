# =============================================================================
# test_horizon_with_bodies.py
#
# an excised kerr-schild box carrying an immersed body also carries the
# horizon body, whose ledger books the accretion through the diagnostic shell
# around the excised core. its shape slot follows the user bodies', so the
# unrefined run with a light gravitating body evolves and publishes a nonzero
# ledger. the ledger is the root's (single owning level): under mesh refinement
# the root books the shell flux once per accepted root step and every patch
# keeps clear of the shell and its stencil band. a refined run whose patch lies
# outside the band publishes on one device and on host tiles cut through the
# shell, its ledger the same bits on both; a patch across the shell is refused
# by name before any output. the ledger continues across a restart: two steps
# resumed after one book the uninterrupted run's totals, on one device and on
# host tiles.
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
        self.refinement_regions = [[0.5 * ll, 5.0 * ll / 6.0] * 3]

    @property
    def immersed_bodies(self):
        ll = self.domain_radius * self._r_bondi()
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=1e-6, radius=0.0, position=(0.5 * ll, 0.5 * ll, 0.5 * ll),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.1 * ll))]


class ExcisedWithBody(ExcisedRefinedWithBody):
    def setup(self) -> None:
        super().setup()
        self.refinement_enabled = False


class ExcisedRefinedAcrossShell(ExcisedRefinedWithBody):
    """the patch over the corner box from r = 3.5 M: across the diagnostic shell at r = 3 M"""

    def setup(self) -> None:
        super().setup()
        ll = self.bounds[0][1]
        self.refinement_regions = [[ll / 3.0, 0.75 * ll] * 3]
'''


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "excised_refined.py"
    path.write_text(FIXTURE)
    return path


def _run(fixture: Path, cls: str, out: Path, gpus: int, steps: int = 2, resume: Path | None = None) -> subprocess.CompletedProcess:
    cuts = ", decompose=[[12], [], []]" if gpus == 2 else ""
    resume_arg = f", checkpoint_file={str(resume)!r}" if resume else ""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixture.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from excised_refined import {cls} as P\n"
        f"runner.run(P(resolution=(24, 24, 24), domain_radius=0.4, excision_radius=1.6, gpus={gpus}{cuts}, data_directory=sys.argv[1], checkpoint_interval=1e9{resume_arg}), compute_mode='cpu', max_steps={steps})\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))


def _final(done: subprocess.CompletedProcess, out: Path, label: str) -> Path:
    assert done.returncode == 0, f"{label} failed:\n{done.stderr[-3000:]}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final


def _ledger(path: Path) -> tuple[float, float]:
    """the horizon body's (accreted mass, accreted energy): the last body, after the user bodies"""
    import h5py

    with h5py.File(path, "r") as f:
        bodies = f["bodies"]
        return float(bodies["total_accreted_mass"][-1]), float(bodies["total_accreted_energy"][-1])


@pytest.mark.parametrize("gpus", [1, 2])
def test_an_excised_run_with_a_body_books_the_horizon_ledger(tmp_path: Path, fixture: Path, gpus: int) -> None:
    final = _final(_run(fixture, "ExcisedWithBody", tmp_path / "out", gpus), tmp_path / "out", f"the excised run with a body on {gpus} device(s)")
    mass, energy = _ledger(final)
    assert mass != 0.0 and energy != 0.0, f"the horizon booked ({mass}, {energy}); its ledger is empty"


def test_a_refined_excised_run_books_the_roots_ledger_on_every_driver(tmp_path: Path, fixture: Path) -> None:
    single = _final(_run(fixture, "ExcisedRefinedWithBody", tmp_path / "single", 1), tmp_path / "single", "the refined run on one device")
    mass, energy = _ledger(single)
    assert mass != 0.0 and energy != 0.0, f"the refined run booked ({mass}, {energy}); its ledger is empty"
    tiles = _final(_run(fixture, "ExcisedRefinedWithBody", tmp_path / "tiles", 2), tmp_path / "tiles", "the refined run on two host tiles")
    assert _ledger(tiles) == (mass, energy), f"two host tiles book {_ledger(tiles)} against the single grid's {(mass, energy)}"


@pytest.mark.parametrize("gpus", [1, 2])
def test_a_refined_patch_across_the_diagnostic_shell_is_refused_by_name(tmp_path: Path, fixture: Path, gpus: int) -> None:
    out = tmp_path / "out"
    done = _run(fixture, "ExcisedRefinedAcrossShell", out, gpus)
    assert done.returncode != 0, f"the refined run across the shell ran on {gpus} device(s)"
    assert "meets the horizon ledger's diagnostic shell" in " ".join(done.stderr.split()), done.stderr[-1500:]
    assert not list(out.glob("*.h5")), "output was written before the refusal"


@pytest.mark.parametrize("gpus", [1, 2])
def test_the_horizon_ledger_continues_across_a_restart(tmp_path: Path, fixture: Path, gpus: int) -> None:
    whole = _final(_run(fixture, "ExcisedRefinedWithBody", tmp_path / "whole", gpus, steps=2), tmp_path / "whole", "two steps")
    first = _final(_run(fixture, "ExcisedRefinedWithBody", tmp_path / "first", gpus, steps=1), tmp_path / "first", "one step")
    assert _ledger(first)[0] != 0.0, "the first step booked nothing; the restart carries no ledger to continue"
    resumed = _final(_run(fixture, "ExcisedRefinedWithBody", tmp_path / "resumed", gpus, steps=2, resume=first), tmp_path / "resumed", "the resumed step")
    assert _ledger(resumed) == _ledger(whole), f"{gpus} device(s): resumed ledger {_ledger(resumed)} against the uninterrupted {_ledger(whole)}"
