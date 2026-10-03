# =============================================================================
# test_launch_tracers.py
#
# the discrete tracer population on launched workers against the single grid.
# a 2d gas drifting across a box with outflow faces carries a seeded
# population; gas entering through the inner x face injects new tracers. on
# the newtonian and the isothermal gas, two workers, a 2x2 launch and a launch
# resumed onto three uneven workers publish the single grid's gas and tracer
# population bit for bit (the population compared by id), and a troubled cell
# flagged against a cut corrects alike on one worker and on three. the ito
# schemes and tracers beside immersed bodies are refused by name before any
# output.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_tracers.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_tiled_tracers import GROUP, RESOLUTION, STEPS, VARIANTS, _assert_same, _final, _iteration, _population, _run

pytestmark = pytest.mark.simulation

LAUNCHED = ["DiscreteDrift", "IsoDrift", "SparseDrift"]
EXTRA = '''

class BodyDrift(DiscreteDrift):
    """the drift with a softened point mass at the box center"""

    @computed_field
    @property
    def immersed_bodies(self) -> list:
        from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig

        return [
            ImmersedBodyConfig(
                capability=BodyCapability.GRAVITATIONAL,
                mass=0.01,
                radius=0.0,
                position=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                gravitational=GravitationalProperties(softening_length=0.05),
            )
        ]
'''


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS + EXTRA)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _launch_run(variants: Path, problem: str, out: Path, layout: Path, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> subprocess.CompletedProcess:
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)


def _launch(variants: Path, problem: str, out: Path, layout: Path, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> Path:
    done = _launch_run(variants, problem, out, layout, steps, checkpoint, inject)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


@pytest.mark.parametrize("problem", LAUNCHED)
def test_launched_workers_carry_the_single_grid_tracer_population(tmp_path: Path, variants: Path, problem: str) -> None:
    group = GROUP[problem]
    want = _run(problem, tmp_path / "single", 1, STEPS)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch(variants, problem, tmp_path / "two", two, STEPS), want, group, f"{problem}: two workers")
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2))
    _assert_same(_launch(variants, problem, tmp_path / "four", four, STEPS), want, group, f"{problem}: a 2x2 launch")
    first = _launch(variants, problem, tmp_path / "first", two, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _launch(variants, problem, tmp_path / "resumed", three, STEPS - _iteration(first), checkpoint=first)
    assert _iteration(resumed) == STEPS
    _assert_same(resumed, want, group, f"{problem}: resumed onto three uneven workers")


def test_a_troubled_cell_against_a_cut_moves_the_tracers_alike(tmp_path: Path, variants: Path) -> None:
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[], [10, 21]])
    # the inflow cell beside the cut at y index 10, in the first stage of step 3: its first-order
    # flux changes the mass entering through the inner x face, which the injection books
    inject = "trouble:3:0:0:9"
    clean = _launch(variants, "DiscreteDrift", tmp_path / "clean", one, STEPS)
    flagged = _launch(variants, "DiscreteDrift", tmp_path / "flagged", one, STEPS, inject=inject)
    a, b = _population(clean, "tracers"), _population(flagged, "tracers")
    assert any(not np.array_equal(x, b[n]) for n, x in a.items()), "the flag left the population unchanged; the correction arm is vacuous"
    _assert_same(_launch(variants, "DiscreteDrift", tmp_path / "corrected", three, STEPS, inject=inject), flagged, "tracers", "corrected on three")


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        ("Ito2Drift", "the ito2 tracer scheme"),
        ("BodyDrift", "a tracer population with immersed bodies"),
    ],
)
def test_a_tracer_setting_the_workers_do_not_carry_is_refused(tmp_path: Path, variants: Path, problem: str, reason: str) -> None:
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    done = _launch_run(variants, problem, tmp_path / "out", two, 1)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert f"{reason} is outside the distributed first release" in flat, done.stderr[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"
