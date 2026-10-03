# =============================================================================
# test_tiled_equilibrium.py
#
# a declared stationary target on host tiles. the isentropic atmosphere in a
# point mass's gravity, adiabatic and isothermal, on one unrefined grid: its
# discrete imbalance is read off one tiled probe stage and added back at every
# stage, so two tiles and three uneven tiles publish the single grid's state
# bit for bit, fresh and resumed from a checkpoint written on two tiles. the
# target changes the run (the same atmosphere without it lands elsewhere), and
# the tiles probe at the single grid's own timestep, bit for bit. a
# refined target and a perturbation over a seeded target on tiles are refused
# by name before any output.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_equilibrium.py
# =============================================================================

import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 8
CELLS = 96
VARIANTS = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.newtonian.refined_atmosphere import RefinedAtmosphere
from simbi_configs.examples.newtonian.refined_isothermal_atmosphere import RefinedIsothermalAtmosphere


class Atmosphere(RefinedAtmosphere):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class AtmosphereWithoutTarget(Atmosphere):
    seed_from_equilibrium: Annotated[bool, ProblemParam(False, description="unseeded")]

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return {}


class IsothermalAtmosphere(RefinedIsothermalAtmosphere):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class PerturbedAtmosphere(Atmosphere):
    @computed_field
    @property
    def perturbation_expressions(self) -> dict:
        x1 = expr.coords(1)[0]
        return x1._graph.compile([0.01 * expr.sin(6.0 * x1), 0.0 * x1, 0.0 * x1]).serialize_equilibrium(dim=1)
'''
# three uneven tiles of the 96 cells
UNEVEN = ", decompose=[[30, 70]]"


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _run_raw(problem: str, out: Path, gpus: int, steps: int, extra: str = "", checkpoint: Path | None = None) -> subprocess.CompletedProcess:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution=({CELLS},), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}{resume}), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)


PROBE = re.compile(r"SIMBI equilibrium probe: level=0 dt_bits=([0-9a-f]{16})")


def _run(problem: str, out: Path, gpus: int, steps: int = STEPS, extra: str = "", checkpoint: Path | None = None) -> Path:
    path, _ = _run_probed(problem, out, gpus, steps, extra, checkpoint)
    return path


def _run_probed(problem: str, out: Path, gpus: int, steps: int = STEPS, extra: str = "", checkpoint: Path | None = None) -> tuple[Path, set[str]]:
    """the final checkpoint and the probe timesteps the run reported, as f64 bits"""
    done = _run_raw(problem, out, gpus, steps, extra, checkpoint)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0], set(PROBE.findall(done.stderr))


def _iteration(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"])


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert "level_0/conserved/den" in ys, f"{label}: the reference publishes no density; the comparison is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("problem", ["Atmosphere", "IsothermalAtmosphere"])
def test_tiles_hold_the_declared_target_as_the_single_grid(tmp_path: Path, variants: Path, problem: str) -> None:
    want, probe = _run_probed(problem, tmp_path / "single", 1)
    assert len(probe) == 1, f"{problem}: the single grid reported probe steps {probe}"
    if problem == "Atmosphere":
        bare = _all_levels(_run("AtmosphereWithoutTarget", tmp_path / "bare", 1))
        assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], bare["level_0/conserved/den"]), "the target left the run unchanged"
    # the tiles' probe step, the smallest of their cfl candidates, is the single grid's own
    two, two_probe = _run_probed(problem, tmp_path / "two", 2)
    assert two_probe == probe, f"{problem}: two tiles probed at {two_probe}, the single grid at {probe}"
    _assert_same(two, want, f"{problem}: two tiles")
    uneven, uneven_probe = _run_probed(problem, tmp_path / "uneven", 3, extra=UNEVEN)
    assert uneven_probe == probe, f"{problem}: three uneven tiles probed at {uneven_probe}, the single grid at {probe}"
    _assert_same(uneven, want, f"{problem}: three uneven tiles")


def test_tiles_resume_against_the_declared_target(tmp_path: Path, variants: Path) -> None:
    want = _run("Atmosphere", tmp_path / "single", 1)
    first = _run("Atmosphere", tmp_path / "first", 2, steps=STEPS // 2)
    resumed = _run("Atmosphere", tmp_path / "resumed", 3, steps=STEPS - _iteration(first), extra=UNEVEN, checkpoint=first)
    assert _iteration(resumed) == STEPS
    _assert_same(resumed, want, "resumed onto three uneven tiles")


def test_a_perturbation_over_a_seeded_target_on_tiles_is_refused(tmp_path: Path, variants: Path) -> None:
    done = _run_raw("PerturbedAtmosphere", tmp_path / "tiles", 2, 1)
    assert done.returncode != 0, "the tiles ran a perturbation over a seeded target"
    assert "the tiles lay the perturbation before the target is seeded" in " ".join(done.stderr.split()), done.stderr[-1500:]
    assert not list((tmp_path / "tiles").glob("*.h5")), "output was written before the refusal"
