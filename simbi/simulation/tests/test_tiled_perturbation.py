# =============================================================================
# test_tiled_perturbation.py
#
# a declared initial perturbation on every driver. the blast, with and without
# its refined box, and an isothermal channel declare a density and velocity
# delta (and a pressure delta on the adiabatic gas) varying in both
# coordinates; the single grid lays it over every level's own cell centers.
# host tiles and two launched workers publish the single grid's state, every
# level bit for bit, at the start and after a few steps. the perturbation
# changes the run: the unperturbed problem lands elsewhere. every arm runs on
# the session's backend and placement (parity_harness), its single-device
# references and host tiles as fresh processes; the gate on single devices
# alone involves no launch and runs under the local placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_perturbation.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 6
VARIANTS = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.isothermal.isothermal_channel import IsothermalChannel
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast


def _delta(pressure: bool) -> dict:
    x1, x2 = expr.coords(2)
    wave = expr.sin(2.0 * math.pi * x1) * expr.cos(2.0 * math.pi * x2)
    outputs = [0.05 * wave, 0.02 * wave, -0.02 * wave] + ([0.01 * wave] if pressure else [])
    graph = x1._graph
    return graph.compile([o if isinstance(o, expr.Expr) else expr.constant(o, graph) for o in outputs]).serialize_equilibrium(dim=2)


class PerturbedBlast(RefinedBlast):
    @computed_field
    @property
    def perturbation_expressions(self) -> dict:
        return _delta(pressure=True)


class FlatPerturbedBlast(PerturbedBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class FlatBlast(RefinedBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class PerturbedChannel(IsothermalChannel):
    @computed_field
    @property
    def perturbation_expressions(self) -> dict:
        return _delta(pressure=False)
'''
ARMS = {
    "refined": ("PerturbedBlast", (32, 32, 1)),
    "flat": ("FlatPerturbedBlast", (32, 32, 1)),
    "isothermal": ("PerturbedChannel", (32, 32)),
}


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _published(out: Path) -> tuple[Path, Path]:
    """the initial checkpoint and the final one"""
    start = [p for p in out.glob("*.h5") if "final" not in p.name]
    final = sorted(out.glob("*final*.h5"))
    assert len(start) == 1 and len(final) == 1, f"expected an initial and a final checkpoint in {out}: {sorted(out.glob('*.h5'))}"
    return start[0], final[0]


def _single(problem: str, resolution: tuple, out: Path, gpus: int = 1) -> tuple[Path, Path]:
    """`problem` on the session's backend in a fresh process, one device or `gpus` host tiles"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), "
        f"compute_mode=sys.argv[2], max_steps={STEPS})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{problem} on {gpus} device(s)", devices=gpus)
    return _published(out)


def _launch(variants: Path, problem: str, resolution: tuple, out: Path, workers: int, shape: tuple) -> tuple[Path, Path]:
    """`problem` launched on `workers` workers partitioned by `shape` on the session's backend and
    placement, with no experimental opt-in"""
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    args = ["--max-steps", str(STEPS), "--resolution", ",".join(str(r) for r in resolution), "--checkpoint-interval", "1e9"]
    ph.launch(script, out, workers=workers, shape=shape, args=args, env=ph.environment())
    return _published(out)


def _assert_same(got: tuple[Path, Path], want: tuple[Path, Path], label: str) -> None:
    for which, a, b in [("start", got[0], want[0]), ("final", got[1], want[1])]:
        xs, ys = _all_levels(a), _all_levels(b)
        assert xs.keys() == ys.keys(), f"{label} {which}: datasets {sorted(set(xs) ^ set(ys))} differ"
        for name, x in xs.items():
            np.testing.assert_array_equal(x, ys[name], err_msg=f"{label} {which}: {name}")


@ph.local_only
def test_the_perturbation_changes_the_run(tmp_path: Path, variants: Path) -> None:
    perturbed = _all_levels(_single("FlatPerturbedBlast", (32, 32, 1), tmp_path / "perturbed")[0])
    plain = _all_levels(_single("FlatBlast", (32, 32, 1), tmp_path / "plain")[0])
    assert not np.array_equal(perturbed["level_0/conserved/den"], plain["level_0/conserved/den"]), "the perturbation left the start unchanged"


@pytest.mark.parametrize("arm", list(ARMS))
def test_tiles_and_launched_workers_lay_the_single_grid_perturbation(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution = ARMS[arm]
    want = _single(problem, resolution, tmp_path / "single")
    ph.disagreement(_assert_same, _single(problem, resolution, tmp_path / "tiles", gpus=2), want, f"{arm}: two host tiles")
    got = _launch(variants, problem, resolution, tmp_path / "launched", 2, (2, 1))
    ph.disagreement(_assert_same, got, want, f"{arm}: two launched workers")
