# =============================================================================
# test_launch_driven_boundary.py
#
# a driven (dynamic) face on launched workers. the x1-inner face of the blast,
# with and without its refined box, of an isothermal channel and of the
# magnetized orszag-tang vortex prescribes an inflow whose density and
# transverse velocity vary along the face and in time. launched on two workers cut across the face (each holding half of it)
# and cut along it (one holding all of it), every level publishes the single
# grid's state bit for bit. the prescription changes the run: the same problem
# with an outflow face lands elsewhere. a magnetized face prescribing in-plane
# vectors alone on the 2.5d grid, whose kernels fill three components, is
# refused by name. the single-grid runs and the launches run on the session's
# backend and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_driven_boundary.py
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

STEPS = 8
VARIANTS = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi.types import BoundaryCondition
from simbi_configs.examples.isothermal.isothermal_channel import IsothermalChannel
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

DRIVEN = [BoundaryCondition.DYNAMIC, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW]
OUTFLOW = [BoundaryCondition.OUTFLOW] * 4


def _inflow(pressure: bool) -> dict:
    """density and transverse velocity varying along the face and in time"""
    x1, x2 = expr.coords(2)
    t = expr.variable("t", x1)
    rho = 1.0 + 0.2 * expr.sin(2.0 * math.pi * x2 + 20.0 * t)
    vy = 0.05 * expr.sin(2.0 * math.pi * x2)
    outputs = [rho, 0.3, vy] + ([1.0] if pressure else [])
    return expr.boundary(outputs, dim=2)


class DrivenBlast(RefinedBlast):
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(DRIVEN, description="driven")]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        return _inflow(pressure=True)


class FlatDrivenBlast(DrivenBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class FlatOutflowBlast(FlatDrivenBlast):
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(OUTFLOW, description="outflow")]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        return {}


class DrivenOrszagTang(NewtonianOrszagTang):
    """the magnetized 2.5d grid carries three-component velocity and field, so its driven face
    prescribes [rho, v_0..v_2, pre, B_0..B_2]"""
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(DRIVEN, description="driven")]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        x1, x2 = expr.coords(2)
        t = expr.variable("t", x1)
        rho = 1.0 + 0.2 * expr.sin(2.0 * math.pi * x2 + 20.0 * t)
        vy = 0.05 * expr.sin(2.0 * math.pi * x2)
        by = 0.02 * expr.cos(2.0 * math.pi * x2)
        return expr.boundary([rho, 0.3, vy, 0.0, 1.0, 0.1, by, 0.0], dim=3)


class PlanarDrivenOrszagTang(DrivenOrszagTang):
    """the same face prescribing in-plane vectors alone"""

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        return expr.boundary([1.0, 0.3, 0.0, 1.0, 0.1, 0.0], dim=2)


class DrivenChannel(IsothermalChannel):
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(DRIVEN, description="driven")]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        return _inflow(pressure=False)
'''
ARMS = {
    "refined": ("DrivenBlast", (32, 32, 1)),
    "flat": ("FlatDrivenBlast", (32, 32, 1)),
    "isothermal": ("DrivenChannel", (32, 32)),
    "magnetized": ("DrivenOrszagTang", (32, 32, 1)),
}


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _run_single(problem: str, resolution: tuple, out: Path, steps: int, check: bool = True):
    """the single grid of `problem` on the session's backend"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode=sys.argv[2], max_steps={steps})\n"
    )
    return ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{problem} on one device", timeout=900, check=check)


def _single(problem: str, resolution: tuple, out: Path) -> Path:
    _run_single(problem, resolution, out, STEPS)
    return _final(out)


def _launch(variants: Path, problem: str, resolution: tuple, out: Path, workers: int, shape: tuple) -> Path:
    """the launch of `problem` on the session's backend and placement without the experimental
    opt-in; its final checkpoint"""
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    args = ["--max-steps", str(STEPS), "--resolution", ",".join(str(r) for r in resolution), "--checkpoint-interval", "1e9"]
    ph.launch(script, out, workers, shape=shape, args=args, env=ph.environment(), timeout=900)
    return _final(out)


def _assert_same_levels(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _assert_same(got: Path, want: Path, label: str) -> None:
    """every level equal bit for bit, a difference raised as a NumericalDisagreement"""
    ph.disagreement(_assert_same_levels, got, want, f"{label}, {ph.BACKEND}")


def test_the_driven_face_changes_the_run(tmp_path: Path, variants: Path) -> None:
    driven = _all_levels(_single("FlatDrivenBlast", (32, 32, 1), tmp_path / "driven"))
    outflow = _all_levels(_single("FlatOutflowBlast", (32, 32, 1), tmp_path / "outflow"))
    assert not np.array_equal(driven["level_0/conserved/den"], outflow["level_0/conserved/den"]), "the prescription left the run unchanged"


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)], ids=["across-the-face", "along-the-face"])
@pytest.mark.parametrize("arm", list(ARMS))
def test_launched_workers_drive_the_face_as_the_single_grid(tmp_path: Path, variants: Path, arm: str, shape: tuple) -> None:
    problem, resolution = ARMS[arm]
    want = _single(problem, resolution, tmp_path / "single")
    _assert_same(_launch(variants, problem, resolution, tmp_path / "launched", 2, shape), want, f"{arm} {shape}: two launched workers")


def test_a_magnetized_face_prescribing_planar_vectors_is_refused(tmp_path: Path, variants: Path) -> None:
    done = _run_single("PlanarDrivenOrszagTang", (32, 32, 1), tmp_path / "out", 1, check=False)
    assert done.returncode != 0
    assert "three-component velocity and field" in done.stderr.replace("\n", " "), done.stderr[-1500:]
    assert not list((tmp_path / "out").glob("*.h5")), "output was written before the refusal"
