# =============================================================================
# test_launch_mesh_motion.py
#
# mesh motion on every driver. a 2d cartesian blast on a moving mesh, the
# motion given two ways: python callables a(t), a_dot(t), which on a cartesian
# grid carry the mesh at a constant rate, and a traced expression law a(t)
# that expands it homologously, evaluated at every stage's entry time. two host
# tiles, two launched workers and a launch resumed onto three uneven workers
# publish the single grid's state and scale factor bit for bit; the motion
# changes the run (the static mesh lands elsewhere). a moving mesh with
# refinement or on a magnetized grid is refused by name on one device and on
# host tiles, given as callables or as an expression law alike.
# the single-grid runs, the host tiles and the launches run on the session's
# backend and placement (parity_harness); the in-process refusals run on the
# local placement alone.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_mesh_motion.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 8
RESOLUTION = (32, 32, 1)
VARIANTS = '''
from typing import Annotated, Callable, Optional

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast


class FlatBlast(RefinedBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class TranslatingBlast(FlatBlast):
    """a cartesian mesh carried at a constant rate: a(t) = 1 + 0.5 t"""

    @computed_field
    @property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        return lambda t: 1.0 + 0.5 * t

    @computed_field
    @property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
        return lambda _t: 0.5


class ExpandingBlast(FlatBlast):
    """a homologous expansion with an accelerating traced law: a(t) = 1 + 2 t + 30 t^2"""

    @computed_field
    @property
    def scale_factor_expressions(self) -> dict:
        t = expr.variable("t", expr.ExprGraph())
        a = 1.0 + 2.0 * t + 30.0 * t * t
        return t._graph.compile([a, a.diff(t)]).serialize_motion()
'''
ARMS = ["TranslatingBlast", "ExpandingBlast"]
UNWIRED = '''
from typing import Callable, Optional

from pydantic import computed_field

import simbi.expression as expr
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast


def _law():
    t = expr.variable("t", expr.ExprGraph())
    a = 1.0 + 2.0 * t + 30.0 * t * t
    return t._graph.compile([a, a.diff(t)]).serialize_motion()


class RefinedExpanding(RefinedBlast):
    @computed_field
    @property
    def scale_factor_expressions(self) -> dict:
        return _law()


class MagnetizedExpanding(NewtonianOrszagTang):
    @computed_field
    @property
    def scale_factor_expressions(self) -> dict:
        return _law()


class MagnetizedTranslating(NewtonianOrszagTang):
    @computed_field
    @property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        return lambda t: 1.0 + 0.5 * t

    @computed_field
    @property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
        return lambda _t: 0.5
'''
UNWIRED_ARMS = {
    "RefinedExpanding": "mesh motion is single-grid only (not wired with refinement)",
    "MagnetizedExpanding": "mesh motion is not wired for MHD (comoving-field convention pending)",
    "MagnetizedTranslating": "mesh motion is not wired for MHD (comoving-field convention pending)",
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


def _single(problem: str, out: Path, steps: int = STEPS, gpus: int = 1) -> Path:
    """`problem` on the session's backend over `gpus` devices, one device or in-process host tiles"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={RESOLUTION!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode=sys.argv[2], max_steps={steps})\n"
    )
    # host tiles beyond the visible devices share them
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{problem} on {gpus} device(s)", env=env, devices=gpus)
    return _final(out)


def _launch(variants: Path, problem: str, out: Path, steps: int, workers: int, cuts=None, shape=None,
            checkpoint: Path | None = None) -> Path:
    """`simbi launch` of `problem` on the session's backend and placement, every worker exiting zero
    without the experimental opt-in; its final checkpoint"""
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    args = ["--max-steps", str(steps), "--resolution", ",".join(str(r) for r in RESOLUTION), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return ph.launch(script, out, workers, cuts=cuts, shape=shape, args=args, env=ph.environment()).final


def _state(path: Path) -> dict[str, np.ndarray]:
    out = _all_levels(path)
    with h5py.File(path, "r") as f:
        for key, value in f["level_0"].attrs.items():
            if "scale" in key or key in ("a", "adot"):
                out[f"attr/{key}"] = np.asarray(value)
    return out


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _state(got), _state(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _same(got: Path, want: Path, label: str) -> None:
    """every level and the scale factor bit for bit, a difference raised as a NumericalDisagreement"""
    ph.disagreement(_assert_same, got, want, f"{label}, {ph.BACKEND}")


@pytest.mark.parametrize("problem", ARMS)
def test_launched_workers_move_the_mesh_as_the_single_grid(tmp_path: Path, variants: Path, problem: str) -> None:
    want = _single(problem, tmp_path / "single")
    static = _all_levels(_single("FlatBlast", tmp_path / "static"))
    assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], static["level_0/conserved/den"]), "the motion left the run unchanged"
    _same(_single(problem, tmp_path / "tiles", gpus=2), want, f"{problem}: two host tiles")
    _same(_launch(variants, problem, tmp_path / "launched", STEPS, 2, shape=(2, 1)), want, f"{problem}: two launched workers")
    first = _launch(variants, problem, tmp_path / "first", STEPS // 2, 2, shape=(2, 1))
    # `max_steps` caps the absolute root iteration: the checkpoint sits at STEPS // 2 and the
    # resumed launch takes the remaining STEPS - STEPS // 2 steps
    resumed = _launch(variants, problem, tmp_path / "resumed", STEPS, 3, cuts=[[10, 21], []], checkpoint=first)
    _same(resumed, want, f"{problem}: resumed onto three uneven workers")


# runs on one device and on in-process host tiles alone
@ph.local_only
@pytest.mark.parametrize("problem", list(UNWIRED_ARMS))
@pytest.mark.parametrize("gpus", [1, 2])
def test_an_unwired_moving_mesh_is_refused_by_name_on_every_in_process_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                               problem: str, gpus: int) -> None:
    (tmp_path / "unwired.py").write_text(UNWIRED)
    out = tmp_path / "out"
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from unwired import {problem}\n"
        f"runner.run({problem}(resolution=(32, 32, 1), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode=sys.argv[2], max_steps=2)\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1", PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    done = ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{problem} on {gpus} device(s)", timeout=600, env=env, devices=gpus,
                     check=False)
    assert done.returncode != 0, f"{problem} on {gpus} device(s) ran:\n{done.stdout[-1500:]}"
    assert UNWIRED_ARMS[problem] in " ".join(done.stderr.split()), done.stderr[-2000:]
    assert "panicked" not in done.stderr, f"{problem}: refused by a panic, not by name:\n{done.stderr[-2000:]}"
    assert not list(out.glob("*.h5")), f"{problem}: a refused run published a checkpoint"
