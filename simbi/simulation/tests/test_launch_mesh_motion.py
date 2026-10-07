# =============================================================================
# test_launch_mesh_motion.py
#
# mesh motion on every driver. a 2d cartesian blast on a moving mesh, the
# motion given two ways: python callables a(t), a_dot(t), which on a cartesian
# grid carry the mesh at a constant rate, and a traced expression law a(t)
# that expands it homologously, evaluated at every stage's entry time. two host
# tiles, two launched workers and a launch resumed onto three uneven workers
# publish the single grid's state and scale factor bit for bit; the motion
# changes the run (the static mesh lands elsewhere).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_mesh_motion.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
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
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={RESOLUTION!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(variants: Path, problem: str, out: Path, layout: Path, steps: int, checkpoint: Path | None = None) -> Path:
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


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


@pytest.mark.parametrize("problem", ARMS)
def test_launched_workers_move_the_mesh_as_the_single_grid(tmp_path: Path, variants: Path, problem: str) -> None:
    want = _single(problem, tmp_path / "single")
    static = _all_levels(_single("FlatBlast", tmp_path / "static"))
    assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], static["level_0/conserved/den"]), "the motion left the run unchanged"
    _assert_same(_single(problem, tmp_path / "tiles", gpus=2), want, f"{problem}: two host tiles")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch(variants, problem, tmp_path / "launched", two, STEPS), want, f"{problem}: two launched workers")
    first = _launch(variants, problem, tmp_path / "first", two, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    # `max_steps` caps the absolute root iteration: the checkpoint sits at STEPS // 2 and the
    # resumed launch takes the remaining STEPS - STEPS // 2 steps
    resumed = _launch(variants, problem, tmp_path / "resumed", three, STEPS, checkpoint=first)
    _assert_same(resumed, want, f"{problem}: resumed onto three uneven workers")
