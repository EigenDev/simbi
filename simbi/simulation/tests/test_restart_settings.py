# =============================================================================
# test_restart_settings.py
#
# settings carried across a restart on every driver. each problem runs
# uninterrupted on the single grid; the single grid, host tiles and launched
# workers run half the steps, publish, and resume from their own final
# checkpoint for the rest.
# the resumed run reaches the single grid's clock and state, every level bit
# for bit. the settings:
# - the user timestep clamp max_dt;
# - a driven face whose prescription varies in time (the resumed clock feeds it);
# - a passive scalar;
# - a declared perturbation, laid on the fresh start alone;
# - the isothermal alpha viscosity around a central body (host tiles; the
#   launch refuses bodies).
# the bondi sink's body ledger resumes on two tiles as on the single grid
# resumed from its own checkpoint of the same step.
#
# run: pytest -m simulation simbi/simulation/tests/test_restart_settings.py
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
VARIANTS = '''
import math
from typing import Annotated, Optional

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi.types import BoundaryCondition, Regime, Solver
from simbi.types.bodies import BondedAssembly
from simbi_configs.examples.ibm.roche_passage import RochePassage
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

DRIVEN = [BoundaryCondition.DYNAMIC, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW]


class FlatBlast(RefinedBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class ClampedBlast(FlatBlast):
    max_dt: Annotated[float, ProblemParam(1e-6, description="clamp")]


class DrivenBlast(RefinedBlast):
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(DRIVEN, description="driven")]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> dict:
        x1, x2 = expr.coords(2)
        t = expr.variable("t", x1)
        rho = 1.0 + 0.2 * expr.sin(2.0 * math.pi * x2 + 20.0 * t)
        return expr.boundary([rho, 0.3, 0.05 * expr.sin(2.0 * math.pi * x2), 1.0], dim=2)


class FlatDrivenBlast(DrivenBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class DyedBlast(RefinedBlast):
    def passive_scalar(self):
        def dye():
            nx, ny, nz = self.resolution
            for _kk in range(nz):
                for _jj in range(ny):
                    for ii in range(nx):
                        yield 1.0 if ii < nx // 2 else 0.25

        return dye()


class PerturbedBlast(FlatBlast):
    @computed_field
    @property
    def perturbation_expressions(self) -> dict:
        x1, x2 = expr.coords(2)
        wave = expr.sin(2.0 * math.pi * x1) * expr.cos(2.0 * math.pi * x2)
        return x1._graph.compile([0.05 * wave, 0.02 * wave, -0.02 * wave, 0.01 * wave]).serialize_equilibrium(dim=2)


class IsothermalAlphaPrimary(RochePassage):
    viscosity_alpha: Annotated[float, ProblemParam(0.1, description="alpha")]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
    # isothermal hydro evaluates the hlle flux
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, description="hlle")]
    sound_speed: Annotated[float, ProblemParam(0.3, description="sound speed")]

    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        return None

    def initial_primitive_state(self):
        adiabatic = super().initial_primitive_state()

        def rows():
            for row in adiabatic():
                yield tuple(row[:-1])

        return rows
'''
BLAST = (32, 32, 1)
# (problem, resolution, launched as well as on host tiles)
CASES = {
    "max_dt": ("ClampedBlast", BLAST, True),
    "driven-flat": ("FlatDrivenBlast", BLAST, True),
    "driven-refined": ("DrivenBlast", BLAST, True),
    "dye-refined": ("DyedBlast", BLAST, True),
    "perturbation": ("PerturbedBlast", BLAST, True),
    "isothermal-alpha": ("IsothermalAlphaPrimary", (64, 64), False),
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


def _iteration(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"])


def _run(problem: str, resolution: tuple, out: Path, gpus: int, steps: int, checkpoint: Path | None = None) -> Path:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{resume}), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(variants: Path, problem: str, resolution: tuple, out: Path, layout: Path, steps: int, checkpoint: Path | None = None) -> Path:
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in resolution), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("case", list(CASES))
def test_a_resumed_run_reaches_the_uninterrupted_single_grid(tmp_path: Path, variants: Path, case: str) -> None:
    problem, resolution, launchable = CASES[case]
    want = _run(problem, resolution, tmp_path / "single", 1, STEPS)
    assert _iteration(want) == STEPS
    # the single grid resumed from its own checkpoint (its step budget counts from the start)
    single_first = _run(problem, resolution, tmp_path / "single-first", 1, STEPS // 2)
    _assert_same(_run(problem, resolution, tmp_path / "single-resumed", 1, STEPS, checkpoint=single_first), want, f"{case}: the single grid resumed")
    # host tiles and launched workers count their step budget from the checkpoint they resume
    first = _run(problem, resolution, tmp_path / "tiles-first", 2, STEPS // 2)
    resumed = _run(problem, resolution, tmp_path / "tiles-resumed", 2, STEPS - _iteration(first), checkpoint=first)
    assert _iteration(resumed) == STEPS, f"{case}: the resumed tiles ended at iteration {_iteration(resumed)}"
    _assert_same(resumed, want, f"{case}: two host tiles resumed")
    if launchable:
        two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
        first = _launch(variants, problem, resolution, tmp_path / "launch-first", two, STEPS // 2)
        three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
        resumed = _launch(variants, problem, resolution, tmp_path / "launch-resumed", three, STEPS - _iteration(first), checkpoint=first)
        _assert_same(resumed, want, f"{case}: resumed onto three launched workers")


def test_the_body_ledger_resumes_on_tiles_as_on_the_single_grid(tmp_path: Path) -> None:
    from simbi.simulation.tests.test_tiled_body_ledger import U, _groups, _pull

    def bondi(out: Path, gpus: int, steps: int, checkpoint: Path | None = None) -> Path:
        resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
        snippet = (
            "import sys\n"
            "from simbi.simulation import runner\n"
            "from simbi_configs.examples.newtonian.bondi import SphericalBondiTest\n"
            f"p = SphericalBondiTest(base_resolution=24, gpus={gpus}, data_directory=sys.argv[1]{resume})\n"
            "p.checkpoint_interval = 1e9\n"
            f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
        )
        env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
        done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
        assert done.returncode == 0, f"bondi on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
        return _final(out)

    # a resumed run's checkpoint carries the history of its own segment on every driver, so the
    # reference is the single grid resumed from its own checkpoint of the same step
    single_first = bondi(tmp_path / "single-first", 1, STEPS // 2)
    want_path = bondi(tmp_path / "single", 1, STEPS, checkpoint=single_first)
    first = bondi(tmp_path / "first", 2, STEPS // 2)
    assert _iteration(first) == _iteration(single_first)
    resumed = bondi(tmp_path / "resumed", 2, STEPS - _iteration(first), checkpoint=first)
    assert _iteration(resumed) == _iteration(want_path) == STEPS
    want, cells = _groups(want_path)
    got, _ = _groups(resumed)
    assert got.keys() == want.keys(), f"datasets {sorted(set(got) ^ set(want))} differ"
    assert len(want["body_diagnostics/time"]) == STEPS - _iteration(single_first), "the resumed history does not cover its segment"
    for name in ("body_diagnostics/time", "body_diagnostics/dt"):
        np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    # the same reassociation bound as the fresh tiled ledger
    bound = cells * U
    pull, arm = _pull(want_path)
    for name, x in want.items():
        scale = pull if "force" in name else pull * arm if "torque" in name else np.abs(x).max()
        worst = np.abs(got[name] - x).max()
        assert worst <= bound * scale, f"{name}: resumed tiles differ by {worst:.3e} against {bound:.1e} of {scale:.3e}"
