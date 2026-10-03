# =============================================================================
# test_locally_isothermal_tiles.py
#
# a locally isothermal closure cs^2(x, y) = p / rho, frozen at the initial
# state, on host tiles and launched workers. a 2d gas at rest on outflow faces
# with a closure varying along both axes: two and three uneven host tiles, a
# 2x2 launch, a launch resumed onto three uneven workers, and a troubled cell
# corrected against a cut each publish the single grid's state bit for bit.
# the closure changes the run (the globally isothermal gas on the same rows
# lands elsewhere). the isothermal orszag-tang vortex carries the closure on
# two launched workers and resumed onto three uneven ones, periodic and with
# outflow faces. refinement with the closure on launch is refused in
# test_launch_isothermal.
#
# run: pytest -m simulation simbi/simulation/tests/test_locally_isothermal_tiles.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 8
RESOLUTION = (32, 32, 1)
VARIANTS = '''
import math
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi.types import BoundaryCondition
from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang
from simbi_configs.examples.isothermal.refined_locally_iso import RefinedLocallyIso


class GlobalIso(RefinedLocallyIso):
    """rho(x, y) at rest on outflow faces, the closure the uniform sound speed"""

    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="outflow")
    ]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="end")]

    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        return False

    def rows(self):
        ni, nj, _ = self.resolution
        for jj in range(nj):
            for ii in range(ni):
                x, y = (ii + 0.5) / ni, (jj + 0.5) / nj
                rho = 1.0 + 0.2 * math.cos(2.0 * math.pi * y)
                cs2 = 0.5 + 0.3 * math.sin(2.0 * math.pi * x) * math.cos(3.0 * math.pi * y)
                yield rho, cs2

    def initial_primitive_state(self):
        def gas():
            for rho, _cs2 in self.rows():
                yield (rho, 0.0, 0.0)

        return gas


class LocalIso(GlobalIso):
    """the same gas with the closure cs^2(x, y) carried in the pressure slot"""

    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        return True

    def initial_primitive_state(self):
        def gas():
            for rho, cs2 in self.rows():
                yield (rho, 0.0, 0.0, cs2 * rho)

        return gas


class LocalImhd(IsothermalOrszagTang):
    """the isothermal orszag-tang vortex with the closure cs^2 = 1 + 0.1 sin(v_x)"""

    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        return True

    def initial_primitive_state(self):
        gas, *faces = super().initial_primitive_state()

        def rows():
            for row in gas():
                yield (*row, row[0] * (1.0 + 0.1 * math.sin(row[1])))

        return (rows, *faces)


class OutflowLocalImhd(LocalImhd):
    boundary_conditions: Annotated[
        list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="outflow")
    ]
'''


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _single(problem: str, out: Path, gpus: int = 1, extra: str = "") -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={RESOLUTION!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}), compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} failed:\n{done.stderr[-3000:]}"
    return _final(out)


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


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def test_tiles_and_workers_carry_the_local_closure_as_the_single_grid(tmp_path: Path, variants: Path) -> None:
    want = _single("LocalIso", tmp_path / "single")
    uniform = _all_levels(_single("GlobalIso", tmp_path / "uniform"))
    assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], uniform["level_0/conserved/den"]), "the closure left the run unchanged"
    _assert_same(_single("LocalIso", tmp_path / "tiles", gpus=2), want, "two host tiles")
    _assert_same(_single("LocalIso", tmp_path / "uneven", gpus=3, extra=", decompose=[[10, 21], []]"), want, "three uneven host tiles")
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2))
    _assert_same(_launch(variants, "LocalIso", tmp_path / "launched", four, STEPS), want, "a 2x2 launch")
    first = _launch(variants, "LocalIso", tmp_path / "first", four, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    _assert_same(_launch(variants, "LocalIso", tmp_path / "resumed", three, STEPS - STEPS // 2, checkpoint=first), want, "resumed onto three uneven workers")


def test_a_troubled_cell_against_a_cut_corrects_alike_under_the_local_closure(tmp_path: Path, variants: Path) -> None:
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    # the cell beside the cut at x index 10, in the first stage of step 3
    inject = "trouble:3:0:9:12"
    clean = _all_levels(_launch(variants, "LocalIso", tmp_path / "clean", one, STEPS))
    flagged = _launch(variants, "LocalIso", tmp_path / "flagged", one, STEPS, inject=inject)
    assert any(not np.array_equal(x, _all_levels(flagged)[n]) for n, x in clean.items()), "the flag left the run unchanged; the correction arm is vacuous"
    _assert_same(_launch(variants, "LocalIso", tmp_path / "corrected", three, STEPS, inject=inject), flagged, "corrected on three")



@pytest.mark.parametrize("problem", ["LocalImhd", "OutflowLocalImhd"])
def test_isothermal_mhd_workers_carry_the_local_closure_as_the_single_grid(tmp_path: Path, variants: Path, problem: str) -> None:
    want = _single(problem, tmp_path / "single")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch(variants, problem, tmp_path / "two", two, STEPS), want, f"{problem}: two workers")
    first = _launch(variants, problem, tmp_path / "first", two, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    _assert_same(_launch(variants, problem, tmp_path / "resumed", three, STEPS - STEPS // 2, checkpoint=first), want, f"{problem}: resumed onto three uneven workers")
