# =============================================================================
# test_tiled_tracers.py
#
# lagrangian tracers on host tiles against the single grid. a 2d gas drifting
# across a box with outflow faces carries a seeded population; gas entering
# through the inner x face injects new tracers. for the discrete scheme and the
# second- and third-order ito schemes, two tiles and three uneven tiles
# publish the single grid's gas and tracer population bit for bit (the
# population compared by id; storage order is a layout choice), fresh and
# resumed from a checkpoint written on two tiles. the isothermal gas carries
# the discrete population the same way; a population of two seeds on three
# tiles leaves a tile with none, which still keys the ancestry blend with the
# population's seed. both schemes ride a grid graded geometrically along
# both axes the same way. the traced kelvin-helmholtz example (ito-3, cohorts,
# doubly periodic) runs on two tiles as on one grid.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_tracers.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 10
SEEDED = 400
RESOLUTION = (32, 32, 1)
VARIANTS = '''
import math
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam
from simbi.types import CellSpacing, Regime, TracerScheme
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast


class Drift(RefinedBlast):
    """gas at uniform pressure drifting in +x with a sheared cross flow; mass enters at x = -0.5"""

    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]

    @computed_field
    @property
    def n_tracers(self) -> int:
        return 400

    def initial_primitive_state(self):
        def gas():
            nx, ny, _ = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    x, y = (ii + 0.5) / nx, (jj + 0.5) / ny
                    rho = 1.0 + 0.3 * math.sin(2.0 * math.pi * y)
                    yield (rho, 1.0, 0.2 * math.sin(2.0 * math.pi * x), 1.0)

        return gas


class DiscreteDrift(Drift):
    pass


class Ito2Drift(Drift):
    @computed_field
    @property
    def tracer_scheme(self) -> TracerScheme:
        return TracerScheme.ITO2


class Ito3Drift(Drift):
    @computed_field
    @property
    def tracer_scheme(self) -> TracerScheme:
        return TracerScheme.ITO3


class IsoDrift(Drift):
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
    sound_speed: Annotated[float, ProblemParam(1.0, description="sound speed")]

    def initial_primitive_state(self):
        adiabatic = super().initial_primitive_state()

        def gas():
            for row in adiabatic():
                yield row[:-1]

        return gas


class GradedDrift(DiscreteDrift):
    """the drift on cells widening geometrically along x and narrowing along y"""

    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.GEOMETRIC, description="graded x")]
    x1_spacing_ratio: Annotated[float, ProblemParam(1.03, description="x growth")]
    x2_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.GEOMETRIC, description="graded y")]
    x2_spacing_ratio: Annotated[float, ProblemParam(0.98, description="y growth")]


class GradedIto2Drift(GradedDrift, Ito2Drift):
    """the ito drift on the graded cells"""


class SparseDrift(Drift):
    @computed_field
    @property
    def n_tracers(self) -> int:
        return 2


class SparseIto2Drift(Ito2Drift):
    @computed_field
    @property
    def n_tracers(self) -> int:
        return 2
'''
GROUP = {
    "DiscreteDrift": "tracers",
    "Ito2Drift": "continuous_tracers",
    "Ito3Drift": "continuous_tracers",
    "IsoDrift": "tracers",
    "SparseDrift": "tracers",
    "SparseIto2Drift": "continuous_tracers",
    "GradedDrift": "tracers",
    "GradedIto2Drift": "continuous_tracers",
}
# the injection arm needs a population fine enough that ten steps of inflow spawn a tracer
INJECTING = {"DiscreteDrift", "Ito2Drift", "Ito3Drift", "IsoDrift", "GradedDrift", "GradedIto2Drift"}


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


def _run(problem: str, out: Path, gpus: int, steps: int, extra: str = "", checkpoint: Path | None = None) -> Path:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={RESOLUTION!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}{resume}), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _population(path: Path, group: str) -> dict[str, np.ndarray]:
    """every dataset and attribute of the tracer group, the per-tracer rows ordered by id"""
    with h5py.File(path, "r") as f:
        g = f[group]
        order = np.argsort(g["id"][:], kind="stable")
        n = len(order)
        out = {f"attr/{k}": np.asarray(v) for k, v in g.attrs.items()}
        for name, ds in g.items():
            data = ds[:]
            out[name] = data[order] if data.shape[:1] == (n,) else data
    return out


def _assert_same(got: Path, want: Path, group: str, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    ps, qs = _population(got, group), _population(want, group)
    assert ps.keys() == qs.keys(), f"{label}: tracer fields {sorted(set(ps) ^ set(qs))} differ"
    for name, p in ps.items():
        np.testing.assert_array_equal(p, qs[name], err_msg=f"{label}: {group}/{name}")


@pytest.mark.parametrize("problem", list(GROUP))
def test_tiles_carry_the_single_grid_tracer_population(tmp_path: Path, variants: Path, problem: str) -> None:
    group = GROUP[problem]
    want = _run(problem, tmp_path / "single", 1, STEPS)
    if problem in INJECTING:
        with h5py.File(want, "r") as f:
            grown = int(f[group].attrs["n_tracers"])
        assert grown > SEEDED, f"the inflow injected no tracer ({grown}); the injection arm is vacuous"
    _assert_same(_run(problem, tmp_path / "two", 2, STEPS), want, group, f"{problem}: two tiles")
    _assert_same(_run(problem, tmp_path / "uneven", 3, STEPS, extra=", decompose=[[10, 21], []]"), want, group, f"{problem}: three uneven tiles")


@pytest.mark.parametrize("problem", list(GROUP))
def test_tiles_resume_the_tracer_population(tmp_path: Path, variants: Path, problem: str) -> None:
    group = GROUP[problem]
    want = _run(problem, tmp_path / "single", 1, STEPS)
    # the single grid resumed from its own checkpoint (its step budget counts from the start)
    single_first = _run(problem, tmp_path / "single-first", 1, STEPS // 2)
    _assert_same(_run(problem, tmp_path / "single-resumed", 1, STEPS, checkpoint=single_first), want, group, f"{problem}: the single grid resumed")
    first = _run(problem, tmp_path / "first", 2, STEPS // 2)
    if problem in INJECTING:
        with h5py.File(first, "r") as f:
            assert int(f[group].attrs["n_tracers"]) > SEEDED, "the checkpoint holds no injected tracer; the resume arm is vacuous"
    resumed = _run(problem, tmp_path / "resumed", 3, STEPS - _iteration(first), extra=", decompose=[[10, 21], []]", checkpoint=first)
    assert _iteration(resumed) == STEPS
    _assert_same(resumed, want, group, f"{problem}: resumed onto three uneven tiles")


def test_tiles_carry_the_traced_kelvin_helmholtz_population(tmp_path: Path) -> None:
    # ito-3 tracers with cohorts on a doubly periodic box: positions folded across the seams
    # land as on the single grid, and a position inside the domain keeps its bits
    from simbi_configs.examples.newtonian.traced_kh import TracedKelvinHelmholtz

    def run(gpus: int, out: Path) -> Path:
        snippet = (
            "import sys\n"
            "from simbi.simulation import runner\n"
            f"from {TracedKelvinHelmholtz.__module__} import TracedKelvinHelmholtz\n"
            f"runner.run(TracedKelvinHelmholtz(resolution=(32, 32), tracers=400, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode='cpu', max_steps=6)\n"
        )
        env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
        done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
        assert done.returncode == 0, f"traced kelvin-helmholtz on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
        return _final(out)

    want = run(1, tmp_path / "single")
    _assert_same(run(2, tmp_path / "two"), want, "continuous_tracers", "traced kelvin-helmholtz on two tiles")
