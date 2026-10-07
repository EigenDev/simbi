# =============================================================================
# test_motion_restart.py
#
# a resumed moving mesh continues the scale factor it integrated. the callable
# motion integrates a at the constant rate sampled at the start (a += a_dot dt),
# so a callable whose own curve departs from that line (a(t) = exp(20 t)) is
# the case where re-sampling the callable at the resume time would move the
# mesh off its integrated path. the uniform sphere runs twenty steps
# uninterrupted and ten plus ten from its checkpoint on one device and on two
# host tiles; the resumed run matches bit for bit in every grid dataset, its
# clock, and the recorded scale factor and rate.
#
# run: pytest -m simulation simbi/simulation/tests/test_motion_restart.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.simulation

STEPS = 20
SPLIT = 10
SPHERE = '''
import math
from typing import Callable, Optional

from simbi_configs.examples.newtonian.uniform_sphere import UniformSphere


class ExpSphere(UniformSphere):
    """the uniform sphere under a(t) = exp(rate t), a curve the integrated line departs from"""

    @property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        return lambda t: math.exp(self.expansion_rate * t)

    @property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
        return lambda t: self.expansion_rate * math.exp(self.expansion_rate * t)
'''


def _run(tmp: Path, out: Path, gpus: int, steps: int, checkpoint: Path | None = None) -> Path:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp)!r})\n"
        "from simbi.simulation import runner\n"
        "from exp_sphere import ExpSphere\n"
        f"p = ExpSphere(nr=64, expansion_rate=20.0, gpus={gpus}, data_directory=sys.argv[1]{resume})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    assert done.returncode == 0, f"the sphere on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _state(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.startswith("level_"):
                out[name] = obj[()]
        f.visititems(visit)
        out["level_0@time"] = np.asarray(f["level_0"].attrs["time"])
        for name in ("scale_factor", "scale_factor_dot"):
            out[f"metadata@{name}"] = np.asarray(f["metadata"].attrs[name])
    return out


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
def test_a_resumed_moving_mesh_continues_its_integrated_scale_factor(tmp_path: Path, gpus: int) -> None:
    (tmp_path / "exp_sphere.py").write_text(SPHERE)
    want = _run(tmp_path, tmp_path / "full", gpus, STEPS)
    first = _run(tmp_path, tmp_path / "first", gpus, SPLIT)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
        t, a = float(f["level_0"].attrs["time"]), float(f["metadata"].attrs["scale_factor"])
    import math

    # premise: the callable's own curve departs from the integrated scale factor at the write
    assert abs(math.exp(20.0 * t) - a) > 1e-6, f"a = {a} equals the callable's exp(20 t) = {math.exp(20.0 * t)}; the gate is vacuous"
    # the step cap is an absolute root iteration on every driver: the resume from written_at runs to STEPS
    assert 0 < written_at < STEPS
    xs, ys = _state(_run(tmp_path, tmp_path / "resumed", gpus, STEPS, checkpoint=first)), _state(want)
    assert xs.keys() == ys.keys(), f"datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{gpus} device(s): {name}")
