# =============================================================================
# test_order_backend.py
#
# `--order 1` must run end-to-end. the python frontend maps order=1 to
# reconstruction=PCM + timestepping=RK1, so the rust backend must accept "rk1"
# (forward euler) and apply theta=0 (PCM has no separate kernel — it is the
# theta-MC limiter at zero slope). a backend missing either mapping dies with
# `ValueError: unknown timestepping 'rk1'`. requires the built cpu_ext backend;
# skipped in its absence.
# =============================================================================
import glob
import os
import tempfile

import pytest

from simbi.simulation import runner
from simbi.types.input import Reconstruction, TimeStepping

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)


def _order_problem(order: int, data_dir: str):
    from simbi_configs.examples.newtonian.kh import KelvinHelmholtz

    p = KelvinHelmholtz.from_cli(
        ["--resolution", "32,32", "--solver", "hlle", "--order", str(order)]
    )
    p.end_time = 0.01
    p.data_directory = data_dir
    p.checkpoint_interval = 1.0
    return p


def test_order1_emits_pcm_rk1() -> None:
    # the python side of the mapping — no backend required.
    p = _order_problem(1, "data/")
    assert p.reconstruction is Reconstruction.PCM
    assert p.timestepping is TimeStepping.RK1


@needs_backend
def test_order1_runs_through_rust_backend() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        runner.run(_order_problem(1, d), compute_mode="cpu", max_steps=400)
        # final snapshot follows the <resolution>.chkpt.final[.unit].h5 convention.
        assert glob.glob(os.path.join(d, "*.chkpt.final*.h5"))


@needs_backend
def test_order2_runs_through_rust_backend() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = d + "/"
        runner.run(_order_problem(2, d), compute_mode="cpu", max_steps=400)
        # final snapshot follows the <resolution>.chkpt.final[.unit].h5 convention.
        assert glob.glob(os.path.join(d, "*.chkpt.final*.h5"))
