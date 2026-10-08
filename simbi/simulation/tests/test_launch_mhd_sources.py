# =============================================================================
# test_launch_mhd_sources.py
#
# a user source on a flat magnetized run, on every driver. the orszag-tang
# vortex under a uniform force (which does work on the moving gas, so the
# energy equation feels it on the energy-carrying regimes) is the single
# grid's run bit for bit on two host tiles, on two launched workers and on a
# two-by-two launch whose every tile holds a cut corner, for newtonian and
# isothermal MHD. the force changes the run (the same vortex without it lands
# elsewhere), so agreement is not the agreement of two runs that ignore it.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_mhd_sources.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import MHD_STEPS, _assert_same_state, _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_isothermal import _script_launch

pytestmark = pytest.mark.simulation

RESOLUTION = (16, 16, 1)
ARMS = {
    "nmhd": ("simbi_configs.examples.newtonian.nmhd_orszag_tang", "NewtonianOrszagTang"),
    "imhd": ("simbi_configs.examples.isothermal.imhd_orszag_tang", "IsothermalOrszagTang"),
}


def _script(tmp_path: Path, arm: str, sourced: bool) -> Path:
    module, cls = ARMS[arm]
    body = "    pass\n"
    if sourced:
        body = (
            "    @property\n"
            "    def source_expressions(self):\n"
            "        return [expr.force([0.05, -0.1, 0.0], dim=3)]\n"
        )
    path = tmp_path / f"{arm}_{'sourced' if sourced else 'plain'}.py"
    path.write_text(f"import simbi.expression as expr\nfrom {module} import {cls}\n\n\nclass Vortex({cls}):\n{body}")
    return path


def _single(script: Path, out: Path, gpus: int = 1) -> Path:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(script.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from {script.stem} import Vortex\n"
        f"runner.run(Vortex(resolution={RESOLUTION!r}, gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9), "
        "compute_mode='cpu', max_steps=int(sys.argv[2]))\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out), str(MHD_STEPS)], capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"{script.stem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _differs(a: Path, b: Path) -> bool:
    try:
        _assert_same_state(a, b, "premise")
    except AssertionError:
        return True
    return False


@pytest.mark.parametrize("arm", list(ARMS))
def test_a_flat_magnetized_user_source_is_the_single_grids_on_every_driver(tmp_path: Path, arm: str) -> None:
    sourced = _script(tmp_path, arm, sourced=True)
    reference = _single(sourced, tmp_path / "single")
    plain = _single(_script(tmp_path, arm, sourced=False), tmp_path / "plain")
    assert _differs(reference, plain), f"{arm}: the source left the run unchanged; the gate is vacuous"
    _assert_same_state(_single(sourced, tmp_path / "tiles", gpus=2), reference, f"{arm} on two host tiles")
    for workers, shape in [(2, (2, 1, 1)), (4, (2, 2, 1))]:
        layout = _layout(tmp_path / f"{workers}.toml", workers=workers, shape=shape)
        launched = _script_launch(str(sourced), RESOLUTION, tmp_path / f"launch{workers}", layout, MHD_STEPS, workers=workers)
        _assert_same_state(launched, reference, f"{arm} on {workers} launched workers")
