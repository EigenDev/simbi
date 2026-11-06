# =============================================================================
# test_outward_wall_surface_regression.py
#
# the inviscid SPB torus with the outward-only outer wall (simbi_configs/examples/newtonian/
# spb_torus.py, run N, 84 x 44, outward_outer) from its state at step 2061947, t = 9.6007
# (src/crates/symbi/tests/fixtures/outward_wall_surface_n84_step2061947.bin), through the
# production frontend:
# - the restart file rebuilt from the fixture resumes the captured state bit for bit
# - the run's method is the recorded one (solver, reconstruction, limiter, cfl, closure, walls)
# - the regression: every accepted state through t = 9.62 stays admissible. the recorded run
#   fails at step 2062990, t = 9.605410191424152, where two surface cells at r = 1.326,
#   theta = 1.321 and 1.821 reach negative internal energy and the next cfl timestep is not
#   positive; this gate fails there until that failure is resolved
#
# run: pytest -m simulation simbi/simulation/tests/test_outward_wall_surface_regression.py
# =============================================================================
import json
import shutil
import struct
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "src/crates/symbi/tests/fixtures/outward_wall_surface_n84_step2061947.bin"
METHOD = json.loads(FIXTURE.with_suffix(".json").read_text())
STEP = 2061947
FAILED_STEP = 2062990
FAILED_TIME = 9.605410191424152
T_END = 9.62
CONSERVED = ("den", "m1", "m2", "m3", "nrg")
PRIMITIVE = ("rho", "v1", "v2", "v3", "pre")
SETUP = "run='N', zones_per_decade=32, polar_zones=44, orbits=3, snapshot_orbits=0.001, outward_outer=True"


def _fixture() -> tuple[float, int, dict[str, np.ndarray]]:
    raw = FIXTURE.read_bytes()
    time, step, _, rows, cols = struct.unpack("<5d", raw[:40])
    values = np.frombuffer(raw[40:], dtype="<f8")
    n = int(rows) * int(cols)
    assert values.size == 10 * n, f"the fixture holds {values.size} values, not 10 x {n}"
    arrays = {name: values[k * n:(k + 1) * n].reshape(int(rows), int(cols)) for k, name in enumerate(CONSERVED + PRIMITIVE)}
    return time, int(step), arrays


def _run(out: Path, max_steps: int, checkpoint: Path | None = None, check: bool = True):
    restart = f", checkpoint_file={str(checkpoint)!r}" if checkpoint else ""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'simbi_configs/examples/newtonian')!r})\n"
        "from spb_torus import SpbTorus\n"
        "from simbi.simulation import runner\n"
        f"runner.run(SpbTorus(data_directory=sys.argv[1], {SETUP}{restart}), compute_mode={ph.BACKEND!r}, max_steps={max_steps})\n"
    )
    return ph.python(ph.snippet_args(snippet, str(out)), f"outward-wall torus to step {max_steps}", check=check)


def _restart_file(tmp_path: Path) -> Path:
    """a step-0 checkpoint of the same configuration carrying the fixture's state, time and step"""
    _run(tmp_path / "base", 1)
    target = tmp_path / "resume.h5"
    shutil.copy(next((tmp_path / "base").glob("*.chkpt.000_000.h5")), target)
    time, step, arrays = _fixture()
    with h5py.File(target, "r+") as f:
        level = f["level_0"]
        for name in CONSERVED:
            level[f"conserved/{name}"][...] = arrays[name]
        for name in PRIMITIVE:
            level[f"partition_0/hydro/primitives/{name}"][...] = arrays[name]
        level.attrs["time"] = time
        level.attrs["iteration"] = step
    return target


def _interior(path: Path) -> tuple[float, int, dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as f:
        level = f["level_0"]
        fields = {k: level[f"conserved/{k}"][()] for k in CONSERVED}
        meta = {k: f["metadata"].attrs[k] for k in f["metadata"].attrs}
        return float(level.attrs["time"]), int(level.attrs["iteration"]), fields, meta


def test_the_outward_wall_torus_surface_stays_admissible_past_the_recorded_failure(tmp_path: Path) -> None:
    resume = _restart_file(tmp_path)
    time, step, arrays = _fixture()
    ng = 2

    # the resumed state: a run whose step cap is the fixture's step evolves nothing and writes
    # the restored state
    _run(tmp_path / "at_cap", STEP, resume)
    t0, it0, held, meta = _interior(next((tmp_path / "at_cap").glob("*.chkpt.final.h5")))
    assert (t0, it0) == (time, step), f"resumed at (t, step) = ({t0!r}, {it0})"
    for name in CONSERVED:
        got, want = held[name][ng:-ng, ng:-ng], arrays[name][ng:-ng, ng:-ng]
        assert np.array_equal(got.view(np.uint64), want.view(np.uint64)), f"{name}: the resumed state differs from the fixture"

    # the method: every recorded property but the build that recorded it
    for key, want in METHOD.items():
        if key != "build_git_sha":
            assert str(meta[key]) == want, f"method {key}: {meta[key]} against the recorded {want}"

    assert str(meta["outward_only_walls"]) == "0,1,0,0", f"the walls the resumed run evolves: {meta['outward_only_walls']}"

    # the regression
    done = _run(tmp_path / "window", FAILED_STEP + 20000, resume, check=False)
    crashed = list((tmp_path / "window").glob("*.chkpt.crashed.h5"))
    if crashed:
        t_fail, it_fail, _, _ = _interior(crashed[0])
        where = "the recorded failure" if (it_fail, t_fail) == (FAILED_STEP, FAILED_TIME) else "a failure other than the recorded one"
        pytest.fail(f"{where}: the run stopped at step {it_fail}, t = {t_fail!r}:\n{done.stderr[-1500:]}")
    assert done.returncode == 0, done.stderr[-2000:]
    t_last, _, _, _ = _interior(next((tmp_path / "window").glob("*.chkpt.final.h5")))
    assert t_last >= T_END, f"the window ended at t = {t_last}, short of t = {T_END}"
