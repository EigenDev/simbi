# =============================================================================
# test_outward_wall_surface_regression.py
#
# the inviscid SPB torus with the outward-only outer wall (simbi_configs/examples/newtonian/
# spb_torus.py, run N, 84 x 44, outward_outer) from its state at step 2061947, t = 9.6007
# (src/crates/symbi/tests/fixtures/outward_wall_surface_n84_step2061947.bin), through the
# production frontend:
# - the restart file rebuilt from the fixture resumes the captured state bit for bit
# - the run's method is the recorded one (solver, reconstruction, limiter, cfl, closure, walls)
# - the regression: the recorded run's stage 0 of step 2062989 (entry t = 9.605405683896917) left
#   the surface cell [68, 18] at r = 1.326, theta = 1.321 with negative internal energy. on the
#   production path the first-order flux correction repairs such stages: the window through
#   t = 9.62 completes with every accepted stage admissible, the correction firing and freezing no
#   cell, and mass, angular momentum and E + rho Phi closing against the receipts from the
#   fixture's state
# - a restart inside the window continues it bit for bit
# - four host tiles, and four launched workers, cut through the faces of cell [68, 18] equal the
#   single device bit for bit
# a failed recovery's stop is gated on a planted unrecoverable cell in
# src/crates/symbi/tests/lifted_recovery_acceptance.rs
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
FAILED_STEP = 2062989
FAILED_TIME = 9.605405683896917
FAILED_EVIDENCE = ("recovery failed at step 2062989 stage 0", "first cell [68, 18]")
WINDOW_STEPS = 4500
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


def _run(out: Path, max_steps: int, checkpoint: Path | None = None, check: bool = True, extra: str = "", devices: int = 1):
    restart = f", checkpoint_file={str(checkpoint)!r}" if checkpoint else ""
    snippet = (
        "import sys, json, dataclasses\n"
        f"sys.path.insert(0, {str(REPO / 'simbi_configs/examples/newtonian')!r})\n"
        "from spb_torus import SpbTorus\n"
        "from simbi.simulation import runner\n"
        f"result = runner.run(SpbTorus(data_directory=sys.argv[1], {SETUP}{restart}{extra}), compute_mode={ph.BACKEND!r}, max_steps={max_steps})\n"
        "print('GUARDS ' + json.dumps(dataclasses.asdict(result.diagnostics.guards)))\n"
    )
    return ph.python(ph.snippet_args(snippet, str(out)), f"outward-wall torus to step {max_steps}", check=check, devices=devices)


def _guards(done) -> dict:
    line = next(l for l in done.stdout.splitlines() if l.startswith("GUARDS "))
    return json.loads(line[len("GUARDS "):])


def _restart_file(tmp_path: Path) -> Path:
    """a step-0 checkpoint of the same configuration carrying the fixture's state, time and step,
    on the level and in the metadata alike"""
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
        # the run clock a restart starts from: every driver resumes the fixture's time and step
        f["metadata"].attrs["time"] = time
        f["metadata"].attrs["iteration"] = np.uint64(step)
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
    end = STEP + WINDOW_STEPS
    done = _run(tmp_path / "window", end, resume, check=False)
    crashed = list((tmp_path / "window").glob("*.chkpt.crashed.h5"))
    if crashed:
        t_fail, it_fail, _, _ = _interior(crashed[0])
        recorded = (it_fail, t_fail) == (FAILED_STEP, FAILED_TIME) and all(e in done.stderr for e in FAILED_EVIDENCE)
        where = "the recorded failure" if recorded else "a failure other than the recorded one"
        pytest.fail(f"{where}: the run stopped at step {it_fail}, t = {t_fail!r}:\n{done.stderr[-1500:]}")
    assert done.returncode == 0, done.stderr[-2000:]
    final = next((tmp_path / "window").glob("*.chkpt.final.h5"))
    t_last, it_last, fields, meta = _interior(final)
    assert t_last >= T_END and it_last == end, f"the window ended at step {it_last}, t = {t_last}"
    guards = _guards(done)
    assert guards["troubled_cells"]["total"] > 0, f"the correction never fired: {guards}"
    assert guards["frozen_cells"]["total"] == 0, f"the correction froze cells, waiving their conservation: {guards}"
    # the receipts from the fixture's state: the window's change of each total plus its wall outflow.
    # each of the window's steps rounds the totals of 3696 cells at relative u, so the residual stays
    # below the steps times 4 u of each total
    sys.path.insert(0, str(REPO / "scripts"))
    import torus_receipts
    from spb_torus_analysis import Grid, geometry

    grid = Grid(84, 44, *geometry(final))
    interior = lambda a: {k: v[2:-2, 2:-2] for k, v in a.items()}
    first, last = torus_receipts.totals(interior(arrays), grid), torus_receipts.totals(interior(fields), grid)
    ledger = torus_receipts.ledger(meta)
    assert ledger.stages == 2 * WINDOW_STEPS, ledger.stages
    assert ledger.mass_in[1] == 0.0, f"{ledger.mass_in[1]} entered through the outward-only wall during the corrected window"
    for k, (name, outflow) in enumerate([("mass", ledger.mass.sum()), ("angular momentum", ledger.angular_momentum.sum()), ("E + rho Phi", ledger.energy.sum())]):
        residual = last[k] - first[k] + outflow
        assert abs(residual) <= WINDOW_STEPS * 4 * 1.1e-16 * abs(first[k]), f"{name}: residual {residual:e}"


def test_a_restart_inside_the_corrected_window_continues_it_bit_for_bit(tmp_path: Path) -> None:
    resume = _restart_file(tmp_path)
    end, middle = STEP + WINDOW_STEPS, FAILED_STEP + 200
    _run(tmp_path / "whole", end, resume)
    _run(tmp_path / "first", middle, resume)
    _run(tmp_path / "second", end, next((tmp_path / "first").glob("*.chkpt.final.h5")))
    whole, second = (next((tmp_path / d).glob("*.chkpt.final.h5")) for d in ("whole", "second"))
    ph.same_checkpoint(second, whole, "the window resumed inside the correction against the uninterrupted window",
                       exclude=("body_diagnostics",), expect=("level_0",))


def test_tiles_cut_through_the_corrected_cell_equal_the_single_device(tmp_path: Path) -> None:
    resume = _restart_file(tmp_path)
    end = STEP + WINDOW_STEPS
    single = _run(tmp_path / "single", end, resume)
    # cuts on the cell's low radial and low polar faces: the corrected cell [68, 18] touches both
    tiles = _run(tmp_path / "tiles", end, resume, extra=", gpus=4, decompose=[[68], [18]]", devices=4)
    assert _guards(single)["troubled_cells"]["total"] > 0, "the correction never fired; the cut checks nothing"
    a, b = (next((tmp_path / d).glob("*.chkpt.final.h5")) for d in ("tiles", "single"))
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert int(fa["level_0"].attrs["iteration"]) == int(fb["level_0"].attrs["iteration"]) == end
        for name in CONSERVED:
            x, y = fa[f"level_0/conserved/{name}"][()], fb[f"level_0/conserved/{name}"][()]
            assert x.shape == y.shape and np.array_equal(x[2:-2, 2:-2].view(np.uint64), y[2:-2, 2:-2].view(np.uint64)), f"{name}: the tiles differ from the single device"


def test_workers_cut_through_the_corrected_cell_equal_the_single_device(tmp_path: Path) -> None:
    resume = _restart_file(tmp_path)
    end = STEP + WINDOW_STEPS
    single = _run(tmp_path / "single", end, resume)
    assert _guards(single)["troubled_cells"]["total"] > 0, "the correction never fired; the cut checks nothing"
    args = ["--run", "N", "--zones-per-decade", "32", "--polar-zones", "44", "--orbits", "3", "--snapshot-orbits", "0.001",
            "--outward-outer", "--checkpoint", str(resume), "--max-steps", str(end)]
    run = ph.launch(REPO / "simbi_configs/examples/newtonian/spb_torus.py", tmp_path / "workers", workers=4, cuts=[[68], [18]], args=args)
    assert set(run.reports()) == {0, 1, 2, 3}, f"workers reporting: {run.reports()}"
    a, b = run.final, next((tmp_path / "single").glob("*.chkpt.final.h5"))
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        assert int(fa["level_0"].attrs["iteration"]) == int(fb["level_0"].attrs["iteration"]) == end
        for name in CONSERVED:
            x, y = fa[f"level_0/conserved/{name}"][()], fb[f"level_0/conserved/{name}"][()]
            assert x.shape == y.shape and np.array_equal(x[2:-2, 2:-2].view(np.uint64), y[2:-2, 2:-2].view(np.uint64)), f"{name}: the workers differ from the single device"
