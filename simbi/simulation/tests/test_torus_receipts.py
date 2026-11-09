# =============================================================================
# test_torus_receipts.py
#
# the production conservation receipts of the viscous torus and its outward-only outer wall,
# through the python frontend on a small grid of the Stone, Pringle & Begelman configuration
# (simbi_configs/examples/newtonian/spb_torus.py, run D: R_B = 0.02):
# - the outward-only outer wall passes no mass inward at any stage, while the plain outflow wall
#   of the same setup draws ambient gas in
# - mass, angular momentum and E + rho Phi close against the ledger at every checkpoint: the wall
#   transport, the viscous torque and the stress work through the walls
# - the shear moves angular momentum and heats the gas; without it the viscous terms are zero
# - a restart continues the receipts bit for bit
# rejected attempts stay out of the accepted totals: the step transaction's own tests in
# symbi-sim/src/step_transaction.rs book receipts through a scripted retried step
# - the batched kernels book the per-channel kernels' receipts bit for bit
#   (SIMBI_LEDGER_REFERENCE=channels), on a bit-identical state
# - the host reference (SIMBI_LEDGER_REFERENCE=host) books receipts on the same evolved state as
#   the kernel path, each closing at roundoff
# - the frontend refuses the wall outside the planes it is baked for
#
# run: pytest -m simulation simbi/simulation/tests/test_torus_receipts.py
# =============================================================================
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import torus_receipts  # noqa: E402

ORBIT = 2.0 * math.pi
# 38 x 24 cells over [0.02, 4] x [0, pi]; 0.008 orbits is a few hundred steps
GRID = "run={run!r}, zones_per_decade=16, polar_zones=24, orbits={orbits}, snapshot_orbits=0.002"


def _run(out: Path, run: str, orbits: float = 0.008, extra: str = "", checkpoint: Path | None = None, env: dict | None = None) -> Path:
    restart = f", checkpoint_file={str(checkpoint)!r}" if checkpoint else ""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'simbi_configs/examples/newtonian')!r})\n"
        "from spb_torus import SpbTorus\n"
        "from simbi.simulation import runner\n"
        f"runner.run(SpbTorus(data_directory=sys.argv[1], {GRID.format(run=run, orbits=orbits)}{extra}{restart}), compute_mode={ph.BACKEND!r})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), f"spb {run} {extra}", env=env)
    return out


def test_the_outward_only_outer_wall_passes_no_mass_inward(tmp_path: Path) -> None:
    _, plain = torus_receipts.closure_series(_run(tmp_path / "plain", "N"))
    _, outward = torus_receipts.closure_series(_run(tmp_path / "outward", "N", extra=", outward_outer=True"))
    # the plain outflow wall of this setup draws the ambient gas inward; the comparison needs it
    assert plain[-1].ledger.mass_in[1] > 0.0, "the plain outer wall drew no mass inward; the setup tests nothing"
    for row in outward[1:]:
        assert row.ledger.mass_in[1] == 0.0, f"t = {row.time}: {row.ledger.mass_in[1]} entered through the outward-only wall"
        assert row.ledger.mass[1] >= 0.0
    # the wall leaves the inner wall's condition as it was: the inner wall still books its transport
    assert outward[-1].ledger.mass[0] != 0.0 and plain[-1].ledger.mass[0] != 0.0


@pytest.mark.parametrize("run", ["D", "N"])
def test_the_receipts_close_mass_angular_momentum_and_energy(tmp_path: Path, run: str) -> None:
    _, rows = torus_receipts.closure_series(_run(tmp_path / run, run))
    first = rows[0]
    assert len(rows) >= 4 and rows[-1].time > 0.0
    for row in rows[1:]:
        dm, dl, de = torus_receipts.residuals(first, row)
        # the bound is round-off: each of the run's few hundred steps rounds the totals of its
        # 912 cells at relative 1e-16, so the drift stays below 1e-12 of each initial total
        assert abs(dm) <= 1e-12 * first.totals[0], f"t = {row.time}: mass residual {dm}"
        assert abs(dl) <= 1e-12 * first.totals[1], f"t = {row.time}: angular momentum residual {dl}"
        assert abs(de) <= 1e-12 * abs(first.totals[2]), f"t = {row.time}: energy residual {de}"
    last = rows[-1].ledger
    assert last.mass[0] != 0.0 and last.energy[0] != 0.0, "no transport through the inner wall; the closure checks the interior alone"


def test_the_shear_transports_and_heats_and_its_absence_books_nothing(tmp_path: Path) -> None:
    # runs A and N share the torus, the grid and the walls; only the stress differs
    grid, viscous = torus_receipts.closure_series(_run(tmp_path / "A", "A"))
    _, inviscid = torus_receipts.closure_series(_run(tmp_path / "N", "N"))
    v, n = viscous[-1].ledger, inviscid[-1].ledger
    assert viscous[-1].time == inviscid[-1].time
    assert v.viscous_steps > 0 and v.viscous_heating > 0.0, f"{v.viscous_steps} viscous steps heated {v.viscous_heating}"
    # the stress carries angular momentum out of the inner torus
    from spb_torus_analysis import read

    def inner_angular_momentum(run: str) -> float:
        final = sorted(p for p in (tmp_path / run).glob("*.chkpt.*.h5"))[-1]
        fields = read(final)[1]
        lever = grid.r[None, :] * np.sin(grid.theta_centroid)[:, None]
        return float((fields["m3"] * lever * grid.volume * (grid.r[None, :] < 1.0)).sum())

    assert inner_angular_momentum("A") < inner_angular_momentum("N"), "the shear moved no angular momentum out of r < 1"
    # the operator's total energy change is the stress work through the walls. each viscous step
    # differences two volume sums of E, each rounded at relative 2.2e-16, so the receipt carries
    # at most 4 x 2.2e-16 x sum(E V) per step
    energy = float((read(sorted((tmp_path / "A").glob("*.chkpt.*.h5"))[-1])[1]["nrg"] * grid.volume).sum())
    assert abs(v.viscous_energy_change - v.viscous_work_in) <= v.viscous_steps * 4.0 * 2.2e-16 * energy
    assert (n.viscous_steps, n.viscous_heating, n.viscous_torque_in, n.viscous_work_in, n.viscous_energy_change) == (0, 0.0, 0.0, 0.0, 0.0)


def test_a_restart_continues_the_receipts_bit_for_bit(tmp_path: Path) -> None:
    from spb_torus_analysis import read

    whole = _run(tmp_path / "whole", "D")
    # a mid-run checkpoint of the same run is an exact prefix of it; a shorter run's end time would
    # clamp its last step and give another state
    numbered = sorted(p for p in whole.glob("*.chkpt.*.h5") if ".final." not in p.name)
    resume = numbered[len(numbered) // 2]
    seeded = torus_receipts.ledger(read(resume)[2])
    assert seeded is not None and seeded.stages > 0, f"{resume.name} carries no receipts to continue"
    resumed = _run(tmp_path / "resumed", "D", checkpoint=resume)
    a_final, b_final = whole / numbered[-1].name, resumed / numbered[-1].name
    ta, _, meta_a = read(a_final)
    tb, _, meta_b = read(b_final)
    assert ta == tb, (ta, tb)
    assert torus_receipts.ledger(meta_b).stages > seeded.stages, "the resumed run booked nothing"
    assert str(meta_a["conservation_ledger"]) == str(meta_b["conservation_ledger"]), "the resumed receipts differ from the uninterrupted run's"
    # the body diagnostics history of a resumed run holds the steps the resumed run took
    ph.same_checkpoint(b_final, a_final, "the resumed state against the uninterrupted run", exclude=("body_diagnostics",), expect=("level_0",))
    # the final checkpoint carries the same receipts, so a restart from it continues them too
    assert str(read(resumed / "38x24.chkpt.final.h5")[2]["conservation_ledger"]) == str(meta_b["conservation_ledger"])


def test_the_host_reference_books_the_same_receipts_on_the_same_state(tmp_path: Path) -> None:
    import os

    from spb_torus_analysis import read

    kernel = _run(tmp_path / "kernel", "A")
    host = _run(tmp_path / "host", "A", env={**os.environ, "SIMBI_LEDGER_REFERENCE": "host"})
    finals = [sorted(p for p in d.glob("*.chkpt.*.h5") if ".final." not in p.name)[-1] for d in (kernel, host)]
    # the receipts read the state; they leave it untouched under either path
    ph.same_checkpoint(finals[1], finals[0], "the host reference's state against the kernel path's", exclude=("metadata", "body_diagnostics"),
                       expect=("level_0",))
    ledgers = [torus_receipts.ledger(read(p)[2]) for p in finals]
    assert ledgers[0].stages == ledgers[1].stages > 0
    for d in (kernel, host):
        _, rows = torus_receipts.closure_series(d)
        for row in rows[1:]:
            dm, dl, de = torus_receipts.residuals(rows[0], row)
            assert abs(dm) <= 1e-12 * rows[0].totals[0] and abs(dl) <= 1e-12 * rows[0].totals[1] and abs(de) <= 1e-12 * abs(rows[0].totals[2]), (d.name, dm, dl, de)
    assert ledgers[0].viscous_heating > 0.0 and ledgers[1].viscous_heating > 0.0


def test_the_batched_receipts_equal_the_per_channel_kernels_bit_for_bit(tmp_path: Path) -> None:
    import os

    from spb_torus_analysis import read

    batched = _run(tmp_path / "batched", "A")
    channels = _run(tmp_path / "channels", "A", env={**os.environ, "SIMBI_LEDGER_REFERENCE": "channels"})
    finals = [sorted(p for p in d.glob("*.chkpt.*.h5") if ".final." not in p.name)[-1] for d in (batched, channels)]
    ph.same_checkpoint(finals[1], finals[0], "the per-channel receipts' state against the batched path's", exclude=("metadata", "body_diagnostics"),
                       expect=("level_0",))
    metas = [read(p)[2] for p in finals]
    assert torus_receipts.ledger(metas[0]).stages > 0, "the batched run booked no receipts"
    for key in ("conservation_ledger", "conservation_attempted", "conservation_window", "conservation_corrections"):
        if key == "conservation_ledger":
            assert key in metas[0], "the checkpoint carries no receipts"
        if key in metas[0] or key in metas[1]:
            assert str(metas[0][key]) == str(metas[1][key]), f"{key}: the batched receipts differ from the per-channel kernels'"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("flags", "outward_only_walls names one flag per face of the plane, four"),
        ("components", "the wall is baked for the adiabatic newtonian gas on the spherical and cylindrical planes with three velocity components"),
    ],
)
def test_the_wall_outside_its_baked_planes_is_refused_by_name(tmp_path: Path, change: str, message: str) -> None:
    out = tmp_path / "run"
    edit = {
        "flags": "p.outward_only_walls = [False, True]\n",
        "components": "p.velocity_components = 2\n",
    }[change]
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'simbi_configs/examples/newtonian')!r})\n"
        "from spb_torus import SpbTorus\n"
        "from simbi.simulation import runner\n"
        f"p = SpbTorus(data_directory=sys.argv[1], {GRID.format(run='N', orbits=0.008)}, outward_outer=True)\n"
        f"{edit}"
        f"runner.run(p, compute_mode={ph.BACKEND!r})\n"
    )
    done = ph.python(ph.snippet_args(snippet, str(out)), f"outward wall {change}, expecting a refusal", check=False)
    flat = " ".join(done.stderr.split())
    assert done.returncode != 0, f"{change}: the run succeeded"
    assert message in flat, f"{change}: {flat[-1500:]}"
    assert not list(out.glob("*.h5")), f"{change}: output was written before the refusal"
