# =============================================================================
# test_launch_isothermal.py
#
# the isothermal closures through `simbi launch`: isothermal hydro on the
# shear layer and isothermal MHD on the orszag-tang vortex, both globally
# isothermal, against the single-grid run, resumed onto a changed partition,
# and with the correction fired against the cut; a locally isothermal closure
# with refinement is refused by name, since the refined worker levels seed no
# closure field. the flat locally isothermal launch is gated in
# test_locally_isothermal_tiles. the single-grid references and the launches
# run on the session's backend and placement (parity_harness); `_script_launch`
# and `_script_reference` serve the other launch modules the same way.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_isothermal.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang
from simbi_configs.examples.isothermal.isothermal_kh import IsothermalKelvinHelmholtz
from simbi_configs.examples.isothermal.refined_locally_iso import RefinedLocallyIso

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import (
    MHD_STEPS,
    STEPS,
    _assert_same_state,
    _interior,
    _max_div_b,
    bounded_processes,  # noqa: F401  the autouse process guard
)

pytestmark = pytest.mark.simulation
# -----------------------------------------------------------------------------
# the isothermal closures through the public path: isothermal hydro on the shear layer,
# isothermal MHD on the orszag-tang vortex, both globally isothermal; a locally isothermal
# closure with refinement is refused by name, since the refined worker levels seed no closure
# field.
# -----------------------------------------------------------------------------

REFERENCE_SNIPPET = """
import sys
from simbi.simulation import runner
from {module} import {cls}
out, steps, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
runner.run({cls}(resolution={resolution!r}, data_directory=out, checkpoint_interval=1e9{extra}), compute_mode=mode, max_steps=steps)
"""


def _launch(script: str, resolution: tuple[int, ...], out: Path, steps: int, workers: int, cuts=None, shape=None,
            checkpoint: Path | None = None, env: dict[str, str] | None = None, flags: list[str] | None = None,
            staging_mb: int = 1) -> Path:
    """`simbi launch` of `script` on the session's backend and placement, every worker exiting zero
    without the experimental opt-in; `env` adds variables (an injected fault, an audit) to this
    process's environment. its final checkpoint"""
    args = ["--max-steps", str(steps), "--resolution", ",".join(str(r) for r in resolution), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    args += flags or []
    environ = dict(ph.environment(), **(env or {}))
    run = ph.launch(script, out, workers, cuts=cuts, shape=shape, args=args, env=environ, timeout=600, staging_mb=staging_mb)
    return run.final


def _script_launch(script: str, resolution: tuple[int, ...], out: Path, layout: Path, steps: int, workers: int, checkpoint: Path | None = None, env: dict[str, str] | None = None, flags: list[str] | None = None) -> Path:
    """`_launch` over the workers, partition and staging a layout file names; the session's
    placement writes its own layout from them"""
    with open(layout, "rb") as f:
        spec = tomllib.load(f)
    assert spec["execution"]["workers"] == workers, f"{layout} names {spec['execution']['workers']} workers, the call {workers}"
    checkpoint_spec = spec.get("checkpoint", {})
    assert "directory" not in checkpoint_spec, f"{layout} names a checkpoint directory; the launch writes into {out}"
    partition = spec["partition"]
    return _launch(script, resolution, out, steps, workers, cuts=partition.get("cuts"), shape=partition.get("shape"),
                   checkpoint=checkpoint, env=env, flags=flags, staging_mb=checkpoint_spec.get("staging_mb", 1))


def _script_reference(module: str, cls: str, resolution: tuple[int, ...], out: Path, steps: int, fields: str = "") -> Path:
    """the single-grid run of `cls` on the session's backend, one device; `fields` are extra
    constructor arguments, `nu=0.05`"""
    snippet = REFERENCE_SNIPPET.format(module=module, cls=cls, resolution=resolution, extra=f", {fields}" if fields else "")
    ph.python(ph.snippet_args(snippet, str(out), str(steps), ph.BACKEND), "reference", timeout=600)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _same_state(got: Path, want: Path, label: str) -> None:
    """the whole checkpoint bit for bit, a difference raised as a NumericalDisagreement"""
    ph.disagreement(_assert_same_state, got, want, f"{label}, {ph.BACKEND}")


ISO_KH_SCRIPT = str(Path(IsothermalKelvinHelmholtz.__module__.replace(".", "/") + ".py"))
IMHD_OT_SCRIPT = str(Path(IsothermalOrszagTang.__module__.replace(".", "/") + ".py"))
# a two-dimensional locally isothermal problem on a refined mesh
LOCAL_ISO_SCRIPT = str(Path(RefinedLocallyIso.__module__.replace(".", "/") + ".py"))


def test_an_isothermal_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _script_reference(IsothermalKelvinHelmholtz.__module__, "IsothermalKelvinHelmholtz", (32, 32), tmp_path / "single", STEPS)
    launched = _launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "two", STEPS, workers=2, shape=(2, 1))
    _same_state(launched, reference, "two isothermal workers")


def test_an_isothermal_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path) -> None:
    reference = _script_reference(IsothermalKelvinHelmholtz.__module__, "IsothermalKelvinHelmholtz", (32, 32), tmp_path / "single", STEPS)
    partial = _launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "first", STEPS // 2, workers=2, shape=(2, 1))
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # the step cap is an absolute root iteration: the resume from written_at runs to STEPS
    resumed = _launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "three", STEPS, workers=3, cuts=[[10, 21], []], checkpoint=partial)
    _same_state(resumed, reference, "isothermal, resumed on three workers")


def test_an_isothermal_launch_with_a_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:15:10"}
    clean = _launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "clean", STEPS, workers=1, shape=(1, 1))
    reference = _launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "one", STEPS, workers=1, shape=(1, 1), env=inject)
    launched = _launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "two", STEPS, workers=2, shape=(2, 1), env=inject)
    with h5py.File(clean, "r") as fc, h5py.File(reference, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the flag left the one-worker run unchanged; the gate is vacuous"
    _same_state(launched, reference, "two isothermal workers, trouble against the cut")


def test_an_isothermal_magnetized_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _script_reference(IsothermalOrszagTang.__module__, "IsothermalOrszagTang", (32, 32, 1), tmp_path / "single", MHD_STEPS)
    launched = _launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "two", MHD_STEPS, workers=2, shape=(2, 1), env={"SIMBI_FABRIC_AUDIT": "all"})
    with h5py.File(reference, "r") as f:
        assert "level_0/partition_0/hydro/magnetic/B1/data" in f
    _same_state(launched, reference, "two isothermal magnetized workers")
    assert _max_div_b(launched) < 1e-10


def test_an_isothermal_magnetized_launch_with_a_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:15:10", "SIMBI_FABRIC_AUDIT": "all"}
    clean = _launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "clean", MHD_STEPS, workers=1, shape=(1, 1))
    reference = _launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "one", MHD_STEPS, workers=1, shape=(1, 1), env=inject)
    launched = _launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "two", MHD_STEPS, workers=2, shape=(2, 1), env=inject)
    with h5py.File(clean, "r") as fc, h5py.File(reference, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the flag left the one-worker run unchanged; the gate is vacuous"
    _same_state(launched, reference, "two isothermal magnetized workers, trouble against the cut")
    assert _max_div_b(launched) < 1e-10


def test_a_refined_locally_isothermal_launch_matches_the_single_hierarchy_fresh_and_resumed(tmp_path: Path) -> None:
    # the closure varies across the fine box and the cuts; every worker rebuilds the whole patch's
    # closure at startup and slices its tiles' cells from it, as the single hierarchy holds them
    want = _script_reference(RefinedLocallyIso.__module__, "RefinedLocallyIso", (32, 32, 1), tmp_path / "single", 4)
    with h5py.File(want, "r") as f:
        names: list[str] = []
        f["level_1"].visit(lambda name: names.append(name) if name.endswith("iso_cs2") else None)
        fine = f["level_1"][names[0]][()]
    assert np.ptp(fine) > 1e-3 * np.abs(fine).max(), "the fine closure is uniform; the comparison is vacuous"
    four = [[12], [14]]
    got = _launch(LOCAL_ISO_SCRIPT, (32, 32, 1), tmp_path / "four", 4, workers=4, cuts=four)
    _same_levels(got, want, "locally isothermal refined launch on four workers")
    first = _launch(LOCAL_ISO_SCRIPT, (32, 32, 1), tmp_path / "first", 2, workers=2, shape=(2, 1))
    resumed = _launch(LOCAL_ISO_SCRIPT, (32, 32, 1), tmp_path / "resumed", 4, workers=4, cuts=four, checkpoint=first)
    _same_levels(resumed, want, "locally isothermal refined launch resumed onto four workers")


def _same_levels(got: Path, want: Path, label: str) -> None:
    """every level bit for bit, a difference raised as a NumericalDisagreement"""
    ph.disagreement(_assert_same_levels, got, want, f"{label}, {ph.BACKEND}")


def _assert_same_levels(got: Path, want: Path, label: str) -> None:
    # every cell and face dataset of every level whole, the halos included: every writer publishes
    # the conserved halo from the halo primitives the boundary and exchange machinery filled
    def datasets(path: Path) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            for level in (k for k in f.keys() if k.startswith("level_")):
                f[level].visititems(lambda n, obj, lv=level: out.__setitem__(f"{lv}/{n}", obj[()])
                                    if isinstance(obj, h5py.Dataset) and not n.startswith("mesh") else None)
        return out

    xs, ys = datasets(got), datasets(want)
    assert any(name.startswith("level_1/") for name in ys), f"{label}: the reference has no refined level; the gate is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
