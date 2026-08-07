# =============================================================================
# test_scale_invariant_seed_backend.py
#
# the velocity seed's trip through the BACKEND: the mode table crossing the
# python/rust boundary and the per-level application on a real hierarchy. these
# evolve, so they live apart from the seed's pure-python invariance gates -- a
# module that calls the driver is marked `simulation` wholesale, and folding
# these in would pull those fast gates out of the default battery.
#
# what only an end-to-end run can see:
# - the payload's shape contract in both directions. the seedless payload (an
#   empty mode table and an empty taper) is what EVERY config emits by default,
#   so a backend that rejects it breaks every run in the repository rather than
#   just the seeded ones.
# - that the seed is applied per level rather than to the root and prolonged.
#   the two are indistinguishable from python: both leave the same mode table in
#   the exec dict.
# =============================================================================

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "simbi_configs"
    / "science"
    / "simbi_projects"
    / "porous_turbulent_accretor.py"
)

_EPSILON = 1.0 / 16.0
# the smallest grid whose ladder still builds several levels, which is what the
# per-level claim needs to be visible at all.
_BASE_RESOLUTION = 32
_ZONES_PER_BONDI = 48
_BONDI_TIMES = 0.02


def _problem_class():
    if not _CONFIG.is_file():
        pytest.skip("the porous accretor config is not present in this checkout")
    spec = importlib.util.spec_from_file_location("_pta_backend", _CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pta_backend"] = module
    spec.loader.exec_module(module)
    return module.PorousTurbulentAccretor


def _tiny_run(tmp_path: Path, **overrides):
    """a one-step refined run, returning its first checkpoint."""
    from simbi.reader import read_checkpoint
    from simbi.simulation import runner

    problem = _problem_class()(
        initial_profile="hydrostatic",
        base_resolution=_BASE_RESOLUTION,
        target_zones_per_bondi=_ZONES_PER_BONDI,
        total_bondi_times=_BONDI_TIMES,
        data_directory=tmp_path,
        **overrides,
    )
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
        runner.run(problem, compute_mode="cpu", max_steps=1)
    written = sorted(tmp_path.rglob("*.chkpt.*.h5"))
    assert written, "the run wrote no checkpoint"
    return problem, read_checkpoint(str(written[0])).unwrap()


def _level_speed(checkpoint, level: int) -> float:
    primitives = checkpoint.levels[level].partitions[0].hydro.primitives
    components = np.stack(
        [np.asarray(primitives[name].data) for name in ("v1", "v2", "v3")]
    )
    return float(np.sqrt(np.mean(np.sum(components**2, axis=0))))


def test_backend_lands_more_seed_power_on_every_finer_level(tmp_path) -> None:
    """each finer level covers a smaller radius, where v_K is larger and more
    octaves survive their cutoff, so the stored speed must RISE down the ladder.
    a seed delivered to the root and prolonged carries no octave the root cannot
    represent and would give a flat or falling profile -- which is precisely the
    initial condition the band-limited seed produced, and the reason the science
    window went unseeded."""
    _, checkpoint = _tiny_run(tmp_path, seed_epsilon=_EPSILON)
    assert checkpoint.num_levels > 2, (
        f"the ladder built {checkpoint.num_levels} level(s); this gate needs a "
        "hierarchy to compare across"
    )
    speeds = [_level_speed(checkpoint, lv) for lv in range(checkpoint.num_levels)]
    assert speeds[0] > 0.0, "the root level carries no seed at all"
    for lv in range(1, len(speeds)):
        assert speeds[lv] > speeds[lv - 1], (
            f"level {lv} carries speed {speeds[lv]:.5g} against "
            f"{speeds[lv - 1]:.5g} on level {lv - 1} (profile "
            f"{np.array2string(np.array(speeds), precision=4)}); the finer level "
            "holds no content beyond the prolongation"
        )


def test_an_unseeded_run_still_builds(tmp_path) -> None:
    """the seedless payload -- an empty mode table AND an empty taper -- is the
    default every config emits, so it must cross the boundary untouched. a
    backend that applied the taper's shape contract before checking for modes
    would fail every run in the repository, seeded or not."""
    _, checkpoint = _tiny_run(tmp_path, seed_epsilon=0.0, turb_mach=0.0)
    assert _level_speed(checkpoint, 0) == 0.0, (
        "an unseeded, unturbulent initial condition carries velocity"
    )
