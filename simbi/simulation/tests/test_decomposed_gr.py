# =============================================================================
# test_decomposed_gr.py
#
# relativistic gas and magnetized gas on curved spacetimes on host tiles. each
# fixture runs on one device, on two tiles and on three or four uneven tiles;
# every level dataset matches the single grid bit for bit. the fixtures cover
# the baked GR charts: the radial schwarzschild kerr-schild line (with and
# without excision), the (R, z) meridional plane, the 3d kerr-schild box (with
# and without excision, and excised under a refined slab clear of the core whose
# uneven layout leaves the tiles around the core without a fine level), the 2d
# spherical kerr torus, and the magnetized
# radial line, kerr frame dragging, spherical torus and cartesian torus. each
# reference run moves its density off the initial state, so the comparison
# exercises the evolution.
#
# run: pytest -m simulation simbi/simulation/tests/test_decomposed_gr.py
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

STEPS = 6
# name: (module, class, constructor fields, uneven cuts)
CASES = {
    "bondi-line-excised": ("simbi_configs.examples.grhd.gr_bondi", "GrBondi", "resolution=128, excision_radius=1.4", [[40, 90]]),
    "bondi-line": ("simbi_configs.examples.grhd.gr_bondi", "GrBondi", "resolution=128, excision_radius=0.0", [[40, 90]]),
    "bondi-ks-line": ("simbi_configs.examples.grhd.gr_bondi_ks", "GrBondiKS", "resolution=128", [[40, 90]]),
    "meridional-plane": ("simbi_configs.examples.grhd.gr_cylindrical_ks_bh", "GrCylindricalKsBH", "resolution=(32, 32)", [[10, 21], []]),
    "box": ("simbi_configs.examples.grhd.gr_bondi_cartesian", "GrBondiCartesian", "resolution=(24, 24, 24), domain_radius=0.12, excision_radius=0.0", [[11], [13], []]),
    "box-excised": ("simbi_configs.examples.grhd.gr_bondi_cartesian", "GrBondiCartesian", "resolution=(24, 24, 24), domain_radius=0.12, excision_radius=1.6", [[11], [13], []]),
    # a refined slab y >= 1.8 clear of the excised core, spanning the x cut; the uneven layout leaves
    # the tiles around the core without a fine level
    "box-excised-refined": ("simbi_configs.examples.grhd.gr_bondi_cartesian", "GrBondiCartesian", "resolution=(24, 24, 24), domain_radius=0.24, excision_radius=1.6, refinement_enabled=True, refinement_max_levels=2, refinement_ratios=[2], refinement_regions=[[-3.6, 3.6, 1.8, 3.6, -3.6, 3.6]]", [[11], [13], []]),
    "kerr-torus": ("simbi_configs.examples.grhd.gr_fishbone_moncrief", "GrFishboneMoncrief", "nr=48, npolar=32", [[15, 33], []]),
    "magnetized-ks-line": ("simbi_configs.examples.grmhd.gr_bondi_ks_magnetized", "GrBondiKsMagnetized", "resolution=128", [[40, 90]]),
    "magnetized-michel-line": ("simbi_configs.examples.grmhd.gr_michel_magnetized", "GrMichelMagnetized", "resolution=128", [[40, 90]]),
    "kerr-dragging": ("simbi_configs.examples.grmhd.gr_kerr_dragging", "GrKerrDragging", "nr=48, npolar=32", [[15, 33], []]),
    "magnetized-kerr-torus": ("simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd", "GrFishboneMoncriefMhd", "nr=48, npolar=32", [[15, 33], []]),
    "magnetized-cartesian-torus": ("simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd_cartesian", "GrFishboneMoncriefMhdCartesian", "resolution=(24, 24, 24)", [[11], [13], []]),
}


def _run(case: str, out: Path, gpus: int, cuts=None) -> Path:
    module, cls, fields, _ = CASES[case]
    extra = f", decompose={cuts!r}" if cuts is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {cls}\n"
        f"p = {cls}({fields}, gpus={gpus}, data_directory=sys.argv[1]{extra})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=1800, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    assert done.returncode == 0, f"{case} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert "level_0/conserved/den" in ys, f"{label}: the reference publishes no density; the comparison is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("case", list(CASES))
def test_curved_spacetime_host_tiles_match_the_single_grid(tmp_path: Path, case: str) -> None:
    want = _run(case, tmp_path / "single", 1)
    initial = [p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name]
    assert initial, f"{case}: the reference wrote no initial checkpoint to measure the evolution against"
    with h5py.File(initial[0], "r") as f0, h5py.File(want, "r") as f1:
        assert not np.array_equal(f0["level_0/conserved/den"][()], f1["level_0/conserved/den"][()]), f"{case}: the density never moved; the gate is vacuous"
    _assert_same(_run(case, tmp_path / "two", 2), want, f"{case} on two host tiles")
    cuts = CASES[case][3]
    tiles = int(np.prod([len(axis) + 1 for axis in cuts]))
    _assert_same(_run(case, tmp_path / "uneven", tiles, cuts), want, f"{case} on {tiles} uneven host tiles")
