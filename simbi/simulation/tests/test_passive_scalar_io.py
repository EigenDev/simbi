# =============================================================================
# test_passive_scalar_io.py
#
# the passive scalar end to end through the python chain: the dyed KH example
# runs, the checkpoint carries `chi` in both the primitive and conserved
# groups, the concentration stays in [0, 1], and the dyed region is where the
# config painted it.
# =============================================================================
import glob
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi_configs.examples.newtonian.dyed_kh import DyedKelvinHelmholtz


@pytest.mark.simulation
def test_dyed_checkpoint_carries_chi():
    p = DyedKelvinHelmholtz()
    d = tempfile.mkdtemp() + "/"
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    runner.run(p, compute_mode="cpu", max_steps=150)

    final = glob.glob(d + "*final*.h5")
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as f:
        prim = f["level_0/partition_0/hydro/primitives"]
        cons = f["level_0/conserved"]
        assert "chi" in prim, f"primitives missing chi: {sorted(prim.keys())}"
        assert "chi" in cons, f"conserved missing chi: {sorted(cons.keys())}"
        chi = prim["chi"][:]
        assert np.isfinite(chi).all()
        assert chi.min() > -1e-12 and chi.max() < 1.0 + 1e-12, (
            f"chi left [0, 1]: [{chi.min()}, {chi.max()}]"
        )
        # the dye sits in the central shear layer after 5 steps: interior rows
        # carry chi ~ 1, edge rows chi ~ 0 (ghost band excluded by halo).
        halo = int(f["level_0/mesh"].attrs.get("halo_width", 0))
        core = chi[halo:-halo, halo:-halo] if halo else chi
        ny = core.shape[0]
        assert core[ny // 2, :].mean() > 0.9, "central layer lost its dye"
        assert core[2, :].mean() < 0.1, "ambient rows gained dye"
        # the dye must EVOLVE, not merely persist: the seeded field is an exact
        # 0/1 step, so a frozen chi (the failure mode: a stage driver missing
        # the chi phase) is binary to the bit. any transport — even the tiny
        # early-time seed-noise advection — pulls edge cells measurably off
        # exact 0/1.
        dev = float(np.abs(core - np.round(core)).max())
        assert dev > 1e-8, (
            f"chi did not evolve: max deviation from the binary IC is {dev} "
            f"after 150 steps"
        )


@pytest.mark.simulation
def test_undyed_checkpoint_carries_no_chi():
    from simbi_configs.examples.newtonian.kh import KelvinHelmholtz

    p = KelvinHelmholtz()
    d = tempfile.mkdtemp() + "/"
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    runner.run(p, compute_mode="cpu", max_steps=2)
    final = glob.glob(d + "*final*.h5")
    assert final
    with h5py.File(final[0]) as f:
        prim = f["level_0/partition_0/hydro/primitives"]
        assert "chi" not in prim, "undyed run wrote a chi dataset"
