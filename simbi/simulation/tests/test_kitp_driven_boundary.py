# =============================================================================
# test_kitp_driven_boundary.py
#
# the kitp thin-disk config (Duffell+2024 comparison) runs with DYNAMIC
# (dirichlet-to-equilibrium) boundaries on all four faces plus the buffer
# sponge, under the locally isothermal closure. this gate drives the full
# stack a few steps at reduced resolution and pins the three failure modes the
# combination exposed:
# - the driven prescription must lower against the isothermal regime spec
#   ([rho, vx, vy], no pressure slot) and fill EVERY ghost cell, corners
#   included (an interior-clamped driven band left corner ghosts at rho = 0,
#   which a viscous 3x3 stencil reads as gas);
# - the ghost temperature must be the clamped continuation of the initial
#   p/rho; the constructor's uniform cs^2 = 1 would book a ~1000x
#   spurious wall pressure into every boundary flux on a cold disk edge;
# - the evolved state stays finite and positive.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np

from simbi.simulation import runner
from simbi_configs.science.simbi_projects.kitp_comp import (
    BinaryThinDiskSimulation,
)


def test_kitp_disk_steps_with_driven_boundaries() -> None:
    d = tempfile.mkdtemp() + "/"
    p = BinaryThinDiskSimulation(
        zones_per_separation=10,
        data_directory=d,
        checkpoint_interval=1.0e30,
    )
    runner.run(p, compute_mode="cpu", max_steps=5)

    assert not glob.glob(os.path.join(d, "*crashed*")), "kitp run crashed"
    final = glob.glob(os.path.join(d, "*final*.h5"))
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as h:
        prim = h["level_0/partition_0/hydro/primitives"]
        rho = prim["rho"][:]
        assert np.isfinite(rho).all(), "non-finite density"
        # the checkpoint includes the ghost halo: a zero anywhere means an
        # unwritten ghost (the corner blocks are the regression target).
        assert float(rho.min()) > 0.0, (
            f"density non-positive (min {rho.min():.3e}) — unwritten ghost cells?"
        )
        if "pre" in prim:
            pre = prim["pre"][:]
            assert np.isfinite(pre).all(), "non-finite pressure"
            assert float(pre.min()) > 0.0, "pressure non-positive"
            # the local closure p = cs^2(x) rho with cs^2 = (H/r)^2 |Phi| is cold
            # everywhere (cs^2 <= ~5e-3 at the cavity edge); a ghost carrying the
            # constructor's uniform cs^2 = 1 shows up as p/rho ~ 1.
            cs2 = pre / rho
            assert float(cs2.max()) < 0.05, (
                f"p/rho max {cs2.max():.3e}: a ghost kept the uniform cs^2 = 1 "
                f"instead of the clamped local temperature"
            )
