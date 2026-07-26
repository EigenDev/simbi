# =============================================================================
# test_tracer_io.py
#
# the tracer chain end to end through the python runner: n_tracers on a problem
# seeds a mass-weighted population, the checkpoint carries the `tracers` group
# (owners, derived positions, ids, flags, weight), the population moved from
# its seed, and an untraced run writes
# no group.
# =============================================================================
import glob
import tempfile

import h5py
import numpy as np
import pytest
from pydantic import computed_field

from simbi.simulation import runner
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz

N_TRACERS = 400


class TracedKH(KelvinHelmholtz):
    @computed_field
    @property
    def n_tracers(self) -> int:
        return N_TRACERS


@pytest.mark.simulation
def test_traced_checkpoint_carries_a_moving_population():
    p = TracedKH()
    d = tempfile.mkdtemp() + "/"
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    runner.run(p, compute_mode="cpu", max_steps=120)

    final = glob.glob(d + "*final*.h5")
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as f:
        assert "tracers" in f, f"no tracers group: {sorted(f.keys())}"
        g = f["tracers"]
        assert int(g.attrs["n_tracers"]) == N_TRACERS
        x = g["position"][:]
        assert x.shape == (N_TRACERS, 2)
        assert np.isfinite(x).all()
        assert float(g["weight"][0]) > 0.0
        # the shear flow must have MOVED the population off its stratified
        # seed lattice: on the seed, every tracer x-coordinate sits at a
        # cell-relative stratum; after 120 steps of +-0.5 shear the
        # x-distribution decorrelates from any lattice. cheap detector: the
        # population's x-spread of fractional cell coordinates is non-lattice.
        ids = g["id"][:]
        owners = g["owner"][:]
        assert ids.dtype == np.dtype("uint64")
        assert owners.dtype == np.dtype("uint64")
        assert len(owners) == N_TRACERS
        assert int(g.attrs["next_id"]) == N_TRACERS
        assert len(np.unique(ids)) == N_TRACERS, "tracer ids must be unique"
        dx = 1.0 / 256  # kh domain [-0.5, 0.5] over 256 cells
        frac = np.mod((x[:, 0] + 0.5) / dx, 1.0)
        assert frac.std() > 0.05, (
            f"tracer x-fractions still lattice-locked (std {frac.std():.4f}): "
            "the population did not advect"
        )


@pytest.mark.simulation
def test_untraced_checkpoint_has_no_tracer_group():
    p = KelvinHelmholtz()
    d = tempfile.mkdtemp() + "/"
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    runner.run(p, compute_mode="cpu", max_steps=2)
    final = glob.glob(d + "*final*.h5")
    assert final
    with h5py.File(final[0]) as f:
        assert "tracers" not in f, "untraced run wrote a tracers group"
