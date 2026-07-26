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

    initial_directory = tempfile.mkdtemp() + "/"
    p.data_directory = initial_directory
    runner.run(p, compute_mode="cpu", max_steps=1)
    initial_final = glob.glob(initial_directory + "*final*.h5")
    assert initial_final, "no initial checkpoint written"
    with h5py.File(initial_final[0]) as checkpoint:
        initial_ids = checkpoint["tracers/id"][:]
        initial_owners = checkpoint["tracers/owner"][:]

    p.data_directory = d
    runner.run(p, compute_mode="cpu", max_steps=120)

    final = glob.glob(d + "*final*.h5")
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as f:
        assert "tracers" in f, f"no tracers group: {sorted(f.keys())}"
        g = f["tracers"]
        assert int(g.attrs["n_tracers"]) == N_TRACERS
        assert "cohort" in g
        assert np.asarray(g["cohort"]).dtype == np.dtype("uint64")
        x = g["position"][:]
        assert x.shape == (N_TRACERS, 2)
        assert np.isfinite(x).all()
        assert float(g["weight"][0]) > 0.0
        ids = g["id"][:]
        owners = g["owner"][:]
        assert ids.dtype == np.dtype("uint64")
        assert owners.dtype == np.dtype("uint64")
        assert len(owners) == N_TRACERS
        assert int(g.attrs["next_id"]) == N_TRACERS
        assert len(np.unique(ids)) == N_TRACERS, "tracer ids must be unique"
        np.testing.assert_array_equal(ids, initial_ids)
        assert np.any(owners != initial_owners), (
            "the shear flow did not move any authoritative tracer owner"
        )

        # live positions are derived cell centroids, not independently advected
        # state. every cartesian coordinate must therefore sit at half a cell.
        dx = 1.0 / 256  # kh domain [-0.5, 0.5] over 256 cells
        frac = np.mod((x[:, 0] + 0.5) / dx, 1.0)
        np.testing.assert_allclose(frac, 0.5, atol=1.0e-12)


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
