# =============================================================================
# test_tracers.py
#
# ownership-native tracer checkpoint loading. exact uint64 identities and
# owners must never pass through floating point, and deterministic spawn state
# remains available to python consumers.
# =============================================================================

import h5py
import numpy as np

from simbi.viz.tracers import load_tracers


def test_load_tracers_preserves_exact_ownership_and_spawn_state(tmp_path):
    path = tmp_path / "tracers.h5"
    ids = np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64)
    owners = np.array([2**56 + 7, 2**62], dtype=np.uint64)
    with h5py.File(path, "w") as checkpoint:
        group = checkpoint.create_group("tracers")
        group.attrs["run_seed"] = np.uint64(2**64 - 3)
        group.attrs["next_id"] = np.uint64(2**63 + 9)
        group.attrs["injection_remainder"] = 0.125
        group.create_dataset("position", data=np.array([[0.25], [0.75]]))
        group.create_dataset("id", data=ids)
        group.create_dataset("owner", data=owners)
        group.create_dataset("escaped", data=np.array([0.0, 0.0]))
        group.create_dataset("crossed_sink", data=np.array([0.0, 1.0]))
        group.create_dataset("crossing_time", data=np.array([0.0, 0.5]))
        group.create_dataset("weight", data=np.array([0.01]))

    cloud = load_tracers(str(path))

    assert cloud is not None
    np.testing.assert_array_equal(cloud.id, ids)
    np.testing.assert_array_equal(cloud.owner, owners)
    assert cloud.id.dtype == np.dtype("uint64")
    assert cloud.owner.dtype == np.dtype("uint64")
    assert cloud.run_seed == 2**64 - 3
    assert cloud.next_id == 2**63 + 9
    assert cloud.injection_remainder == 0.125
