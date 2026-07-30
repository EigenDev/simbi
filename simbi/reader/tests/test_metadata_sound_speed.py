# =============================================================================
# test_metadata_sound_speed.py
#
# isothermal checkpoints carry the constant sound speed in metadata so
# readers can reconstruct p = cs^2 rho; energy-regime files omit the attr
# and the field reads back as None.
# =============================================================================

import h5py
import pytest

from simbi.reader.io import read_metadata

REQUIRED_ATTRS = {
    "time": 0.0,
    "dt": 1e-3,
    "dlogt": 0.0,
    "tend": 1.0,
    "iteration": 0,
    "checkpoint_index": 0,
    "gamma": 1.0,
    "cfl": 0.3,
    "plm_theta": 1.5,
    "dimensions": 3,
    "coord_system": "cartesian",
    "halo_radius": 2,
    "is_mhd": False,
    "regime": "isothermal",
    "solver": "hlle",
    "reconstruction": "plm",
    "timestepping": "rk2",
}


def _write_meta(path, extra_attrs):
    with h5py.File(path, "w") as f:
        g = f.create_group("metadata")
        for k, v in {**REQUIRED_ATTRS, **extra_attrs}.items():
            g.attrs[k] = v


def test_sound_speed_attr_round_trips(tmp_path) -> None:
    path = tmp_path / "iso.h5"
    _write_meta(path, {"sound_speed": 0.75})
    with h5py.File(path, "r") as f:
        meta = read_metadata(f["metadata"]).unwrap()
    assert meta.sound_speed == pytest.approx(0.75)


def test_missing_sound_speed_reads_as_none(tmp_path) -> None:
    path = tmp_path / "legacy.h5"
    _write_meta(path, {})
    with h5py.File(path, "r") as f:
        meta = read_metadata(f["metadata"]).unwrap()
    assert meta.sound_speed is None


def test_boundary_conditions_read_from_scalar_metadata(tmp_path) -> None:
    path = tmp_path / "boundaries.h5"
    _write_meta(
        path,
        {"boundary_conditions": "periodic,periodic,reflecting,outflow"},
    )
    with h5py.File(path, "r") as f:
        meta = read_metadata(f["metadata"]).unwrap()
    assert meta.boundary_conditions == (
        "periodic",
        "periodic",
        "reflecting",
        "outflow",
    )
