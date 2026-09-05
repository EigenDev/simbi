# =============================================================================
# test_magnetic_slip_config.py
#
# the public path of the magnetic-slip sink: a python `MagneticSlipProperties` on an
# accreting body reaches the backend as the `magnetic.slip` wire, is refused where the
# operator has no meaning (a surface that removes no mass, degenerate scales), and drives a
# cpu run whose field departs from the magnetically transparent sink, from a fresh start
# and across a checkpoint restart.
# =============================================================================
import glob
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi.simulation.problem import ConfigError
from simbi.types.bodies import (
    AccretionProperties,
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
    MagneticSlipProperties,
    RigidProperties,
)
from simbi_configs.examples.ibm.magnetic_slip_disk_2p5d import MagneticSlipDisk2p5d
from simbi_configs.examples.ibm.magnetic_slip_sink import MagneticSlipBondiSink

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

RES = 16


def _slip(**overrides) -> MagneticSlipProperties:
    kw = dict(diffusivity_ratio=2.0, shell_width=0.05, field_regularization=0.01)
    kw.update(overrides)
    return MagneticSlipProperties(**kw)


def _accretor(**accretion) -> ImmersedBodyConfig:
    return ImmersedBodyConfig(
        capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
        mass=1.0,
        radius=0.1,
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        gravitational=GravitationalProperties(softening_length=0.05),
        accretion=AccretionProperties(accretion_radius=0.1, **accretion),
    )


def test_slip_scales_are_validated_at_construction() -> None:
    for bad in (
        dict(shell_width=0.0),
        dict(diffusivity_ratio=-1.0),
        dict(field_regularization=math.nan),
        dict(slip_length_ratio=0.0),
        dict(placement=math.inf),
    ):
        with pytest.raises(ConfigError):
            _slip(**bad)
    ok = _slip()
    assert ok.slip_length_ratio == 1.0 and ok.placement == 0.0


def test_slip_requires_a_mass_removing_surface() -> None:
    # a plain drain, a torque-free drain, and a porous surface with porosity > 0 remove mass.
    replace(_accretor(), magnetic=_slip())
    replace(_accretor(torque_free_xi=0.5), magnetic=_slip())
    replace(_accretor(porosity=0.5), magnetic=_slip())
    with pytest.raises(ConfigError):
        replace(_accretor(porosity=0.0), magnetic=_slip())
    with pytest.raises(ConfigError):
        ImmersedBodyConfig(
            capability=BodyCapability.RIGID,
            mass=1.0,
            radius=0.1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            rigid=RigidProperties(inertia=1.0, apply_no_slip=True),
            magnetic=_slip(),
        )
    with pytest.raises(ConfigError):
        ImmersedBodyConfig(
            capability=BodyCapability.GRAVITATIONAL,
            mass=1.0,
            radius=0.1,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            gravitational=GravitationalProperties(softening_length=0.05),
            magnetic=_slip(),
        )


class _TransparentSink(MagneticSlipBondiSink):
    """the same bondi sink with no magnetic coupling: the control the slip run departs from."""

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [replace(b, magnetic=None) for b in super().immersed_bodies]


def _problem(cls, directory: Path, **kw):
    return cls(
        resolution=(RES, RES, RES),
        r_acc_scale=2.0,
        shell_cells=2.0,
        data_directory=directory,
        **kw,
    )


def _final(directory: Path) -> Path:
    files = list(Path(directory).glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {directory}, found {files}"
    return files[0]


def _interior(directory: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(_final(directory), "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        for nm in names:
            arr = prims[nm][...]
            sl = tuple(slice((n - RES) // 2, (n - RES) // 2 + RES) for n in arr.shape)
            out[nm] = arr[sl]
    return out


@needs_backend
def test_slip_from_config_genuinely_acts() -> None:
    slip_dir = Path(tempfile.mkdtemp())
    control_dir = Path(tempfile.mkdtemp())
    runner.run(_problem(MagneticSlipBondiSink, slip_dir), compute_mode="cpu", max_steps=20)
    runner.run(_problem(_TransparentSink, control_dir), compute_mode="cpu", max_steps=20)
    assert not glob.glob(os.path.join(slip_dir, "*crashed*")), "the slip run crashed"
    slip = _interior(slip_dir, ("rho", "b1", "pre"))
    control = _interior(control_dir, ("rho", "b1", "pre"))
    for nm, arr in slip.items():
        assert np.all(np.isfinite(arr)), f"non-finite {nm} in the slip run"
    bscale = np.abs(control["b1"]).max()
    assert bscale > 1e-6, "the field never developed; the comparison is vacuous"
    db = np.abs(slip["b1"] - control["b1"]).max()
    assert db > 1e-6 * bscale, (
        f"the magnetic slip left the field bit-near-identical to the transparent sink ({db:e}); "
        "the config value is being dropped by the builder chain"
    )


def _assert_tree_equal(first: h5py.Group, second: h5py.Group) -> None:
    assert set(first.keys()) == set(second.keys())
    for name in first:
        if isinstance(first[name], h5py.Group):
            _assert_tree_equal(first[name], second[name])
        else:
            np.testing.assert_array_equal(first[name][...], second[name][...])


@needs_backend
def test_slip_restart_matches_the_uninterrupted_run(tmp_path) -> None:
    continuous = tmp_path / "continuous"
    split = tmp_path / "split"
    restarted = tmp_path / "restarted"
    runner.run(_problem(MagneticSlipBondiSink, continuous), compute_mode="cpu", max_steps=6)
    runner.run(_problem(MagneticSlipBondiSink, split), compute_mode="cpu", max_steps=3)
    runner.run(
        _problem(MagneticSlipBondiSink, restarted, checkpoint_file=_final(split)),
        compute_mode="cpu",
        max_steps=6,
    )
    with h5py.File(_final(continuous)) as a, h5py.File(_final(restarted)) as b:
        assert a["metadata"].attrs["iteration"] == 6
        assert b["metadata"].attrs["iteration"] == 6
        assert a["metadata"].attrs["time"] == b["metadata"].attrs["time"]
        _assert_tree_equal(a["level_0"], b["level_0"])


# ---- the 2.5D public path -------------------------------------------------------------------------


class _TransparentDisk(MagneticSlipDisk2p5d):
    """the same vertically magnetized plane with no magnetic coupling."""

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [replace(b, magnetic=None) for b in super().immersed_bodies]


def _disk(cls, directory: Path, **kw):
    return cls(resolution=(RES, RES, 1), r_acc_scale=2.0, shell_cells=2.0, data_directory=directory, **kw)


def _interior_2d(directory: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(_final(directory), "r") as h:
        prims = h["level_0/partition_0/hydro/primitives"]
        for nm in names:
            arr = prims[nm][...]
            sl = tuple(slice((n - RES) // 2, (n - RES) // 2 + RES) if n > RES else slice(None) for n in arr.shape)
            out[nm] = arr[sl]
    return out


@needs_backend
def test_2p5d_slip_from_config_genuinely_acts() -> None:
    slip_dir = Path(tempfile.mkdtemp())
    control_dir = Path(tempfile.mkdtemp())
    runner.run(_disk(MagneticSlipDisk2p5d, slip_dir), compute_mode="cpu", max_steps=20)
    runner.run(_disk(_TransparentDisk, control_dir), compute_mode="cpu", max_steps=20)
    assert not glob.glob(os.path.join(slip_dir, "*crashed*")), "the 2.5D slip run crashed"
    slip = _interior_2d(slip_dir, ("rho", "b3", "pre"))
    control = _interior_2d(control_dir, ("rho", "b3", "pre"))
    for nm, arr in slip.items():
        assert np.all(np.isfinite(arr)), f"non-finite {nm} in the 2.5D slip run"
    # the vertical field is the channel the slip acts on in this configuration.
    bscale = np.abs(control["b3"]).max()
    assert bscale > 1e-6, "the vertical field vanished; the comparison is vacuous"
    db = np.abs(slip["b3"] - control["b3"]).max()
    assert db > 1e-6 * bscale, (
        f"the 2.5D magnetic slip left the vertical field bit-near-identical to the transparent sink "
        f"({db:e}); the config value is being dropped by the builder chain"
    )


@needs_backend
def test_2p5d_slip_restart_matches_the_uninterrupted_run(tmp_path) -> None:
    continuous = tmp_path / "continuous"
    split = tmp_path / "split"
    restarted = tmp_path / "restarted"
    runner.run(_disk(MagneticSlipDisk2p5d, continuous), compute_mode="cpu", max_steps=6)
    runner.run(_disk(MagneticSlipDisk2p5d, split), compute_mode="cpu", max_steps=3)
    runner.run(
        _disk(MagneticSlipDisk2p5d, restarted, checkpoint_file=_final(split)),
        compute_mode="cpu",
        max_steps=6,
    )
    with h5py.File(_final(continuous)) as a, h5py.File(_final(restarted)) as b:
        assert a["metadata"].attrs["iteration"] == 6
        assert b["metadata"].attrs["iteration"] == 6
        assert a["metadata"].attrs["time"] == b["metadata"].attrs["time"]
        _assert_tree_equal(a["level_0"], b["level_0"])
