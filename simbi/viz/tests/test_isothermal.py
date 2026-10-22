# =============================================================================
# test_isothermal.py
#
# isothermal plots use the checkpoint's sound speed and newtonian velocities.
# synthetic checkpoints exercise pressure reconstruction, halos, and slicing.
# =============================================================================

import h5py
import numpy as np
import pytest

from simbi.reader.adapter import SimData
from simbi.reader.computation import FieldComputationError
from simbi.reader.io import read_checkpoint
from simbi.viz.config import PlotConfig, VisualizationConfig
from simbi.viz.pipeline.plot_data import create_plot_data


def checkpoint(tmp_path, *, regime="isothermal", cs=2.0, local_cs2=None, ndim=3):
    shape = (2, 4, 6)[-ndim:]
    rho = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    path = tmp_path / "isothermal.h5"
    with h5py.File(path, "w") as file:
        file.attrs["format_version"] = "2.0"
        meta = file.create_group("metadata")
        meta.attrs.update(
            time=0.0, dt=0.01, dlogt=0.0, tend=1.0, iteration=0,
            checkpoint_index=0, gamma=1.0, cfl=0.3, plm_theta=1.5,
            dimensions=ndim, coord_system="cartesian", halo_radius=1,
            is_mhd=regime == "imhd", regime=regime, eos="ideal",
            solver="hlle", reconstruction="plm", timestepping="rk2",
        )
        if cs is not None:
            meta.attrs["sound_speed"] = cs
        level = file.create_group("level_0")
        mesh = level.create_group("mesh")
        mesh["global_cells"] = shape
        mesh.attrs["halo_width"] = 1
        geometry = mesh.create_group("geometry")
        geometry.attrs["metric"] = "cartesian"
        for dd in range(ndim):
            geometry.create_group(f"dim_{dd}").attrs.update(start=-1.0, end=1.0)
        partition = level.create_group("partition_0")
        partition["owned_start"] = (0,) * ndim
        partition["owned_fin"] = shape
        primitives = partition.create_group("hydro/primitives")
        primitives["rho"] = np.pad(rho, 1, constant_values=99.0)
        # speeds above one are valid in newtonian code units; the 1d inflow
        # also checks that mach uses speed rather than signed velocity.
        velocity = (-3.0, 4.0, 12.0)
        for ii in range(3 if regime == "imhd" else ndim):
            primitives[f"v{ii + 1}"] = np.pad(
                np.full(shape, velocity[ii]), 1, constant_values=99.0
            )
        if local_cs2 is not None:
            level["iso_cs2"] = np.pad(local_cs2, 1, constant_values=99.0)
    return SimData(read_checkpoint(str(path)).unwrap()), rho


@pytest.mark.parametrize("regime,ndim,speed", [
    ("isothermal", 1, 3.0), ("isothermal", 3, 13.0), ("imhd", 2, 13.0),
])
def test_isothermal_mach_uses_newtonian_speed(tmp_path, regime, ndim, speed):
    data, rho = checkpoint(tmp_path, regime=regime, ndim=ndim)
    for name in ("mach", "mach_static"):
        np.testing.assert_allclose(data.get_field(name), np.full(rho.shape, speed / 2.0))


@pytest.mark.parametrize("cs", [None, 2.0])
def test_mach_slice_uses_local_sound_speed_before_metadata(tmp_path, cs):
    # an even number of x2 cells puts zero exactly on a vertex, avoiding
    # a nearest-vertex tie when selecting the slice.
    cs2 = np.arange(48, dtype=float).reshape(2, 4, 6) + 1.0
    data, rho = checkpoint(tmp_path, cs=cs, local_cs2=cs2)
    config = VisualizationConfig(
        plot=PlotConfig(plot_type="multidim", fields=["mach"], ndim=3, slice={"x2": 0.0})
    )
    result = create_plot_data(data, ["mach"], config)
    assert result.dimensions == 2
    np.testing.assert_allclose(result.fields[0].values, 13.0 / np.sqrt(cs2[:, 2, :]))
    np.testing.assert_allclose(data.get_field("ptot"), cs2 * rho)


@pytest.mark.parametrize("regime", ["isothermal", "imhd"])
def test_isothermal_density_and_momentum_are_newtonian(tmp_path, regime):
    data, rho = checkpoint(tmp_path, regime=regime)
    np.testing.assert_allclose(data.get_field("W"), 1.0)
    np.testing.assert_allclose(data.get_field("D"), rho)
    np.testing.assert_allclose(data.get_field("m1"), -3.0 * rho)
    np.testing.assert_allclose(data.get_field("u1"), -3.0)
    np.testing.assert_allclose(data.get_field("enthalpy_density"), rho)
    np.testing.assert_allclose(data.get_field("enthalpy"), 1.0)


def test_missing_isothermal_sound_speed_explains_the_missing_data(tmp_path):
    data, rho = checkpoint(tmp_path, cs=None)
    with pytest.raises(FieldComputationError, match="sound_speed"):
        data.get_field("mach")
    np.testing.assert_allclose(data.get_field("D"), rho)


def test_isothermal_energy_reports_that_no_energy_equation_was_evolved(tmp_path):
    data, _ = checkpoint(tmp_path)
    with pytest.raises(FieldComputationError, match="energy equation"):
        data.get_field("energy")


def test_refinement_level_uses_its_own_sound_speed(tmp_path):
    cs2 = np.full((2, 4, 6), 4.0)
    checkpoint(tmp_path, local_cs2=cs2)
    path = tmp_path / "isothermal.h5"
    with h5py.File(path, "r+") as file:
        file.copy("level_0", "level_1")
        file["level_1/iso_cs2"][1:-1, 1:-1, 1:-1] = 16.0
    data = SimData(read_checkpoint(str(path)).unwrap())
    np.testing.assert_allclose(data.get_field("mach", level=0), 13.0 / 2.0)
    np.testing.assert_allclose(data.get_field("mach", level=1), 13.0 / 4.0)


def test_isothermal_mach_does_not_use_reporting_gamma(tmp_path):
    checkpoint(tmp_path)
    path = tmp_path / "isothermal.h5"
    with h5py.File(path, "r+") as file:
        file["metadata"].attrs["gamma"] = 5.0 / 3.0
    data = SimData(read_checkpoint(str(path)).unwrap())
    np.testing.assert_allclose(data.get_field("mach"), 13.0 / 2.0)


@pytest.mark.parametrize("regime,energy", [("newtonian", 43.5), ("nmhd", 45.5)])
def test_adiabatic_newtonian_fields_keep_their_energy_and_sound_speed(regime, energy):
    from simbi.reader.computation import gamma_law_t, labframe_energy_density, sound_speed

    rho, pre = np.array([3.0]), np.array([6.0])
    vel = [np.array([3.0]), np.array([4.0]), np.array([0.0])]
    bfield = [np.array([0.0]), np.array([0.0]), np.array([2.0])]
    eos = gamma_law_t(2.0)
    np.testing.assert_allclose(sound_speed(rho, pre, eos, regime), 2.0)
    np.testing.assert_allclose(
        labframe_energy_density(rho, pre, vel, bfield, eos, regime), energy
    )
