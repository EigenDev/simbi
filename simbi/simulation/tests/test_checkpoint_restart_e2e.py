# =============================================================================
# test_checkpoint_restart_e2e.py
#
# production python-runner restart identity. a bounded run resumed through
# `checkpoint_file` must match uninterrupted hydro and tracer state exactly,
# even when the restart config's initial-condition generator is incompatible.
# =============================================================================

from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pytest
from pydantic import computed_field

from simbi import ProblemParam
from simbi.simulation.runner import run, to_execution_dict
from simbi.types.bodies import (
    AccretionProperties,
    BodyCapability,
    ImmersedBodyConfig,
)
from simbi.types.typing import GasStateGenerator, InitialStateType
from simbi.simulation.tests.fixtures.mhd_energy_conservation import (
    MhdEnergyConservation,
)
from simbi_configs.examples.newtonian.sod import SodProblem


class RestartProbe(SodProblem):
    density_scale: Annotated[
        float,
        ProblemParam(1.0, description="initial-condition restart trap"),
    ]

    @computed_field
    @property
    def n_tracers(self) -> int:
        return 128

    def initial_primitive_state(self) -> InitialStateType:
        scale = self.density_scale

        def gas_state() -> GasStateGenerator:
            dx = 1.0 / self.resolution
            for ii in range(self.resolution):
                if ii * dx < 0.5:
                    yield (scale, 0.0, 1.0)
                else:
                    yield (0.125 * scale, 0.0, 0.1)

        return gas_state


class AccretionRestartProbe(RestartProbe):
    resolution: Annotated[int, ProblemParam(64, description="grid resolution")]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION,
                mass=1.0,
                radius=0.08,
                position=(0.5, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                accretion=AccretionProperties(
                    accretion_radius=0.08,
                    sink_rate=10.0,
                ),
            )
        ]


def _final(directory: Path) -> Path:
    files = list(directory.glob("*final*.h5"))
    assert len(files) == 1
    return files[0]


def _assert_dataset_group_equal(first: h5py.Group, second: h5py.Group) -> None:
    assert set(first.keys()) == set(second.keys())
    for name in first:
        np.testing.assert_array_equal(first[name][...], second[name][...])


def _assert_tree_equal(first: h5py.Group, second: h5py.Group) -> None:
    assert set(first.keys()) == set(second.keys())
    for name in first:
        if isinstance(first[name], h5py.Group):
            _assert_tree_equal(first[name], second[name])
        else:
            np.testing.assert_array_equal(first[name][...], second[name][...])


def test_execution_dict_serializes_restart_path(tmp_path):
    checkpoint = tmp_path / "restart.h5"

    payload = to_execution_dict(RestartProbe(checkpoint_file=checkpoint))

    assert payload["checkpoint_file"] == str(checkpoint)


@pytest.mark.simulation
def test_python_runner_restart_matches_uninterrupted_state(tmp_path):
    continuous_dir = tmp_path / "continuous"
    split_dir = tmp_path / "split"
    restarted_dir = tmp_path / "restarted"

    run(
        RestartProbe(data_directory=continuous_dir),
        compute_mode="cpu",
        max_steps=6,
    )
    run(
        RestartProbe(data_directory=split_dir),
        compute_mode="cpu",
        max_steps=3,
    )
    checkpoint = _final(split_dir)
    run(
        RestartProbe(
            data_directory=restarted_dir,
            checkpoint_file=checkpoint,
            density_scale=9.0,
        ),
        compute_mode="cpu",
        max_steps=6,
    )

    with (
        h5py.File(_final(continuous_dir)) as continuous,
        h5py.File(_final(restarted_dir)) as restarted,
    ):
        _assert_dataset_group_equal(
            continuous["level_0/conserved"],
            restarted["level_0/conserved"],
        )
        _assert_dataset_group_equal(continuous["tracers"], restarted["tracers"])
        assert dict(continuous["tracers"].attrs) == dict(restarted["tracers"].attrs)
        assert continuous["metadata"].attrs["iteration"] == 6
        assert restarted["metadata"].attrs["iteration"] == 6
        assert (
            continuous["metadata"].attrs["time"]
            == restarted["metadata"].attrs["time"]
        )


@pytest.mark.simulation
def test_python_runner_restart_preserves_staggered_mhd_state(tmp_path):
    continuous_dir = tmp_path / "mhd-continuous"
    split_dir = tmp_path / "mhd-split"
    restarted_dir = tmp_path / "mhd-restarted"
    common = {"resolution": (32, 8, 1)}

    run(
        MhdEnergyConservation(data_directory=continuous_dir, **common),
        compute_mode="cpu",
        max_steps=6,
    )
    run(
        MhdEnergyConservation(data_directory=split_dir, **common),
        compute_mode="cpu",
        max_steps=3,
    )
    checkpoint = _final(split_dir)
    run(
        MhdEnergyConservation(
            data_directory=restarted_dir,
            checkpoint_file=checkpoint,
            b0=9.0,
            **common,
        ),
        compute_mode="cpu",
        max_steps=6,
    )

    with (
        h5py.File(_final(continuous_dir)) as continuous,
        h5py.File(_final(restarted_dir)) as restarted,
    ):
        _assert_tree_equal(continuous["level_0"], restarted["level_0"])
        assert continuous["metadata"].attrs["iteration"] == 6
        assert restarted["metadata"].attrs["iteration"] == 6


@pytest.mark.simulation
def test_python_runner_restart_preserves_body_state(tmp_path):
    continuous_dir = tmp_path / "body-continuous"
    split_dir = tmp_path / "body-split"
    restarted_dir = tmp_path / "body-restarted"

    run(
        AccretionRestartProbe(data_directory=continuous_dir),
        compute_mode="cpu",
        max_steps=6,
    )
    run(
        AccretionRestartProbe(data_directory=split_dir),
        compute_mode="cpu",
        max_steps=3,
    )
    checkpoint = _final(split_dir)
    run(
        AccretionRestartProbe(
            data_directory=restarted_dir,
            checkpoint_file=checkpoint,
            density_scale=9.0,
        ),
        compute_mode="cpu",
        max_steps=6,
    )

    with (
        h5py.File(_final(continuous_dir)) as continuous,
        h5py.File(_final(restarted_dir)) as restarted,
    ):
        _assert_tree_equal(continuous["bodies"], restarted["bodies"])
        np.testing.assert_array_equal(
            continuous["level_0/conserved/den"],
            restarted["level_0/conserved/den"],
        )
        assert continuous["bodies/total_accreted_mass"][0] > 0.0
