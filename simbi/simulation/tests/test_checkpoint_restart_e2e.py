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
    GravitationalProperties,
    ImmersedBodyConfig,
)
from simbi.types import Regime
from simbi.types.input import normalize_regime
from simbi.types.typing import GasStateGenerator, InitialStateType
from simbi.simulation.tests.fixtures.mhd_energy_conservation import (
    MhdEnergyConservation,
)
from simbi.types.input import Limiter
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
                capability=BodyCapability.ACCRETION
                | BodyCapability.GRAVITATIONAL,
                mass=1.0,
                radius=0.08,
                position=(0.5, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                gravitational=GravitationalProperties(
                    softening_length=0.08,
                ),
                accretion=AccretionProperties(
                    accretion_radius=0.08,
                ),
            )
        ]


class TwoAccretorRestartProbe(AccretionRestartProbe):
    @computed_field
    @property
    def n_tracers(self) -> int:
        return 2048

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION
                | BodyCapability.GRAVITATIONAL,
                mass=1.0,
                radius=0.08,
                position=(position, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                gravitational=GravitationalProperties(
                    softening_length=0.08,
                ),
                accretion=AccretionProperties(
                    accretion_radius=0.08,
                ),
            )
            for position in (0.35, 0.65)
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


# a gradient (neumann) wall is recorded by kind alone; the restart takes its coefficients from
# the configuration and continues the run exactly
@pytest.mark.simulation
def test_python_runner_restart_continues_a_gradient_walled_run(tmp_path):
    from simbi_configs.examples.newtonian.gradient_box import GradientBox

    run(GradientBox(resolution=(16, 16), data_directory=tmp_path / "continuous"), compute_mode="cpu", max_steps=6)
    run(GradientBox(resolution=(16, 16), data_directory=tmp_path / "split"), compute_mode="cpu", max_steps=3)
    run(
        GradientBox(resolution=(16, 16), data_directory=tmp_path / "restarted", checkpoint_file=_final(tmp_path / "split")),
        compute_mode="cpu",
        max_steps=6,
    )
    with h5py.File(_final(tmp_path / "continuous")) as continuous, h5py.File(_final(tmp_path / "restarted")) as restarted:
        assert continuous["metadata"].attrs["time"] == restarted["metadata"].attrs["time"]
        _assert_dataset_group_equal(continuous["level_0/conserved"], restarted["level_0/conserved"])


# a two-dimensional cylindrical checkpoint records its plane; the disk (r, phi) chart
# survives the restart, so the resumed run continues on the same geometry
@pytest.mark.simulation
def test_python_runner_restart_keeps_the_disk_plane_of_a_cylindrical_run(tmp_path):
    from simbi_configs.examples.newtonian.nmhd_rotor_cyl import NewtonianRotorCyl

    problem = dict(resolution=(16, 16, 1))
    run(NewtonianRotorCyl(**problem, data_directory=tmp_path / "continuous"), compute_mode="cpu", max_steps=6)
    run(NewtonianRotorCyl(**problem, data_directory=tmp_path / "split"), compute_mode="cpu", max_steps=3)
    with h5py.File(_final(tmp_path / "split")) as split:
        assert split["metadata"].attrs["cyl_plane"] == "rphi"
    run(
        NewtonianRotorCyl(**problem, data_directory=tmp_path / "restarted", checkpoint_file=_final(tmp_path / "split")),
        compute_mode="cpu",
        max_steps=6,
    )
    with h5py.File(_final(tmp_path / "continuous")) as continuous, h5py.File(_final(tmp_path / "restarted")) as restarted:
        assert continuous["metadata"].attrs["time"] == restarted["metadata"].attrs["time"]
        _assert_dataset_group_equal(continuous["level_0/conserved"], restarted["level_0/conserved"])
        _assert_tree_equal(continuous["level_0/partition_0/hydro/magnetic"], restarted["level_0/partition_0/hydro/magnetic"])


# a checkpoint names its regime in the configuration vocabulary; the bare "mhd" that earlier
# adiabatic Newtonian MHD checkpoints recorded is a legacy slug the reader maps to nmhd.
def test_legacy_mhd_checkpoint_slug_restarts_as_newtonian_mhd() -> None:
    assert normalize_regime("mhd") == "nmhd"
    assert Regime(normalize_regime("mhd")) is Regime.NMHD
    for current in Regime:
        assert normalize_regime(current.value) == current.value


# the relativistic and the Newtonian MHD regimes both restart to the uninterrupted state: the
# staggered field, the conserved state, and the regime slug the checkpoint records all round-trip.
@pytest.mark.simulation
@pytest.mark.parametrize("regime", [Regime.RMHD, Regime.NMHD])
def test_python_runner_restart_preserves_staggered_mhd_state(tmp_path, regime):
    continuous_dir = tmp_path / "mhd-continuous"
    split_dir = tmp_path / "mhd-split"
    restarted_dir = tmp_path / "mhd-restarted"
    common = {"resolution": (32, 8, 1), "regime": regime}

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


@pytest.mark.simulation
def test_python_runner_preserves_distinct_body_accretion_reservoirs(tmp_path):
    split_dir = tmp_path / "two-body-split"
    restarted_dir = tmp_path / "two-body-restarted"
    run(
        TwoAccretorRestartProbe(data_directory=split_dir),
        compute_mode="cpu",
        max_steps=3,
    )
    checkpoint = _final(split_dir)
    prefix = np.uint64((1 << 62) | (1 << 61))
    with h5py.File(checkpoint) as split:
        owners = split["tracers/owner"][:]
        assert np.count_nonzero(owners == prefix) > 0
        assert np.count_nonzero(owners == (prefix | np.uint64(1))) > 0

    run(
        TwoAccretorRestartProbe(
            data_directory=restarted_dir,
            checkpoint_file=checkpoint,
        ),
        compute_mode="cpu",
        max_steps=6,
    )
    with h5py.File(_final(restarted_dir)) as restarted:
        owners = restarted["tracers/owner"][:]
        assert np.count_nonzero(owners == prefix) > 0
        assert np.count_nonzero(owners == (prefix | np.uint64(1))) > 0


def _density(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.concatenate([f[name][...].ravel() for name in _density_datasets(f)])


def _density_datasets(f: h5py.File) -> list[str]:
    names: list[str] = []
    f.visit(lambda name: names.append(name) if name.endswith("rho") and isinstance(f[name], h5py.Dataset) else None)
    assert names, "the checkpoint holds no density dataset; the comparison is vacuous"
    return sorted(names)


@pytest.mark.simulation
def test_the_van_leer_selection_reaches_the_kernel(tmp_path):
    # the run records the kernel spelling of its limiter and evolves a different state from
    # the minmod family it was formerly handed
    from simbi.types.input import Limiter

    run(SodProblem(data_directory=tmp_path / "minmod"), compute_mode="cpu", max_steps=20)
    run(SodProblem(limiter=Limiter.VAN_LEER, data_directory=tmp_path / "vanleer"), compute_mode="cpu", max_steps=20)
    with h5py.File(_final(tmp_path / "vanleer"), "r") as f:
        assert float(f["metadata"].attrs["plm_theta"]) < 0.0, "the run recorded the minmod family"
    assert not np.array_equal(_density(_final(tmp_path / "minmod")), _density(_final(tmp_path / "vanleer")))


class VanLeerSod(SodProblem):
    """the sod problem declaring the van leer limiter as its own default"""

    limiter: Annotated[
        Limiter,
        ProblemParam(Limiter.VAN_LEER, cli=True, description="PLM slope limiter"),
    ]


@pytest.mark.simulation
@pytest.mark.parametrize(
    "recorded, resumed",
    [(VanLeerSod, SodProblem), (SodProblem, VanLeerSod)],
    ids=["vanleer_file_minmod_config", "minmod_file_vanleer_config"],
)
def test_a_restart_continues_the_recorded_limiter_bitwise(tmp_path, recorded, resumed):
    # a run resumed from a config whose own default is the other limiter family continues the
    # family its checkpoint recorded, in the solver: it matches the uninterrupted run exactly
    run(recorded(data_directory=tmp_path / "continuous"), compute_mode="cpu", max_steps=12)
    run(resumed(data_directory=tmp_path / "other"), compute_mode="cpu", max_steps=12)
    assert not np.array_equal(_density(_final(tmp_path / "continuous")), _density(_final(tmp_path / "other"))), (
        "the two limiter families evolve alike on this setup; the restart comparison is vacuous"
    )
    run(recorded(data_directory=tmp_path / "split"), compute_mode="cpu", max_steps=6)
    run(resumed(data_directory=tmp_path / "restarted", checkpoint_file=_final(tmp_path / "split")), compute_mode="cpu", max_steps=12)
    with h5py.File(_final(tmp_path / "restarted"), "r") as f, h5py.File(_final(tmp_path / "continuous"), "r") as g:
        assert float(f["metadata"].attrs["plm_theta"]) == float(g["metadata"].attrs["plm_theta"]), "the resumed run recorded another limiter"
    np.testing.assert_array_equal(_density(_final(tmp_path / "continuous")), _density(_final(tmp_path / "restarted")))


@pytest.mark.simulation
def test_a_ppm_restart_continues_the_recorded_flatten_bitwise(tmp_path):
    # a flattened ppm run resumed from a config that leaves the dials at their defaults
    # matches the uninterrupted flattened run exactly
    from simbi.types.input import Reconstruction

    ppm = {"reconstruction": Reconstruction.PPM}
    dials = {"ppm_flatten_onset": 0.015, "ppm_flatten_full": 0.05}
    run(SodProblem(**ppm, data_directory=tmp_path / "plain"), compute_mode="cpu", max_steps=6)
    run(SodProblem(**ppm, **dials, data_directory=tmp_path / "continuous"), compute_mode="cpu", max_steps=6)
    assert not np.array_equal(_density(_final(tmp_path / "plain")), _density(_final(tmp_path / "continuous"))), (
        "the flatten never acted on this setup; the restart comparison is vacuous"
    )
    run(SodProblem(**ppm, **dials, data_directory=tmp_path / "split"), compute_mode="cpu", max_steps=3)
    run(
        SodProblem(**ppm, data_directory=tmp_path / "restarted", checkpoint_file=_final(tmp_path / "split")),
        compute_mode="cpu",
        max_steps=6,
    )
    np.testing.assert_array_equal(_density(_final(tmp_path / "continuous")), _density(_final(tmp_path / "restarted")))
