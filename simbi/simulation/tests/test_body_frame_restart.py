# =============================================================================
# test_body_frame_restart.py
#
# checkpoints record the frame their body positions are written in
# (`bodies/position_frame = "meridional"`: cartesian components, the (x, z) pair
# on the 2D spherical chart). a checkpoint without the record predates that
# frame: its penalizing bodies on the 2D spherical chart had their masks placed
# with the two components read as (z, x).
#   - a restart from such a checkpoint with a penalizing body off the symmetry
#     line x = z is refused by name, naming the swapped placement;
#   - the same checkpoint with its frame recorded restarts;
#   - a body on the symmetry line restarts without the record, since both
#     readings place it at the same point.
# =============================================================================

import math
from pathlib import Path
from typing import Annotated

import h5py
import pytest

from simbi import ProblemParam, SimbiProblem
from simbi.simulation.runner import run
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.bodies import (
    AccretionProperties,
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class MeridionalSink(SimbiProblem):
    """a drain on the 2D spherical chart at a configurable meridional (x, z)."""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="adiabatic index")]
    body_x: Annotated[float, ProblemParam(2.5, description="sink x")]
    body_z: Annotated[float, ProblemParam(1.0, description="sink z")]
    resolution: Annotated[tuple[int, int], ProblemParam((24, 24), description="zones")]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(1.0, 5.0), (0.05, math.pi - 0.05)], description="domain"),
    ]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.SPHERICAL, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[
        BoundaryCondition, ProblemParam(BoundaryCondition.OUTFLOW, description="walls")
    ]
    end_time: Annotated[float, ProblemParam(1.0, description="end time")]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            for _ in range(self.resolution[0] * self.resolution[1]):
                yield (1.0, 0.0, 0.0, 1.0)

        return gas_state

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
                mass=1.0e-3,
                radius=0.0,
                position=(self.body_x, self.body_z),
                velocity=(0.0, 0.0),
                gravitational=GravitationalProperties(softening_length=0.1),
                accretion=AccretionProperties(accretion_radius=0.5),
            )
        ]


def _final(directory: Path) -> Path:
    finals = sorted(directory.glob("*.chkpt.final*.h5"))
    assert finals, f"no final checkpoint in {directory}"
    return finals[0]


def _checkpoint(tmp_path: Path, name: str, x: float, z: float, keep_frame: bool) -> Path:
    run(MeridionalSink(data_directory=tmp_path / name, body_x=x, body_z=z), compute_mode="cpu", max_steps=2)
    final = _final(tmp_path / name)
    with h5py.File(final, "r+") as f:
        assert f["bodies"].attrs["position_frame"] in (b"meridional", "meridional")
        if not keep_frame:
            del f["bodies"].attrs["position_frame"]
    return final


def _restart(tmp_path: Path, name: str, x: float, z: float, checkpoint: Path) -> None:
    run(
        MeridionalSink(data_directory=tmp_path / name, body_x=x, body_z=z, checkpoint_file=checkpoint),
        compute_mode="cpu",
        max_steps=4,
    )


@pytest.mark.simulation
def test_an_unrecorded_frame_with_an_off_line_body_is_refused(tmp_path):
    legacy = _checkpoint(tmp_path, "legacy", 2.5, 1.0, keep_frame=False)
    with pytest.raises(Exception, match="predates the meridional body frame"):
        _restart(tmp_path, "resumed", 2.5, 1.0, legacy)


@pytest.mark.simulation
def test_a_recorded_frame_restarts(tmp_path):
    current = _checkpoint(tmp_path, "current", 2.5, 1.0, keep_frame=True)
    _restart(tmp_path, "resumed", 2.5, 1.0, current)
    with h5py.File(_final(tmp_path / "resumed")) as f:
        assert f["metadata"].attrs["iteration"] == 4


@pytest.mark.simulation
def test_a_body_on_the_symmetry_line_restarts_without_the_record(tmp_path):
    legacy = _checkpoint(tmp_path, "legacy", 2.0, 2.0, keep_frame=False)
    _restart(tmp_path, "resumed", 2.0, 2.0, legacy)
    with h5py.File(_final(tmp_path / "resumed")) as f:
        assert f["metadata"].attrs["iteration"] == 4
