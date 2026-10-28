# =============================================================================
# test_binary_reference_frame.py
#
# the reference frame of a prescribed circular binary, from the problem's body
# system to the evolved bodies on the single device. q = m2/m1 = 0.5, separation
# a = 0.8, total mass 1; body 1 (the heavier) starts at (a q/(1+q), 0), body 2 at
# (-a/(1+q), 0), and omega = sqrt(M / a^3).
# - inertial: the components follow the circle at omega, each with velocity
#   omega times its own radius, tangential and counterclockwise, and the
#   center-of-mass momentum vanishes
# - corotating: the frame turns with the orbit, so the components rest at their
#   initial positions with zero velocity at every checkpoint
# - stationary: the backend holds the positions fixed
# - refused by name before any output: an eccentric orbit, a frame name outside
#   inertial / corotating / stationary, and a corotating binary moving under its
#   own dynamics (prescribed_motion = false, or a two-way coupled component)
#
# run: pytest -m simulation simbi/simulation/tests/test_binary_reference_frame.py
# =============================================================================
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

SEPARATION, MASS_RATIO, TOTAL_MASS = 0.8, 0.5, 1.0
OMEGA = math.sqrt(TOTAL_MASS / SEPARATION**3)
A1 = SEPARATION * MASS_RATIO / (1.0 + MASS_RATIO)
A2 = SEPARATION - A1
M1 = TOTAL_MASS / (1.0 + MASS_RATIO)
M2 = TOTAL_MASS - M1
STEPS = 4

BINARY = '''
import math
import sys
from typing import Annotated, Optional

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.simulation import runner
from simbi.types import CoordSystem, Regime
from simbi.types.bodies import BinaryComponentConfig, BinaryConfig, BodySystemConfig, GravitationalSystemConfig


def _component(two_way: bool) -> BinaryComponentConfig:
    return BinaryComponentConfig(mass=0.5, radius=0.0, is_an_accretor=True, softening_length=0.1,
                                 two_way_coupling=two_way, accretion_radius=0.15)


class Binary(SimbiProblem):
    """a prescribed circular binary, q = 0.5, at the center of a uniform gas at rest on [-2, 2]^2"""

    resolution: Annotated[tuple[int, int], ProblemParam((24, 24), description="grid")]
    bounds: list[tuple[float, float]] = [(-2.0, 2.0)] * 2
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    regime: Regime = Regime.NEWTONIAN
    adiabatic_index: float = 5.0 / 3.0
    end_time: float = 10.0

    @computed_field
    @property
    def body_system(self) -> Optional[BodySystemConfig]:
        return GravitationalSystemConfig(
            prescribed_motion={prescribed}, reference_frame={frame!r}, system_type="binary",
            binary_config=BinaryConfig(semi_major={separation}, eccentricity={eccentricity}, mass_ratio={mass_ratio},
                                       total_mass={total_mass}, components=[_component(False), _component({two_way})]),
        )

    def initial_primitive_state(self):
        def gas_state():
            for _ in range(math.prod(self.resolution)):
                yield (1.0, 0.0, 0.0, 1.0)

        return gas_state


if __name__ == "__main__":
    p = Binary(data_directory=sys.argv[1])
    p.checkpoint_interval = 1e-9
    runner.run(p, compute_mode="cpu", max_steps={steps})
'''


def _script(frame: str, prescribed: bool = True, eccentricity: float = 0.0, two_way: bool = False) -> str:
    return BINARY.format(frame=frame, prescribed=prescribed, eccentricity=eccentricity, two_way=two_way, separation=SEPARATION,
                         mass_ratio=MASS_RATIO, total_mass=TOTAL_MASS, steps=STEPS)


def _module(tmp_path: Path, **settings) -> str:
    """the case as a module file: the runner records the problem class's source"""
    path = tmp_path / "binary_case.py"
    path.write_text(_script(**settings))
    return str(path)


def _run(tmp_path: Path, **settings) -> list[Path]:
    out = tmp_path / "run"
    ph.python([_module(tmp_path, **settings), str(out)], f"binary {settings}")
    files = sorted(out.glob("*.h5"), key=_time)
    assert len(files) >= STEPS, f"{len(files)} checkpoints; every step writes one"
    return files


def _time(path: Path) -> float:
    with h5py.File(path, "r") as f:
        return float(f["level_0"].attrs["time"])


def _bodies(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as f:
        return np.asarray(f["bodies/position"][()]), np.asarray(f["bodies/velocity"][()]), float(f["level_0"].attrs["time"])


INITIAL = np.array([[A1, 0.0], [-A2, 0.0]])


def test_a_corotating_binary_rests_at_its_initial_positions(tmp_path: Path) -> None:
    files = _run(tmp_path, frame="corotating")
    assert _time(files[-1]) > 0.0, "the run never advanced; the frame is never exercised"
    for path in files:
        position, velocity, t = _bodies(path)
        np.testing.assert_array_equal(position, INITIAL, err_msg=f"t = {t}: a corotating component moved")
        np.testing.assert_array_equal(velocity, np.zeros_like(velocity), err_msg=f"t = {t}: a corotating component moves in its frame")


def test_an_inertial_binary_follows_its_circle_at_each_components_own_speed(tmp_path: Path) -> None:
    files = _run(tmp_path, frame="inertial")
    for path in files:
        position, velocity, t = _bodies(path)
        phase = OMEGA * t
        rotation = np.array([[math.cos(phase), -math.sin(phase)], [math.sin(phase), math.cos(phase)]])
        # the positions are rotated by omega dt each step; the accumulated angle differs from
        # omega t by roundoff
        np.testing.assert_allclose(position, INITIAL @ rotation.T, rtol=0.0, atol=1e-12, err_msg=f"t = {t}: positions")
        for body, radius in ((0, A1), (1, A2)):
            r, v = position[body], velocity[body]
            assert math.isclose(np.linalg.norm(v), OMEGA * radius, rel_tol=1e-12), f"t = {t}, body {body}: |v| = {np.linalg.norm(v)} against omega r = {OMEGA * radius}"
            assert abs(np.dot(r, v)) < 1e-12, f"t = {t}, body {body}: the velocity has a radial part"
            assert r[0] * v[1] - r[1] * v[0] > 0.0, f"t = {t}, body {body}: the orbit runs clockwise"
        momentum = M1 * velocity[0] + M2 * velocity[1]
        assert np.abs(momentum).max() < 1e-12, f"t = {t}: center-of-mass momentum {momentum}"
    assert _time(files[-1]) > 0.0


def test_a_stationary_binary_holds_its_positions(tmp_path: Path) -> None:
    files = _run(tmp_path, frame="stationary")
    assert _time(files[-1]) > 0.0
    for path in files:
        position, _, t = _bodies(path)
        np.testing.assert_array_equal(position, INITIAL, err_msg=f"t = {t}: a stationary component moved")


REFUSED = {
    "eccentric-inertial": ({"frame": "inertial", "eccentricity": 0.2}, "eccentricity of 0.2"),
    "eccentric-corotating": ({"frame": "corotating", "eccentricity": 0.2}, "eccentricity of 0.2"),
    "unknown-frame": ({"frame": "com"}, "reference_frame 'com'"),
    "dynamic-corotating": ({"frame": "corotating", "prescribed": False}, "prescribed_motion = false"),
    "two-way-corotating": ({"frame": "corotating", "two_way": True}, "two-way coupled component"),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_an_unsupported_binary_is_refused_by_name_before_any_output(tmp_path: Path, case: str) -> None:
    settings, message = REFUSED[case]
    out = tmp_path / "run"
    done = ph.python([_module(tmp_path, **settings), str(out)], f"binary {case}, expecting a refusal", check=False)
    flat = " ".join(done.stderr.split())
    assert done.returncode != 0, f"{case}: the run succeeded"
    assert message in flat, f"{case}: {flat[-1500:]}"
    assert not list(out.glob("*.h5")), f"{case}: output was written before the refusal"
