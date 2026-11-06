# =============================================================================
# test_axisymmetric_rotation.py
#
# the newtonian gas with three velocity components on a two-dimensional axisymmetric
# plane, spherical (r, theta) and cylindrical (R, z), through the frontend, under the
# adiabatic and the isothermal closure:
# - a point mass on the axis accelerates a uniform gas at rest by its closed-form
#   newtonian gravity, the azimuthal momentum staying exactly zero. the compact
#   softening is newtonian outside its length, which lies inside the inner radius
# - a run carrying anything outside the verified scope is refused by name before
#   any output: a body on the cartesian plane, the full stress on a plane with three
#   velocity components, the azimuthal shear without its named face conditions or
#   on the cartesian plane
#
# run: pytest -m simulation simbi/simulation/tests/test_axisymmetric_rotation.py
# =============================================================================
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

STEPS = 4
CASE = '''
import math
import sys
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.simulation import runner
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig


class Plane(SimbiProblem):
    """a uniform gas with three velocity components about a point mass"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 32), description="grid")]
    bounds: list[tuple[float, float]] = {bounds}
    coord_system: CoordSystem = {coord}
{closure}    velocity_components: Annotated[int | None, ProblemParam(3, description="three velocity components")]
    end_time: float = 10.0
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="walls")]
{extra}
    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [ImmersedBodyConfig(capability=BodyCapability.{capability}, mass=1.0, velocity=(0.0, 0.0), position={position}, radius=0.0,
                                   gravitational=GravitationalProperties(softening_length=0.5, softening_kind="compact"){accretion})]

    def initial_primitive_state(self):
        def gas_state():
            for _ in range(math.prod(self.resolution)):
                yield (1.0, 0.0, 0.0, 0.0{pressure})

        return gas_state


if __name__ == "__main__":
    p = Plane(data_directory=sys.argv[1])
    p.checkpoint_interval = 1e-9
    runner.run(p, compute_mode={backend!r}, max_steps={steps})
'''
CLOSURES = {
    "adiabatic": dict(closure="    regime: Regime = Regime.NEWTONIAN\n    adiabatic_index: float = 5.0 / 3.0\n", pressure=", 100.0"),
    "isothermal": dict(
        closure='    regime: Regime = Regime.ISOTHERMAL\n    adiabatic_index: float | None = None\n'
                '    sound_speed: Annotated[float | None, ProblemParam(10.0, description="sound speed")]\n',
        pressure="",
    ),
    "locally-isothermal": dict(
        closure='    regime: Regime = Regime.ISOTHERMAL\n    adiabatic_index: float | None = None\n'
                '    sound_speed: Annotated[float | None, ProblemParam(10.0, description="ambient sound speed")]\n'
                '    @computed_field\n    @property\n    def locally_isothermal(self) -> bool:\n        return True\n\n'
                '    @computed_field\n    @property\n    def ambient_sound_speed(self) -> float:\n        return 10.0\n\n',
        pressure=", 100.0",
    ),
}
PLANES = {
    "spherical": dict(bounds="[(1.0, 2.0), (math.pi / 4.0, 3.0 * math.pi / 4.0)]", coord="CoordSystem.SPHERICAL"),
    "cylindrical": dict(bounds="[(1.0, 2.0), (-0.5, 0.5)]", coord="CoordSystem.CYLINDRICAL"),
    "cartesian": dict(bounds="[(1.0, 2.0), (-0.5, 0.5)]", coord="CoordSystem.CARTESIAN"),
}


def _script(tmp_path: Path, plane: str, closure: str = "adiabatic", extra: str = "", capability: str = "GRAVITATIONAL", position: str = "(0.0, 0.0)", accretion: str = "") -> str:
    path = tmp_path / "plane_case.py"
    path.write_text(CASE.format(backend=ph.BACKEND, extra=extra, capability=capability, position=position, accretion=accretion, steps=STEPS, **PLANES[plane], **CLOSURES[closure]))
    return str(path)


def _final(out: Path) -> tuple[h5py.File, int]:
    f = h5py.File(max(out.glob("*.h5"), key=lambda p: h5py.File(p, "r")["level_0"].attrs["time"]), "r")
    return f, int(f["level_0/mesh"].attrs["halo_width"])


@pytest.mark.parametrize("closure", list(CLOSURES))
@pytest.mark.parametrize("plane", ["spherical", "cylindrical"])
def test_a_point_mass_on_the_axis_accelerates_the_gas_by_newtonian_gravity(tmp_path: Path, plane: str, closure: str) -> None:
    out = tmp_path / "run"
    ph.python([_script(tmp_path, plane, closure), str(out)], f"a point mass on the {plane} plane, {closure}")
    f, ng = _final(out)
    t = float(f["level_0"].attrs["time"])
    assert t > 0.0, "the run never advanced"
    conserved = f["level_0/conserved"]
    interior = lambda name: conserved[name][()][ng:-ng, ng:-ng]
    den, m = interior("den"), [interior(f"m{k}") for k in (1, 2, 3)]
    nj, ni = den.shape
    (lo0, hi0, lo1, hi1) = (1.0, 2.0, math.pi / 4.0, 3.0 * math.pi / 4.0) if plane == "spherical" else (1.0, 2.0, -0.5, 0.5)
    checked = 0
    for jj in range(4, nj - 4):
        x1 = lo1 + (jj + 0.5) * (hi1 - lo1) / nj
        for ii in range(4, ni - 4):
            x0 = lo0 + (ii + 0.5) * (hi0 - lo0) / ni
            v = [m[k][jj, ii] / den[jj, ii] for k in range(3)]
            if plane == "spherical":
                want = {0: -t / x0**2, 1: 0.0}
                azimuthal = v[2]
            else:
                d3 = math.hypot(x0, x1) ** 3
                want = {0: -t * x0 / d3, 2: -t * x1 / d3}
                azimuthal = v[1]
            for slot, value in want.items():
                # the pressure's acoustic response enters at order t^2 c_s / dx below the source
                assert abs(v[slot] - value) < 0.02 * t / x0**2, f"{plane} cell ({ii}, {jj}), slot {slot}: {v[slot]} against {value}"
            assert azimuthal == 0.0, f"{plane} cell ({ii}, {jj}): azimuthal velocity {azimuthal}"
            checked += 1
    assert checked > 200, f"{checked} cells checked"


VISCOSITY = "    @computed_field\n    @property\n    def viscosity(self) -> float:\n        return 1.0e-3\n"
SHEAR = '    viscous_stress: Annotated[str | None, ProblemParam("azimuthal_shear", description="the azimuthal shear")]\n'
REFUSED = {
    "off-axis": ({"position": "(0.5, 0.5)"}, "a bare point mass at rest on the symmetry axis"),
    "full-stress": ({"extra": VISCOSITY}, "viscous transport other than the azimuthal shear"),
    "shear-without-faces": ({"extra": SHEAR + VISCOSITY}, "names one condition per domain face"),
    "hllc": ({"extra": '    solver: str = "hllc"\n'}, "a solver other than hlle"),
    "ppm": ({"extra": '    reconstruction: str = "ppm"\n'}, "ppm reconstruction requires a flat cartesian chart"),
}


@pytest.mark.parametrize("plane", ["spherical", "cylindrical"])
@pytest.mark.parametrize("case", list(REFUSED))
def test_a_run_outside_the_verified_scope_is_refused_by_name_before_any_output(tmp_path: Path, plane: str, case: str) -> None:
    settings, message = REFUSED[case]
    out = tmp_path / "run"
    done = ph.python([_script(tmp_path, plane, **settings), str(out)], f"{plane} {case}, expecting a refusal", check=False)
    flat = " ".join(done.stderr.split())
    assert done.returncode != 0, f"{case}: the run succeeded"
    assert message in flat, f"{case}: {flat[-1200:]}"
    assert not list(out.glob("*.h5")), f"{case}: output was written before the refusal"


@pytest.mark.parametrize("closure", list(CLOSURES))
def test_a_body_on_the_cartesian_plane_is_refused_by_name(tmp_path: Path, closure: str) -> None:
    out = tmp_path / "run"
    done = ph.python([_script(tmp_path, "cartesian", closure), str(out)], "a body on the cartesian plane, expecting a refusal", check=False)
    flat = " ".join(done.stderr.split())
    assert done.returncode != 0, "the run succeeded"
    assert "a body on the cartesian plane" in flat, flat[-1200:]
    assert not list(out.glob("*.h5")), "output was written before the refusal"


def test_the_shear_on_the_cartesian_plane_is_refused_by_name(tmp_path: Path) -> None:
    out = tmp_path / "run"
    faces = '    viscous_boundaries: Annotated[list[str] | None, ProblemParam(["zero_torque"] * 4, description="faces")]\n'
    script = Path(_script(tmp_path, "cartesian", extra=SHEAR + faces + VISCOSITY))
    # the plane alone: a body would be refused first, by its own name
    script.write_text(script.read_text().replace("return [ImmersedBodyConfig(", "return [] and [ImmersedBodyConfig("))
    done = ph.python([str(script), str(out)], "the shear on the cartesian plane, expecting a refusal", check=False)
    flat = " ".join(done.stderr.split())
    assert done.returncode != 0, "the run succeeded"
    assert "the azimuthal shear on the cartesian plane" in flat, flat[-1200:]
