# =============================================================================
# test_launch_magnetized_charts.py
#
# the magnetized regimes on the curvilinear grids no earlier launch gate
# covered: newtonian, isothermal and special-relativistic MHD on the
# one-dimensional spherical and cylindrical grids, and special-relativistic MHD
# on the (r, z) plane, the (r, phi) disk, the three-dimensional spherical shell
# and the three-dimensional cylindrical annulus. each case's single-grid run
# carries a field the run evolves (the shear winds the toroidal field, the
# flow compresses the axial one) and moves the gas on both sides of every cut;
# four workers publish its state bit for bit, every face and halo included,
# and a run checkpointed halfway on four workers resumes onto three unevenly
# owned tiles to the uninterrupted single-grid state. every arm runs on the
# session's backend and placement (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_magnetized_charts.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import MHD_STEPS, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined import _launch, _reference

pytestmark = pytest.mark.simulation

FIXTURES = '''
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi_configs.examples.newtonian.nmhd_jet_rz import NewtonianJetRz
from simbi_configs.examples.newtonian.nmhd_orszag_tang_3d import NewtonianOrszagTang3D
from simbi_configs.examples.newtonian.nmhd_polar_shell_3d import NewtonianPolarShell3D
from simbi_configs.examples.newtonian.nmhd_spherical_field import NewtonianSphericalField
from simbi_configs.examples.srmhd.rmhd_orszag_tang import OrszagTang


class RadialField(NewtonianSphericalField):
    """a pressure bump in sheared rotation on the radial line, B_r = 0.3 (1.5 / r)^2 on the radial
    faces (divergence-free), a uniform toroidal B_phi the shear winds"""

    def initial_primitive_state(self):
        def gas_state():
            nr, _, _ = self.resolution
            (r_lo, r_hi), _ = self.bounds[0], self.bounds[1]
            dr = (r_hi - r_lo) / nr
            for ii in range(nr):
                r = r_lo + (ii + 0.5) * dr
                b = 0.3 * math.exp(-(((r - 1.5) / 0.1) ** 2))
                yield self.row(1.0 + b, 0.1 * 1.5 / r, 1.0 + b)

        def b_field(bn):
            nr, nt, nk = self.resolution
            (r_lo, r_hi) = self.bounds[0]
            dr = (r_hi - r_lo) / nr
            for _kk in range(nk + (bn == "bz")):
                for _jj in range(nt + (bn == "by")):
                    for ii in range(nr + (bn == "bx")):
                        if bn == "bx":
                            yield 0.3 * (1.5 / (r_lo + ii * dr)) ** 2
                        elif bn == "by":
                            yield 0.0
                        else:
                            yield 0.2

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))

    def row(self, rho, v_phi, pre):
        return (rho, 0.0, 0.0, v_phi, pre)


class IsothermalRadialField(RadialField):
    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="regime")]
    sound_speed: Annotated[float, ProblemParam(1.0, description="isothermal sound speed")]

    def row(self, rho, v_phi, pre):
        return (rho, 0.0, 0.0, v_phi)


class RelativisticRadialField(RadialField):
    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="regime")]


class JetLine(NewtonianJetRz):
    """the jet's radial line: on one axis the two conditions name its two faces, outflow on both"""

    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW], description="r faces")]


class IsothermalJet(JetLine):
    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="regime")]
    sound_speed: Annotated[float, ProblemParam(1.0, description="isothermal sound speed")]

    def initial_primitive_state(self):
        gas, *fields = super().initial_primitive_state()

        def isothermal():
            for row in gas():
                yield row[:-1]

        return (isothermal, *fields)


class RelativisticJet(JetLine):
    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="regime")]


class RelativisticVortexRz(OrszagTang):
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="(r, z)")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.1, 1.0), (0.0, 1.0)], description="bounds")]


class RelativisticVortexDisk(OrszagTang):
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.PLANAR_CYLINDRICAL, description="(r, phi)")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.1, 1.0), (0.0, 2.0 * math.pi)], description="bounds")]


class RelativisticPolarShell(NewtonianPolarShell3D):
    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="regime")]


class RelativisticVortexAnnulus(NewtonianOrszagTang3D):
    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="regime")]
    v0: Annotated[float, ProblemParam(0.5, description="velocity scale")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="(r, phi, z)")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.1, 1.0), (0.0, 2.0 * math.pi), (0.0, 1.0)], description="bounds")]
'''

LINE = dict(resolution=(64, 1, 1), cuts=[[16, 32, 48]], restart=[[10, 41]])
PLANE = dict(resolution=(32, 32, 1), cuts=[[16], [16]], restart=[[10, 21], []])
# the pole rows cut into two azimuth slabs, so every axis ring spans two tiles
SHELL = dict(resolution=(8, 8, 16), cuts=[[], [4], [8]], restart=[[], [], [5, 11]])
ANNULUS = dict(resolution=(8, 8, 8), cuts=[[4], [], [4]], restart=[[3, 5], [], []])
CASES = {
    "nmhd-spherical-line": ("RadialField", LINE),
    "imhd-spherical-line": ("IsothermalRadialField", LINE),
    "rmhd-spherical-line": ("RelativisticRadialField", LINE),
    "nmhd-cylindrical-line": ("JetLine", LINE),
    "imhd-cylindrical-line": ("IsothermalJet", LINE),
    "rmhd-cylindrical-line": ("RelativisticJet", LINE),
    "rmhd-rz-plane": ("RelativisticVortexRz", PLANE),
    "rmhd-disk": ("RelativisticVortexDisk", PLANE),
    "rmhd-shell-3d": ("RelativisticPolarShell", SHELL),
    "rmhd-annulus-3d": ("RelativisticVortexAnnulus", ANNULUS),
}


@pytest.fixture
def fixtures(tmp_path: Path, monkeypatch) -> Path:
    source = tmp_path / "src"
    source.mkdir()
    (source / "magnetized_charts.py").write_text(FIXTURES)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(source), str(Path.cwd()), os.environ.get("PYTHONPATH", "")]))
    return source


def _script(fixtures: Path, cls: str) -> str:
    # the launch evolves the one problem its script defines
    path = fixtures / f"launched_{cls.lower()}.py"
    path.write_text(f"from magnetized_charts import {cls} as Base\n\n\nclass Launched{cls}(Base):\n    pass\n")
    return str(path)


def _interior(f: h5py.File, name: str) -> np.ndarray:
    data = f[name][()]
    ng = int(f["level_0/mesh"].attrs["halo_width"])
    return data[tuple(slice(ng, -ng) if n > 2 * ng else slice(None) for n in data.shape)]


def _assert_exercised(initial: Path, final: Path, cuts: list[list[int]], label: str) -> None:
    with h5py.File(initial, "r") as f0, h5py.File(final, "r") as f1:
        magnetic = [k for k in f1["level_0/partition_0/hydro/primitives"].keys() if k.startswith("b")]
        assert magnetic, f"{label}: the run publishes no magnetic field"
        strength = max(float(np.abs(f1[f"level_0/partition_0/hydro/primitives/{k}"][()]).max()) for k in magnetic)
        assert strength > 0.05, f"{label}: the field peaks at {strength:e}; the magnetized physics is idle"
        evolved = any(not np.array_equal(f0[f"level_0/partition_0/hydro/primitives/{k}"][()], f1[f"level_0/partition_0/hydro/primitives/{k}"][()]) for k in magnetic)
        assert evolved, f"{label}: the field never evolved"
        den0, den1 = _interior(f0, "level_0/conserved/den"), _interior(f1, "level_0/conserved/den")
        assert np.isfinite(den1).all(), f"{label}: the final density is not finite"
    changed = den0 != den1
    ndim = den1.ndim
    for ax, positions in enumerate(cuts):
        for cut in positions:
            # the stored arrays run the last simulation axis first
            along = ndim - 1 - ax
            below = np.take(changed, range(cut - 2, cut), axis=along)
            above = np.take(changed, range(cut, cut + 2), axis=along)
            assert below.any() and above.any(), f"{label}: the gas beside the cut at {cut} on axis {ax} never moved"


@pytest.mark.parametrize("name", list(CASES))
def test_a_magnetized_chart_launches_as_the_single_grid_runs_fresh_and_resumed(tmp_path: Path, fixtures: Path, name: str) -> None:
    cls, case = CASES[name]
    resolution = case["resolution"]
    reference = _reference("magnetized_charts", cls, resolution, tmp_path / "single", MHD_STEPS)
    initial = sorted(p for p in (tmp_path / "single").glob("*.h5") if "final" not in p.name)[0]
    _assert_exercised(initial, reference, case["cuts"], name)
    script = _script(fixtures, cls)
    launched = _launch(script, resolution, tmp_path / "four", MHD_STEPS, 4, cuts=case["cuts"])
    ph.same_checkpoint(launched, reference, f"{name}: four workers")
    partial = _launch(script, resolution, tmp_path / "first", MHD_STEPS // 2, 4, cuts=case["cuts"])
    with h5py.File(partial, "r") as f:
        assert 0 < int(f["level_0"].attrs["iteration"]) < MHD_STEPS
    resumed = _launch(script, resolution, tmp_path / "three", MHD_STEPS, 3, cuts=case["restart"], checkpoint=partial)
    ph.same_checkpoint(resumed, reference, f"{name}: resumed onto three uneven workers")
