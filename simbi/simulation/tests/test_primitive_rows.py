# =============================================================================
# test_primitive_rows.py
#
# the primitive rows a configuration may yield, and where each entry lands. a
# row is rho, one velocity per momentum component the dispatched arm carries,
# then the pressure; an isothermal row ends with an optional p/cs entry and an
# isothermal MHD row likewise. the momentum components equal the grid dimension
# except on the kerr-schild 2D charts that lift the azimuthal velocity (the
# (r, theta) plane takes either form, spinning kerr and the (R, z) plane require
# the lift) and on MHD, which carries three everywhere. every supported layout
# seeds distinct, exactly representable values and the initial checkpoint
# returns each one in its own slot; a row of any other width is refused by name
# before any output, so no entry is shifted into another slot.
#
# run: pytest -m simulation simbi/simulation/tests/test_primitive_rows.py
# =============================================================================

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.simulation

PROBE = '''
import json
import os
from functools import partial
from typing import Annotated, Any

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Spacetime

OV: dict[str, Any] = json.loads(os.environ["ROW_PROBE"])
RES = tuple(OV["res"])


class RowProbe(SimbiProblem):
    """a uniform box on one chart whose rows come from ROW_PROBE"""

    resolution: Annotated[tuple[int, ...], ProblemParam(RES, cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([tuple(b) for b in OV["bounds"]], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem(OV.get("coord", "cartesian")), description="chart")]
    spacetime: Annotated[Spacetime, ProblemParam(Spacetime(OV.get("spacetime", "minkowski")), description="metric")]
    schwarzschild_mass: Annotated[float, ProblemParam(OV.get("mass", 0.0), description="mass")]
    kerr_spin: Annotated[float, ProblemParam(OV.get("spin", 0.0), description="spin")]
    regime: Annotated[Regime, ProblemParam(Regime(OV["regime"]), description="regime")]
    adiabatic_index: Annotated[float | None, ProblemParam(OV.get("gamma", 5.0 / 3.0), description="gamma")]
    sound_speed: Annotated[float | None, ProblemParam(OV.get("cs"), description="cs")]
    boundary_conditions: Annotated[list[Any], ProblemParam([BoundaryCondition.OUTFLOW], description="walls")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]

    def initial_primitive_state(self):
        n = 1
        for r in RES:
            n *= r
        row = tuple(OV["row"])

        def gas():
            for _ in range(n):
                yield row

        if self.regime in (Regime.NMHD, Regime.RMHD, Regime.IMHD):
            r3 = list(RES) + [1] * (3 - len(RES))

            def faces(k):
                count = 1
                for ax in range(3):
                    count *= r3[ax] + (1 if ax == k else 0)
                return (0.0 for _ in range(count))

            return gas, partial(faces, 0), partial(faces, 1), partial(faces, 2)
        return gas
'''
RHO, PRE = 1.25, 2.5
VEL = (0.0625, -0.03125, 0.015625)
FLAT = {"bounds": [[1.0, 2.0], [0.25, 1.25], [0.0, 1.0]]}
KS_SPH = {"spacetime": "schwarzschild_ks", "mass": 1.0, "bounds": [[4.0, 6.0], [0.5, 1.5]]}
KERR_SPH = {"spacetime": "kerr_ks", "mass": 1.0, "spin": 0.5, "bounds": [[4.0, 6.0], [0.5, 1.5]]}
KS_CYL = {"spacetime": "schwarzschild_ks", "mass": 1.0, "coord": "cylindrical", "bounds": [[4.0, 6.0], [-1.0, 1.0]]}
KS_BOX = {"spacetime": "schwarzschild_ks", "mass": 1.0, "bounds": [[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]]}


def _case(regime: str, dims: int, nvel: int, chart: dict, pressure: bool = True, trailing: float | None = None, **extra) -> dict:
    res = [8] * dims if regime not in ("nmhd", "rmhd", "imhd") or dims == 3 else [8, 8, 1]
    row = [RHO, *VEL[:nvel]] + ([PRE] if pressure else []) + ([trailing] if trailing is not None else [])
    bounds = chart["bounds"][: len(res)] if len(chart["bounds"]) >= len(res) else chart["bounds"] + [[0.0, 1.0]]
    settings = {**chart, **extra, "regime": regime, "res": res, "row": row, "bounds": bounds}
    if regime in ("isothermal", "imhd"):
        settings.update(gamma=None, cs=1.0)
    return settings


# name: (settings, velocity components carried, whether a pressure is stored)
SUPPORTED = {
    "newtonian-cartesian-1d": (_case("newtonian", 1, 1, FLAT), 1, True),
    "newtonian-cartesian-2d": (_case("newtonian", 2, 2, FLAT), 2, True),
    "newtonian-cartesian-3d": (_case("newtonian", 3, 3, FLAT), 3, True),
    "newtonian-spherical-2d": (_case("newtonian", 2, 2, FLAT, coord="spherical"), 2, True),
    "newtonian-spherical-3d": (_case("newtonian", 3, 3, {"bounds": [[1.0, 2.0], [0.5, 1.5], [0.0, 1.0]]}, coord="spherical"), 3, True),
    "newtonian-cylindrical-rz": (_case("newtonian", 2, 2, FLAT, coord="cylindrical"), 2, True),
    "newtonian-cylindrical-rphi": (_case("newtonian", 2, 2, FLAT, coord="planar_cylindrical"), 2, True),
    "rhd-cartesian-2d": (_case("rhd", 2, 2, FLAT, gamma=4.0 / 3.0), 2, True),
    "rhd-ks-line": (_case("rhd", 1, 1, KS_SPH, coord="spherical", gamma=4.0 / 3.0), 1, True),
    "rhd-ks-plane": (_case("rhd", 2, 2, KS_SPH, coord="spherical", gamma=4.0 / 3.0), 2, True),
    "rhd-ks-plane-lifted": (_case("rhd", 2, 3, KS_SPH, coord="spherical", gamma=4.0 / 3.0), 3, True),
    "rhd-kerr-plane-lifted": (_case("rhd", 2, 3, KERR_SPH, coord="spherical", gamma=4.0 / 3.0), 3, True),
    "rhd-ks-meridional-lifted": (_case("rhd", 2, 3, KS_CYL, gamma=4.0 / 3.0), 3, True),
    "rhd-ks-box": (_case("rhd", 3, 3, KS_BOX, gamma=4.0 / 3.0), 3, True),
    "isothermal-cartesian-2d": (_case("isothermal", 2, 2, FLAT, pressure=False), 2, False),
    "isothermal-cartesian-2d-trailing": (_case("isothermal", 2, 2, FLAT, pressure=False, trailing=1.0), 2, False),
    "isothermal-spherical-2d": (_case("isothermal", 2, 2, FLAT, pressure=False, coord="spherical"), 2, False),
    "nmhd-cartesian-2.5d": (_case("nmhd", 2, 3, FLAT), 3, True),
    "imhd-cartesian-2.5d": (_case("imhd", 2, 3, FLAT, pressure=False), 3, False),
    "imhd-cartesian-2.5d-trailing": (_case("imhd", 2, 3, FLAT, pressure=False, trailing=1.0), 3, False),
}
# name: (settings, row width the refusal names)
UNSUPPORTED = {
    "newtonian-spherical-2d-lifted": (_case("newtonian", 2, 3, FLAT, coord="spherical"), 5),
    "newtonian-cartesian-2d-lifted": (_case("newtonian", 2, 3, FLAT), 5),
    "newtonian-cylindrical-rz-lifted": (_case("newtonian", 2, 3, FLAT, coord="cylindrical"), 5),
    "rhd-cartesian-2d-lifted": (_case("rhd", 2, 3, FLAT, gamma=4.0 / 3.0), 5),
    "rhd-kerr-plane": (_case("rhd", 2, 2, KERR_SPH, coord="spherical", gamma=4.0 / 3.0), 4),
    "isothermal-cartesian-2d-two-trailing": (_case("isothermal", 2, 2, FLAT, pressure=False, trailing=1.0) | {"row": [RHO, *VEL[:2], 1.0, 1.0]}, 5),
    "nmhd-cartesian-2.5d-trailing": (_case("nmhd", 2, 3, FLAT, trailing=1.0), 6),
}


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "row_probe.py"
    path.write_text(PROBE)
    return path


def _run(probe: Path, settings: dict, out: Path) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(probe.parent)!r})\n"
        "from simbi.simulation import runner\n"
        "from row_probe import RowProbe\n"
        "runner.run(RowProbe(data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps=0)\n"
    )
    env = dict(os.environ, ROW_PROBE=json.dumps(settings))
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=env)


def _primitives(out: Path) -> dict[str, np.ndarray]:
    files = sorted(out.glob("*.h5"))
    assert files, f"no checkpoint in {out}"
    found: dict[str, np.ndarray] = {}
    with h5py.File(files[0], "r") as f:
        level = f["level_0"]
        ng = int(level["mesh"].attrs["halo_width"])

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and "/primitives/" in f"/{name}":
                data = obj[()]
                found[name.rsplit("/", 1)[-1]] = data[tuple(slice(ng, -ng) if s > 2 * ng else slice(None) for s in data.shape)]

        level.visititems(visit)
    return found


@pytest.mark.parametrize("case", list(SUPPORTED))
def test_every_entry_of_a_supported_row_lands_in_its_own_slot(tmp_path: Path, probe: Path, case: str) -> None:
    settings, nvel, stored_pressure = SUPPORTED[case]
    done = _run(probe, settings, tmp_path / "out")
    assert done.returncode == 0, f"{case}: the run failed:\n{done.stderr[-2000:]}"
    prims = _primitives(tmp_path / "out")
    names = ["rho"] + [f"v{k + 1}" for k in range(nvel)] + (["pre"] if stored_pressure else [])
    assert set(names) <= set(prims), f"{case}: stored primitives {sorted(prims)} lack {sorted(set(names) - set(prims))}"
    assert f"v{nvel + 1}" not in prims, f"{case}: a velocity beyond the row's {nvel} components is stored"
    want = dict(zip(names, [RHO, *VEL[:nvel]] + ([PRE] if stored_pressure else [])))
    # the initial checkpoint stores the primitives recovered from the seeded conserved state; the
    # recovery round-trips each entry to its round-off, and a shifted slot misplaces an entry by
    # its whole magnitude (the seeded values are pairwise distinct by at least 1.5e-2)
    for name, value in want.items():
        got = prims[name]
        assert np.allclose(got, value, rtol=1e-12, atol=1e-14), f"{case}: {name} holds {got.ravel()[:4]}, seeded {value}"


@pytest.mark.parametrize("case", list(UNSUPPORTED))
def test_a_row_of_an_unsupported_width_is_refused_by_name(tmp_path: Path, probe: Path, case: str) -> None:
    settings, width = UNSUPPORTED[case]
    done = _run(probe, settings, tmp_path / "out")
    assert done.returncode != 0, f"{case}: the run accepted a {width}-entry row:\n{done.stdout[-1500:]}"
    flat = " ".join(done.stderr.split())
    # the backend names the widths the dispatched arm reads; where the front door fixes the exact
    # width (flat cartesian hydro, MHD) it refuses the row first, with the width it expects
    named = f"a primitive row of {width} entries" in flat or "must have EXACTLY" in flat
    assert named, f"{case}: refusal not named:\n{flat[-2000:]}"
    assert not list((tmp_path / "out").glob("*.h5")), f"{case}: a refused run published a checkpoint"
