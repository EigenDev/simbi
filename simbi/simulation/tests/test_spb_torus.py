# =============================================================================
# test_spb_torus.py
#
# the Stone, Pringle & Begelman torus configuration and its analysis
# (simbi_configs/examples/newtonian/spb_torus.py, scripts/spb_torus_analysis.py):
# - each run's grid, clock and stress follow the paper's table 1, and runs E, F and G hold run
#   A's torus mass
# - the initial state is the polytropic torus at constant specific angular momentum in the
#   ambient medium: P = A rho^gamma and v_phi = 1 / R inside, p_0 = rho_0 / r at rest outside
# - the analysis of a short run gives inflow plus outflow equal to the net rate at every radius,
#   rates equal to an independent sum over the checkpoint's primitives, and the torus mass of
#   the initial state
#
# run: pytest -m simulation simbi/simulation/tests/test_spb_torus.py
# =============================================================================
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "simbi_configs/examples/newtonian"))
import spb_torus  # noqa: E402


def test_each_run_follows_the_papers_table() -> None:
    for run, (scale, density_power, radius_power, per_decade, ratio, _, gamma, orbits, _) in spb_torus.RUNS.items():
        p = spb_torus.SpbTorus(run=run)
        decades = math.log10(spb_torus.OUTER_RADIUS * ratio)
        assert p.resolution == (round(per_decade * decades) + 1, 192), run
        assert p.bounds == [(1.0 / ratio, 4.0), (0.0, math.pi)], run
        assert p.adiabatic_index == gamma and p.end_time == 2.0 * math.pi * orbits, run
        assert (p.viscosity, p.viscosity_density_power, p.viscosity_radius_power) == (scale, density_power, radius_power), run
        assert (p.viscous_stress is None) == (scale == 0.0), run
    assert spb_torus.SpbTorus(run="A").resolution == (168, 192), "the paper's standard 168 radial zones"
    assert spb_torus.SpbTorus(run="B").resolution[0] == 334, "the paper's high resolution 334 radial zones"
    with pytest.raises(Exception):
        spb_torus.SpbTorus(run="Z")


def test_runs_e_f_and_g_hold_run_a_torus_mass() -> None:
    def mass(run: str) -> float:
        _, _, _, _, _, distortion, gamma, _, _ = spb_torus.RUNS[run]
        n = 1.0 / (gamma - 1.0)
        return spb_torus.polytropic_constant(run) ** (-n) * spb_torus.unit_mass(gamma, distortion)

    reference = mass("A")
    for run in ("E", "F", "G"):
        assert abs(mass(run) / reference - 1.0) < 1e-12, run
    assert spb_torus.polytropic_constant("A") == (1.0 / 2.5) * 0.125 / 2.25


def test_the_initial_state_is_the_torus_in_the_ambient_medium() -> None:
    p = spb_torus.SpbTorus(run="A", polar_zones=24)
    rows = list(p.initial_primitive_state()())
    nr, nt = p.resolution
    assert len(rows) == nr * nt
    a = spb_torus.polytropic_constant("A")
    inside = 0
    for index, (rho, vr, vt, vp, pressure) in enumerate(rows):
        jj, ii = divmod(index, nr)
        lo, hi = 0.01 * 400.0 ** (ii / nr), 0.01 * 400.0 ** ((ii + 1) / nr)
        r, theta = 0.5 * (lo + hi), (jj + 0.5) * math.pi / nt
        assert vr == 0.0 and vt == 0.0
        if rho > spb_torus.AMBIENT_DENSITY:
            inside += 1
            assert abs(pressure / (a * rho ** (5.0 / 3.0)) - 1.0) < 1e-13
            assert abs(vp * r * math.sin(theta) - 1.0) < 1e-13
        else:
            assert rho == spb_torus.AMBIENT_DENSITY and vp == 0.0
            assert abs(pressure * r / spb_torus.AMBIENT_DENSITY - 1.0) < 1e-13
    assert inside > 50, f"{inside} torus cells"


def test_the_analysis_of_a_short_run_closes(tmp_path: Path) -> None:
    out = tmp_path / "D"
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'simbi_configs/examples/newtonian')!r})\n"
        "from spb_torus import SpbTorus\n"
        "from simbi.simulation import runner\n"
        f"runner.run(SpbTorus(data_directory=sys.argv[1], run='D', polar_zones=48, orbits=0.0004, snapshot_orbits=0.0001), compute_mode={ph.BACKEND!r})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), "the short spb run")
    done = subprocess.run([sys.executable, str(REPO / "scripts/spb_torus_analysis.py"), str(out), "--from-orbit", "0.0001", "--fit", "0.03,0.3"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    result = np.load(out / "spb_analysis.npz")
    assert np.allclose(result["inflow"] + result["outflow"], result["net"], rtol=0.0, atol=1e-15)
    assert np.all(result["inflow"] <= 0.0) and np.all(result["outflow"] >= 0.0)
    assert np.any(result["inflow"] < 0.0), "no inflow anywhere; the closure checks nothing"
    # the last series row against the primitives the last checkpoint stores
    last = max(out.glob("*.chkpt.0*.h5"))
    with h5py.File(last, "r") as f:
        ng = int(f["level_0/mesh"].attrs["halo_width"])
        prim = f["level_0/partition_0/hydro/primitives"]
        rho, vr = prim["rho"][()][ng:-ng, ng:-ng], prim["v1"][()][ng:-ng, ng:-ng]
    nj, ni = rho.shape
    faces = 0.02 * 200.0 ** (np.arange(ni + 1) / ni)
    a, b = faces[:-1], faces[1:]
    r = (b**4 - a**4) / 4.0 / ((b**3 - a**3) / 3.0)
    t_face = np.arange(nj + 1) * math.pi / nj
    weight = np.cos(t_face[:-1]) - np.cos(t_face[1:])
    net = 2.0 * math.pi * r**2 * (rho * vr * weight[:, None]).sum(axis=0) / (float(result["torus_mass"]) / (2.0 * math.pi))
    series = result["series"][-1]
    probes = [0, int(np.argmin(np.abs(r - 0.05))), int(np.argmin(np.abs(r - 0.1)))]
    for k, value in zip(probes, series[1:]):
        assert abs(net[k] - value) <= 1e-12 * max(abs(value), 1e-30) + 1e-18, (k, net[k], value)
    # the torus mass: the polytropic torus's integral, sampled on the run's cells
    mass = float(result["torus_mass"])
    n = 1.5
    analytic = spb_torus.polytropic_constant("D") ** (-n) * spb_torus.unit_mass(5.0 / 3.0, 1.125)
    assert abs(mass / analytic - 1.0) < 0.02, f"torus mass {mass} against the integral {analytic}"
