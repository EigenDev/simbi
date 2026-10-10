# =============================================================================
# test_shared_admission.py
#
# one semantic admission for every driver. a setting the dispatched kernels
# never apply is refused by name, before any output, on the single grid, on host
# tiles and through `simbi launch` alike:
# - a magnetic slip body on several tiles (the tiled marches step no slip);
# - a bonded fragment assembly outside newtonian/isothermal hydro;
# - resistivity on a hydro regime;
# - tracer cohorts on a refined hierarchy;
# - viscosity on the relativistic gas;
# - a neumann wall on a magnetized regime (no gradient ghost fill);
# - the balanced reconstruction outside the newtonian regime;
# - a non-hlle solver on isothermal hydro;
# - mesh refinement of a one-dimensional magnetized grid;
# - the balanced reconstruction with an immersed body across a periodic wrap (the
#   periodic images sit at different body potentials).
# a driven face whose prescription does not lower against the regime is refused at
# admission, naming the face, on every driver including the refined tiles, which
# otherwise build each level's kernels after the run is admitted. a periodic
# declaration on one face of an axis cut into tiles is refused at admission: the one
# tile wraps onto itself and its neighbor's seam is never filled.
# the launch reads the tile count from its layout, so a config declaring
# several devices launches on a cpu build; a refined mesh on a curved spacetime
# outside the relativistic gas on the kerr-schild box,
# which the launched workers do not build, is refused by the launch scope before
# any output.
#
# run: pytest -m simulation simbi/simulation/tests/test_shared_admission.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

PROBE = '''
import json
import os
from functools import partial
from typing import Annotated, Any

from pydantic import computed_field

from simbi import Neumann, ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.bodies import BodyCapability, BondedAssembly, BondMaterial, ImmersedBodyConfig, RigidProperties

OV: dict[str, Any] = json.loads(os.environ["ADMISSION_PROBE"])
RES = tuple(OV.get("res", [16, 16]))


def _bcs():
    return [Neumann(**b) if isinstance(b, dict) else BoundaryCondition(b) for b in OV.get("bcs", ["outflow"])]


class Probe(SimbiProblem):
    """a small box whose settings come from ADMISSION_PROBE"""

    resolution: Annotated[tuple[int, ...], ProblemParam(RES, cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0)] * len(RES), description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime(OV.get("regime", "newtonian")), description="regime")]
    solver: Annotated[Solver, ProblemParam(Solver(OV.get("solver", "hlle")), description="solver")]
    sound_speed: Annotated[float | None, ProblemParam(OV.get("cs"), description="cs")]
    adiabatic_index: Annotated[float | None, ProblemParam(None if OV.get("regime") == "isothermal" else 5.0 / 3.0, description="gamma")]
    wb_reconstruction: Annotated[bool, ProblemParam(OV.get("wb", False), description="balanced")]
    boundary_conditions: Annotated[list[Any], ProblemParam(_bcs(), description="walls")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(OV.get("gpus", 1), cli=True, description="devices")]
    refinement_enabled: Annotated[bool, ProblemParam(bool(OV.get("regions")), description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(1 + len(OV.get("regions", [])), description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam(OV.get("regions", []), description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2] * len(OV.get("regions", [])), description="ratios")]

    @computed_field
    @property
    def viscosity(self) -> float:
        return OV.get("viscosity", 0.0)

    @computed_field
    @property
    def resistivity(self) -> float:
        return OV.get("resistivity", 0.0)

    @computed_field
    @property
    def bx1_inner_expressions(self) -> Any:
        return OV.get("driven", {})

    @computed_field
    @property
    def n_tracers(self) -> int:
        return OV.get("n_tracers", 0)

    @computed_field
    @property
    def immersed_bodies(self) -> list[Any]:
        if not OV.get("body"):
            return []
        return [ImmersedBodyConfig(capability=BodyCapability.RIGID, mass=0.0, radius=0.1, position=(0.5, 0.5),
                                   velocity=(0.0, 0.0), rigid=RigidProperties(inertia=1.0, apply_no_slip=True))]

    @computed_field
    @property
    def bonded_assembly(self) -> Any:
        if not OV.get("bonded"):
            return None
        return BondedAssembly(positions=[[0.4, 0.5, 0.0], [0.6, 0.5, 0.0]], masses=[1.0, 1.0], radii=[0.08, 0.08],
                              bonds=[[0, 1]], bond_material=BondMaterial(k_n=200.0))

    def tracer_cohort(self):
        if not OV.get("cohort"):
            return None
        n = 1
        for r in RES:
            n *= r
        return (ii % 3 for ii in range(n))

    def initial_primitive_state(self):
        n = 1
        for r in RES:
            n *= r
        row = tuple(OV.get("row", [1.0, 0.1, 0.0, 1.0]))

        def gas():
            for _ in range(n):
                yield row

        if self.regime in (Regime.NMHD, Regime.RMHD, Regime.IMHD):
            r3 = list(RES) + [1] * (3 - len(RES))
            bval = OV.get("b", [0.1, 0.0, 0.0])

            def faces(k):
                count = 1
                for ax in range(3):
                    count *= r3[ax] + (1 if ax == k else 0)
                return (bval[k] for _ in range(count))

            return gas, partial(faces, 0), partial(faces, 1), partial(faces, 2)
        return gas
'''
MHD = {"regime": "nmhd", "res": [16, 16, 1], "row": [1.0, 0.1, 0.0, 0.0, 1.0]}
# name: (settings, refusal every driver names)
REFUSED = {
    "bonded-mhd": ({**MHD, "bonded": True}, "bonded_assembly (2 fragments) is wired for newtonian/isothermal hydro; got regime 'nmhd'"),
    "bonded-rhd": ({"regime": "rhd", "bonded": True}, "bonded_assembly (2 fragments) is wired for newtonian/isothermal hydro; got regime 'rhd'"),
    "resistive-hydro": ({"resistivity": 0.1}, "resistivity is an MHD operator"),
    "cohort-refined": ({"n_tracers": 100, "cohort": True, "regions": [[0.25, 0.75, 0.25, 0.75]]}, "tracer_cohort seeding on a refined hierarchy"),
    "viscous-rhd": ({"regime": "rhd", "viscosity": 0.01}, "viscous transport is not built for the rhd regime"),
    "neumann-mhd": ({**MHD, "bcs": [{"rho": 5.0, "pressure": 5.0}, "outflow", "outflow", "outflow"]}, "a neumann or robin wall with the nmhd regime"),
    "balanced-isothermal": ({"regime": "isothermal", "cs": 1.0, "row": [1.0, 0.1, 0.0], "wb": True}, "wb_reconstruction with the isothermal regime"),
    "balanced-mhd": ({**MHD, "wb": True}, "wb_reconstruction with the nmhd regime"),
    "hllc-isothermal": ({"regime": "isothermal", "cs": 1.0, "row": [1.0, 0.1, 0.0], "solver": "hllc"}, "solver = 'hllc' with the isothermal regime"),
    "balanced-body-periodic": ({"wb": True, "body": True, "bcs": ["periodic"]}, "a periodic axis with an immersed body cannot use the balance-aware reconstruction"),
    "refined-mhd-line": ({"regime": "nmhd", "res": [32], "row": [1.0, 0.1, 0.0, 0.0, 1.0], "regions": [[0.25, 0.75]]}, "mhd refinement requires a 3d or 2.5d cartesian grid"),
}


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "admission_probe.py"
    path.write_text(PROBE)
    return path


def _env(settings: dict, host_tiles: bool = False) -> dict:
    """the probe settings; host tiles run several devices on the cpu build, a launch takes its tile
    count from the layout alone"""
    import json

    env = dict(os.environ, ADMISSION_PROBE=json.dumps(settings))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    env.pop("SYMBI_GPU_OVERSUBSCRIBE", None)
    if host_tiles:
        env["SYMBI_GPU_OVERSUBSCRIBE"] = "1"
    return env


def _single(probe: Path, settings: dict, out: Path, gpus: int = 1) -> subprocess.CompletedProcess:
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(probe.parent)!r})\n"
        "from simbi.simulation import runner\n"
        "from admission_probe import Probe\n"
        f"runner.run(Probe(gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps=1)\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=_env(settings, host_tiles=gpus > 1))


def _launch(probe: Path, settings: dict, out: Path, layout: Path, script: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script or probe), "--layout", str(layout), "--max-steps", "1",
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env(settings))


def _assert_refused(done: subprocess.CompletedProcess, out: Path, reason: str, label: str) -> None:
    assert done.returncode != 0, f"{label}: the run succeeded:\n{done.stdout[-1500:]}"
    flat = " ".join((done.stdout + done.stderr).split())
    assert reason in flat, f"{label}: refusal not named:\n{flat[-2000:]}"
    assert not list(out.glob("*.h5")), f"{label}: a refused run published a checkpoint"


def _layout_for(tmp_path: Path, settings: dict) -> Path:
    dims = len(settings.get("res", [16, 16]))
    return _layout(tmp_path / "two.toml", workers=2, shape=(2,) + (1,) * (dims - 1))


@pytest.mark.parametrize("case", list(REFUSED))
def test_a_setting_no_kernel_applies_is_refused_by_name_on_every_driver(tmp_path: Path, probe: Path, case: str) -> None:
    settings, reason = REFUSED[case]
    _assert_refused(_single(probe, settings, tmp_path / "single"), tmp_path / "single", reason, f"{case} on one device")
    _assert_refused(_single(probe, settings, tmp_path / "tiles", gpus=2), tmp_path / "tiles", reason, f"{case} on two host tiles")
    _assert_refused(_launch(probe, settings, tmp_path / "launched", _layout_for(tmp_path, settings)), tmp_path / "launched", reason, f"{case} launched")


def test_a_magnetic_slip_body_is_refused_on_launched_workers_as_on_host_tiles(tmp_path: Path, probe: Path) -> None:
    from simbi_configs.examples.ibm.magnetic_slip_disk_2p5d import MagneticSlipDisk2p5d

    script = Path(MagneticSlipDisk2p5d.__module__.replace(".", "/") + ".py")
    reason = "a magnetic slip body with several tiles awaits its tiled slip step"
    for workers, shape in ((1, (1, 1)), (2, (2, 1))):
        layout = _layout(tmp_path / f"w{workers}.toml", workers=workers, shape=shape)
        cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
               "--resolution", "32,32,1", "--data-directory", str(tmp_path / f"out{workers}"), "--checkpoint-interval", "1e9"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_env({}))
        _assert_refused(done, tmp_path / f"out{workers}", reason, f"slip body on {workers} launched worker(s)")


def test_a_launch_reads_its_tile_count_from_the_layout(tmp_path: Path, probe: Path) -> None:
    # a config declaring two devices launches on the host workers its layout names
    settings = {"gpus": 2}
    done = _launch(probe, settings, tmp_path / "launched", _layout_for(tmp_path, settings))
    assert done.returncode == 0, f"the launch failed:\n{done.stderr[-2000:]}"
    assert list((tmp_path / "launched").glob("*final*.h5")), "the launch published no final checkpoint"


def test_a_refined_magnetized_curved_spacetime_is_refused_by_the_launch_scope(tmp_path: Path, probe: Path) -> None:
    # the launched refined workers build the relativistic gas on the kerr-schild box alone; the
    # magnetized gas on the same chart stays outside
    script = tmp_path / "refined_torus.py"
    script.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd_cartesian import GrFishboneMoncriefMhdCartesian\n"
        "\n"
        "class RefinedTorus(GrFishboneMoncriefMhdCartesian):\n"
        "    refinement_enabled: Annotated[bool, ProblemParam(True, description='refined')]\n"
        "    refinement_max_levels: Annotated[int, ProblemParam(2, description='levels')]\n"
        "    refinement_ratios: Annotated[list, ProblemParam([2], description='ratios')]\n"
        "    refinement_regions: Annotated[list, ProblemParam([[10.0, 20.0, 10.0, 20.0, 10.0, 20.0]], description='box')]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", "16,16,16", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    env = _env({})
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path.cwd()), env.get("PYTHONPATH", "")])
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    _assert_refused(done, tmp_path / "out", "spacetime with mesh refinement is outside the distributed first release", "refined magnetized kerr-schild launch")


# a dirichlet prescription of the right arity whose first node names no operation: python
# validation and the json parse accept it, the lowering against the regime does not
UNLOWERABLE = {
    "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
    "nodes": [{"op": "NO_SUCH_OP"}, {"op": "CONSTANT", "value": 0.0},
              {"op": "CONSTANT", "value": 0.0}, {"op": "CONSTANT", "value": 1.0}],
}


def _assert_admission_refusal(done: subprocess.CompletedProcess, out: Path, reason: str, label: str) -> None:
    _assert_refused(done, out, reason, label)
    flat = " ".join((done.stdout + done.stderr).split())
    assert "panicked" not in flat, f"{label}: refused by a panic rather than at admission:\n{flat[-2000:]}"


def test_an_unlowerable_driven_face_is_refused_at_admission_on_every_driver(tmp_path: Path, probe: Path) -> None:
    settings = {"bcs": ["dynamic", "outflow", "outflow", "outflow"], "driven": UNLOWERABLE,
                "regions": [[0.0, 0.5, 0.25, 0.75]]}
    reason = "driven boundary 0 lower"
    _assert_admission_refusal(_single(probe, settings, tmp_path / "single"), tmp_path / "single", reason, "refined single grid")
    _assert_admission_refusal(_single(probe, settings, tmp_path / "tiles", gpus=2), tmp_path / "tiles", reason, "refined host tiles")
    _assert_admission_refusal(_launch(probe, settings, tmp_path / "launched", _layout_for(tmp_path, settings)), tmp_path / "launched", reason, "refined launch")


def test_a_one_sided_periodic_axis_cut_into_tiles_is_refused_at_admission(tmp_path: Path, probe: Path) -> None:
    settings = {"bcs": ["periodic", "outflow", "outflow", "outflow"]}
    reason = "periodic on one face of axis 0"
    # one tile wraps onto itself across the whole domain: the single grid runs
    done = _single(probe, settings, tmp_path / "single")
    assert done.returncode == 0, f"the single grid refused a one-sided periodic axis:\n{done.stderr[-2000:]}"
    _assert_admission_refusal(_single(probe, settings, tmp_path / "tiles", gpus=2), tmp_path / "tiles", reason, "host tiles cut across x")
    _assert_admission_refusal(_launch(probe, settings, tmp_path / "launched", _layout_for(tmp_path, settings)), tmp_path / "launched", reason, "launch cut across x")
