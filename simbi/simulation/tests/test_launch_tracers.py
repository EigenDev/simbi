# =============================================================================
# test_launch_tracers.py
#
# tracer populations on launched workers against the single grid. a 2d gas
# drifting across a box with outflow faces carries a seeded population; gas
# entering through the inner x face injects new tracers. for the discrete
# scheme (newtonian and isothermal gas) and the second- and third-order ito
# schemes, dense, two-seed and geometrically graded, two workers, a 2x2 launch and a
# launch resumed onto three uneven workers publish the single grid's gas and
# tracer population bit for bit, compared by id: positions, ownership, noise
# counters, the injection remainder and the id counter. ito tracers crossing a
# cut and a corner of an uneven 2x2 launch land as on the single grid, and the
# traced kelvin-helmholtz example (ito-3, cohorts, periodic) launches as it
# runs on one grid. a troubled cell flagged against a cut corrects alike on one
# worker and on three. tracers in the cells an accreting sink drains cross into
# its reservoir on host tiles and launched workers as on one grid, discrete
# and ito alike. the newtonian, relativistic and isothermal magnetized
# orszag-tang vortices carry the discrete population, and the newtonian one
# the ito-3 population, on host tiles and launched workers as on one grid.
# tracers with mesh refinement are refused by name before any output.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_tracers.py
# =============================================================================

import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_tiled_tracers import GROUP, INJECTING, RESOLUTION, STEPS, VARIANTS, _assert_same, _final, _iteration, _population, _run

pytestmark = pytest.mark.simulation

LAUNCHED = ["DiscreteDrift", "IsoDrift", "SparseDrift", "Ito2Drift", "Ito3Drift", "SparseIto2Drift", "GradedDrift", "GradedIto2Drift"]
EXTRA = '''
from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang
from simbi_configs.examples.srmhd.rmhd_orszag_tang import OrszagTang as RmhdOrszagTang

class SinkDrift(DiscreteDrift):
    """the drift through an accreting point mass at the box center: tracers in the drained cells
    cross into the sink's reservoir"""

    @computed_field
    @property
    def immersed_bodies(self) -> list:
        from simbi.types.bodies import AccretionProperties, BodyCapability, GravitationalProperties, ImmersedBodyConfig

        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
                mass=0.01,
                radius=0.0,
                position=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                gravitational=GravitationalProperties(softening_length=0.02),
                accretion=AccretionProperties(accretion_radius=0.1),
            )
        ]


class SinkIto2Drift(SinkDrift, Ito2Drift):
    """the ito drift through the accreting sink"""


class RefinedTracedDrift(DiscreteDrift):
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]


class TracedNmhd(NewtonianOrszagTang):
    @computed_field
    @property
    def n_tracers(self) -> int:
        return 400


class TracedRmhd(RmhdOrszagTang):
    @computed_field
    @property
    def n_tracers(self) -> int:
        return 400


class TracedImhd(IsothermalOrszagTang):
    @computed_field
    @property
    def n_tracers(self) -> int:
        return 400


class Ito3Nmhd(TracedNmhd):
    @computed_field
    @property
    def tracer_scheme(self) -> TracerScheme:
        return TracerScheme.ITO3


class DiagonalIto3Drift(Ito3Drift):
    """a dense ito-3 population in gas drifting along the diagonal: tracers within a step of the
    corner of two cuts cross both in one migration"""

    @computed_field
    @property
    def n_tracers(self) -> int:
        return 20000

    def initial_primitive_state(self):
        def gas():
            nx, ny, _ = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    y = (jj + 0.5) / ny
                    yield (1.0 + 0.3 * math.sin(2.0 * math.pi * y), 1.0, 1.0, 1.0)

        return gas
'''


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS + EXTRA)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _launch_run(variants: Path, problem: str, out: Path, layout: Path, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> subprocess.CompletedProcess:
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)


def _launch(variants: Path, problem: str, out: Path, layout: Path, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> Path:
    done = _launch_run(variants, problem, out, layout, steps, checkpoint, inject)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


@pytest.mark.parametrize("problem", LAUNCHED)
def test_launched_workers_carry_the_single_grid_tracer_population(tmp_path: Path, variants: Path, problem: str) -> None:
    group = GROUP[problem]
    want = _run(problem, tmp_path / "single", 1, STEPS)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch(variants, problem, tmp_path / "two", two, STEPS), want, group, f"{problem}: two workers")
    four = _layout(tmp_path / "four.toml", workers=4, shape=(2, 2))
    _assert_same(_launch(variants, problem, tmp_path / "four", four, STEPS), want, group, f"{problem}: a 2x2 launch")
    first = _launch(variants, problem, tmp_path / "first", two, STEPS // 2)
    if group == "continuous_tracers" and problem in INJECTING:
        # the resume carries the stochastic continuation, not positions alone
        with h5py.File(first, "r") as f:
            g = f[group]
            assert np.any(g["random_counter"][:] > 0), "no tracer drew noise before the checkpoint; the resume arm is vacuous"
            assert float(g.attrs["injection_remainder"]) != 0.0, "the checkpoint carries no injection remainder"
            assert int(g.attrs["next_id"]) > int(g.attrs["n_tracers"]) - 1
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _launch(variants, problem, tmp_path / "resumed", three, STEPS - _iteration(first), checkpoint=first)
    assert _iteration(resumed) == STEPS
    _assert_same(resumed, want, group, f"{problem}: resumed onto three uneven workers")


MAGNETIZED = {"TracedNmhd": "tracers", "TracedRmhd": "tracers", "TracedImhd": "tracers", "Ito3Nmhd": "continuous_tracers"}


@pytest.mark.parametrize("problem", list(MAGNETIZED))
def test_magnetized_tiles_and_workers_carry_the_single_grid_tracer_population(tmp_path: Path, variants: Path, problem: str) -> None:
    group = MAGNETIZED[problem]
    want = _run(problem, tmp_path / "single", 1, STEPS)
    _assert_same(_run(problem, tmp_path / "tiles", 2, STEPS), want, group, f"{problem}: two host tiles")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch(variants, problem, tmp_path / "two", two, STEPS), want, group, f"{problem}: two workers")
    first = _launch(variants, problem, tmp_path / "first", two, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _launch(variants, problem, tmp_path / "resumed", three, STEPS - _iteration(first), checkpoint=first)
    _assert_same(resumed, want, group, f"{problem}: resumed onto three uneven workers")
    # the population moved: the gate compares a population the magnetized transport displaced
    start, end = _population(first, group), _population(want, group)
    assert not np.array_equal(start["owner"], end["owner"]), f"{problem}: no tracer changed cell; the arm is vacuous"


def test_a_troubled_cell_against_a_cut_moves_the_tracers_alike(tmp_path: Path, variants: Path) -> None:
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[], [10, 21]])
    # the inflow cell beside the cut at y index 10, in the first stage of step 3: its first-order
    # flux changes the mass entering through the inner x face, which the injection books
    inject = "trouble:3:0:0:9"
    clean = _launch(variants, "DiscreteDrift", tmp_path / "clean", one, STEPS)
    flagged = _launch(variants, "DiscreteDrift", tmp_path / "flagged", one, STEPS, inject=inject)
    a, b = _population(clean, "tracers"), _population(flagged, "tracers")
    assert any(not np.array_equal(x, b[n]) for n, x in a.items()), "the flag left the population unchanged; the correction arm is vacuous"
    _assert_same(_launch(variants, "DiscreteDrift", tmp_path / "corrected", three, STEPS, inject=inject), flagged, "tracers", "corrected on three")


MIGRATIONS = re.compile(r"SIMBI launch tracers: worker=\d+/\d+ migrations=(\d+) corner_migrations=(\d+)")


def test_ito_tracers_crossing_a_cut_and_a_corner_land_as_on_the_single_grid(tmp_path: Path, variants: Path) -> None:
    want = _run("DiagonalIto3Drift", tmp_path / "single", 1, STEPS)
    four = _layout(tmp_path / "four.toml", workers=4, cuts=[[12], [12]])
    done = _launch_run(variants, "DiagonalIto3Drift", tmp_path / "four", four, STEPS)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    _assert_same(_final(tmp_path / "four"), want, "continuous_tracers", "DiagonalIto3Drift: an uneven 2x2 launch")
    # every worker counts the session's moves from the same assembled records
    counts = {(int(m), int(c)) for m, c in MIGRATIONS.findall(done.stderr)}
    assert len(counts) == 1, f"the workers report different migration counts: {counts}"
    moved, corner = counts.pop()
    assert moved > 0, "no ito tracer crossed a cut; the cut arm is vacuous"
    assert corner > 0, f"no single migration crossed both cuts ({moved} crossed one); the corner arm is vacuous"


def test_the_traced_kelvin_helmholtz_example_launches_as_it_runs(tmp_path: Path) -> None:
    from simbi_configs.examples.newtonian.traced_kh import TracedKelvinHelmholtz

    module = TracedKelvinHelmholtz.__module__
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import TracedKelvinHelmholtz\n"
        "runner.run(TracedKelvinHelmholtz(resolution=(32, 32), tracers=400, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode='cpu', max_steps=6)\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(tmp_path / "single")], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    want = _final(tmp_path / "single")
    script = str(Path(module.replace(".", "/") + ".py"))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(two), "--max-steps", "6", "--resolution", "32,32",
           "--tracers", "400", "--data-directory", str(tmp_path / "two"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    _assert_same(_final(tmp_path / "two"), want, "continuous_tracers", "traced kelvin-helmholtz on two workers")


SINKS = {"SinkDrift": "tracers", "SinkIto2Drift": "continuous_tracers"}


@pytest.mark.parametrize("problem", list(SINKS))
def test_tracers_cross_into_a_sink_on_tiles_and_workers_as_on_the_single_grid(tmp_path: Path, variants: Path, problem: str) -> None:
    group = SINKS[problem]
    want = _run(problem, tmp_path / "single", 1, STEPS)
    with h5py.File(want, "r") as f:
        absorbed = int(np.count_nonzero(f[group]["crossed_sink"][:]))
    assert absorbed > 0, f"{problem}: no tracer crossed into the sink; the arm is vacuous"
    # a tracer the sink absorbed is frozen at its crossing: its position, crossing time and
    # noise counter at the half-way checkpoint hold to the end
    half, end = _population(_run(problem, tmp_path / "half", 1, STEPS // 2), group), _population(want, group)
    frozen = np.nonzero(half["crossed_sink"])[0]
    assert frozen.size > 0, f"{problem}: no tracer was absorbed by the half-way checkpoint; the freeze arm is vacuous"
    at = np.searchsorted(end["id"], half["id"][frozen])
    for name in ["position", "crossing_time"] + (["random_counter"] if group == "continuous_tracers" else []):
        np.testing.assert_array_equal(end[name][at], half[name][frozen], err_msg=f"{problem}: an absorbed tracer's {name} moved")
    _assert_same(_run(problem, tmp_path / "tiles", 2, STEPS), want, group, f"{problem}: two host tiles")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch(variants, problem, tmp_path / "two", two, STEPS), want, group, f"{problem}: two workers")
    first = _launch(variants, problem, tmp_path / "first", two, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _launch(variants, problem, tmp_path / "resumed", three, STEPS - _iteration(first), checkpoint=first)
    _assert_same(resumed, want, group, f"{problem}: resumed onto three uneven workers")


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        ("RefinedTracedDrift", "a tracer population with mesh refinement"),
    ],
)
def test_a_tracer_setting_the_workers_do_not_carry_is_refused(tmp_path: Path, variants: Path, problem: str, reason: str) -> None:
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    done = _launch_run(variants, problem, tmp_path / "out", two, 1)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert f"{reason} is outside the distributed first release" in flat, done.stderr[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"
