# =============================================================================
# test_launch_bodies.py
#
# immersed bodies on launched workers. launched workers sum every tile's body
# feedback in flat tile order, the order host tiles sum it in, so two launched
# workers publish what two host tiles publish, bit for bit: every grid field,
# the bodies group, the per-step body history and the body diagnostics file.
# host tiles reach the single grid within the reassociation bound of their
# cross-tile sums (test_tiled_body_ledger, test_isothermal_tile_settings). the
# arms:
# - a shaped rigid wall in a driven wind tunnel, adiabatic and isothermal;
# - a bonded fragment cluster in the same wind;
# - an accreting bondi sink, fresh, and resumed onto three uneven workers
#   against three host tiles cut at the same cells;
# - the adiabatic alpha viscosity about a central body.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_bodies.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 6
VARIANTS = '''
import importlib
from typing import Annotated, Optional

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi.types import Regime
from simbi.types.bodies import BondedAssembly
from simbi_configs.examples.ibm.roche_passage import RochePassage
from simbi_configs.examples.ibm.rubble_wind import RubbleWind
from simbi_configs.examples.ibm.wind_tunnel import WindTunnel
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


def _isothermal(base):
    class Isothermal(base):
        regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="isothermal")]
        sound_speed: Annotated[float, ProblemParam(3.0, description="sound speed")]

        @computed_field
        @property
        def bx1_inner_expressions(self) -> dict:
            freestream = importlib.import_module(base.__module__).RHO_INF
            return expr.boundary([freestream, self.freestream_velocity(), 0.0], dim=2)

        def initial_primitive_state(self):
            adiabatic = super().initial_primitive_state()

            def rows():
                for row in adiabatic():
                    yield tuple(row[:-1])

            return rows

    return Isothermal


IsoWindTunnel = _isothermal(WindTunnel)


class AlphaPrimary(RochePassage):
    viscosity_alpha: Annotated[float, ProblemParam(0.1, description="alpha")]

    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        return None


class Bondi(SphericalBondiTest):
    base_resolution: Annotated[int, ProblemParam(24, description="base")]
'''
# (problem, resolution passed to the run, layout cuts or shape)
ARMS = {
    "wall": ("WindTunnel", (64, 32)),
    "wall-isothermal": ("IsoWindTunnel", (64, 32)),
    "fragments": ("RubbleWind", (64, 32)),
    "alpha": ("AlphaPrimary", (64, 64)),
    "sink": ("Bondi", None),
}


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _tiles(problem: str, resolution: tuple | None, out: Path, steps: int, checkpoint: Path | None = None, cuts: list | None = None) -> Path:
    fields = [f"resolution={resolution!r}"] if resolution is not None else []
    gpus = 2
    if cuts is not None:
        fields.append(f"decompose={cuts!r}")
        gpus = 1
        for axis in cuts:
            gpus *= len(axis) + 1
    if checkpoint is not None:
        fields.append(f"checkpoint_file={str(checkpoint)!r}")
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"p = {problem}(gpus={gpus}, data_directory=sys.argv[1]{''.join(', ' + f for f in fields)})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} on {gpus} host tiles failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(variants: Path, problem: str, resolution: tuple | None, out: Path, layout: Path, steps: int, checkpoint: Path | None = None) -> Path:
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if resolution is not None:
        cmd += ["--resolution", ",".join(str(r) for r in resolution)]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


def _body_groups(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as f:
        for group in ("bodies", "body_diagnostics"):
            if group in f:
                for name in f[group]:
                    out[f"{group}/{name}"] = np.asarray(f[group][name][()])
    return out


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    bx, by = _body_groups(got), _body_groups(want)
    assert by, f"{label}: the reference publishes no body state; the gate is vacuous"
    assert bx.keys() == by.keys(), f"{label}: body datasets {sorted(set(bx) ^ set(by))} differ"
    for name, x in bx.items():
        np.testing.assert_array_equal(x, by[name], err_msg=f"{label}: {name}")


def _assert_same_diagnostics(got: Path, want: Path, label: str) -> None:
    a, b = got / "diagnostics.dat", want / "diagnostics.dat"
    assert b.exists(), f"{label}: the host tiles wrote no body diagnostics"
    assert a.exists(), f"{label}: the launch wrote no body diagnostics"
    assert a.read_text() == b.read_text(), f"{label}: the body diagnostics differ"


@pytest.mark.parametrize("arm", list(ARMS))
def test_launched_workers_publish_the_host_tiles_bodies(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution = ARMS[arm]
    want = _tiles(problem, resolution, tmp_path / "tiles", STEPS)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1) if resolution is not None else (2, 1, 1))
    got = _launch(variants, problem, resolution, tmp_path / "launched", two, STEPS)
    _assert_same(got, want, f"{arm}: two launched workers")
    if arm == "sink":
        _assert_same_diagnostics(tmp_path / "launched", tmp_path / "tiles", arm)


def test_a_sink_resumed_onto_three_uneven_workers_publishes_the_host_tiles_ledger(tmp_path: Path, variants: Path) -> None:
    # the reference sums over the same three tiles, cut at the same cells
    cuts = [[7, 15], [], []]
    first = _tiles("Bondi", None, tmp_path / "tiles-first", STEPS // 2)
    want = _tiles("Bondi", None, tmp_path / "tiles", STEPS - STEPS // 2, checkpoint=first, cuts=cuts)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=cuts)
    got = _launch(variants, "Bondi", None, tmp_path / "launched", three, STEPS - STEPS // 2, checkpoint=first)
    _assert_same(got, want, "sink resumed onto three uneven workers")
