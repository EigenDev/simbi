# =============================================================================
# test_launch_gr_excision.py
#
# relativistic gas on a black hole's horizon-penetrating charts through
# `simbi launch`, with horizon excision. the bondi inflow onto a schwarzschild
# hole in the kerr-schild box (3d) and on the radial line (1d) launches on two
# workers and on uneven workers whose cuts pass through the excised region, and
# resumes onto another partition, publishing the single grid's state bit for
# bit; the excision changes the run. host tiles resumed from a checkpoint keep
# its stored primitives, the excision floor included, as the single grid does.
# a tracer population on the curved spacetime is refused by name before any
# output.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_gr_excision.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 8
BOX = ("simbi_configs.examples.grhd.gr_bondi_cartesian", "GrBondiCartesian")
BOX_RESOLUTION = (24, 24, 24)
BOX_FIELDS = {"domain_radius": "0.12", "excision_radius": "1.6"}
LINE = ("simbi_configs.examples.grhd.gr_bondi", "GrBondi")
LINE_RESOLUTION = (128,)
LINE_FIELDS = {"excision_radius": "1.4"}


def _script(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _flags(fields: dict[str, str]) -> list[str]:
    return [item for name, value in fields.items() for item in (f"--{name.replace('_', '-')}", value)]


def _kwargs(fields: dict[str, str]) -> str:
    return ", ".join(f"{name}={value}" for name, value in fields.items())


def _iteration(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"])


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert "level_0/conserved/den" in ys, f"{label}: the reference publishes no density; the comparison is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize(
    ("case", "resolution", "fields", "uneven"),
    [
        (BOX, BOX_RESOLUTION, BOX_FIELDS, [[11], [13], []]),
        (LINE, LINE_RESOLUTION, LINE_FIELDS, [[40, 90]]),
    ],
    ids=["box", "line"],
)
def test_an_excised_relativistic_launch_matches_the_single_grid(tmp_path: Path, case, resolution, fields, uneven) -> None:
    module, cls = case
    # the radial problem takes its cell count as a scalar
    declared = resolution if len(resolution) > 1 else resolution[0]
    want = _script_reference(module, cls, declared, tmp_path / "single", STEPS, fields=_kwargs(fields))
    bare = _script_reference(module, cls, declared, tmp_path / "bare", STEPS, fields=_kwargs({**fields, "excision_radius": "0.0"}))
    assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], _all_levels(bare)["level_0/conserved/den"]), (
        "the excision left the run unchanged; the gate is vacuous"
    )
    shape = (2,) + (1,) * (len(resolution) - 1)
    two = _layout(tmp_path / "two.toml", workers=2, shape=shape)
    _assert_same(_script_launch(_script(module), resolution, tmp_path / "two", two, STEPS, workers=2, flags=_flags(fields)), want, "two workers")
    workers = int(np.prod([len(cuts) + 1 for cuts in uneven]))
    many = _layout(tmp_path / "uneven.toml", workers=workers, cuts=uneven)
    _assert_same(
        _script_launch(_script(module), resolution, tmp_path / "uneven", many, STEPS, workers=workers, flags=_flags(fields)),
        want,
        f"{workers} uneven workers",
    )
    first = _script_launch(_script(module), resolution, tmp_path / "first", two, STEPS // 2, workers=2, flags=_flags(fields))
    written_at = _iteration(first)
    assert 0 < written_at < STEPS
    # a resumed launch takes `max_steps` further steps
    resumed = _script_launch(
        _script(module), resolution, tmp_path / "resumed", many, STEPS - written_at, workers=workers, checkpoint=first, flags=_flags(fields)
    )
    _assert_same(resumed, want, f"resumed onto {workers} uneven workers")


def test_a_tracer_population_on_a_curved_spacetime_is_refused(tmp_path: Path) -> None:
    module, cls = LINE
    script = tmp_path / "traced.py"
    script.write_text(
        "from pydantic import computed_field\n"
        f"from {module} import {cls}\n\n\n"
        f"class Traced({cls}):\n"
        "    @computed_field\n"
        "    @property\n"
        "    def n_tracers(self) -> int:\n"
        "        return 100\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2,))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", "128", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch ran tracers on a curved spacetime:\n{done.stdout}"
    flat = " ".join((done.stdout + done.stderr).split())
    assert "a tracer population on the schwarzschild_ks spacetime is outside the distributed first release" in flat, flat[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


def _tiles(out: Path, gpus: int, steps: int, checkpoint: Path | None = None, decompose=None) -> Path:
    module, cls = LINE
    extra = "".join(
        [f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else "", f", decompose={decompose!r}" if decompose is not None else ""]
    )
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {cls}\n"
        f"runner.run({cls}(resolution={LINE_RESOLUTION[0]}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1], "
        f"{_kwargs(LINE_FIELDS)}{extra}), compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600, env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    assert done.returncode == 0, f"{gpus} host tiles failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def test_excised_host_tiles_resume_on_the_stored_primitives(tmp_path: Path) -> None:
    module, cls = LINE
    want = _script_reference(module, cls, LINE_RESOLUTION[0], tmp_path / "single", STEPS, fields=_kwargs(LINE_FIELDS))
    _assert_same(_tiles(tmp_path / "fresh", 2, STEPS), want, "two host tiles")
    first = _tiles(tmp_path / "first", 2, STEPS // 2)
    written_at = _iteration(first)
    assert 0 < written_at < STEPS
    # a resumed run on tiles takes `max_steps` further steps
    _assert_same(_tiles(tmp_path / "resumed", 3, STEPS - written_at, checkpoint=first, decompose=[[40, 90]]), want, "resumed onto three uneven host tiles")
