# =============================================================================
# test_refined_decomposed_numerics.py
#
# a refined run on two host tiles carries the numerics the single hierarchy
# carries, and reproduces it bit for bit, both levels included:
# - ppm with the convergence-gated flattening (quartic prolongation), shown to
#   differ from plm;
# - a passive scalar seeded non-uniformly, shown to be carried.
# a census, which the tiled drivers do not reduce across tiles, is refused by
# name on host tiles and through `simbi launch` rather than left unrecorded.
#
# run: pytest -m simulation simbi/simulation/tests/test_refined_decomposed_numerics.py
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import simbi.expression as expr
from simbi.simulation import runner
from simbi.simulation.tests.test_launch import _layout
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

pytestmark = pytest.mark.simulation

RESOLUTION = (16, 16, 1)
STEPS = 3


class DyedRefinedBlast(RefinedBlast):
    def passive_scalar(self):
        def dye():
            nx, ny, _ = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    yield 0.25 + 0.5 * ((ii + 0.5) / nx) + 0.1 * ((jj + 0.5) / ny)

        return dye()


def _census():
    g = expr.ExprGraph()
    mass = expr.density(g) * expr.cell_volume(g)
    return expr.Census(name="total", axes=[], values={"mass": mass}).serialize()


class CensusedRefinedBlast(RefinedBlast):
    @property
    def census_expressions(self):
        return [_census()]


def _run(cls, out: Path, gpus: int, **kwargs) -> Path:
    runner.run(cls(resolution=RESOLUTION, gpus=gpus, data_directory=out, **kwargs), compute_mode="cpu", max_steps=STEPS)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _datasets(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for level in (k for k in f.keys() if k.startswith("level_")):
            f[level].visit(lambda n, level=level: out.__setitem__(f"{level}/{n}", f[level][n][()]) if isinstance(f[level][n], h5py.Dataset) and not n.startswith("mesh") else None)
    return out


def _assert_same(a: Path, b: Path, label: str) -> None:
    xs, ys = _datasets(a), _datasets(b)
    assert xs.keys() == ys.keys(), f"{label}: dataset sets differ: {sorted(xs.keys() ^ ys.keys())}"
    for name, x in xs.items():
        assert x.shape == ys[name].shape and np.array_equal(x, ys[name]), f"{label}: {name} differs"


@pytest.fixture
def host_tiles(monkeypatch):
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")


def test_ppm_with_flattening_on_two_refined_tiles_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    ppm = dict(reconstruction="ppm", ppm_flatten_onset=0.05, ppm_flatten_full=0.5)
    single = _run(RefinedBlast, tmp_path / "single", 1, **ppm)
    plm = _run(RefinedBlast, tmp_path / "plm", 1)
    assert any(not np.array_equal(x, _datasets(plm)[n]) for n, x in _datasets(single).items()), "ppm left the run unchanged; the gate is vacuous"
    tiles = _run(RefinedBlast, tmp_path / "tiles", 2, **ppm)
    _assert_same(tiles, single, "two refined tiles under ppm")


def test_a_passive_scalar_on_two_refined_tiles_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    single = _run(DyedRefinedBlast, tmp_path / "single", 1)
    assert any("chi" in n for n in _datasets(single)), "the single hierarchy wrote no dye; the gate is vacuous"
    tiles = _run(DyedRefinedBlast, tmp_path / "tiles", 2)
    _assert_same(tiles, single, "two refined tiles carrying a dye")


def test_a_census_on_host_tiles_is_refused_by_name(tmp_path: Path, host_tiles) -> None:
    with pytest.raises(Exception, match="a census with gpus > 1 awaits a cross-tile census reduction"):
        _run(CensusedRefinedBlast, tmp_path / "tiles", 2, refinement_enabled=False)
    assert not list((tmp_path / "tiles").glob("*final*.h5")), "a refused run published a final checkpoint"


def test_a_census_through_launch_is_refused_by_name(tmp_path: Path) -> None:
    script = tmp_path / "censused.py"
    script.write_text(
        "import simbi.expression as expr\n"
        "from simbi_configs.examples.newtonian.kh import KelvinHelmholtz\n\n"
        "class CensusedKh(KelvinHelmholtz):\n"
        "    @property\n"
        "    def census_expressions(self):\n"
        "        g = expr.ExprGraph()\n"
        "        return [expr.Census(name='total', axes=[], values={'mass': expr.density(g) * expr.cell_volume(g)}).serialize()]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1",
           "--resolution", "16,16", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    assert "a census is outside the distributed first release" in " ".join(done.stderr.split()), done.stderr
