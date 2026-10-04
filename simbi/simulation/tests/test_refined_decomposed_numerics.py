# =============================================================================
# test_refined_decomposed_numerics.py
#
# a refined run on two host tiles carries the numerics the single hierarchy
# carries, and reproduces it bit for bit, both levels included:
# - ppm with the convergence-gated flattening (quartic prolongation), shown to
#   differ from plm;
# - a passive scalar seeded non-uniformly, shown to be carried.
# - a second nested refinement level spanning the cut, fresh and resumed from a
#   checkpoint written on the tiles halfway, every level included;
# - periodic kelvin-helmholtz layers on four tiles whose ragged cuts wrap both
#   axes, the root closed across its seams, shown to differ from outflow walls.
# - the refined isothermal blob on two, three uneven and six tiles cut along
#   both axes through the fine box; a locally isothermal refined run on tiles is
#   refused by name before any output.
# a census, which the tiled drivers do not reduce across tiles, is refused by
# name on host tiles and through `simbi launch` rather than left unrecorded, and
# so are immersed bodies with nested levels on host tiles.
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
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig
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


def _run(cls, out: Path, gpus: int, steps: int = STEPS, resolution=RESOLUTION, **kwargs) -> Path:
    runner.run(cls(resolution=resolution, gpus=gpus, data_directory=out, **kwargs), compute_mode="cpu", max_steps=steps)
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


# the second region spans the cut at x = 0 that two tiles place, with no face on it
NESTED = dict(refinement_max_levels=3, refinement_regions=[[-0.25, 0.25, -0.25, 0.25], [-0.125, 0.125, -0.125, 0.125]], refinement_ratios=[2, 2])


class GravitatingRefinedBlast(RefinedBlast):
    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=0.1, radius=0.0, position=(0.0, 0.0, 0.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]


def _level_2_written(path: Path) -> None:
    with h5py.File(path, "r") as f:
        assert "level_2" in f, f"{path.name} holds no second refined level"


def test_a_second_nested_level_on_two_refined_tiles_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    single = _run(RefinedBlast, tmp_path / "single", 1, **NESTED)
    _level_2_written(single)
    tiles = _run(RefinedBlast, tmp_path / "tiles", 2, **NESTED)
    _level_2_written(tiles)
    _assert_same(tiles, single, "two refined tiles, three levels")


def test_a_second_nested_level_resumed_on_two_refined_tiles_matches_the_uninterrupted_run(tmp_path: Path, host_tiles) -> None:
    single = _run(RefinedBlast, tmp_path / "single", 1, **NESTED)
    partial = _run(RefinedBlast, tmp_path / "first", 2, steps=1, **NESTED)
    _level_2_written(partial)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # the tiled driver counts its step budget from the checkpoint it resumes
    resumed = _run(RefinedBlast, tmp_path / "resumed", 2, steps=STEPS - written_at, checkpoint_file=str(partial), **NESTED)
    _assert_same(resumed, single, "two refined tiles, three levels, resumed")


def test_immersed_bodies_with_nested_levels_on_host_tiles_are_refused_by_name(tmp_path: Path, host_tiles) -> None:
    with pytest.raises(Exception, match="immersed bodies with 3 nested refinement levels and gpus > 1"):
        _run(GravitatingRefinedBlast, tmp_path / "tiles", 2, **NESTED)
    assert not list((tmp_path / "tiles").glob("*final*.h5")), "a refused run published a final checkpoint"


def _fresh_kh(out: Path, gpus: int, **kwargs) -> Path:
    """the refined kelvin-helmholtz run in a fresh interpreter: the example draws its noise from a
    module-level generator, so every arm starts from the same seed only in its own process"""
    fields = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        "from simbi_configs.examples.newtonian.refined_kh import RefinedKelvinHelmholtz\n"
        f"runner.run(RefinedKelvinHelmholtz(resolution=(64, 64), gpus={gpus}, data_directory=sys.argv[1]{', ' + fields if fields else ''}), "
        f"compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, f"the kh run failed:\n{done.stderr[-3000:]}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final


def test_a_periodic_refined_run_on_four_tiles_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    # the fine box spans root cells 4..60 in x and 8..56 in y of the 64 x 64 root, clear of the seams
    single = _fresh_kh(tmp_path / "single", 1)
    walled = _fresh_kh(tmp_path / "walled", 1, boundary_conditions="outflow")
    assert any(not np.array_equal(x, _datasets(walled)[n]) for n, x in _datasets(single).items()), "the seams left the run unchanged; the wrap is never exercised"
    tiles = _fresh_kh(tmp_path / "tiles", 4, decompose=[[26], [36]])
    # the conserved halo ring holds no state (no stage reads it and no exchange moves it); on a
    # periodic root the single writer leaves it zero and the tiled writer fills it, so it is
    # trimmed here and every other dataset is compared whole
    xs, ys = _datasets(tiles), _datasets(single)
    assert xs.keys() == ys.keys(), f"dataset sets differ: {sorted(xs.keys() ^ ys.keys())}"
    with h5py.File(single, "r") as f:
        ng = {lv: int(f[lv]["mesh"].attrs["halo_width"]) for lv in f if lv.startswith("level_")}
    for name, x in xs.items():
        level, _, field = name.partition("/")
        if field.startswith("conserved/"):
            h = ng[level]
            x, y = x[h:-h, h:-h], ys[name][h:-h, h:-h]
        else:
            y = ys[name]
        assert x.shape == y.shape and np.array_equal(x, y), f"four refined tiles wrapping both axes: {name} differs"


def test_refined_isothermal_tiles_carry_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    from simbi_configs.examples.isothermal.refined_isothermal_blob import RefinedIsothermalBlob

    # the fine box covers root cells 8..24 of 32 on both axes; every cut lies four or more cells
    # from its faces
    resolution = (32, 32)
    want = _run(RefinedIsothermalBlob, tmp_path / "single", 1, resolution=resolution)
    assert any(name.startswith("level_1/") for name in _datasets(want)), "the single hierarchy published no fine level"
    for gpus, decompose in ((2, None), (3, [[12, 20], []]), (6, [[12, 20], [14]])):
        extra = {} if decompose is None else {"decompose": decompose}
        got = _run(RefinedIsothermalBlob, tmp_path / f"tiles{gpus}", gpus, resolution=resolution, **extra)
        _assert_same(got, want, f"refined isothermal blob on {gpus} tiles")


def test_a_locally_isothermal_refined_run_on_tiles_is_refused(tmp_path: Path, host_tiles) -> None:
    from simbi_configs.examples.isothermal.refined_locally_iso import RefinedLocallyIso

    with pytest.raises(Exception, match="a locally isothermal closure with mesh refinement and gpus > 1 awaits the whole-patch closure"):
        _run(RefinedLocallyIso, tmp_path / "tiles", 2, resolution=(32, 32))
    assert not list((tmp_path / "tiles").glob("*.h5")), "output was written before the refusal"
