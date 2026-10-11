# =============================================================================
# test_refined_decomposed_numerics.py
#
# a refined run on two host tiles carries the numerics the single hierarchy
# carries, and reproduces it bit for bit, both levels included:
# - ppm with the convergence-gated flattening (quartic prolongation), shown to
#   differ from plm;
# - nine tiles whose center tile lies wholly under the patch;
# - a passive scalar seeded non-uniformly, shown to be carried.
# - a second nested refinement level spanning the cut, fresh and resumed from a
#   checkpoint written on the tiles halfway, every level included;
# - periodic kelvin-helmholtz layers on four tiles whose ragged cuts wrap both
#   axes, the root closed across its seams, shown to differ from outflow walls;
#   the fine box widened over the whole periodic x axis on two and four tiles
#   cut across x, its level wrapping across the tiles at the seam.
# - the refined isothermal blob on two, three uneven and six tiles cut along
#   both axes through the fine box; a locally isothermal refined run on tiles is
#   refused by name before any output.
# a census, which the tiled drivers do not reduce across tiles, is refused by
# name on host tiles and through `simbi launch` rather than left unrecorded.
# a gravitating body over two nested levels on two host tiles publishes every
# level of the single hierarchy's state bit for bit.
# a gravitating draining sink under two nested levels of a 2.5D newtonian MHD
# run on two host tiles, cut short of the finest level with a pressure pulse
# on the cut, publishes every level, every staggered face, the body state and
# the body history of the single hierarchy bit for bit, on two host tiles and on
# two launched workers; the field evolves and changes the gas against a
# zero-field run. immersed bodies over nested levels on magnetized host tiles
# and launched workers outside that scope are refused by name.
#
# run: pytest -m simulation simbi/simulation/tests/test_refined_decomposed_numerics.py
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import math
from dataclasses import replace
from functools import partial
from typing import Annotated, Any

import h5py
import numpy as np
import pytest

import simbi.expression as expr
from simbi.simulation import runner
from simbi.simulation.tests.test_launch import _layout
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.bodies import AccretionProperties, BodyCapability, GravitationalProperties, ImmersedBodyConfig, MagneticProperties
from simbi_configs.examples.ibm.magnetized_sink import MagnetizedBondiSink
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


def test_a_tile_wholly_under_the_patch_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    # the patch spans root cells 8..24 of 32; the cuts at 12 and 20 on both axes leave the center
    # tile wholly under the fine level, its root evolving no cell of its own
    single = _run(RefinedBlast, tmp_path / "single", 1, resolution=(32, 32, 1))
    tiles = _run(RefinedBlast, tmp_path / "tiles", 9, resolution=(32, 32, 1), decompose=[[12, 20], [12, 20]])
    _assert_same(tiles, single, "nine refined tiles, the center one wholly under the patch")


def test_a_passive_scalar_on_two_refined_tiles_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    single = _run(DyedRefinedBlast, tmp_path / "single", 1)
    assert any("chi" in n for n in _datasets(single)), "the single hierarchy wrote no dye; the gate is vacuous"
    tiles = _run(DyedRefinedBlast, tmp_path / "tiles", 2)
    _assert_same(tiles, single, "two refined tiles carrying a dye")


def test_a_census_on_host_tiles_is_refused_by_name(tmp_path: Path, host_tiles) -> None:
    with pytest.raises(Exception, match="a census with several tiles awaits a cross-tile census reduction"):
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
    assert "a census with several tiles awaits a cross-tile census reduction" in " ".join(done.stderr.split()), done.stderr
    assert not list((tmp_path / "out").glob("*.h5")), "a refused launch published a checkpoint"


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
    # the step cap is an absolute root iteration: the resume from written_at runs to STEPS
    resumed = _run(RefinedBlast, tmp_path / "resumed", 2, steps=STEPS, checkpoint_file=str(partial), **NESTED)
    _assert_same(resumed, single, "two refined tiles, three levels, resumed")


def test_immersed_bodies_with_nested_levels_on_host_tiles_match_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    single = _run(GravitatingRefinedBlast, tmp_path / "single", 1, **NESTED)
    _level_2_written(single)
    _assert_same(_run(GravitatingRefinedBlast, tmp_path / "tiles", 2, **NESTED), single, "two refined tiles, three levels, a gravitating body")


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


def test_a_patch_spanning_a_cut_periodic_axis_on_host_tiles_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    # the kh fine box widened over the whole periodic x axis, the tiles cut across x: the patch has
    # no face along x, and its level wraps across the tiles at the seam as the root does
    spanning = dict(refinement_regions=[[-0.5, 0.5, -0.375, 0.375]])
    single = _fresh_kh(tmp_path / "single", 1, **spanning)
    walled = _fresh_kh(tmp_path / "walled", 1, boundary_conditions="outflow", **spanning)
    assert any(not np.array_equal(x, _datasets(walled)[n]) for n, x in _datasets(single).items()), "the seam left the spanning run unchanged"
    for gpus, cuts in ((2, [[26], []]), (4, [[20], [36]])):
        tiles = _fresh_kh(tmp_path / f"tiles{gpus}", gpus, decompose=cuts, **spanning)
        xs, ys = _datasets(tiles), _datasets(single)
        assert xs.keys() == ys.keys(), f"dataset sets differ: {sorted(xs.keys() ^ ys.keys())}"
        with h5py.File(single, "r") as f:
            ng = {lv: int(f[lv]["mesh"].attrs["halo_width"]) for lv in f if lv.startswith("level_")}
        for name, x in xs.items():
            level, _, field = name.partition("/")
            y = ys[name]
            if field.startswith("conserved/"):
                h = ng[level]
                x, y = x[h:-h, h:-h], y[h:-h, h:-h]
            assert x.shape == y.shape and np.array_equal(x, y), f"{gpus} refined tiles cut across the periodic axis: {name} differs"


SPANNING_KH = {
    "viscous": (
        "viscous_kh",
        "class Variant(RefinedKelvinHelmholtz):\n"
        "    @computed_field\n"
        "    @property\n"
        "    def viscosity(self) -> float:\n"
        "        return 0.001\n",
    ),
    "ito2": (
        "ito_kh",
        "class Variant(RefinedKelvinHelmholtz):\n"
        "    @computed_field\n"
        "    @property\n"
        "    def n_tracers(self) -> int:\n"
        "        return 200\n\n"
        "    @computed_field\n"
        "    @property\n"
        "    def tracer_scheme(self) -> TracerScheme:\n"
        "        return TracerScheme.ITO2\n",
    ),
}


def _spanning_kh_variant(tmp_path: Path, name: str) -> str:
    stem, body = SPANNING_KH[name]
    (tmp_path / f"{stem}.py").write_text(
        "from pydantic import computed_field\n"
        "from simbi.types import TracerScheme\n"
        "from simbi_configs.examples.newtonian.refined_kh import RefinedKelvinHelmholtz\n\n\n" + body
    )
    return stem


def _run_spanning_kh(tmp_path: Path, stem: str, out: Path, gpus: int, **kwargs) -> Path:
    """the variant's kh run with its fine box widened over the whole periodic x axis, in a fresh
    interpreter: the example draws its noise from a module-level generator"""
    fields = "".join(f", {k}={v!r}" for k, v in kwargs.items())
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {stem} import Variant\n"
        f"runner.run(Variant(resolution=(64, 64), gpus={gpus}, refinement_regions=[[-0.5, 0.5, -0.375, 0.375]], data_directory=sys.argv[1]{fields}), "
        f"compute_mode='cpu', max_steps={STEPS})\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{stem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final


@pytest.mark.parametrize("name", list(SPANNING_KH))
def test_a_patch_spanning_a_cut_periodic_axis_carries_the_corner_reading_operators_on_host_tiles(tmp_path: Path, host_tiles, name: str) -> None:
    # the viscous cross derivatives and the continuous tracers' coefficient interpolation read the
    # fine halo cells where the seam meets a coarse-fine face: the periodic image of the prolonged
    # ghost rows on the single hierarchy, the fine exchange's on the tiles
    from simbi.simulation.tests.test_decomposed_mhd import _population

    stem = _spanning_kh_variant(tmp_path, name)
    want = _run_spanning_kh(tmp_path, stem, tmp_path / "single", 1)
    plain = _fresh_kh(tmp_path / "plain", 1, refinement_regions=[[-0.5, 0.5, -0.375, 0.375]])
    if name == "viscous":
        assert any(not np.array_equal(x, _datasets(plain)[n]) for n, x in _datasets(want).items()), "the viscosity left the run unchanged; the gate is vacuous"
    else:
        population = _population(want, "continuous_tracers")
        levels = population["owner"].astype(np.uint64) >> np.uint64(56)
        assert 1 in set(levels.tolist()), "no tracer lives on the fine level; the interpolation there is never read"
    for gpus, cuts in ((2, [[26], []]), (4, [[20], [36]])):
        got = _run_spanning_kh(tmp_path, stem, tmp_path / f"tiles{gpus}", gpus, decompose=cuts)
        _assert_same(got, want, f"{name} kh spanning x on {gpus} tiles cut at {cuts}")
        if name != "viscous":
            tiled = _population(got, "continuous_tracers")
            assert tiled.keys() == population.keys(), "the tracer fields differ"
            for field, values in population.items():
                np.testing.assert_array_equal(tiled[field], values, err_msg=f"{gpus} tiles: continuous_tracers/{field}")


class NestedMagnetizedSink(MagnetizedBondiSink):
    """the magnetized sink under two nested fine levels"""

    def setup(self) -> None:
        super().setup()
        half = self.bounds[0][1]
        self.refinement_enabled = True
        self.refinement_max_levels = 3
        self.refinement_ratios = [2, 2]
        self.refinement_regions = [[-0.75 * half, 0.75 * half] * 3, [-0.4 * half, 0.4 * half] * 3]


def test_magnetized_bodies_with_nested_levels_on_host_tiles_are_refused_by_name(tmp_path: Path, host_tiles) -> None:
    with pytest.raises(Exception, match="immersed bodies with 3 nested refinement levels on magnetized tiles"):
        _run(NestedMagnetizedSink, tmp_path / "out", 2, resolution=(16, 16, 16))
    assert not list((tmp_path / "out").glob("*.h5")), "a refused run published a checkpoint"


SINK_N = 64
# the root cell the cut between the two tiles lies on: the first tile holds levels 0 and 1 alone
SINK_CUT = 14
# beta ~ 1e5 outside the sink: the drain keeps the finest density near 1e-4. at field_scale 0.01 the
# drained interior reaches vacuum (rho ~ 3e-8) by t = 0.037 and the single hierarchy halts
SINK_FIELD = 0.003


class NestedSink2p5d(SimbiProblem):
    """a gravitating draining sink under two nested levels of a sheared 2.5D field"""

    field_scale: Annotated[float, ProblemParam(SINK_FIELD, description="field strength")]
    magnetic_resistivity: Annotated[float | None, ProblemParam(None, description="the sink's ohmic resistivity")]
    resolution: Annotated[tuple[int, ...], ProblemParam((SINK_N, SINK_N, 1), description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="regime")]
    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    boundary_conditions: Annotated[list[Any], ProblemParam([BoundaryCondition("outflow")], description="walls")]
    end_time: Annotated[float, ProblemParam(0.05, checkpoint_safe=True, description="t")]
    checkpoint_interval: Annotated[float, ProblemParam(1.0e9, checkpoint_safe=True, description="dt out")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.0625, 0.9375, 0.0625, 0.9375], [0.3125, 0.6875, 0.3125, 0.6875]], description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description="ratios")]

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        sink = ImmersedBodyConfig(capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL, mass=0.01, radius=0.08,
                                  position=(0.5, 0.5), velocity=(0.0, 0.0),
                                  gravitational=GravitationalProperties(softening_length=0.05),
                                  accretion=AccretionProperties(accretion_radius=0.08))
        if self.magnetic_resistivity is None:
            return [sink]
        return [replace(sink, magnetic=MagneticProperties(resistivity=self.magnetic_resistivity))]

    # face-normal components each independent of their own axis: discretely divergence-free
    def _bx(self, y: float) -> float:
        return self.field_scale * (0.4 + 0.2 * math.sin(2.0 * math.pi * y + 0.3))

    def _by(self, x: float) -> float:
        return self.field_scale * (0.2 + 0.2 * math.cos(2.0 * math.pi * x + 0.7))

    def initial_primitive_state(self):
        dx = 1.0 / SINK_N

        def gas():
            for jj in range(SINK_N):
                y = (jj + 0.5) * dx
                for ii in range(SINK_N):
                    x = (ii + 0.5) * dx
                    bump = 0.3 * math.exp(-(((x - 0.5) / 0.07) ** 2 + ((y - 0.48) / 0.09) ** 2))
                    # a pressure pulse centered on the cut, whose waves cross it on the first refined level
                    pulse = 0.5 * math.exp(-(((x - SINK_CUT * dx) / 0.04) ** 2 + ((y - 0.5) / 0.1) ** 2))
                    yield (1.0 + bump, 0.0, 0.0, 0.0, 1.0 + bump + pulse)

        def faces(k):
            for _kk in range(1 + (k == 2)):
                for jj in range(SINK_N + (k == 1)):
                    for ii in range(SINK_N + (k == 0)):
                        if k == 0:
                            yield self._bx((jj + 0.5) * dx)
                        elif k == 1:
                            yield self._by((ii + 0.5) * dx)
                        else:
                            yield 0.1 * self.field_scale

        return gas, partial(faces, 0), partial(faces, 1), partial(faces, 2)


def _run_sink(out: Path, gpus: int, **kwargs) -> Path:
    # the run reaches its end time in ~170 root steps; the step cap stays clear of it
    decompose = dict(decompose=[[SINK_CUT], []]) if gpus > 1 else {}
    return _run(NestedSink2p5d, out, gpus, steps=1000, resolution=(SINK_N, SINK_N, 1), **decompose, **kwargs)


def _body_groups(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for group in ("bodies", "body_diagnostics"):
            f[group].visit(lambda n, g=group: out.__setitem__(f"{g}/{n}", f[g][n][()]) if isinstance(f[g][n], h5py.Dataset) else None)
    return out


def _interior(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        level = name.split("/")[0]
        h = int(f[level]["mesh"].attrs["halo_width"])
        return f[name][()][h:-h, h:-h]


def test_a_magnetized_nested_sink_on_two_host_tiles_matches_the_single_hierarchy(tmp_path: Path, host_tiles) -> None:
    single = _run_sink(tmp_path / "single", 1)
    _level_2_written(single)
    with h5py.File(single, "r") as f:
        assert float(f["bodies/total_accreted_mass"][0]) > 1.0e-2, "the sink drained too little; the gate is vacuous"
        # gas crosses the cut on level 1: the x-momentum in the two level-1 columns beside it
        # (measured 4.5e-3)
        h = int(f["level_1"]["mesh"].attrs["halo_width"])
        # level 1 starts at root cell 4; the cut lies between its cells beside - 1 and beside
        beside = 2 * (SINK_CUT - 4)
        crossing = np.abs(f["level_1/conserved/m1"][()][h:-h, beside - 1 + h:beside + 1 + h]).max()
    assert crossing > 1.0e-3, f"the level-1 momentum beside the cut peaks at {crossing:e}; the cut couples nothing"
    # the field acts: the finest field grows about eightfold from its initial peak 0.6 field_scale,
    # and the run without a field differs in the finest density by up to 63%
    finest_b = np.abs(_interior(single, "level_2/conserved/b1")).max()
    assert finest_b > 4.0 * 0.6 * SINK_FIELD, f"the finest field peaks at {finest_b:e}; it is passive"
    unmagnetized = _run_sink(tmp_path / "unmagnetized", 1, field_scale=0.0)
    rho, rho_0 = _interior(single, "level_2/conserved/den"), _interior(unmagnetized, "level_2/conserved/den")
    change = np.max(np.abs(rho - rho_0) / rho_0)
    assert change > 0.25, f"the field changes the finest density by at most {change:e} relative; it couples nothing"

    tiles = _run_sink(tmp_path / "tiles", 2)
    _assert_same(tiles, single, "a magnetized nested sink on two refined tiles")
    want, got = _body_groups(single), _body_groups(tiles)
    assert want.keys() == got.keys(), f"body datasets differ: {sorted(want.keys() ^ got.keys())}"
    for name, x in want.items():
        assert x.shape == got[name].shape and np.array_equal(x, got[name]), f"a magnetized nested sink on two refined tiles: {name} differs"


# one class attribute per variant, set in the launched script's subclass
SINK_VARIANTS = {
    "hlld": ("from simbi.types import Solver", "solver: Annotated[Solver, ProblemParam(Solver.HLLD, description='solver')]"),
    "resistive_sink": ("", "magnetic_resistivity: Annotated[float | None, ProblemParam(1.0e-2, description='eta')]"),
}


def _launch_sink(tmp_path: Path, out: Path, variant: str | None = None) -> subprocess.CompletedProcess:
    """the sink through `simbi launch` on two workers cut at root cell 14, without the experimental opt-in"""
    imports, attribute = SINK_VARIANTS[variant] if variant else ("", "pass")
    script = tmp_path / f"launched_sink_{variant or 'gated'}.py"
    script.write_text(
        "from typing import Annotated\n\nfrom simbi import ProblemParam\n"
        "from simbi.simulation.tests.test_refined_decomposed_numerics import NestedSink2p5d\n"
        f"{imports}\n\n\nclass LaunchedSink(NestedSink2p5d):\n    {attribute}\n"
    )
    layout = _layout(tmp_path / f"layout_{variant or 'gated'}.toml", 2, cuts=[[SINK_CUT], []])
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1000",
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)


def test_a_magnetized_nested_sink_through_launch_matches_the_single_hierarchy(tmp_path: Path) -> None:
    single = _run_sink(tmp_path / "single", 1)
    done = _launch_sink(tmp_path, tmp_path / "launched")
    assert done.returncode == 0, f"launch failed:\n{done.stderr[-3000:]}"
    (launched,) = sorted((tmp_path / "launched").glob("*final*.h5"))
    _level_2_written(launched)
    _assert_same(launched, single, "a magnetized nested sink on two launched workers")
    want, got = _body_groups(single), _body_groups(launched)
    assert want.keys() == got.keys(), f"body datasets differ: {sorted(want.keys() ^ got.keys())}"
    for name, x in want.items():
        assert x.shape == got[name].shape and np.array_equal(x, got[name]), f"a magnetized nested sink on two launched workers: {name} differs"


@pytest.mark.parametrize("variant", sorted(SINK_VARIANTS))
def test_magnetized_nested_bodies_outside_the_gated_scope_are_refused_by_name_through_launch(tmp_path: Path, variant: str) -> None:
    out = tmp_path / "out"
    done = _launch_sink(tmp_path, out, variant)
    flat = " ".join((done.stdout + done.stderr).split())
    assert done.returncode != 0, f"{variant}: an ungated magnetized nested sink ran on launched workers"
    assert "immersed bodies with 3 nested refinement levels on magnetized launched workers" in flat, f"{variant}: refusal not named:\n{flat[-2000:]}"
    assert not list(out.glob("*.h5")), f"{variant}: a refused launch published a checkpoint"


@pytest.mark.parametrize("variant", [dict(solver=Solver.HLLD), dict(magnetic_resistivity=1.0e-2)], ids=["hlld", "resistive_sink"])
def test_magnetized_nested_bodies_outside_the_gated_scope_are_refused_by_name(tmp_path: Path, host_tiles, variant) -> None:
    with pytest.raises(Exception, match="immersed bodies with 3 nested refinement levels on magnetized tiles"):
        _run_sink(tmp_path / "out", 2, **variant)
    assert not list((tmp_path / "out").glob("*.h5")), "a refused run published a checkpoint"


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
