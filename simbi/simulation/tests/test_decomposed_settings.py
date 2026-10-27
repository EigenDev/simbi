# =============================================================================
# test_decomposed_settings.py
#
# the settings a decomposed hydro run must honor or refuse. a viscous run on
# two devices equals the viscous single-grid run and differs from the inviscid
# one; an excised relativistic run on two devices equals the excised
# single-grid run bit for bit and differs from the unexcised one; the synge perfect gas
# on two devices equals the synge single-grid run and differs from the ideal
# gas. a refined patch face on the device cut under the production halo
# equals the single grid bit for bit, uninterrupted and resumed; a face one
# cell off the cut, and a magnetized and a relativistic face on
# the cut are refused by name, as are a magnetized level-2 face on the cut of
# level 1 and one under a level 1 spanning a periodic axis. every arm runs as a
# fresh process on the session's backend (parity_harness), the decomposed ones
# as host tiles; host tiles alone involve no launch, so every gate runs under
# the local placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_decomposed_settings.py
#      (SIMBI_PARITY_BACKEND selects the session, parity_harness.py)
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

VISCOUS = "simbi_configs/examples/newtonian/viscous_shear.py"
BONDI = "simbi_configs/examples/grhd/gr_bondi_cartesian.py"
SYNGE = "simbi_configs/examples/srhd/synge_blast.py"


def _command(script: str, out: Path, ngpus: int, flags: list[str]) -> list[str]:
    """the interpreter arguments of the public run command on the session's backend"""
    return ["-m", "simbi.cli", "run", script, "--mode", ph.BACKEND, "--ngpus", str(ngpus),
            "--checkpoint-interval", "1e9", "--data-directory", str(out), *flags]


def _run(script: str, out: Path, ngpus: int, flags: list[str]) -> Path:
    ph.python(_command(script, out, ngpus, flags), f"{Path(script).stem} on {ngpus} device(s)", devices=ngpus)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _density(path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(path, "r") as f:
        level = f["level_0"]
        ng = int(level["mesh"].attrs["halo_width"])
        data = level["conserved/den"][()]
        return data[(slice(ng, -ng),) * data.ndim], int(level.attrs["iteration"])


def _conserved(path: Path) -> tuple[dict[str, np.ndarray], int]:
    with h5py.File(path, "r") as f:
        level = f["level_0"]
        return {name: data[()] for name, data in level["conserved"].items()}, int(level.attrs["iteration"])


@ph.local_only
def test_a_viscous_run_on_two_devices_is_viscous(tmp_path: Path) -> None:
    base = ["--resolution", "32,32,1", "--end-time", "0.05"]
    one = _run(VISCOUS, tmp_path / "v1", 1, base + ["--nu", "0.05"])
    viscous_1, _ = _density(one)
    inviscid_1, _ = _density(_run(VISCOUS, tmp_path / "i1", 1, base + ["--nu", "0.0"]))
    assert np.abs(viscous_1 - inviscid_1).max() > 0.0, "viscosity does not act on this setup; the gate is vacuous"
    ph.same_checkpoint(_run(VISCOUS, tmp_path / "v2", 2, base + ["--nu", "0.05"]), one, "a viscous run on two devices")


@ph.local_only
def test_an_excised_run_on_two_devices_is_excised(tmp_path: Path) -> None:
    base = ["--resolution", "24,24,24", "--end-time", "0.6", "--domain-radius", "0.12"]
    excised = _run(BONDI, tmp_path / "e1", 1, base + ["--excision-radius", "1.6"])
    one, _ = _conserved(excised)
    bare, _ = _conserved(_run(BONDI, tmp_path / "n1", 1, base + ["--excision-radius", "0.0"]))
    assert not np.array_equal(one["den"], bare["den"]), "excision does not act on this setup; the gate is vacuous"
    # every tile evaluates the metric at whole-grid cell centers (the global origin plus its index
    # offset), so the excised relativistic march is the single grid's, bit for bit
    ph.same_checkpoint(_run(BONDI, tmp_path / "e2", 2, base + ["--excision-radius", "1.6"]), excised, "an excised run on two devices")


@ph.local_only
def test_the_synge_gas_on_two_devices_is_the_synge_gas(tmp_path: Path) -> None:
    base = ["--resolution", "32,32", "--end-time", "0.02"]
    one = _run(SYNGE, tmp_path / "s1", 1, base)
    synge_1, _ = _density(one)
    ideal_1, _ = _density(_run(SYNGE, tmp_path / "i1", 1, base + ["--eos", "ideal", "--adiabatic-index", "1.4"]))
    assert np.abs(synge_1 - ideal_1).max() > 0.0, "the closure does not act on this setup; the gate is vacuous"
    ph.same_checkpoint(_run(SYNGE, tmp_path / "s2", 2, base), one, "the synge gas on two devices")


def _patch_on_cut_script(path: Path, x_lo: float, base: str = "newtonian.refined_blast:RefinedBlast", y: tuple[float, float] = (-0.25, 0.25),
                         x_hi: float = 0.25) -> str:
    """the refined problem `base` (module:class under simbi_configs.examples) whose fine patch
    starts at `x_lo`; the two-device cut of the symmetric blast domain lies at x = 0"""
    module, name = base.split(":")
    path.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        f"from simbi_configs.examples.{module} import {name}\n\n\n"
        f"class PatchAgainstTheCut({name}):\n"
        "    refinement_regions: Annotated[\n"
        "        list[list[float]],\n"
        f"        ProblemParam([[{x_lo!r}, {x_hi!r}, {y[0]!r}, {y[1]!r}]], description='the fine patch'),\n"
        "    ]\n"
    )
    return str(path)


def _refused(script: str, out: Path, flags: list[str]) -> str:
    done = ph.python(_command(script, out, 2, flags), f"{Path(script).stem} on two devices, expecting a refusal", devices=2, check=False)
    assert done.returncode != 0, f"the run succeeded:\n{done.stdout[-1500:]}"
    assert not list(out.glob("*.h5")), "output was written before the refusal"
    return " ".join(done.stderr.split())


# the coarse-fine prolongation's reach in root cells past a patch face: half the fine halo (one
# cell deeper than the root's) rounded up, plus the stencil half-width of 2
PROLONG_REACH = 4


def _capped(script: str, out: Path, gpus: int, steps: int, flags: list[str], checkpoint: Path | None = None) -> Path:
    """the problem class of `script` from the command-line `flags` on `gpus` devices, capped at the
    absolute root iteration `steps` and resumed from `checkpoint` when given; its final checkpoint"""
    resume = f"p.checkpoint_file = {str(checkpoint)!r}\n" if checkpoint is not None else ""
    snippet = (
        "import importlib, pathlib, sys\n"
        "from simbi.simulation import runner\n"
        "script = pathlib.Path(sys.argv[1])\n"
        "sys.path.insert(0, str(script.parent))\n"
        "module = importlib.import_module(script.stem)\n"
        f"p = module.PatchAgainstTheCut.from_cli({flags!r})\n"
        f"p.gpus = {gpus}\n"
        "p.checkpoint_interval = 1e9\n"
        "p.data_directory = sys.argv[2]\n"
        + resume
        + f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, script, str(out)), f"{Path(script).stem} on {gpus} device(s) to step {steps}", devices=gpus)
    (final,) = sorted(out.glob("*final*.h5"))
    return final


@ph.local_only
@pytest.mark.parametrize("order", ["2", "3"], ids=["plm", "ppm"])
def test_a_patch_face_on_the_device_cut_matches_the_single_grid(tmp_path: Path, order: str) -> None:
    """a patch face lying on the device cut prolongs its fine ghosts from parent cells past the
    root halo, gathered from the tile across the cut each root step: two devices equal one bit
    for bit over every level, uninterrupted and resumed from a checkpoint the two-device run
    wrote halfway"""
    flags = ["--resolution", "32,32,1", "--order", order]
    on_cut = _patch_on_cut_script(tmp_path / "on_cut.py", 0.0)
    one = _capped(on_cut, tmp_path / "one", 1, 6, flags)
    with h5py.File(one, "r") as f:
        halo = int(f["level_0/mesh"].attrs["halo_width"])
        assert "level_1" in f, "the single grid wrote no refined level; the gate is vacuous"
        assert int(f["level_0"].attrs["iteration"]) == 6
    assert halo < PROLONG_REACH, f"the root halo ({halo}) holds the prolongation reach; the cut face reads no gathered cell"
    ph.same_checkpoint(_capped(on_cut, tmp_path / "two", 2, 6, flags), one, "a patch face on the cut", expect=["level_1"])
    half = _capped(on_cut, tmp_path / "half", 2, 3, flags)
    resumed = _capped(on_cut, tmp_path / "resumed", 2, 6, flags, checkpoint=half)
    ph.same_checkpoint(resumed, one, "a patch face on the cut, resumed", expect=["level_1"])


@ph.local_only
def test_a_patch_face_near_the_device_cut_is_refused_by_name(tmp_path: Path) -> None:
    """a patch face one root cell past the cut prolongs its fine ghosts from the root halo, which
    the plm stencil reaches past: the run is refused with the clearance to move the face by, and
    the same patch starting two root cells past the cut (the reach inside the root halo) runs."""
    from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

    flags = ["--resolution", "32,32,1", "--end-time", "1e-6"]
    (x_min, x_max), _ = RefinedBlast(resolution=(32, 32, 1)).bounds[:2]
    dx = (x_max - x_min) / 32
    one_cell_past = _patch_on_cut_script(tmp_path / "one_past.py", dx)
    flat = _refused(one_cell_past, tmp_path / "too_close", flags)
    assert "prolongation stencil reaches" in flat and "move the patch face" in flat, flat[-1500:]
    two_cells_past = _patch_on_cut_script(tmp_path / "past_cut.py", 2.0 * dx)
    _run(two_cells_past, tmp_path / "admitted", 2, flags)


@ph.local_only
def test_a_patch_face_on_the_device_cut_prolonging_further_inputs_is_refused_by_name(tmp_path: Path) -> None:
    """the gathered parent stencil carries the newtonian gas primitives and the dye: a magnetized
    level (cell and face field) and a relativistic level keep the root halo requirement, and each
    is refused by name before any output. an isothermal level runs on a cut face
    (test_isothermal_face_on_cut)"""
    flags = ["--resolution", "32,32,1", "--end-time", "1e-6"]
    magnetized = _patch_on_cut_script(tmp_path / "mhd.py", 0.0, "newtonian.refined_mhd_blast:RefinedMhdBlast")
    flat = _refused(magnetized, tmp_path / "mhd", flags)
    assert "lies on a decomposition cut" in flat and "beyond the root halo" in flat, flat[-1500:]
    relativistic = _patch_on_cut_script(tmp_path / "rhd.py", 0.5, "srhd.refined_synge_blast:RefinedSyngeBlast", (0.25, 0.75), 0.75)
    flat = _refused(relativistic, tmp_path / "rhd", ["--resolution", "32,32", "--end-time", "1e-6"])
    assert "lies on a decomposition cut" in flat and "this level is relativistic" in flat, flat[-1500:]


def _nested_script(path: Path, level_1: list[float], level_2: list[float], base: str = "newtonian.refined_blast:RefinedBlast",
                   extra: str = "") -> str:
    """the refined problem `base` (module:class under simbi_configs.examples) with level 1 over
    `level_1` and level 2 over `level_2`, each [x_lo, x_hi, y_lo, y_hi]; the two-device cut lies at
    x = 0"""
    module, name = base.split(":")
    path.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi.types import BoundaryCondition\n"
        f"from simbi_configs.examples.{module} import {name}\n\n\n"
        f"class NestedAgainstTheCut({name}):\n"
        "    refinement_max_levels: Annotated[int, ProblemParam(3, description='coarse + 2 fine')]\n"
        "    refinement_regions: Annotated[\n"
        "        list[list[float]],\n"
        f"        ProblemParam([{level_1!r}, {level_2!r}], description='nested patches'),\n"
        "    ]\n"
        "    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description='ratio per jump')]\n"
        + extra
    )
    return str(path)


@ph.local_only
def test_a_level_2_face_on_the_cut_of_level_1_outside_the_gated_scope_is_refused_by_name(tmp_path: Path) -> None:
    """a level-2 face on the cut of level 1 runs on host tiles for the newtonian gas inside the
    domain; a magnetized one carries no edge-field register across the cut, and one under a level 1
    spanning the periodic x axis would gather its stencil across the seam: each is refused by name
    before any output"""
    flags = ["--resolution", "32,32,1", "--end-time", "1e-6"]
    level_2 = [0.0, 0.125, -0.125, 0.125]
    magnetized = _nested_script(tmp_path / "mhd.py", [-0.25, 0.25, -0.25, 0.25], level_2, "newtonian.refined_mhd_blast:RefinedMhdBlast")
    flat = _refused(magnetized, tmp_path / "mhd", flags)
    assert "a level-2 refinement patch face on axis 0 lies on a decomposition cut" in flat and "magnetized" in flat, flat[-1500:]
    periodic = _nested_script(
        tmp_path / "periodic.py",
        [-0.5, 0.5, -0.25, 0.25],
        level_2,
        extra=(
            "    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.PERIODIC, "
            "BoundaryCondition.PERIODIC, BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW], description='walls')]\n"
        ),
    )
    flat = _refused(periodic, tmp_path / "periodic", flags)
    assert "lies on a cut of level 1" in flat and "spans a wrapping axis" in flat, flat[-1500:]
