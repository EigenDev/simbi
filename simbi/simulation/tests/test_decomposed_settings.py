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
# cell off the cut, a magnetized face and an isothermal face on the cut are
# refused by name. every arm runs as a fresh process through the public
# command, the decomposed ones as host tiles on a cpu build.
# =============================================================================
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.checkpoint_compare import assert_same_checkpoint

pytestmark = pytest.mark.simulation

VISCOUS = "simbi_configs/examples/newtonian/viscous_shear.py"
BONDI = "simbi_configs/examples/grhd/gr_bondi_cartesian.py"
SYNGE = "simbi_configs/examples/srhd/synge_blast.py"


def _command(script: str, out: Path, ngpus: int, flags: list[str]) -> list[str]:
    return [sys.executable, "-m", "simbi.cli", "run", script, "--mode", "cpu", "--ngpus", str(ngpus),
            "--checkpoint-interval", "1e9", "--data-directory", str(out), *flags]


def _run(script: str, out: Path, ngpus: int, flags: list[str]) -> Path:
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run(_command(script, out, ngpus, flags), capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{script} on {ngpus} devices failed:\n{done.stderr[-3000:]}"
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


def test_a_viscous_run_on_two_devices_is_viscous(tmp_path: Path) -> None:
    base = ["--resolution", "32,32,1", "--end-time", "0.05"]
    one = _run(VISCOUS, tmp_path / "v1", 1, base + ["--nu", "0.05"])
    viscous_1, _ = _density(one)
    inviscid_1, _ = _density(_run(VISCOUS, tmp_path / "i1", 1, base + ["--nu", "0.0"]))
    assert np.abs(viscous_1 - inviscid_1).max() > 0.0, "viscosity does not act on this setup; the gate is vacuous"
    assert_same_checkpoint(_run(VISCOUS, tmp_path / "v2", 2, base + ["--nu", "0.05"]), one, "a viscous run on two devices")


def test_an_excised_run_on_two_devices_is_excised(tmp_path: Path) -> None:
    base = ["--resolution", "24,24,24", "--end-time", "0.6", "--domain-radius", "0.12"]
    excised = _run(BONDI, tmp_path / "e1", 1, base + ["--excision-radius", "1.6"])
    one, _ = _conserved(excised)
    bare, _ = _conserved(_run(BONDI, tmp_path / "n1", 1, base + ["--excision-radius", "0.0"]))
    assert not np.array_equal(one["den"], bare["den"]), "excision does not act on this setup; the gate is vacuous"
    # every tile evaluates the metric at whole-grid cell centers (the global origin plus its index
    # offset), so the excised relativistic march is the single grid's, bit for bit
    assert_same_checkpoint(_run(BONDI, tmp_path / "e2", 2, base + ["--excision-radius", "1.6"]), excised, "an excised run on two devices")


def test_the_synge_gas_on_two_devices_is_the_synge_gas(tmp_path: Path) -> None:
    base = ["--resolution", "32,32", "--end-time", "0.02"]
    one = _run(SYNGE, tmp_path / "s1", 1, base)
    synge_1, _ = _density(one)
    ideal_1, _ = _density(_run(SYNGE, tmp_path / "i1", 1, base + ["--eos", "ideal", "--adiabatic-index", "1.4"]))
    assert np.abs(synge_1 - ideal_1).max() > 0.0, "the closure does not act on this setup; the gate is vacuous"
    assert_same_checkpoint(_run(SYNGE, tmp_path / "s2", 2, base), one, "the synge gas on two devices")


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
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run(_command(script, out, 2, flags), capture_output=True, text=True, timeout=900, env=env)
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
        + f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, script, str(out)], capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    assert done.returncode == 0, f"{script} on {gpus} device(s) to step {steps} failed:\n{done.stderr[-3000:]}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final


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
    assert_same_checkpoint(_capped(on_cut, tmp_path / "two", 2, 6, flags), one, "a patch face on the cut", expect=["level_1"])
    half = _capped(on_cut, tmp_path / "half", 2, 3, flags)
    resumed = _capped(on_cut, tmp_path / "resumed", 2, 6, flags, checkpoint=half)
    assert_same_checkpoint(resumed, one, "a patch face on the cut, resumed", expect=["level_1"])


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


def test_a_patch_face_on_the_device_cut_prolonging_further_inputs_is_refused_by_name(tmp_path: Path) -> None:
    """the gathered parent stencil carries the gas primitives and the dye: a magnetized level
    (cell and face field) and an isothermal closure (the sound speed) prolong further parent data
    past the root halo, and each is refused by name before any output"""
    flags = ["--resolution", "32,32,1", "--end-time", "1e-6"]
    magnetized = _patch_on_cut_script(tmp_path / "mhd.py", 0.0, "newtonian.refined_mhd_blast:RefinedMhdBlast")
    flat = _refused(magnetized, tmp_path / "mhd", flags)
    assert "lies on a decomposition cut" in flat and "beyond the root halo" in flat, flat[-1500:]
    isothermal = _patch_on_cut_script(tmp_path / "iso.py", 0.5, "isothermal.refined_isothermal_blob:RefinedIsothermalBlob", (0.25, 0.75), 0.75)
    flat = _refused(isothermal, tmp_path / "iso", ["--resolution", "32,32", "--end-time", "1e-6"])
    assert "patch face on a decomposition cut" in flat and "the isothermal closure prolongs" in flat, flat[-1500:]
