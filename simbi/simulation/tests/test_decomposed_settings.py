# =============================================================================
# test_decomposed_settings.py
#
# the settings a decomposed hydro run must honor or refuse. a viscous run on
# two devices equals the viscous single-grid run and differs from the inviscid
# one; an excised relativistic run on two devices equals the excised
# single-grid run bit for bit and differs from the unexcised one; the synge perfect gas
# on two devices equals the synge single-grid run and differs from the ideal
# gas; the balanced reconstruction on several devices is refused by name. every arm runs as a
# fresh process through the public command, the decomposed ones as host tiles
# on a cpu build.
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
KH = "simbi_configs/examples/newtonian/kh.py"
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


def test_the_balanced_reconstruction_on_several_devices_is_refused(tmp_path: Path) -> None:
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    cmd = _command(KH, tmp_path / "wb", 2, ["--resolution", "32,32", "--end-time", "0.001", "--wb-reconstruction"])
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    assert done.returncode != 0
    assert "wb_reconstruction with gpus > 1" in done.stderr.replace("\n", " "), done.stderr[-1500:]
    assert not list((tmp_path / "wb").glob("*.h5")), "output was written before the refusal"


def _patch_on_cut_script(path: Path, x_lo: float) -> str:
    """a refined blast whose fine patch starts at `x_lo`; the two-device cut of the symmetric
    domain lies at x = 0"""
    path.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.newtonian.refined_blast import RefinedBlast\n\n\n"
        "class PatchAgainstTheCut(RefinedBlast):\n"
        "    refinement_regions: Annotated[\n"
        "        list[list[float]],\n"
        f"        ProblemParam([[{x_lo!r}, 0.25, -0.25, 0.25]], description='the fine patch'),\n"
        "    ]\n"
    )
    return str(path)


def test_a_patch_face_on_the_device_cut_is_refused_by_name(tmp_path: Path) -> None:
    """the fine faces on a patch face lying on a cut and the coarse cell their reflux corrects
    would live on different devices; the run is refused with the remedy, and the same patch
    starting two root cells past the cut (the prolongation stencil's reach inside the root
    halo) runs."""
    from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    flags = ["--resolution", "32,32,1", "--end-time", "1e-6"]
    on_cut = _patch_on_cut_script(tmp_path / "on_cut.py", 0.0)
    done = subprocess.run(_command(on_cut, tmp_path / "refused", 2, flags), capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode != 0, f"the run succeeded:\n{done.stdout[-1500:]}"
    flat = " ".join(done.stderr.split())
    assert "lies on a decomposition cut" in flat, done.stderr[-1500:]
    assert not list((tmp_path / "refused").glob("*.h5")), "output was written before the refusal"
    (x_min, x_max), _ = RefinedBlast(resolution=(32, 32, 1)).bounds[:2]
    dx = (x_max - x_min) / 32
    # one root cell past the cut: the fine ghosts' prolongation stencil reaches beyond the root
    # halo across the cut, refused with the clearance to move the face by
    one_cell_past = _patch_on_cut_script(tmp_path / "one_past.py", dx)
    done = subprocess.run(_command(one_cell_past, tmp_path / "too_close", 2, flags), capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode != 0, f"the run succeeded:\n{done.stdout[-1500:]}"
    flat = " ".join(done.stderr.split())
    assert "prolongation stencil reaches" in flat and "move the patch face" in flat, done.stderr[-1500:]
    assert not list((tmp_path / "too_close").glob("*.h5")), "output was written before the refusal"
    two_cells_past = _patch_on_cut_script(tmp_path / "past_cut.py", 2.0 * dx)
    _run(two_cells_past, tmp_path / "admitted", 2, flags)


@pytest.mark.parametrize(
    ("script", "flags", "reason"),
    [
        ("simbi_configs/examples/ibm/magnetic_slip_sink.py", ["--resolution", "16,16,16", "--end-time", "1e-6"],
         "a magnetic slip body with several tiles awaits its tiled slip step"),
    ],
)
def test_a_setting_the_tiles_do_not_carry_is_refused_by_name(tmp_path: Path, script: str, flags: list[str], reason: str) -> None:
    """the tiled drivers step no magnetic slip body: a run asking for one on several devices is
    refused by name and publishes nothing, where it ran without it before."""
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run(_command(script, tmp_path / "out", 2, flags), capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode != 0, f"the run succeeded:\n{done.stdout[-1500:]}"
    assert reason in " ".join(done.stderr.split()), done.stderr[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a refused run published a checkpoint"
