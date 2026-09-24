# =============================================================================
# test_decomposed_settings.py
#
# the settings a decomposed hydro run must honor or refuse. a viscous run on
# two devices equals the viscous single-grid run and differs from the inviscid
# one; an excised relativistic run on two devices equals the excised
# single-grid run at the decomposition's roundoff bound; the balanced
# reconstruction on several devices is refused by name. every arm runs as a
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

pytestmark = pytest.mark.simulation

VISCOUS = "simbi_configs/examples/newtonian/viscous_shear.py"
BONDI = "simbi_configs/examples/grhd/gr_bondi_cartesian.py"
KH = "simbi_configs/examples/newtonian/kh.py"


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


def test_a_viscous_run_on_two_devices_is_viscous(tmp_path: Path) -> None:
    base = ["--resolution", "32,32,1", "--end-time", "0.05"]
    viscous_1, steps_1 = _density(_run(VISCOUS, tmp_path / "v1", 1, base + ["--nu", "0.05"]))
    inviscid_1, _ = _density(_run(VISCOUS, tmp_path / "i1", 1, base + ["--nu", "0.0"]))
    assert np.abs(viscous_1 - inviscid_1).max() > 0.0, "viscosity does not act on this setup; the gate is vacuous"
    viscous_2, steps_2 = _density(_run(VISCOUS, tmp_path / "v2", 2, base + ["--nu", "0.05"]))
    assert steps_2 == steps_1, "the viscous timestep limit reaches the decomposed run"
    np.testing.assert_array_equal(viscous_2, viscous_1)


def test_an_excised_run_on_two_devices_is_excised(tmp_path: Path) -> None:
    base = ["--resolution", "24,24,24", "--end-time", "0.6", "--domain-radius", "0.12", "--excision-radius", "1.6"]
    one, steps_1 = _density(_run(BONDI, tmp_path / "e1", 1, base))
    two, steps_2 = _density(_run(BONDI, tmp_path / "e2", 2, base))
    assert steps_1 == steps_2
    # each tile evaluates the metric from its own origin, so the decomposed relativistic march
    # follows the single grid at roundoff: the bound the decomposed restart gates hold
    scale = max(np.abs(one).max(), np.abs(two).max())
    worst = np.abs(one - two).max()
    assert worst <= 1e-12 * scale, f"excised densities differ by {worst:.3e} against scale {scale:.3e}"


def test_the_balanced_reconstruction_on_several_devices_is_refused(tmp_path: Path) -> None:
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    cmd = _command(KH, tmp_path / "wb", 2, ["--resolution", "32,32", "--end-time", "0.001", "--wb-reconstruction"])
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    assert done.returncode != 0
    assert "wb_reconstruction with gpus > 1" in done.stderr.replace("\n", " "), done.stderr[-1500:]
    assert not list((tmp_path / "wb").glob("*.h5")), "output was written before the refusal"


def test_a_refined_run_with_a_cut_periodic_axis_on_several_devices_is_refused(tmp_path: Path) -> None:
    """the refined multi-device path exchanges level halos under an open topology, so a periodic
    axis cut into tiles would leave the domain seam unfilled; the run is refused by name."""
    from simbi_configs.examples.newtonian.refined_kh import RefinedKelvinHelmholtz

    script = str(Path(RefinedKelvinHelmholtz.__module__.replace(".", "/") + ".py"))
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run(_command(script, tmp_path / "out", 2, ["--resolution", "32,32", "--end-time", "1e-6"]), capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode != 0, f"the run succeeded:\n{done.stdout[-1500:]}"
    flat = " ".join(done.stderr.split())
    assert "cut across gpus > 1 with mesh refinement is outside the refined multi-GPU path" in flat, done.stderr[-1500:]



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
    two_cells_past = _patch_on_cut_script(tmp_path / "past_cut.py", 2.0 * (x_max - x_min) / 32)
    _run(two_cells_past, tmp_path / "admitted", 2, flags)
