# =============================================================================
# test_bonded_restart.py
#
# a bonded fragment assembly resumes the bonds and contacts it has lived
# through. the roche passage pile runs twenty steps uninterrupted and ten plus
# ten from its checkpoint, on one device and on two host tiles; the resumed run
# matches the uninterrupted one bit for bit in every grid dataset, every body
# dataset and the recorded fragment state (bond rest lengths, tangential slip,
# intact flags, contact slip). one arm breaks its bonds before the checkpoint
# (bond strength 1e-9), the other carries intact bonds with accumulated slip.
# the per-step body history restarts empty by design and is not compared.
#
# run: pytest -m simulation simbi/simulation/tests/test_bonded_restart.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.simulation

STEPS = 20
SPLIT = 10
MODULE = "simbi_configs.examples.ibm.roche_passage"
ARMS = {"broken": ", bond_strength=1e-9", "intact": ""}


def _run(out: Path, gpus: int, fields: str, steps: int, checkpoint: Path | None = None) -> Path:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {MODULE} import RochePassage\n"
        f"p = RochePassage(resolution=(64, 64), gpus={gpus}, data_directory=sys.argv[1]{fields}{resume})\n"
        "p.checkpoint_interval = 1e9\n"
        f"runner.run(p, compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    assert done.returncode == 0, f"roche passage on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _datasets(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and not name.startswith("body_diagnostics") and "/mesh/" not in f"/{name}/":
                out[name] = obj[()]
        f.visititems(visit)
    return out


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
@pytest.mark.parametrize("arm", list(ARMS))
def test_a_resumed_assembly_continues_its_bonds_and_contacts(tmp_path: Path, arm: str, gpus: int) -> None:
    fields = ARMS[arm]
    want = _run(tmp_path / "full", gpus, fields, STEPS)
    first = _run(tmp_path / "first", gpus, fields, SPLIT)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
        fragments = f["bodies/fragments"]
        intact = fragments["bond_intact"][()]
        slip = fragments["bond_slip"][()]
        if arm == "broken":
            assert (intact == 0).any(), "no bond had broken at the checkpoint; the broken arm is vacuous"
        else:
            assert np.abs(slip[intact == 1]).max() > 0.0, "no intact bond carried slip at the checkpoint; the arm is vacuous"
    # max_steps caps the absolute root iteration on every driver; the checkpoint sits at or
    # below SPLIT < STEPS, so the resumed run steps from written_at up to STEPS
    assert written_at < STEPS, f"the checkpoint at iteration {written_at} leaves no step to resume"
    resumed = _run(tmp_path / "resumed", gpus, fields, STEPS, checkpoint=first)
    xs, ys = _datasets(resumed), _datasets(want)
    assert any(name.startswith("bodies/fragments/") for name in ys), "the reference records no fragment state"
    assert xs.keys() == ys.keys(), f"datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{arm} on {gpus} device(s): {name}")
