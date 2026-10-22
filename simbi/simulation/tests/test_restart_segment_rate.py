# =============================================================================
# test_restart_segment_rate.py
#
# a restarted run reports the steps its own segment took: the completion line
# counts the steps from the checkpoint's iteration to the end and names the end
# iteration; the sustained rate and the per-phase profile divide by that same
# count. a run resumed at iteration 4 and capped at 7 reports 3 steps to
# iteration 7.
# =============================================================================
import re
from pathlib import Path

import pytest

from simbi.simulation import runner
from simbi_configs.examples.newtonian.sod import SodProblem


def _completion(text: str) -> tuple[str, str]:
    found = re.findall(r"complete — (\d+) steps to iteration (\d+), t = ", text)
    assert found, f"no completion line:\n{text[-2000:]}"
    return found[-1]


@pytest.mark.simulation
def test_a_restarted_segment_counts_its_own_steps(tmp_path: Path, capfd):
    first = tmp_path / "first"
    runner.run(SodProblem(resolution=256, data_directory=first), compute_mode="cpu", max_steps=4)
    steps, end = _completion(capfd.readouterr().out)
    assert (int(steps), int(end)) == (4, 4)
    final = next(first.glob("*final*.h5"))
    runner.run(
        SodProblem(resolution=256, data_directory=tmp_path / "resumed", checkpoint_file=final),
        compute_mode="cpu",
        max_steps=7,
    )
    steps, end = _completion(capfd.readouterr().out)
    assert (int(steps), int(end)) == (3, 7), "the resumed segment counted steps it did not take"
