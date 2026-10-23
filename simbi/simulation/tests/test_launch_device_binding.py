# =============================================================================
# test_launch_device_binding.py
#
# the local launcher gives each gpu worker one device:
# - with no mask in the environment, worker k sees device k
# - under a parent mask (ROCR_, HIP_ or CUDA_VISIBLE_DEVICES) worker k sees its k-th entry
# - more workers than masked devices is refused
# - `simbi launch --gpu` reaches every worker as its compute mode
#
# run: pytest simbi/simulation/tests/test_launch_device_binding.py
# =============================================================================
import pytest

from simbi.simulation.launcher import worker_device_env


def test_unmasked_workers_take_one_device_each() -> None:
    envs = [worker_device_env({}, k, 3) for k in range(3)]
    assert [e["HIP_VISIBLE_DEVICES"] for e in envs] == ["0", "1", "2"]
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["0", "1", "2"]


@pytest.mark.parametrize("mask", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_a_parent_mask_is_divided_among_the_workers(mask: str) -> None:
    parent = {mask: "4, 6,7"}
    assert [worker_device_env(parent, k, 3) for k in range(3)] == [{mask: "4"}, {mask: "6"}, {mask: "7"}]


def test_the_runtime_mask_takes_precedence_over_the_language_masks() -> None:
    parent = {"ROCR_VISIBLE_DEVICES": "2,3", "HIP_VISIBLE_DEVICES": "0,1"}
    assert worker_device_env(parent, 1, 2) == {"ROCR_VISIBLE_DEVICES": "3"}


def test_more_workers_than_masked_devices_is_refused() -> None:
    with pytest.raises(ValueError, match="3 gpu workers but HIP_VISIBLE_DEVICES=0,1 makes 2 devices visible"):
        worker_device_env({"HIP_VISIBLE_DEVICES": "0,1"}, 0, 3)


def _parse(argv: list[str]):
    import argparse

    from simbi.cli.commands.launch.parser import setup_parser

    root = argparse.ArgumentParser(add_help=False)
    setup_parser(root.add_subparsers(dest="command"))
    return root.parse_known_args(["launch", *argv])[0]


def test_launch_takes_the_compute_mode_flags() -> None:
    assert _parse(["kh"]).compute_mode == "cpu"
    assert _parse(["kh", "--gpu"]).compute_mode == "gpu"
    assert _parse(["kh", "--mode", "gpu"]).compute_mode == "gpu"


def test_the_worker_role_runs_in_the_launch_mode(monkeypatch) -> None:
    from argparse import Namespace

    from simbi.cli.commands.launch import executor
    from simbi.simulation import runner

    seen = {}
    monkeypatch.setattr(executor, "require_resolvable", lambda _: None)
    monkeypatch.setattr(runner, "launch_worker", lambda problem, **kw: seen.update(kw))
    args = Namespace(max_steps=4, compute_mode="gpu")
    executor._run_worker_role(args, object(), 1, 2, [0, 1], [[8]], "r", 7, "127.0.0.1", "127.0.0.1", 64)
    assert seen["compute_mode"] == "gpu", f"the worker ran in {seen.get('compute_mode')!r}"
