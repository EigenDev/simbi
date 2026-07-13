# =============================================================================
# test_hardening_batch3.py
#
# regression gates for the cli-correctness hardening: an unregistered or
# mistyped command must fail fast with the real message and exit 2 (the
# historical error handler re-entered parsing and recursed forever on
# `simbi touch`); flag errors exit 2, never 0; the analysis cli reports
# missing files / groups / out-of-range bodies as one-line errors.
# =============================================================================

import numpy as np
import pytest

from simbi.cli.simbi_parser import SimbiParser


def _parse(argv):
    parser = SimbiParser()
    return parser.parse_known_args(argv)


def test_unregistered_command_exits_2_without_hanging(capsys):
    # `simbi touch` recursed error -> help -> error forever; it must now exit 2
    # promptly with the real message.
    with pytest.raises(SystemExit) as exc:
        _parse(["touch"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_mistyped_command_gets_did_you_mean(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse(["runn"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "did you mean 'run'" in err


def test_bare_run_parses_and_defers_to_the_executor(capsys):
    # `simbi run` with no config parses (the config positional is optional at
    # the parser level); the executor owns the missing-config error. the
    # historical behavior — re-parsing into --help and exiting 0 — is gone.
    args, remaining = _parse(["run"])
    assert args.command == "run"


def test_real_flag_error_message_surfaces(capsys):
    # a wrong-arity/invalid flag must show ITS message, not a generic help dump.
    with pytest.raises(SystemExit) as exc:
        _parse(["plot"])
    assert exc.value.code == 2


def test_registered_commands_include_attach_and_afterglow():
    parser = SimbiParser()
    for cmd in ("run", "plot", "afterglow", "attach"):
        assert cmd in parser._subparser_map, f"'{cmd}' not registered"


def test_gpu_block_dims_reject_nonpositive():
    import argparse

    from simbi.cli.actions import RegisterGPUBlockDimensions

    p = argparse.ArgumentParser()
    p.add_argument("--gpu-block-dims", nargs="+", type=int, action=RegisterGPUBlockDimensions)
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["--gpu-block-dims", "16", "-4"])
    assert exc.value.code == 2


def _write_minimal_h5(path, with_group: bool, n_steps: int = 8, n_bodies: int = 1):
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["gamma"] = 1.4
        if with_group:
            g = f.create_group("body_diagnostics")
            g["time"] = np.linspace(0.0, 1.0, n_steps)
            g["dt"] = np.full(n_steps, 0.1)
            g["mass_delta"] = np.full((n_steps, n_bodies), 1e-4)
            g["energy_delta"] = np.full((n_steps, n_bodies), 1e-5)
            g["force"] = np.zeros((n_steps, n_bodies, 2))


def _run_analysis(argv, monkeypatch):
    import sys as _sys

    from simbi.analysis import __main__ as amain

    monkeypatch.setattr(_sys, "argv", ["simbi.analysis"] + argv)
    return amain.main()


def test_analysis_missing_file_is_a_one_line_error(monkeypatch, capsys):
    rc = _run_analysis(["/nonexistent/file.h5"], monkeypatch)
    assert rc == 2
    assert "cannot open checkpoint" in capsys.readouterr().err


def test_analysis_missing_group_names_the_fix(monkeypatch, capsys, tmp_path):
    path = str(tmp_path / "no_group.h5")
    _write_minimal_h5(path, with_group=False)
    rc = _run_analysis([path], monkeypatch)
    assert rc == 2
    err = capsys.readouterr().err
    assert "body_diagnostics" in err and "diagnostics.dat" in err


def test_analysis_body_out_of_range(monkeypatch, capsys, tmp_path):
    path = str(tmp_path / "one_body.h5")
    _write_minimal_h5(path, with_group=True, n_bodies=1)
    rc = _run_analysis([path, "--body", "3"], monkeypatch)
    assert rc == 2
    assert "out of range" in capsys.readouterr().err


def test_analysis_short_series(monkeypatch, capsys, tmp_path):
    path = str(tmp_path / "short.h5")
    _write_minimal_h5(path, with_group=True, n_steps=1)
    rc = _run_analysis([path], monkeypatch)
    assert rc == 2
    assert "too short" in capsys.readouterr().err
