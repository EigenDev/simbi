# =============================================================================
# test_config_resolution.py
#
# regression: `simbi run <name>` resolution of a config NAME to a file.
# - a name matched by exactly one discovered config resolves to that file
# - a name shared by SEVERAL configs must NOT silently run the first one the
#   directory walk visited: non-interactive resolution fails loudly with the
#   full candidate paths; a tty gets a numbered prompt (not exercised here)
# - an explicit .py path that does not exist fails at parse time
# - an unknown name suggests the closest known names before the full listing
# =============================================================================
from argparse import ArgumentTypeError
from pathlib import Path
from unittest import mock

import pytest

from simbi.cli.commands.run import parser as run_parser


def _with_configs(monkeypatch: pytest.MonkeyPatch, paths: list[Path]) -> None:
    monkeypatch.setattr(
        run_parser, "get_available_configs", lambda: [str(p) for p in paths]
    )


def test_unique_name_resolves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "examples" / "sod_test.py"
    _with_configs(monkeypatch, [cfg])
    assert run_parser._validate_config_script("sod_test") == str(cfg)
    # kebab-case variant matches the snake_case stem
    assert run_parser._validate_config_script("sod-test") == str(cfg)


def test_duplicate_names_fail_loudly_when_not_a_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a = tmp_path / "examples" / "kh.py"
    b = tmp_path / "science" / "kh.py"
    _with_configs(monkeypatch, [a, b])
    with mock.patch.object(run_parser.sys.stdin, "isatty", return_value=False):
        with pytest.raises(ArgumentTypeError) as err:
            run_parser._validate_config_script("kh")
    msg = str(err.value)
    assert "ambiguous" in msg
    assert str(a) in msg and str(b) in msg


def test_duplicate_names_prompt_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a = tmp_path / "examples" / "kh.py"
    b = tmp_path / "science" / "kh.py"
    _with_configs(monkeypatch, [a, b])
    with (
        mock.patch.object(run_parser.sys.stdin, "isatty", return_value=True),
        mock.patch.object(run_parser.sys.stderr, "isatty", return_value=True),
        mock.patch("builtins.input", return_value="2"),
    ):
        assert run_parser._validate_config_script("kh") == str(b)


def test_tty_prompt_abort_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a = tmp_path / "examples" / "kh.py"
    b = tmp_path / "science" / "kh.py"
    _with_configs(monkeypatch, [a, b])
    with (
        mock.patch.object(run_parser.sys.stdin, "isatty", return_value=True),
        mock.patch.object(run_parser.sys.stderr, "isatty", return_value=True),
        mock.patch("builtins.input", return_value="q"),
    ):
        with pytest.raises(ArgumentTypeError):
            run_parser._validate_config_script("kh")


def test_missing_explicit_path_fails_at_parse_time(tmp_path: Path) -> None:
    with pytest.raises(ArgumentTypeError) as err:
        run_parser._validate_config_script(str(tmp_path / "nope.py"))
    assert "does not exist" in str(err.value)


def test_explicit_path_that_exists_passes_through(tmp_path: Path) -> None:
    cfg = tmp_path / "my_setup.py"
    cfg.write_text("x = 1\n")
    assert run_parser._validate_config_script(str(cfg)) == str(cfg)


def test_unknown_name_suggests_closest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _with_configs(
        monkeypatch,
        [tmp_path / "kelvin_helmholtz.py", tmp_path / "sod_test.py"],
    )
    with pytest.raises(ArgumentTypeError) as err:
        run_parser._validate_config_script("kelvin-hemholtz")  # typo
    msg = str(err.value)
    assert "did you mean" in msg
    assert "kelvin-helmholtz" in msg
