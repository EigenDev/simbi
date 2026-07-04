# =============================================================================
# test_problem_class_discovery.py
#
# regression for the run-command loader (_discover_problem_classes): a config that
# inherits from an IMPORTED base config must be discovered as runnable. the former
# ast base-NAME scan resolved a base only when it was literally "SimbiProblem" or a
# class defined in the same file, so `class Mine(ImportedBase)` surfaced no runnable
# class ("no SimbiProblem subclasses found"). the import-based loader tests the real
# subclass relationship and returns the classes DEFINED in the script, excluding the
# imported base and any non-SimbiProblem helpers.
# =============================================================================
from pathlib import Path

from simbi.cli.commands.run.executor import _discover_problem_classes


def test_discovers_config_subclassing_an_imported_base(tmp_path: Path) -> None:
    (tmp_path / "base_cfg.py").write_text(
        "from simbi import SimbiProblem\nclass BaseCfg(SimbiProblem):\n    pass\n"
    )
    (tmp_path / "derived_cfg.py").write_text(
        "from base_cfg import BaseCfg\nclass DerivedCfg(BaseCfg):\n    pass\n"
    )
    found = _discover_problem_classes(str(tmp_path / "derived_cfg.py"))
    # the derived config is runnable; the imported base belongs to base_cfg, not
    # this script, so it is NOT surfaced.
    assert [name for name, _ in found] == ["DerivedCfg"]


def test_excludes_non_problem_helpers(tmp_path: Path) -> None:
    (tmp_path / "helpers_cfg.py").write_text(
        "from typing import NamedTuple\n"
        "from simbi import SimbiProblem\n"
        "class State(NamedTuple):\n    x: int\n"
        "class Runnable(SimbiProblem):\n    pass\n"
    )
    found = _discover_problem_classes(str(tmp_path / "helpers_cfg.py"))
    assert [name for name, _ in found] == ["Runnable"]


def test_multiple_runnable_classes_in_source_order(tmp_path: Path) -> None:
    # names deliberately reverse-alphabetical to prove the ordering is by source
    # line, not by name.
    (tmp_path / "multi_cfg.py").write_text(
        "from simbi import SimbiProblem\n"
        "class Second(SimbiProblem):\n    pass\n"
        "class First(SimbiProblem):\n    pass\n"
    )
    found = _discover_problem_classes(str(tmp_path / "multi_cfg.py"))
    assert [name for name, _ in found] == ["Second", "First"]
