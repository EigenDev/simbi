# =============================================================================
# test_config_discovery.py
#
# regression: config discovery must NOT descend into virtualenvs / caches when a
# config tree is a symlink to a repo that carries its own .venv. the original
# `rglob("*.py")` followed `simbi_configs/science -> sibling_repo` straight into
# `.venv/lib/.../site-packages`, listing ~2200 scipy/matplotlib modules as
# "configs". the pruned walk + SimbiProblem content filter keep the listing to
# actual config files.
# =============================================================================
from pathlib import Path

from simbi.cli.actions import _find_configs, _is_config_file

_CONFIG_SRC = "from simbi import SimbiProblem\nclass Foo(SimbiProblem):\n    pass\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_find_configs_prunes_venv_and_caches(tmp_path: Path) -> None:
    root = tmp_path / "simbi_configs"
    # a real config, and a symlinked tree that carries a venv + non-config script.
    _write(root / "examples" / "my_setup.py", _CONFIG_SRC)

    sibling = tmp_path / "science"
    _write(sibling / "real_config.py", _CONFIG_SRC)
    _write(sibling / "analysis.py", "import numpy as np\nprint('not a config')\n")
    _write(
        sibling / ".venv" / "lib" / "site-packages" / "scipy.py",
        "class SimbiProblem: pass\n",  # marker present but lives in a venv
    )
    _write(sibling / "__pycache__" / "real_config.cpython-313.pyc.py", _CONFIG_SRC)
    (root / "science").symlink_to(sibling)

    found = {p.name for p in _find_configs(root)}

    assert "my_setup.py" in found  # bundled config
    assert "real_config.py" in found  # config inside the symlinked tree
    assert "scipy.py" not in found  # pruned: under .venv
    assert "analysis.py" not in found  # filtered: no SimbiProblem marker
    # nothing from a cache dir.
    assert not any("pycache" in str(p).lower() for p in _find_configs(root))


def test_is_config_file_requires_marker_and_public_name(tmp_path: Path) -> None:
    cfg = tmp_path / "good.py"
    cfg.write_text(_CONFIG_SRC)
    assert _is_config_file(cfg)

    # a private / dunder module is never a config, even with the marker.
    priv = tmp_path / "_helper.py"
    priv.write_text(_CONFIG_SRC)
    assert not _is_config_file(priv)

    # a plain script without the marker is excluded.
    plain = tmp_path / "script.py"
    plain.write_text("print('hello')\n")
    assert not _is_config_file(plain)

    # setup.py / conftest.py are tooling.
    for name in ("setup.py", "conftest.py"):
        f = tmp_path / name
        f.write_text(_CONFIG_SRC)
        assert not _is_config_file(f)


def test_is_config_file_recognizes_a_subclass_of_an_imported_base(tmp_path: Path) -> None:
    # a config that extends a base config it imports from simbi_configs carries no
    # literal "SimbiProblem" text, yet is a runnable config. a raw-marker scan would
    # drop it from the registry (so `simbi run <name>` fails); the ast marker keeps it.
    derived = tmp_path / "derived.py"
    derived.write_text(
        "from simbi_configs.examples.grmhd.base import BaseCfg\n"
        "class Derived(BaseCfg):\n    pass\n"
    )
    assert _is_config_file(derived)

    # importing a config without subclassing it (an analysis / plot helper) is NOT a
    # config — the base name must appear as a class base; a bare import or reference does not qualify.
    helper = tmp_path / "analysis.py"
    helper.write_text(
        "from simbi_configs.examples.grmhd.base import BaseCfg\n"
        "print(BaseCfg.__name__)\n"
    )
    assert not _is_config_file(helper)


# =============================================================================
# which ROOTS get searched (the above covers what is a config once a root is walked)
#
# discovery used to locate the checkout through a `gitrepo_home.txt` marker written
# beside the package -- except nothing ever wrote it and it was gitignored, so a fresh
# install fell back to a cwd-local `simbi_configs/` and just reported fewer problems.
# run from anywhere but the checkout root and the shipped library was invisible, with no
# error to explain it. the roots are derived now rather than recorded.
# =============================================================================
import os

from simbi.cli.actions import config_roots, get_available_configs

_CHECKOUT_CONFIGS = Path(__file__).resolve().parents[3] / "simbi_configs"


def test_bundled_configs_are_found_from_an_unrelated_directory(tmp_path, monkeypatch):
    # the regression, exactly: an empty cwd carrying no simbi_configs/ of its own.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIMBI_CONFIG_PATH", raising=False)
    assert not (tmp_path / "simbi_configs").exists()
    assert get_available_configs(), (
        "no configs discovered from a directory with no local simbi_configs/ -- the "
        "shipped library is invisible to `simbi run` outside the checkout"
    )


def test_a_cwd_local_directory_is_also_searched(tmp_path, monkeypatch):
    # the other half: drop a simbi_configs/ beside you and your own problems become
    # name-addressable without passing a path.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIMBI_CONFIG_PATH", raising=False)
    local = tmp_path / "simbi_configs"
    local.mkdir()
    assert local.resolve() in {r.resolve() for r in config_roots()}


def test_the_env_var_takes_priority(tmp_path, monkeypatch):
    # SIMBI_CONFIG_PATH is the explicit override, so it comes first: a user pointing at
    # their own library must not have the shipped one shadow it.
    monkeypatch.chdir(tmp_path)
    mine = tmp_path / "mine"
    mine.mkdir()
    monkeypatch.setenv("SIMBI_CONFIG_PATH", str(mine))
    roots = config_roots()
    assert roots and roots[0].resolve() == mine.resolve(), f"roots were {roots}"


def test_missing_roots_are_skipped_rather_than_raising(tmp_path, monkeypatch):
    # discovery runs on every `simbi run`, so an absent root must never be what kills the
    # command -- which is also why the old marker failed silently instead of loudly.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "SIMBI_CONFIG_PATH",
        os.pathsep.join([str(tmp_path / "nope"), str(tmp_path / "also-nope")]),
    )
    for root in config_roots():
        assert root.is_dir(), f"{root} does not exist but was returned as a root"
    get_available_configs()  # must not raise


def test_the_checkout_sibling_is_among_the_roots(tmp_path, monkeypatch):
    # what replaced the marker: the package sits at <checkout>/simbi and the configs at
    # <checkout>/simbi_configs, so the location is a relative fact about the layout and
    # needs nothing written at install time.
    if not _CHECKOUT_CONFIGS.is_dir():
        return
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIMBI_CONFIG_PATH", raising=False)
    assert _CHECKOUT_CONFIGS.resolve() in {r.resolve() for r in config_roots()}
