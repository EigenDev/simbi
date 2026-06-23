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

    # setup.py / conftest.py are tooling, not configs.
    for name in ("setup.py", "conftest.py"):
        f = tmp_path / name
        f.write_text(_CONFIG_SRC)
        assert not _is_config_file(f)
