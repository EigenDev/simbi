# =============================================================================
# test_dev_clean.py
#
# cleanup must reclaim cargo artifacts without deleting the installed python
# extension; extension removal belongs exclusively to uninstall.
# =============================================================================

from types import SimpleNamespace

import dev


def test_clean_runs_cargo_clean_without_removing_extensions(monkeypatch):
    commands = []
    extension_removals = []

    monkeypatch.setattr(dev, "_require_cargo", lambda: None)
    monkeypatch.setattr(
        dev,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        dev,
        "_remove_extensions",
        lambda: extension_removals.append(True),
    )

    dev.clean_command(SimpleNamespace(all=False, verbose=False))

    assert commands == [
        [
            "cargo",
            "clean",
            "--manifest-path",
            str(dev.SRC / "Cargo.toml"),
        ]
    ]
    assert extension_removals == []
