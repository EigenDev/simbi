# =============================================================================
# simbi/cli/actions.py
#
# custom argparse actions for cli arguments.
# includes actions for compute mode selection, gpu block dimensions,
# version printing, and config discovery.
# =============================================================================
import ast
import os
import warnings
from argparse import SUPPRESS, Action, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, Sequence

from .utils.colors import bcolors

# directories that never hold simbi configs. pruned from the recursive walk so a
# symlinked config tree (e.g., simbi_configs/science -> a sibling repo) does not
# drag in its virtualenv / caches / build artifacts. any hidden dir (leading
# dot, catches .venv/.git/.tox/...) is pruned too.
_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        "site-packages",
        "node_modules",
        "build",
        "dist",
        "venv",
        "egg-info",
    }
)

# a simbi config DEFINES a SimbiProblem subclass — directly, or by subclassing a
# base config it imports from another simbi_configs module. `_defines_a_config`
# recognizes both from the ast without importing (importing every candidate would
# be slow and run module side effects across a symlinked tree). config bases live
# only under simbi_configs, so a class subclassing a simbi_configs import is a
# config subclass; plot/analysis helpers subclass neither and are excluded.
_SIMBI_PROBLEM_BASE = "SimbiProblem"


class ComputeModeAction(Action):
    """Sets computation mode (cpu/gpu/omp)"""

    def __init__(
        self, option_strings: Sequence[str], dest: str, **kwargs: Any
    ) -> None:
        super().__init__(
            option_strings, dest, nargs=0, default=SUPPRESS, **kwargs
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        setattr(namespace, "compute_mode", self.const)


class RegisterGPUBlockDimensions(Action):
    """takes the user input, and sets the environment variables for GPU block dimensions"""

    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(option_strings, dest=dest, **kwargs)

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Sequence[int] | None,
        option_string: str | None = None,
    ):
        if values is not None and any(v <= 0 for v in values):
            parser.error(
                f"{option_string or 'gpu block dims'}: every block dimension must "
                f"be a positive integer (got {list(values)})"
            )
        if values is not None and len(values) == 3:
            os.environ["BLOCK_X"] = str(values[0])
            os.environ["BLOCK_Y"] = str(values[1])
            os.environ["BLOCK_Z"] = str(values[2])
        elif values is not None and len(values) == 2:
            os.environ["BLOCK_X"] = str(values[0])
            os.environ["BLOCK_Y"] = str(values[1])
            os.environ["BLOCK_Z"] = "1"
        elif values is not None and len(values) == 1:
            os.environ["BLOCK_X"] = str(values[0])
            os.environ["BLOCK_Y"] = "1"
            os.environ["BLOCK_Z"] = "1"
        else:
            parser.error(
                f"{option_string or 'gpu block dims'}: specify 1, 2, or 3 "
                "positive integers"
            )


class PrintVersionAction(Action):
    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(
            option_strings, dest, nargs=0, default=SUPPRESS, **kwargs
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        from simbi import __version__ as version

        print(f"SIMBI version {version}")
        parser.exit()


class PrintAvailableConfigsAction(Action):
    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(
            option_strings, dest, nargs=0, default=SUPPRESS, **kwargs
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ):
        available_configs = get_available_configs()
        available_configs = sorted(
            [Path(conf).stem for conf in available_configs]
        )

        print(
            "Available configs are:\n{}".format(
                "".join(
                    f"> {bcolors.BOLD}{conf}{bcolors.ENDC}\n"
                    for conf in available_configs
                )
            )
        )
        parser.exit()


def _defines_a_config(tree: ast.Module) -> bool:
    """true if the module defines a class subclassing SimbiProblem, directly or via
    a base imported from a simbi_configs module (a sibling config it extends)."""
    config_bases: set[str] = {_SIMBI_PROBLEM_BASE}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "simbi_configs" in node.module:
            for alias in node.names:
                config_bases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in config_bases:
                    return True
                if isinstance(base, ast.Attribute) and base.attr in config_bases:
                    return True
    return False


def _is_config_file(path: Path) -> bool:
    """a candidate is a real config if it is an importable, non-private module that
    defines a SimbiProblem subclass (see `_defines_a_config`). dunder / private
    files (setup, conftest, __init__, _helpers) are skipped."""
    stem = path.stem
    if stem.startswith("_") or stem in ("setup", "conftest"):
        return False
    try:
        with warnings.catch_warnings():
            # classifying a candidate must stay silent: the scan inspects structure,
            # it does not import or execute. a config tree's own latent warnings
            # (invalid string escapes, deprecations) are surfaced when that config is
            # actually run, not when every `simbi run` lists the available configs.
            warnings.simplefilter("ignore")
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError, ValueError):
        return False
    return _defines_a_config(tree)


def _find_configs(root: Path) -> list[Path]:
    """walk `root` for config files, PRUNING excluded / hidden directories so a
    symlinked tree's virtualenv and caches are never descended into. follows
    symlinks (the `science` config link is intentional); the prune keeps it cheap.
    """
    root = Path(root)
    if not root.exists():
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _EXCLUDED_DIRS
            and not d.startswith(".")
            and not d.endswith(".egg-info")
        ]
        for fn in filenames:
            if fn.endswith(".py"):
                path = Path(dirpath) / fn
                if _is_config_file(path):
                    found.append(path)
    return found


def get_available_configs():
    # the repo's bundled configs plus a cwd-local `simbi_configs` (real or
    # symlinked), deduplicated by resolved path so the same file isn't listed
    # twice when invoked from the repo root. the repo-home marker is written at
    # install time; an install without it (or an unreadable marker) degrades to
    # cwd-local discovery instead of killing every command with a traceback.
    roots = [Path("simbi_configs")]
    try:
        marker = Path(__file__).resolve().parent.parent / "gitrepo_home.txt"
        githome = marker.read_text().strip()
        if githome:
            roots.insert(0, Path(githome).resolve() / "simbi_configs")
    except OSError:
        pass
    configs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in _find_configs(root):
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                configs.append(path)
    return configs
