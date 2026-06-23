# =============================================================================
# simbi/cli/actions.py
#
# custom argparse actions for cli arguments.
# includes actions for compute mode selection, gpu block dimensions,
# version printing, and config discovery.
# =============================================================================
import os
from argparse import SUPPRESS, Action, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, Sequence

from .utils.colors import bcolors

# directories that never hold simbi configs. pruned from the recursive walk so a
# symlinked config tree (e.g. simbi_configs/science -> a sibling repo) does not
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

# a simbi config DEFINES a SimbiProblem subclass, so the class name appears in
# the file text. content-filtering on this marker keeps the listing precise —
# a symlinked tree's own non-config scripts (plot helpers, analysis) are excluded
# without importing (which would be slow and run module side effects).
_CONFIG_MARKER = "SimbiProblem"


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
        import os

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
            raise ValueError(
                "GPU block dimensions must be specified as 1, 2, or 3 integers."
            )


class print_the_version(Action):
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


class print_available_configs(Action):
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


def _is_config_file(path: Path) -> bool:
    """a candidate is a real config if it is an importable, non-private module
    that references SimbiProblem. dunder / private files (setup, conftest,
    __init__, _helpers) are skipped."""
    stem = path.stem
    if stem.startswith("_") or stem in ("setup", "conftest"):
        return False
    try:
        return _CONFIG_MARKER in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


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
    with open(Path(__file__).resolve().parent.parent / "gitrepo_home.txt") as f:
        githome = f.read()

    # the repo's bundled configs plus a cwd-local `simbi_configs` (real or
    # symlinked), deduplicated by resolved path so the same file isn't listed
    # twice when invoked from the repo root.
    roots = [Path(githome).resolve() / "simbi_configs", Path("simbi_configs")]
    configs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in _find_configs(root):
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                configs.append(path)
    return configs
