from argparse import Action, ArgumentParser, Namespace, SUPPRESS
from typing import Optional, Any, Sequence
from pathlib import Path
from ..detail import bcolors


class ComputeModeAction(Action):
    """Sets computation mode (cpu/gpu/omp)"""

    def __init__(self, option_strings: Sequence[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=SUPPRESS, **kwargs)

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
            os.environ["GPU_BLOCK_X"] = str(values[0])
            os.environ["GPU_BLOCK_Y"] = str(values[1])
            os.environ["GPU_BLOCK_Z"] = str(values[2])
        elif values is not None and len(values) == 2:
            os.environ["GPU_BLOCK_X"] = str(values[0])
            os.environ["GPU_BLOCK_Y"] = str(values[1])
            os.environ["GPU_BLOCK_Z"] = "1"
        elif values is not None and len(values) == 1:
            os.environ["GPU_BLOCK_X"] = str(values[0])
            os.environ["GPU_BLOCK_Y"] = "1"
            os.environ["GPU_BLOCK_Z"] = "1"
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
        available_configs = sorted([Path(conf).stem for conf in available_configs])

        print(
            "Available configs are:\n{}".format(
                "".join(
                    f"> {bcolors.BOLD}{conf}{bcolors.ENDC}\n"
                    for conf in available_configs
                )
            )
        )
        parser.exit()


def get_available_configs():
    with open(Path(__file__).resolve().parent.parent / "gitrepo_home.txt") as f:
        githome = f.read()

    configs_src = Path(githome).resolve() / "simbi_configs"
    pkg_configs = [file for file in configs_src.rglob("*.py")]
    soft_paths = [
        soft_path
        for soft_path in (Path("simbi_configs")).glob("*")
        if soft_path.is_symlink()
    ]
    soft_configs = [file for path in soft_paths for file in path.rglob("*.py")]
    soft_configs += [
        file
        for file in Path("simbi_configs").resolve().rglob("*.py")
        if file not in pkg_configs
    ]

    return pkg_configs + soft_configs
