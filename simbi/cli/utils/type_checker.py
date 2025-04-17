import subprocess
import sys
import time
from simbi import logger, bcolors
from pathlib import Path
from halo import Halo


def type_check_input(file: Path) -> None:
    """Run mypy type checking on input file"""
    logger.info(f"Running type checker on configuration script {file}")
    spinner = Halo(text="Type checking in progress", spinner="dots")
    spinner.start()

    try:
        type_checker = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--strict",
                "--ignore-missing-imports",
                str(file),
            ],
            capture_output=True,
            text=True,
        )

        if type_checker.returncode != 0:
            spinner.stop()
            logger.error(f"Type checking failed: {type_checker.stdout}")
            raise TypeError(
                "\nYour configuration script failed type safety checks. "
                + "Please fix them or run with --no-type-check option"
            )

        output = type_checker.stdout
        spinner.stop_and_persist(
            symbol="\n", text=f"{bcolors.GREEN}{output}{bcolors.ENDC}"
        )
        spinner = Halo(text="Moving forward with simulation setup", spinner="dots")
        spinner.start()
        # wait a few seconds before moving forward
        # to give user a chance to read the message
        time.sleep(1)
        spinner.succeed("Type checking completed successfully")
    except Exception as e:
        spinner.stop()
        raise e
