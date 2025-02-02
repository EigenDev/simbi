import subprocess
import sys
from pathlib import Path

def type_check_input(file: Path) -> None:
    """Run mypy type checking on input file"""
    type_checker = subprocess.run(
        [sys.executable, 
         '-m', 
         'mypy', 
         '--strict',
         '--ignore-missing-imports',
         str(file)]
    )
    
    if type_checker.returncode != 0:
        raise TypeError(
            "\nYour configuration script failed type safety checks. " +
            "Please fix them or run with --no-type-check option"
        )