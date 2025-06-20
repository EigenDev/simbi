import argparse
from typing import Any


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[0;30m"
    DARK_GRAY = "\033[1;30m"
    RED = "\033[0;31m"
    LIGHT_RED = "\033[1;31m"
    GREEN = "\033[0;32m"
    LIGHT_GREEN = "\033[1;32m"
    ORANGE = "\033[0;33m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    LIGHT_BLUE = "\033[1;34m"
    PURPLE = "\033[0;35m"
    LIGHT_PURPLE = "\033[1;35m"
    CYAN = "\033[0;36m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_GRAY = "\033[0;37m"
    WHITE = "\033[1;37m"


__all__ = ["bcolors"]
