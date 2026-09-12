# =============================================================================
# simbi/cli/commands/launch/__init__.py
#
# the `simbi launch` command: one problem evolved by several worker processes
# over the fabric. the parser registers the layout flags; the executor either
# spawns the workers (the launcher role) or runs one worker (the worker role).
# =============================================================================
from .parser import setup_parser

__all__ = ["setup_parser"]
