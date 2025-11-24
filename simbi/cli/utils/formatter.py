# =============================================================================
# simbi/cli/utils/formatter.py
#
# help formatter for cli output. uses rich_argparse if available.
# =============================================================================
try:
    from rich_argparse import RichHelpFormatter as HelpFormatter
except ImportError:
    pass
