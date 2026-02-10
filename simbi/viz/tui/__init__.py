# =============================================================================
# simbi/viz/tui/__init__.py
#
# interactive terminal ui for plot configuration.
# collects plot parameters, then hands off to matplotlib for rendering.
# =============================================================================
from .app import run_plot_tui

__all__ = ["run_plot_tui"]
