# =============================================================================
# test_print_mode_title.py
#
# the print (publication) rendering: `--print` builds a FigureConfig with
# show_time = False, and set_title then emits the bare title — no ", t=..."
# suffix — while the default keeps the time stamp. gated at both layers so a
# regression in either the flag wiring or the title formatting is named.
# =============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simbi.viz.config import FigureConfig
from simbi.viz.formatting import set_title


def _title_for(config: FigureConfig, time: float) -> str:
    fig, ax = plt.subplots()
    try:
        set_title(ax, fig, config, time)
        return ax.get_title()
    finally:
        plt.close(fig)


def test_default_title_carries_the_time_stamp() -> None:
    title = _title_for(FigureConfig(title="Decay", time_units="s"), 3.21)
    assert title == "Decay, t=3.21 s"


def test_print_mode_title_is_bare() -> None:
    title = _title_for(
        FigureConfig(title="Decay", time_units="s", show_time=False), 3.21
    )
    assert title == "Decay"


def test_print_flag_flows_into_the_figure_config() -> None:
    from argparse import Namespace

    from simbi.viz.pipeline.conversion import figure_config_from_args

    on = figure_config_from_args(Namespace(**{"print": True}))
    off = figure_config_from_args(Namespace())
    assert on.show_time is False
    assert off.show_time is True
