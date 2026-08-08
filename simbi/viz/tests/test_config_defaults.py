# =============================================================================
# test_config_defaults.py
#
# a plot built by the cli and the same plot built by a script must render the
# same way. the two reach the config by different routes -- argparse defaults
# on one side, model defaults on the other -- and where those disagree the
# divergence is invisible: both produce a plot, of the same field, at the same
# time, drawn by a different renderer with different performance and different
# bugs.
# =============================================================================
from argparse import Namespace

from simbi.viz.config import RefinementConfig
from simbi.viz.pipeline.conversion import refinement_config_from_args


def test_an_absent_flag_leaves_the_model_default_standing() -> None:
    assert refinement_config_from_args(Namespace()) == RefinementConfig()


def test_the_default_renderer_is_the_quadmesh() -> None:
    """the polygon build is a python loop over cells and costs seconds a frame
    at production resolution; nothing should select it without being asked."""
    assert RefinementConfig().render_mode == "pcolormesh"


def test_an_explicit_flag_still_wins() -> None:
    chosen = refinement_config_from_args(Namespace(render_mode="polygons"))

    assert chosen.render_mode == "polygons"
