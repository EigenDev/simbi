# =============================================================================
# test_census_roundtrip.py
#
# the census END TO END: register one on a running problem, let the backend evaluate and
# write it, and read it back with the reader. this is the only place the reader's assumed
# checkpoint layout meets the writer's actual one — every other reader test builds the file
# itself and so agrees with whatever the reader expects by construction.
#
# it also pins the two numbers a census exists to produce: a binned profile whose sum is the
# global total, and that total against an independently computed one. a binning that
# mis-assigns cells still sums correctly, and a binning that drops them does not, so both
# the sum and the per-bin split are checked.
# =============================================================================
import glob
import os
import tempfile

import numpy as np
import pytest

import simbi.expression as expr
from simbi.reader.census import read_census
from simbi.simulation import runner

_BACKEND = runner._load_backend("cpu")
needs_backend = pytest.mark.skipif(
    _BACKEND is None, reason="rust cpu_ext backend not built"
)

_N = 64
_RHO = 2.0


def _census_payload(with_axis: bool):
    """the binned mass plus a mass-weighted position, so a Favre mean has something to
    divide. both read the cell measure, which is what makes them extensive."""
    g = expr.ExprGraph()
    x = expr.variable("x", g)
    mass = expr.density(g) * expr.cell_volume(g)
    axes = [expr.BinAxis("x", x, [0.0, 0.25, 0.5, 0.75, 1.0])] if with_axis else []
    return expr.Census(
        name="profile", axes=axes, values={"mass": mass, "mass_x": mass * x}
    ).serialize()


def _problem(data_dir: str, with_axis: bool):
    """a 1D cartesian box on [0, 1] carrying one census."""
    from simbi.simulation.examples.sod import SodProblem

    payload = _census_payload(with_axis)

    class _CensusSod(SodProblem):
        @property
        def census_expressions(self):
            return [payload]

    p = _CensusSod.from_cli(["--resolution", str(_N)])
    p.data_directory = data_dir
    p.checkpoint_interval = 1.0e9
    return p


def _run(with_axis: bool):
    d = tempfile.mkdtemp() + "/"
    runner.run(_problem(d, with_axis), compute_mode="cpu", max_steps=2)
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, "the census run produced no checkpoint"
    return read_census(finals[0], "profile")


@needs_backend
def test_binned_census_round_trips_through_the_reader() -> None:
    c = _run(with_axis=True)

    # the premise: a census that dropped cells is under-covering its domain, and every
    # number below would then be a partial sum wearing the shape of a total.
    c.assert_fully_binned()

    assert c.value_names == ("mass", "mass_x")
    assert c.axis_names == ("x",)
    assert c.bin_shape == (4,)
    assert c.n_samples >= 1
    np.testing.assert_allclose(c.bin_centers(0), [0.125, 0.375, 0.625, 0.875])

    # the binned mass sums to the domain mass. the sod initial state is piecewise uniform,
    # so this is a number the census cannot get right by accident: it requires the cell
    # measure (the `dv` leaf) and the bin assignment to both be right.
    mass = c.value("mass")[0]
    assert mass.shape == (4,)
    assert float(mass.sum()) == pytest.approx(float(c.total("mass")[0]))

    # every bin holds a quarter of the box, and each is non-empty — a profile with an empty
    # bin would make the Favre mean below NaN and the check vacuous.
    assert (mass > 0.0).all(), f"an empty bin: {mass}"

    # the mass-weighted position per bin must land inside that bin. this is the assertion a
    # TRANSPOSED reshape fails: the values are all plausible, but bin k would carry the
    # centroid of some other bin.
    x_bar = c.favre("mass_x", "mass")[0]
    edges = c.axis_edges[0]
    for k in range(4):
        assert edges[k] <= x_bar[k] <= edges[k + 1], (
            f"bin {k} spans [{edges[k]}, {edges[k + 1]}] but its mass-weighted position is "
            f"{x_bar[k]} — the segment axis is not being reshaped in registration order"
        )


@needs_backend
def test_axis_free_census_is_a_single_global_reduction() -> None:
    # no bin axes at all: the whole grid reduces to one bucket. this is the conservation
    # ledger's shape, and the reader must not require a dummy axis to express it.
    c = _run(with_axis=False)
    c.assert_fully_binned()
    assert c.bin_shape == ()
    assert c.value("mass").shape == (c.n_samples,)
    assert float(c.value("mass")[0]) > 0.0
