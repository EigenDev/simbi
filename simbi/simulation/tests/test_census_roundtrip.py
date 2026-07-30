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


def _census_payload(with_axis: bool, **controls):
    """the binned mass plus a mass-weighted position, so a Favre mean has something to
    divide. both read the cell measure, which is what makes them extensive."""
    g = expr.ExprGraph()
    x = expr.variable("x", g)
    mass = expr.density(g) * expr.cell_volume(g)
    axes = [expr.BinAxis("x", x, [0.0, 0.25, 0.5, 0.75, 1.0])] if with_axis else []
    return expr.Census(
        name="profile", axes=axes, values={"mass": mass, "mass_x": mass * x}, **controls
    ).serialize()


def _problem(data_dir: str, with_axis: bool, **controls):
    """a 1D cartesian box on [0, 1] carrying one census."""
    from simbi.simulation.examples.sod import SodProblem

    payload = _census_payload(with_axis, **controls)

    class _CensusSod(SodProblem):
        @property
        def census_expressions(self):
            return [payload]

    p = _CensusSod.from_cli(["--resolution", str(_N)])
    p.data_directory = data_dir
    p.checkpoint_interval = 1.0e9
    return p


def _run(with_axis: bool, steps: int = 2, **controls):
    d = tempfile.mkdtemp() + "/"
    runner.run(_problem(d, with_axis, **controls), compute_mode="cpu", max_steps=steps)
    finals = glob.glob(os.path.join(d, "*final*.h5"))
    assert finals, "the census run produced no checkpoint"
    return read_census(finals[0], "profile")


@needs_backend
def test_binned_census_round_trips_through_the_reader() -> None:
    c = _run(with_axis=True)

    # the premise: a census that dropped cells is under-covering its domain, and every
    # number it reports would then be a partial sum wearing the shape of a total.
    c.assert_fully_binned()

    assert c.value_names == ("mass", "mass_x")
    assert c.axis_names == ("x",)
    assert c.bin_shape == (4,)
    assert c.n_rows >= 1
    np.testing.assert_allclose(c.bin_centers(0), [0.125, 0.375, 0.625, 0.875])

    # the binned mass sums to the domain mass. the sod initial state is piecewise uniform,
    # so this is a number the census cannot get right by accident: it requires the cell
    # measure (the `dv` leaf) and the bin assignment to both be right.
    mass = c.value("mass")[0]
    assert mass.shape == (4,)
    assert float(mass.sum()) == pytest.approx(float(c.total("mass")[0]))

    # every bin holds a quarter of the box, and each is non-empty — a profile with an empty
    # bin would make the Favre mean NaN and the check vacuous.
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
    assert c.value("mass").shape == (c.n_rows,)
    assert float(c.value("mass")[0]) > 0.0


# =============================================================================
# the registration CONTROLS, end to end
#
# each of these is an inert default that a stale backend accepts and ignores: the wire fields
# carry serde defaults, so a run declaring them produces entirely correct output at the wrong
# cost or the wrong storage, and nothing in the numbers says so. the only place that can be
# caught is here, where the declaration and the written file meet.
# =============================================================================

_STEPS = 8


@needs_backend
def test_an_accumulating_census_writes_one_row_that_is_the_sum_of_the_samples() -> None:
    plain = _run(with_axis=True, steps=_STEPS)
    acc = _run(with_axis=True, steps=_STEPS, accumulate=True)

    # the premise: the per-sample run must record several rows, or "folded into one" is what
    # the reference did too and the comparison is vacuous.
    assert plain.n_rows > 1, (
        f"the reference run recorded {plain.n_rows} row(s) over {_STEPS} steps; with nothing "
        "to fold, an accumulating run trivially matches it"
    )

    assert acc.accumulated is True, (
        "the checkpoint is not marked accumulating. the wire field carries a serde default, so "
        "a backend that ignored it would write a per-sample history that reads as a valid census"
    )
    assert acc.n_rows == 1
    assert int(acc.n_samples.sum()) == plain.n_rows

    # exactness: the fold is the same additive reduce, so the row IS the sum of the samples.
    np.testing.assert_allclose(
        acc.value("mass")[0], plain.value("mass").sum(axis=0), rtol=1e-12
    )
    # and the count divides it back to the mean the samples would have given.
    np.testing.assert_allclose(
        acc.time_average("mass"), plain.time_average("mass"), rtol=1e-12
    )


@needs_backend
def test_a_declared_sample_interval_reaches_the_backend() -> None:
    dense = _run(with_axis=False, steps=_STEPS)
    # an interval far above the whole run's span: only the first sample is ever due.
    sparse = _run(with_axis=False, steps=_STEPS, sample_interval=1.0e9)

    assert dense.n_rows > 1, (
        f"the unthrottled run recorded {dense.n_rows} row(s); with one sample either way the "
        "interval cannot be shown to do anything"
    )
    assert sparse.n_rows == 1, (
        f"a sample interval of 1e9 over a run of order unity still recorded {sparse.n_rows} "
        "rows. the field carries a serde default, so a backend that dropped it samples every "
        "step and the only symptom is the cost"
    )


@needs_backend
def test_the_declared_cadence_travels_with_the_file() -> None:
    # a single-level run samples identically under either cadence — what is checkable here is
    # that the declaration reached the backend and was recorded, so a consumer reading the file
    # knows which sampling produced it. the per-level BEHAVIOR is gated on a refined hierarchy
    # in the rust suite, where a second level exists to sample at its own rate.
    c = _run(with_axis=False, steps=2, cadence=expr.Cadence.PER_LEVEL_STEP)
    assert c.cadence == "per_level_step"
    assert c.levels == (0,), "a single-level run produces root-level rows only"

    default = _run(with_axis=False, steps=2)
    assert default.cadence == "root_step"

