# =============================================================================
# test_census_reader.py
#
# the census reader against a synthetic checkpoint written in the layout the rust writer
# produces. what is worth gating here is not the plumbing but the two places a reader can be
# quietly wrong: the segment-to-bin reshape (a transposed profile is smooth and plausible)
# and the weighted means (a volume-weighted mean of the same field is a DIFFERENT number
# that also looks reasonable), plus the dropped-cell shortfall, which is invisible in the
# values themselves.
# =============================================================================

import h5py
import numpy as np
import pytest

from simbi.reader.census import (
    Census,
    CensusError,
    census_names,
    read_census,
    read_census_series,
)


def _write(
    path,
    name="shells",
    edges=(np.array([0.0, 1.0, 2.0, 3.0]),),
    axis_names=("r",),
    value_names=("mass", "mass_v", "mass_v2"),
    values=None,
    times=(0.0, 1.0),
    dropped=(0, 0),
    op="add",
    accumulated=None,
    n_samples=None,
    level=None,
):
    n_seg = int(np.prod([len(e) - 1 for e in edges])) if edges else 1
    n_val = len(value_names)
    if values is None:
        values = np.arange(len(times) * n_seg * n_val, dtype=float).reshape(
            len(times), n_seg, n_val
        )
    with h5py.File(path, "w") as h:
        g = h.create_group(f"census/{name}")
        g.attrs["n_segments"] = n_seg
        g.attrs["n_values"] = n_val
        g.attrs["value_names"] = ",".join(value_names)
        g.attrs["op"] = op
        g.attrs["node_count"] = 42
        # written only when asked for, so the reader's handling of a file predating
        # accumulation stays exercised by every other case here.
        if accumulated is not None:
            g.attrs["accumulated"] = int(accumulated)
        if n_samples is not None:
            counts = (
                [int(n_samples)] * len(times)
                if isinstance(n_samples, int)
                else [int(v) for v in n_samples]
            )
            g.create_dataset("n_samples", data=np.asarray(counts, dtype=np.uint64))
            g.create_dataset(
                "t_start", data=np.full(len(times), float(times[0]), dtype=float)
            )
        if level is not None:
            g.create_dataset("level", data=np.asarray(level, dtype=np.uint64))
            g.attrs["cadence"] = "per_level_step"
        g.create_dataset("time", data=np.asarray(times, dtype=float))
        g.create_dataset("values", data=np.asarray(values, dtype=float))
        g.create_dataset("dropped", data=np.asarray(dropped, dtype=np.uint64))
        for k, e in enumerate(edges):
            g.attrs[f"axis{k}_name"] = axis_names[k]
            g.create_dataset(f"axis{k}_edges", data=np.asarray(e, dtype=float))
    return path


def test_reads_names_shape_and_edges(tmp_path):
    p = _write(tmp_path / "c.h5")
    assert census_names(p) == ("shells",)
    c = read_census(p, "shells")
    assert isinstance(c, Census)
    assert c.n_rows == 2
    assert c.bin_shape == (3,)
    assert c.value_names == ("mass", "mass_v", "mass_v2")
    assert c.axis_names == ("r",)
    np.testing.assert_allclose(c.bin_centers(0), [0.5, 1.5, 2.5])
    # (n_rows, *bin_shape, n_values)
    assert c.values.shape == (2, 3, 3)
    assert c.value("mass").shape == (2, 3)


def test_two_axes_reshape_is_last_axis_fastest(tmp_path):
    # the segment axis is stored FLAT. a reader that reshaped it the other way round would
    # produce a transposed profile — perfectly smooth, entirely wrong — so pin the order
    # against a segment table whose value IS its own flat index.
    edges = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0, 3.0]))  # 2 x 3 bins
    flat = np.arange(6, dtype=float).reshape(1, 6, 1)
    p = _write(
        tmp_path / "c.h5",
        edges=edges,
        axis_names=("r", "theta"),
        value_names=("idx",),
        values=flat,
        times=(0.0,),
        dropped=(0,),
    )
    c = read_census(p, "shells")
    assert c.bin_shape == (2, 3)
    got = c.value("idx")[0]
    # last axis varying fastest: segment = i*3 + j
    want = np.arange(6, dtype=float).reshape(2, 3)
    np.testing.assert_allclose(got, want)


def test_favre_mean_differs_from_the_volume_weighted_mean(tmp_path):
    # the reason a census stores sums instead of means. one bin holds two parcels of equal
    # volume but very different mass; the mass-weighted mean is dominated by the heavy one,
    # while the plain average of the two parcel velocities is not. a reader that accumulated
    # or averaged means would report the latter.
    m1, v1 = 9.0, 1.0   # heavy, slow
    m2, v2 = 1.0, 11.0  # light, fast
    mass = m1 + m2
    mass_v = m1 * v1 + m2 * v2
    mass_v2 = m1 * v1 * v1 + m2 * v2 * v2
    values = np.array([[[mass, mass_v, mass_v2]]], dtype=float)
    p = _write(
        tmp_path / "c.h5",
        edges=(np.array([0.0, 1.0]),),
        values=values,
        times=(0.0,),
        dropped=(0,),
    )
    c = read_census(p, "shells")

    favre = c.favre("mass_v", "mass")[0, 0]
    unweighted = 0.5 * (v1 + v2)
    assert favre == pytest.approx(2.0)          # (9*1 + 1*11)/10
    assert unweighted == pytest.approx(6.0)     # the number the wrong reduction gives
    assert abs(favre - unweighted) > 1.0, "the two means coincide here; the gate is vacuous"

    # variance from the same sums: <v^2> - <v>^2 = (9*1 + 1*121)/10 - 4 = 13 - 4 = 9.
    assert c.favre_variance("mass_v2", "mass_v", "mass")[0, 0] == pytest.approx(9.0)


def test_empty_bin_is_nan_not_zero(tmp_path):
    # no matter landed in the bin, so the mean is undefined. zero would read as a physical
    # measurement of a velocity that is actually absent.
    values = np.array([[[0.0, 0.0, 0.0], [4.0, 8.0, 16.0]]], dtype=float)
    p = _write(
        tmp_path / "c.h5",
        edges=(np.array([0.0, 1.0, 2.0]),),
        values=values,
        times=(0.0,),
        dropped=(0,),
    )
    c = read_census(p, "shells")
    got = c.favre("mass_v", "mass")[0]
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(2.0)


def test_dropped_cells_are_reported_not_hidden(tmp_path):
    p = _write(tmp_path / "c.h5", dropped=(0, 7))
    c = read_census(p, "shells")
    c.assert_fully_binned(sample=0)  # that sample is clean
    with pytest.raises(CensusError, match="dropped 7 cell"):
        c.assert_fully_binned()


def test_segment_count_disagreeing_with_the_edges_is_refused(tmp_path):
    # the file's own two descriptions of the bin layout must agree; if they do not, no
    # reshape of the segment axis is trustworthy and guessing one would fabricate a profile.
    p = _write(tmp_path / "c.h5")
    with h5py.File(p, "r+") as h:
        del h["census/shells/axis0_edges"]
        h["census/shells"].create_dataset("axis0_edges", data=np.array([0.0, 1.0, 2.0]))
    with pytest.raises(CensusError, match="stored segments but the axis edges"):
        read_census(p, "shells")


def test_derived_means_are_refused_on_a_non_additive_census(tmp_path):
    # an extremum census carries no sums, so a weighted mean of its accumulators is not a
    # weighted mean of anything.
    p = _write(tmp_path / "c.h5", op="max")
    c = read_census(p, "shells")
    with pytest.raises(CensusError, match="reduces with 'max'"):
        c.favre("mass_v", "mass")
    with pytest.raises(CensusError, match="reduces with 'max'"):
        c.total("mass")


def test_unknown_names_name_what_is_available(tmp_path):
    c = read_census(_write(tmp_path / "c.h5"), "shells")
    with pytest.raises(CensusError, match="no accumulator 'nope'"):
        c.value("nope")
    with pytest.raises(CensusError, match="no census 'missing'"):
        read_census(_write(tmp_path / "d.h5"), "missing")


def test_total_sums_over_bins_only(tmp_path):
    values = np.array([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=float)
    p = _write(tmp_path / "c.h5", values=values, times=(0.0,), dropped=(0,))
    c = read_census(p, "shells")
    np.testing.assert_allclose(c.total("mass"), [6.0])


def test_a_file_predating_accumulation_reads_as_a_per_sample_history(tmp_path):
    # the compatibility case: a checkpoint written before the mode existed carries none of its
    # metadata, and by definition it is not accumulated. defaulting is right here — the sample
    # count of a per-sample history IS its row count — but it must not be defaulted the other way,
    # which would make every old file claim to be one folded row.
    c = read_census(_write(tmp_path / "c.h5", times=(0.0, 1.0, 2.0), dropped=(0, 0, 0)), "shells")
    assert c.accumulated is False
    assert c.n_rows == 3
    assert c.n_samples is None and c.level is None
    assert c.levels == (0,)


def test_an_accumulated_row_averages_back_by_its_sample_count(tmp_path):
    # the whole point of the mode: the samples are gone, so the count is the only thing that can
    # turn the stored running sum back into a mean. an average taken over the ROWS instead would
    # divide by one and report the sum as though it were the average.
    values = np.array([[[10.0], [20.0], [30.0]]])  # one row, three bins, one accumulator
    p = _write(
        tmp_path / "acc.h5",
        value_names=("mass",),
        values=values,
        times=(4.0,),
        dropped=(0,),
        accumulated=True,
        n_samples=5,
    )
    c = read_census(p, "shells")
    assert c.accumulated is True
    assert c.n_rows == 1
    np.testing.assert_array_equal(c.n_samples, [5])
    np.testing.assert_allclose(c.time_average("mass"), [2.0, 4.0, 6.0])
    # and the per-sample form of the same call is the mean over rows, so one name means one thing.
    plain = read_census(
        _write(
            tmp_path / "plain.h5",
            value_names=("mass",),
            values=np.array([[[1.0], [2.0], [3.0]], [[3.0], [4.0], [5.0]]]),
            times=(0.0, 1.0),
        ),
        "shells",
    )
    np.testing.assert_allclose(plain.time_average("mass"), [2.0, 3.0, 4.0])


def test_a_file_claiming_accumulation_with_many_rows_is_refused(tmp_path):
    # the metadata and the data must agree. an accumulating history folds every sample into one
    # row, so a file marked accumulated that stores several is describing something the writer
    # cannot produce — and `time_average` would silently read row zero as though it were the whole
    # segment.
    p = _write(tmp_path / "bad.h5", accumulated=True, n_samples=9, times=(0.0, 1.0))
    with pytest.raises(CensusError, match="accumulating"):
        read_census(p, "shells")


def test_a_time_average_of_an_extremal_census_is_refused(tmp_path):
    # a running max over a segment is already the answer for that segment; dividing it by a sample
    # count is not an average of anything, and the result would look like a perfectly ordinary
    # profile.
    p = _write(tmp_path / "mx.h5", op="max", accumulated=True, n_samples=4,
               value_names=("peak",),
               values=np.array([[[1.0], [2.0], [3.0]]]), times=(1.0,), dropped=(0,))
    c = read_census(p, "shells")
    with pytest.raises(CensusError):
        c.time_average("peak")


def test_per_level_rows_are_selected_before_any_time_series_analysis(tmp_path):
    # rows from different levels are not interchangeable: a level subcycles, so it contributes
    # several rows per root step, each covering only the volume that level resolves. averaging them
    # together weights the answer by the subcycle ratio — a number that is smooth and plausible and
    # is a property of the timestepper rather than the flow.
    p = _write(
        tmp_path / "lv.h5",
        value_names=("mass",),
        values=np.array([[[1.0], [1.0], [1.0]], [[9.0], [9.0], [9.0]], [[9.0], [9.0], [9.0]]]),
        times=(1.0, 0.5, 1.0),
        dropped=(0, 0, 0),
        level=(0, 1, 1),
    )
    c = read_census(p, "shells")
    assert c.cadence == "per_level_step"
    assert c.levels == (0, 1)
    with pytest.raises(CensusError, match="for_level"):
        c.time_average("mass")

    fine = c.for_level(1)
    assert fine.n_rows == 2
    np.testing.assert_allclose(fine.time_average("mass"), [9.0, 9.0, 9.0])
    np.testing.assert_allclose(c.for_level(0).time_average("mass"), [1.0, 1.0, 1.0])
    with pytest.raises(CensusError, match="no rows from level 2"):
        c.for_level(2)


# =============================================================================
# joining a restart chain. the hazard is not the concatenation but the OVERLAP: a history is
# never cleared within a process, so a later checkpoint of the same job repeats every row an
# earlier one holds. an append would count those twice and the duplicates are undetectable
# afterwards, because a repeated sample is numerically indistinguishable from a real one.
# =============================================================================


def test_series_join_does_not_double_count_a_superset(tmp_path):
    # two checkpoints of ONE process: the second holds every row of the first plus more.
    vals = np.arange(4 * 3 * 3, dtype=float).reshape(4, 3, 3)
    early = _write(
        tmp_path / "e.h5", times=(0.0, 1.0), dropped=(0, 0), values=vals[:2]
    )
    late = _write(
        tmp_path / "l.h5",
        times=(0.0, 1.0, 2.0, 3.0),
        dropped=(0, 0, 0, 0),
        values=vals,
    )
    joined = read_census_series([early, late], "shells")
    assert joined.n_rows == 4, (
        f"the join produced {joined.n_rows} rows from a 2-row file and its 4-row superset; "
        "the shared rows were counted twice"
    )
    assert np.array_equal(joined.time, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(joined.values, vals)


def test_series_join_is_idempotent_and_order_free(tmp_path):
    a = _write(tmp_path / "a.h5", times=(0.0, 1.0))
    b = _write(tmp_path / "b.h5", times=(2.0, 3.0))
    once = read_census_series([a, b], "shells")
    assert once.n_rows == 4  # genuinely disjoint segments DO concatenate
    for paths in ([a, b, a, b], [b, a], [a, a, b]):
        again = read_census_series(paths, "shells")
        assert np.array_equal(again.time, once.time)
        assert np.array_equal(again.values, once.values)


def test_series_join_keys_on_level_not_time_alone(tmp_path):
    # a per-level census samples several levels at the SAME time; keying on time alone would
    # collapse them into one row and silently discard every level but one.
    a = _write(
        tmp_path / "a.h5", times=(0.0, 0.0, 1.0, 1.0), dropped=(0,) * 4, level=(0, 1, 0, 1)
    )
    joined = read_census_series([a], "shells")
    assert joined.n_rows == 4, "rows from different levels at one time were merged"
    assert joined.for_level(1).n_rows == 2


def test_series_join_refuses_a_changed_registration(tmp_path):
    a = _write(tmp_path / "a.h5", times=(0.0,), dropped=(0,))
    b = _write(
        tmp_path / "b.h5",
        times=(1.0,),
        dropped=(0,),
        edges=(np.array([0.0, 1.5, 3.0]),),
    )
    with pytest.raises(CensusError, match="bins axis 0 differently"):
        read_census_series([a, b], "shells")

    c = _write(
        tmp_path / "c.h5",
        times=(1.0,),
        dropped=(0,),
        value_names=("mass", "mass_v", "mass_w"),
    )
    with pytest.raises(CensusError, match="accumulators"):
        read_census_series([a, c], "shells")


def test_series_join_refuses_an_accumulating_history(tmp_path):
    # an accumulating row is a reduction over a span, so two rows may be disjoint segments to
    # merge or one may contain the other -- (level, time) cannot tell those apart.
    a = _write(
        tmp_path / "a.h5", times=(1.0,), dropped=(0,), accumulated=True, n_samples=7
    )
    with pytest.raises(CensusError, match="accumulating"):
        read_census_series([a, a], "shells")
