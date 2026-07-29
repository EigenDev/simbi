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

from simbi.reader.census import Census, CensusError, census_names, read_census


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
    assert c.n_samples == 2
    assert c.bin_shape == (3,)
    assert c.value_names == ("mass", "mass_v", "mass_v2")
    assert c.axis_names == ("r",)
    np.testing.assert_allclose(c.bin_centers(0), [0.5, 1.5, 2.5])
    # (n_samples, *bin_shape, n_values)
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
